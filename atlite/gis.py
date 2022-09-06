# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2020 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Functions for Geographic Information System.
"""

import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import scipy.sparse
import geopandas as gpd
import rasterio as rio
import rasterio.warp
import multiprocessing as mp

from collections import OrderedDict
from pathlib import Path
from warnings import warn, catch_warnings, simplefilter
from pyproj import CRS, Transformer
from shapely.ops import transform
from rasterio.warp import reproject, transform_bounds
from rasterio.mask import mask
from rasterio.features import geometry_mask
from scipy.ndimage.morphology import binary_dilation as dilation
from numpy import isin, empty
from shapely.strtree import STRtree
from tqdm import tqdm


import logging

logger = logging.getLogger(__name__)


def get_coords(x, y, time, dx=0.25, dy=0.25, dt="h", **kwargs):
    """
    Create an cutout coordinate system on the basis of slices and step sizes.

    Parameters
    ----------
    x : slice
        Numerical slices with lower and upper bound of the x dimension.
    y : slice
        Numerical slices with lower and upper bound of the y dimension.
    time : slice
        Slice with strings with lower and upper bound of the time dimension.
    dx : float, optional
        Step size of the x coordinate. The default is 0.25.
    dy : float, optional
        Step size of the y coordinate. The default is 0.25.
    dt : str, optional
        Frequency of the time coordinate. The default is 'h'. Valid are all
        pandas offset aliases.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with x, y and time variables, representing the whole coordinate
        system.
    """
    x = slice(*sorted([x.start, x.stop]))
    y = slice(*sorted([y.start, y.stop]))

    ds = xr.Dataset(
        {
            "x": np.round(np.arange(-180, 180, dx), 9),
            "y": np.round(np.arange(-90, 90, dy), 9),
            "time": pd.date_range(start="1959", end="now", freq=dt),
        }
    )
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    ds = ds.sel(x=x, y=y, time=time)
    return ds


def spdiag(v):
    """Create a sparse diagonal matrix from a 1-dimensional array."""
    N = len(v)
    inds = np.arange(N + 1, dtype=np.int32)
    return sp.sparse.csr_matrix((v, inds[:-1], inds), (N, N))


def reproject_shapes(shapes, crs1, crs2):
    """Project a collection of shapes from one crs to another."""
    transformer = Transformer.from_crs(crs1, crs2)

    def _reproject_shape(shape):
        return transform(transformer.transform, shape)

    if isinstance(shapes, pd.Series):
        return shapes.map(_reproject_shape)
    elif isinstance(shapes, dict):
        return OrderedDict((k, _reproject_shape(v)) for k, v in shapes.items())
    else:
        return list(map(_reproject_shape, shapes))


def reproject(shapes, p1, p2):
    """
    Project a collection of shapes from one crs to another.

    Deprecated since version 0.2.
    """
    warn("reproject has been renamed to reproject_shapes", DeprecationWarning)
    return reproject_shapes(shapes, p1, p2)


reproject.__doc__ = reproject_shapes.__doc__


def compute_indicatormatrix(orig, dest, orig_crs=4326, dest_crs=4326):
    """
    Compute the indicatormatrix.

    The indicatormatrix I[i,j] is a sparse representation of the ratio
    of the area in orig[j] lying in dest[i], where orig and dest are
    collections of polygons, i.e.

    A value of I[i,j] = 1 indicates that the shape orig[j] is fully
    contained in shape dest[j].

    Note that the polygons must be in the same crs.

    Parameters
    ----------
    orig : Collection of shapely polygons
    dest : Collection of shapely polygons

    Returns
    -------
    I : sp.sparse.lil_matrix
      Indicatormatrix
    """
    orig = orig.geometry if isinstance(orig, gpd.GeoDataFrame) else orig
    dest = dest.geometry if isinstance(dest, gpd.GeoDataFrame) else dest
    dest = reproject_shapes(dest, dest_crs, orig_crs)
    indicator = sp.sparse.lil_matrix((len(dest), len(orig)), dtype=float)
    tree = STRtree(orig)
    idx = dict((id(o), i) for i, o in enumerate(orig))

    for i, d in enumerate(dest):
        for o in tree.query(d):
            if o.intersects(d):
                j = idx[id(o)]
                area = d.intersection(o).area
                indicator[i, j] = area / o.area

    return indicator


def compute_intersectionmatrix(orig, dest, orig_crs=4326, dest_crs=4326):
    """
    Compute the intersectionmatrix.

    The intersectionmatrix is a sparse matrix with entries (i,j) being one
    if shapes orig[j] and dest[i] are intersecting, and zero otherwise.

    Note that the polygons must be in the same crs.

    Parameters
    ----------
    orig : Collection of shapely polygons
    dest : Collection of shapely polygons

    Returns
    -------
    I : sp.sparse.lil_matrix
      Intersectionmatrix
    """
    orig = orig.geometry if isinstance(orig, gpd.GeoDataFrame) else orig
    dest = dest.geometry if isinstance(dest, gpd.GeoDataFrame) else dest
    dest = reproject_shapes(dest, dest_crs, orig_crs)
    intersection = sp.sparse.lil_matrix((len(dest), len(orig)), dtype=float)
    tree = STRtree(orig)
    idx = dict((id(o), i) for i, o in enumerate(orig))

    for i, d in enumerate(dest):
        for o in tree.query(d):
            j = idx[id(o)]
            intersection[i, j] = o.intersects(d)

    return intersection


class ExclusionContainer:
    """Container for exclusion objects and meta data."""

    def __init__(self, crs=3035, res=100):
        """Initialize a container for excluded areas.

        Parameters
        ----------
        crs : rasterio.CRS/proj.CRS/EPSG, optional
            Base crs of the raster collection. All rasters and geometries
            diverging from this crs will be converted to it.
            The default is 3035.
        res : float, optional
            Resolution of the base raster. All diverging rasters will be
            resampled using the gdal Resampling method 'nearest'.
            The default is 100.
        """
        self.rasters = []
        self.geometries = []
        self.crs = crs
        self.res = res

    def add_raster(
        self,
        raster,
        codes=None,
        buffer=0,
        invert=False,
        nodata=255,
        allow_no_overlap=False,
        crs=None,
    ):
        """
        Register a raster to the ExclusionContainer.

        Parameters
        ----------
        raster : str/rasterio.DatasetReader
            Raster or path to raster which to exclude.
        codes : int/list/function, optional
            Codes in the raster which to exclude. Can be a callable function
            which takes the mask (np.array) as argument and performs a
            elementwise condition (must not change the shape). The function may
            not be an anonymous (lambda) function.
            The default is 1.
        buffer : int, optional
            Buffer around the excluded areas in units of ExclusionContainer.crs.
            Use this to create a buffer around the excluded/included area.
            The default is 0.
        invert : bool, optional
            Whether to exclude (False) or include (True) the specified areas
            of the raster. The default is False.
        allow_no_overlap:
            Allow that a raster and a shape (for which the raster will be used as
            a mask) do not overlap. In this case an array with only `nodata` is
            returned.
        crs : rasterio.CRS/EPSG
            CRS of the raster. Specify this if the raster has invalid crs.
        """
        d = dict(
            raster=raster,
            codes=codes,
            buffer=buffer,
            invert=invert,
            nodata=nodata,
            allow_no_overlap=allow_no_overlap,
            crs=crs,
        )
        self.rasters.append(d)

    def add_geometry(self, geometry, buffer=0, invert=False):
        """
        Register a collection of geometries to the ExclusionContainer.

        Parameters
        ----------
        geometry : str/path/geopandas.GeoDataFrame
            Path to geometries or geometries which to exclude.
        buffer : float, optional
            Buffer around the excluded areas in units of ExclusionContainer.crs.
            The default is 0.
        invert : bool, optional
            Whether to exclude (False) or include (True) the specified areas
            of the geometries. The default is False.
        """
        d = dict(geometry=geometry, buffer=buffer, invert=invert)
        self.geometries.append(d)

    def open_files(self):
        """Open rasters and load geometries."""
        for d in self.rasters:
            raster = d["raster"]
            if isinstance(raster, (str, Path)):
                raster = rio.open(raster)
            else:
                assert isinstance(raster, rio.DatasetReader)
            if not raster.crs.is_valid if raster.crs is not None else True:
                if d["crs"]:
                    raster._crs = CRS(d["crs"])
                else:
                    raise ValueError(
                        f"CRS of {raster} is invalid, please " "provide it."
                    )
            d["raster"] = raster

        for d in self.geometries:
            geometry = d["geometry"]
            if isinstance(geometry, (str, Path)):
                geometry = gpd.read_file(geometry)
            if isinstance(geometry, gpd.GeoDataFrame):
                geometry = geometry.geometry
            assert isinstance(geometry, gpd.GeoSeries)
            assert geometry.crs is not None
            geometry = geometry.to_crs(self.crs)
            if d.get("buffer", 0) and not d.get("_buffered", False):
                geometry = geometry.buffer(d["buffer"])
                d["_buffered"] = True
            d["geometry"] = geometry

    @property
    def all_closed(self):
        """Check whether all files in the raster container are closed."""
        return all(isinstance(d["raster"], (str, Path)) for d in self.rasters) and all(
            isinstance(d["geometry"], (str, Path)) for d in self.geometries
        )

    @property
    def all_open(self):
        """Check whether all files in the raster container are open."""
        return all(
            isinstance(d["raster"], rio.DatasetReader) for d in self.rasters
        ) and all(isinstance(d["geometry"], gpd.GeoSeries) for d in self.geometries)

    def __repr__(self):
        return (
            f"Exclusion Container"
            f"\n registered rasters: {len(self.rasters)} "
            f"\n registered geometry collections: {len(self.geometries)}"
            f"\n CRS: {self.crs} - Resolution: {self.res}"
        )


def padded_transform_and_shape(bounds, res):
    """
    Get the (transform, shape) tuple of a raster with resolution `res` and
    bounds `bounds`.
    """
    left, bottom = [(b // res) * res for b in bounds[:2]]
    right, top = [(b // res + 1) * res for b in bounds[2:]]
    shape = int((top - bottom) / res), int((right - left) / res)
    return rio.Affine(res, 0, left, 0, -res, top), shape


def projected_mask(
    raster, geom, transform=None, shape=None, crs=None, allow_no_overlap=False, **kwargs
):
    """Load a mask and optionally project it to target resolution and shape."""
    nodata = kwargs.get("nodata", 255)
    kwargs.setdefault("indexes", 1)
    if geom.crs != raster.crs:
        geom = geom.to_crs(raster.crs)

    if allow_no_overlap:
        try:
            masked, transform_ = mask(raster, geom, crop=True, **kwargs)
        except ValueError:
            res = raster.res[0]
            transform_, shape = padded_transform_and_shape(geom.total_bounds, res)
            masked = np.full(shape, nodata)
    else:
        masked, transform_ = mask(raster, geom, crop=True, **kwargs)

    if transform is None or (transform_ == transform and shape == masked.shape):
        return masked, transform_

    assert shape is not None and crs is not None
    return rio.warp.reproject(
        masked,
        empty(shape),
        src_crs=raster.crs,
        dst_crs=crs,
        src_transform=transform_,
        dst_transform=transform,
        dst_nodata=nodata,
    )


def pad_extent(src, src_transform, dst_transform, src_crs, dst_crs, **kwargs):
    """
    Pad the extent of `src` by an equivalent of one cell of the target raster.

    This ensures that the array is large enough to not be treated as nodata in
    all cells of the destination raster. If src.ndim > 2, the function expects
    the last two dimensions to be y,x.
    Additional keyword arguments are used in `np.pad()`.
    """
    if src.size == 0:
        return src, src_transform

    left, top, right, bottom = *(src_transform * (0, 0)), *(src_transform * (1, 1))
    covered = transform_bounds(src_crs, dst_crs, left, bottom, right, top)
    covered_res = min(abs(covered[2] - covered[0]), abs(covered[3] - covered[1]))
    pad = int(dst_transform[0] // covered_res * 1.1)

    kwargs.setdefault("mode", "constant")

    if src.ndim == 2:
        return rio.pad(src, src_transform, pad, **kwargs)

    npad = ((0, 0),) * (src.ndim - 2) + ((pad, pad), (pad, pad))
    padded = np.pad(src, npad, **kwargs)
    transform = list(src_transform)
    transform[2] -= pad * transform[0]
    transform[5] -= pad * transform[4]
    return padded, rio.Affine(*transform[:6])


def shape_availability(geometry, excluder):
    """
    Compute the eligible area in one or more geometries.

    Parameters
    ----------
    geometry : geopandas.Series
        Geometry of which the eligible area is computed. If the series contains
        more than one geometry, the eligble area of the combined geometries is
        computed.
    excluder : atlite.gis.ExclusionContainer
        Container of all meta data or objects which to exclude, i.e.
        rasters and geometries.

    Returns
    -------
    masked : np.array
        Mask whith eligible raster cells indicated by 1 and excluded cells by 0.
    transform : rasterion.Affine
        Affine transform of the mask.

    """
    if not excluder.all_open:
        excluder.open_files()
    assert geometry.crs == excluder.crs

    bounds = rio.features.bounds(geometry)
    transform, shape = padded_transform_and_shape(bounds, res=excluder.res)
    masked = geometry_mask(geometry, shape, transform)
    exclusions = masked

    # For the following: 0 is eligible, 1 in excluded
    raster = None
    for d in excluder.rasters:
        # allow reusing preloaded raster with different post-processing
        if raster != d["raster"]:
            raster = d["raster"]
            kwargs_keys = ["allow_no_overlap", "nodata"]
            kwargs = {k: v for k, v in d.items() if k in kwargs_keys}
            masked, transform = projected_mask(
                d["raster"], geometry, transform, shape, excluder.crs, **kwargs
            )
        if d["codes"]:
            if callable(d["codes"]):
                masked_ = d["codes"](masked).astype(bool)
            else:
                masked_ = isin(masked, d["codes"])
        else:
            masked_ = masked.astype(bool)

        if d["invert"]:
            masked_ = ~masked_
        if d["buffer"]:
            iterations = int(d["buffer"] / excluder.res) + 1
            masked_ = dilation(masked_, iterations=iterations)

        exclusions = exclusions | masked_

    for d in excluder.geometries:
        masked = ~geometry_mask(d["geometry"], shape, transform, invert=d["invert"])
        exclusions = exclusions | masked

    warn(
        "Output dtype of shape_availability changed from float to boolean.", UserWarning
    )
    return ~exclusions, transform


def shape_availability_reprojected(
    geometry, excluder, dst_transform, dst_crs, dst_shape
):
    """
    Compute and reproject the eligible area of one or more geometries.

    The function executes `shape_availability` and reprojects the calculated
    mask onto a new raster defined by (dst_transform, dst_crs, dst_shape).
    Before reprojecting, the function pads the mask such all non-nodata data
    points  are projected in full cells of the target raster. The ensures that
    all data within the mask are projected correclty (GDAL inherent 'problem').

    ----------
    geometry : geopandas.Series
        Geometry in which the eligible area is computed. If the series contains
        more than one geometry, the eligble area of the combined geometries is
        computed.
    excluder : atlite.gis.ExclusionContainer
        Container of all meta data or objects which to exclude, i.e.
        rasters and geometries.
    dst_transform : rasterio.Affine
        Transform of the target raster.
    dst_crs : rasterio.CRS/proj.CRS
        CRS of the target raster.
    dst_shape : tuple
        Shape of the target raster.

    masked : np.array
        Average share of available area per grid cell. 0 indicates excluded,
        1 is fully included.
    transform : rasterio.Affine
        Affine transform of the mask.

    """
    masked, transform = shape_availability(geometry, excluder)
    masked, transform = pad_extent(
        masked, transform, dst_transform, excluder.crs, dst_crs
    )
    return rio.warp.reproject(
        masked.astype(np.uint8),
        empty(dst_shape),
        resampling=rio.warp.Resampling.average,
        src_transform=transform,
        dst_transform=dst_transform,
        src_crs=excluder.crs,
        dst_crs=dst_crs,
    )


def _init_process(shapes_, excluder_, dst_transform_, dst_crs_, dst_shapes_):
    global shapes, excluder, dst_transform, dst_crs, dst_shapes
    shapes, excluder = shapes_, excluder_
    dst_transform, dst_crs, dst_shapes = dst_transform_, dst_crs_, dst_shapes_


def _process_func(i):
    args = (excluder, dst_transform, dst_crs, dst_shapes)
    with catch_warnings():
        simplefilter("ignore")
        return shape_availability_reprojected(shapes.loc[[i]], *args)[0]


def compute_availabilitymatrix(
    cutout, shapes, excluder, nprocesses=None, disable_progressbar=False
):
    """
    Compute the eligible share within cutout cells in the overlap with shapes.

    For parallel calculation (nprocesses not None) the excluder must not be
    initialized and all raster references must be strings. Otherwise processes
    are colliding when reading from one common rasterio.DatasetReader.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout which the availability matrix is aligned to.
    shapes : geopandas.Series/geopandas.DataFrame
        Geometries for which the availabilities are calculated.
    excluder : atlite.gis.ExclusionContainer
        Container of all meta data or objects which to exclude, i.e.
        rasters and geometries.
    nprocesses : int, optional
        Number of processes to use for calculating the matrix. The paralle-
        lization can heavily boost the calculation speed. The default is None.
    disable_progressbar: bool, optional
        Disable the progressbar if nprocesses is not None. Then the `map`
        function instead of the `imap` function is used for the multiprocessing
        pool. This speeds up the calculation.

    Returns
    -------
    availabilities : xr.DataArray
        DataArray of shape (|shapes|, |y|, |x|) containing all the eligible
        share of cutout cell (x,y) in the overlap with shape i.

    Notes
    -----
    The rasterio (or GDAL) average downsampling returns different results
    dependent on how the target raster (the cutout raster) is spanned.
    Either it is spanned from the top left going downwards,
    e.g. Affine(0.25, 0, 0, 0, -0.25, 50), or starting in the
    lower left corner and going up, e.g. Affine(0.25, 0, 0, 0, 0.25, 50).
    Here we stick to the top down version which is why we use
    `cutout.transform_r` and flipping the y-axis in the end.

    """
    availability = []
    shapes = shapes.geometry if isinstance(shapes, gpd.GeoDataFrame) else shapes
    shapes = shapes.to_crs(excluder.crs)

    args = (excluder, cutout.transform_r, cutout.crs, cutout.shape)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" gridcells",
        total=len(shapes),
        desc="Compute availability matrix",
    )
    if nprocesses is None:
        with catch_warnings():
            simplefilter("ignore")
            for i in tqdm(shapes.index, **tqdm_kwargs):
                _ = shape_availability_reprojected(shapes.loc[[i]], *args)[0]
                availability.append(_)
    else:
        assert (
            excluder.all_closed
        ), "For parallelization all raster files in excluder must be closed"
        kwargs = {
            "initializer": _init_process,
            "initargs": (shapes, *args),
            "maxtasksperchild": 20,
            "processes": nprocesses,
        }
        with mp.get_context("spawn").Pool(**kwargs) as pool:
            if disable_progressbar:
                availability = list(pool.map(_process_func, shapes.index))
            else:
                availability = list(
                    tqdm(pool.imap(_process_func, shapes.index), **tqdm_kwargs)
                )

    availability = np.stack(availability)[:, ::-1]  # flip axis, see Notes
    coords = [(shapes.index), ("y", cutout.data.y.data), ("x", cutout.data.x.data)]
    return xr.DataArray(availability, coords=coords)


def maybe_swap_spatial_dims(ds, namex="x", namey="y"):
    """Swap order of spatial dimensions according to atlite concention."""
    swaps = {}
    lx, rx = ds.indexes[namex][[0, -1]]
    ly, uy = ds.indexes[namey][[0, -1]]

    if lx > rx:
        swaps[namex] = slice(None, None, -1)
    if uy < ly:
        swaps[namey] = slice(None, None, -1)

    return ds.isel(**swaps) if swaps else ds


def _as_transform(x, y):
    lx, rx = x[[0, -1]]
    ly, uy = y[[0, -1]]

    dx = float(rx - lx) / float(len(x) - 1)
    dy = float(uy - ly) / float(len(y) - 1)

    return rio.Affine(dx, 0, lx - dx / 2, 0, dy, ly - dy / 2)


def regrid(ds, dimx, dimy, **kwargs):
    """
    Interpolate Dataset or DataArray `ds` to a new grid, using rasterio's
    reproject facility.

    See also: https://mapbox.github.io/rasterio/topics/resampling.html

    Parameters
    ----------
    ds : xr.Dataset|xr.DataArray
      N-dim data on a spatial grid
    dimx : pd.Index
      New x-coordinates in destination crs.
      dimx.name MUST refer to x-coord of ds.
    dimy : pd.Index
      New y-coordinates in destination crs.
      dimy.name MUST refer to y-coord of ds.
    **kwargs :
      Arguments passed to rio.wrap.reproject; of note:
      - resampling is one of gis.Resampling.{average,cubic,bilinear,nearest}
      - src_crs, dst_crs define the different crs (default: EPSG 4326, ie latlong)
    """
    namex = dimx.name
    namey = dimy.name

    ds = maybe_swap_spatial_dims(ds, namex, namey)

    src_transform = _as_transform(ds.indexes[namex], ds.indexes[namey])
    dst_transform = _as_transform(dimx, dimy)
    dst_shape = len(dimy), len(dimx)

    kwargs.update(dst_transform=dst_transform)
    kwargs.setdefault("src_crs", CRS.from_epsg(4326))
    kwargs.setdefault("dst_crs", CRS.from_epsg(4326))

    def _reproject(src, **kwargs):
        shape = src.shape[:-2] + dst_shape
        src, trans = pad_extent(
            src,
            src_transform,
            dst_transform,
            kwargs["src_crs"],
            kwargs["dst_crs"],
            mode="edge",
        )

        return rio.warp.reproject(src, empty(shape), src_transform=trans, **kwargs)[0]

    data_vars = ds.data_vars.values() if isinstance(ds, xr.Dataset) else (ds,)
    dtypes = {da.dtype for da in data_vars}
    assert len(dtypes) == 1, "regrid can only reproject datasets with homogeneous dtype"

    return (
        xr.apply_ufunc(
            _reproject,
            ds,
            input_core_dims=[[namey, namex]],
            output_core_dims=[["yout", "xout"]],
            output_dtypes=[dtypes.pop()],
            dask_gufunc_kwargs=dict(
                output_sizes={"yout": dst_shape[0], "xout": dst_shape[1]}
            ),
            dask="parallelized",
            kwargs=kwargs,
        )
        .rename({"yout": namey, "xout": namex})
        .assign_coords(
            **{
                namey: (namey, dimy.data, ds.coords[namey].attrs),
                namex: (namex, dimx.data, ds.coords[namex].attrs),
            }
        )
        .assign_attrs(**ds.attrs)
    )
