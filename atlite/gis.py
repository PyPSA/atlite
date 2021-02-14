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
from collections import OrderedDict
from pathlib import Path
from warnings import warn
from pyproj import CRS, Transformer
import geopandas as gpd
from shapely.ops import transform
import rasterio as rio
import rasterio.warp
from rasterio.warp import reproject, transform_bounds
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.features import geometry_mask
from scipy.ndimage.morphology import binary_dilation as dilation
from numpy import isin, empty, where
from shapely.strtree import STRtree

import logging
logger = logging.getLogger(__name__)


def get_coords(x, y, time, dx=0.25, dy=0.25, dt='h', **kwargs):
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

    ds = xr.Dataset({'x': np.round(np.arange(-180, 180, dx), 9),
                     'y': np.round(np.arange(-90, 90, dy), 9),
                     'time': pd.date_range(start="1979", end="now", freq=dt)})
    ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
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
    ---------
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
    indicator = sp.sparse.lil_matrix((len(dest), len(orig)), dtype=np.float)
    tree = STRtree(orig)
    idx = dict((id(o), i) for i, o in enumerate(orig))

    for i, d in enumerate(dest):
        for o in tree.query(d):
            if o.intersects(d):
                j = idx[id(o)]
                area = d.intersection(o).area
                indicator[i, j] = area / o.area

    return indicator


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
            resmapled using the gdal Resampling method 'nearest'.
            The default is 100.
        """
        self.rasters = []
        self.geometries = []
        self.crs = crs
        self.res = res

    def add_raster(self, raster, codes=1, dilation=0, invert=False, **kwargs):
        """
        Register a raster to the Excluder.

        Parameters
        ----------
        raster : str/rasterio.DatasetReader
            Path to raster or raster which to exclude.
        codes : int/list, optional
            Codes in the raster which to exclude/include. The default is 1.
        dilation : int, optional
            Dilation of the excluded areas in cell. Use this to create a buffer
            around the excluded/included area. The default is 0.
        invert : bool, optional
            Whether to exclude (False) or include (True) the specified areas
            of the raster. The default is False.
        crs : rasterio.CRS/EPSG
            CRS of the raster. Specify this if the raster has crs data missing.
        **kwargs
        """
        d = dict(raster=raster, codes=codes, dilation=dilation,
                 invert=invert, **kwargs)
        self.rasters.append(d)

    def add_geometry(self, geometry, buffer=0, invert=False, **kwargs):
        """
        Register a collection of geometries to the Excluder.

        Parameters
        ----------
        geometry : str/path/geopandas.GeoDataFrame
            Path to geometries or geometries which to exclude.
        buffer : float, optional
            Buffer around the geometries. The default is 0.
        invert : bool, optional
            Whether to exclude (False) or include (True) the specified areas
            of the geometries. The default is False.
        **kwargs
        """
        d = dict(buffer=buffer, invert=invert, **kwargs)
        self.geometries.append(d)


    def initialize(self):
        """Open rasters and load geometries."""
        for d in self.rasters:
            raster = d['raster']
            if isinstance(raster, (str, Path)):
                raster = rio.open(raster)
            else:
                assert isinstance(raster, rio.DatasetReader)
            if 'crs' in d:
                raster._crs = d['crs']
            d['raster'] = raster

        for d in self.geometries:
            geometry = d['geometry']
            if isinstance(geometry, (str, Path)):
                geometry = gpd.read_file(geometry)
            assert isinstance(geometry, gpd.GeoDataFrame)
            geometry.to_crs(self.crs)
            d['geometry'] = geometry.geometry

    def __repr__(self):
        return (f"Exclusion Container"
                f"\n registered rasters: {len(self.rasters)} "
                f"\n registered geometry collections: {len(self.geometries)}"
                f"\n CRS: {self.crs} - Resolution: {self.res}")


def padded_transform_and_shape(bounds, res):
    left, bottom = [(b // res)* res for b in bounds[:2]]
    right, top = [(b // res + 1) * res for b in bounds[2:]]
    shape = int((top - bottom) // res), int((right - left) / res)
    return rio.Affine(res, 0, left, 0, -res, top), shape


def projected_mask(raster, geom, transform=None, shape=None, crs=None, **kwargs):
    """Load a mask and optionally project it to target resolution and shape."""
    kwargs.setdefault('indexes', 1)
    masked, transform_ = mask(raster, geom, crop=True, **kwargs)

    if transform is None or (transform_ == transform):
        return masked, transform_

    assert shape is not None and crs is not None
    return rio.warp.reproject(masked, empty(shape), src_crs=raster.crs, dst_crs=crs,
                     src_transform=transform_, dst_transform=transform)


def pad_extent(values, src_transform, dst_transform, src_crs, dst_crs):
    """Ensure the array is large enough to not be treated as nodata."""
    left, top, right, bottom = *(src_transform*(0,0)), *(src_transform*(1,1))
    covered = transform_bounds(src_crs, dst_crs, left, bottom, right, top)
    covered_res = min(covered[2] - covered[0], covered[3] - covered[1])
    pad = int(dst_transform[0] // covered_res * 1.1)
    return rio.pad(values, src_transform, pad, 'constant', constant_values=0)


def shape_availability(geometry, excluder):
    exclusions = []
    crs = excluder.crs

    bounds = rio.features.bounds(geometry)
    transform, shape = padded_transform_and_shape(bounds, res=excluder.res)
    masked = geometry_mask(geometry, shape, transform).astype(int)
    exclusions.append(masked)

    raster = None
    for d in excluder.rasters:
        # allow reusing preloaded raster with different config
        if raster != d['raster']:
            raster = d['raster']
            nodata = d.get('nodata', 255)
            masked, transform = projected_mask(d['raster'], geometry, transform,
                                               shape, crs, nodata=nodata)
        masked_ = masked
        # ...
        exclusions.append(masked_)
    return (sum(exclusions) == 0).astype(float), transform


def compute_availabilitymatrix(cutout, shapes, excluder):
    availability = []
    names = shapes.get('name', shapes.index)
    shapes = shapes.geometry if isinstance(shapes, gpd.GeoDataFrame) else shapes

    for i in shapes.index:
        masked, transform = shape_availability(shapes[[i]], excluder)
        masked, transform = pad_extent(masked, transform, cutout.transform,
                                       excluder.crs, cutout.crs)

        _ = rio.warp.reproject(masked, empty(cutout.shape), resampling=5,
                               src_transform=transform,
                               dst_transform=cutout.transform,
                               src_crs=excluder.crs, dst_crs=cutout.crs,)[0]
        availability.append(_)
    coords=[('shapes', names), ('y', cutout.data.y), ('x', cutout.data.x),]
    return xr.DataArray(np.stack(availability), coords=coords)




def maybe_swap_spatial_dims(ds, namex='x', namey='y'):
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

    return rio.transform.from_origin(lx, uy, dx, dy)


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

    src_transform = _as_transform(ds.indexes[namex],
                                  ds.indexes[namey])
    dst_transform = _as_transform(dimx, dimy)
    dst_shape = len(dimy), len(dimx)

    kwargs.update(dst_shape=dst_shape,
                  src_transform=src_transform,
                  dst_transform=dst_transform)
    kwargs.setdefault("src_crs", CRS.from_epsg(4326))
    kwargs.setdefault("dst_crs", CRS.from_epsg(4326))

    def _reproject(src, dst_shape, **kwargs):
        dst = np.empty(src.shape[:-2] + dst_shape, dtype=src.dtype)
        rio.warp.reproject(np.asarray(src), dst, **kwargs)
        return dst

    data_vars = ds.data_vars.values() if isinstance(ds, xr.Dataset) else (ds,)
    dtypes = {da.dtype for da in data_vars}
    assert len(dtypes) == 1, \
        "regrid can only reproject datasets with homogeneous dtype"

    return (xr.apply_ufunc(_reproject,
                           ds,
                           input_core_dims=[[namey, namex]],
                           output_core_dims=[['yout', 'xout']],
                           output_dtypes=[dtypes.pop()],
                           dask_gufunc_kwargs =
                               dict(output_sizes={'yout': dst_shape[0],
                                                  'xout': dst_shape[1]}),
                           dask='parallelized',
                           kwargs=kwargs)
            .rename({'yout': namey, 'xout': namex})
            .assign_coords(**{namey: (namey, dimy, ds.coords[namey].attrs),
                              namex: (namex, dimx, ds.coords[namex].attrs)})
            .assign_attrs(**ds.attrs))
