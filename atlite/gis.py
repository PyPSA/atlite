# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Functions for Geographic Information System.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from warnings import catch_warnings, simplefilter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.warp
import scipy as sp
import scipy.sparse
import xarray as xr
from numpy import empty, isin
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import transform_bounds
from scipy.ndimage import binary_dilation as dilation
from shapely.ops import transform
from shapely.strtree import STRtree
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.axes import Axes

    from atlite._types import (
        CrsLike,
        DataArray,
        Dataset,
        GeoDataFrame,
        Geometry,
        GeoSeries,
        NDArray,
        PathLike,
    )

logger = logging.getLogger(__name__)


def get_coords(
    x: slice,
    y: slice,
    time: slice,
    dx: float = 0.25,
    dy: float = 0.25,
    dt: str = "h",
    **kwargs: Any,
) -> Dataset:
    """
    Create cutout coordinates from slices and resolutions.

    Parameters
    ----------
    x : slice
        Bounds of the x dimension.
    y : slice
        Bounds of the y dimension.
    time : slice
        Bounds of the time dimension.
    dx : float, optional
        Step size of the x coordinate.
    dy : float, optional
        Step size of the y coordinate.
    dt : str, optional
        Frequency of the time coordinate.
    **kwargs
        Unused keyword arguments.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``x``, ``y``, and ``time`` coordinates.
    """
    x = slice(*sorted([x.start, x.stop]))
    y = slice(*sorted([y.start, y.stop]))

    ds = xr.Dataset({
        "x": np.round(np.arange(-180, 180, dx), 9),
        "y": np.round(np.arange(-90, 90, dy), 9),
        "time": pd.date_range(start="1940", end="now", freq=dt),
    })
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    ds = ds.sel(x=x, y=y, time=time)
    return cast("Dataset", ds)


def spdiag(v: NDArray | Sequence[float]) -> sp.sparse.csr_matrix:
    """
    Create a sparse diagonal matrix.

    Parameters
    ----------
    v : array-like
        Values placed on the diagonal.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse diagonal matrix with ``v`` on the diagonal.
    """
    N = len(v)
    inds = np.arange(N + 1, dtype=np.int32)
    return sp.sparse.csr_matrix((v, inds[:-1], inds), (N, N))


def reproject_shapes(
    shapes: Iterable[Geometry] | pd.Series | dict[Any, Geometry],
    crs1: CrsLike,
    crs2: CrsLike,
) -> Iterable[Geometry] | pd.Series | OrderedDict[Any, Geometry]:
    """
    Reproject a collection of geometries.

    Parameters
    ----------
    shapes : iterable, pandas.Series, or dict
        Shapes to reproject.
    crs1 : any
        Source coordinate reference system.
    crs2 : any
        Target coordinate reference system.

    Returns
    -------
    iterable, pandas.Series, or collections.OrderedDict
        Reprojected shapes with the same container type where applicable.
    """
    transformer = Transformer.from_crs(crs1, crs2, always_xy=True)

    def _reproject_shape(shape: Geometry) -> Geometry:
        return transform(transformer.transform, shape)

    if isinstance(shapes, pd.Series):
        return shapes.map(_reproject_shape)
    if isinstance(shapes, dict):
        return OrderedDict((k, _reproject_shape(v)) for k, v in shapes.items())
    return list(map(_reproject_shape, shapes))


def compute_indicatormatrix(
    orig: GeoDataFrame | GeoSeries | Iterable[Geometry],
    dest: GeoDataFrame | GeoSeries | Iterable[Geometry],
    orig_crs: CrsLike = 4326,
    dest_crs: CrsLike = 4326,
) -> sp.sparse.lil_matrix:
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
        Origin polygons.
    dest : Collection of shapely polygons
        Destination polygons.
    orig_crs : int or CRS, default 4326
        CRS of the origin polygons.
    dest_crs : int or CRS, default 4326
        CRS of the destination polygons.

    Returns
    -------
    I : sp.sparse.lil_matrix
      Indicatormatrix

    """
    orig = orig.geometry if isinstance(orig, gpd.GeoDataFrame) else orig
    dest = dest.geometry if isinstance(dest, gpd.GeoDataFrame) else dest
    dest = reproject_shapes(dest, dest_crs, orig_crs)
    orig_list: list[Any] | pd.Series = (
        list(orig) if not isinstance(orig, pd.Series) else orig
    )
    dest_list: list[Any] | pd.Series = (
        list(dest) if not isinstance(dest, pd.Series) else dest
    )
    indicator = sp.sparse.lil_matrix((len(dest_list), len(orig_list)), dtype=float)
    tree = STRtree(orig_list)
    idx = {hash(o.wkt): i for i, o in enumerate(orig_list)}

    for i, d in enumerate(dest_list):
        for o in tree.query(d):
            # STRtree query returns a list of indices for shapely >= v2.0
            if isinstance(o, (int | np.integer)):
                o = orig_list[o]
            if o.intersects(d):
                j = idx[hash(o.wkt)]
                area = d.intersection(o).area
                indicator[i, j] = area / o.area

    return indicator


def compute_intersectionmatrix(
    orig: GeoDataFrame | GeoSeries | Iterable[Geometry],
    dest: GeoDataFrame | GeoSeries | Iterable[Geometry],
    orig_crs: CrsLike = 4326,
    dest_crs: CrsLike = 4326,
) -> sp.sparse.lil_matrix:
    """
    Compute the intersectionmatrix.

    The intersectionmatrix is a sparse matrix with entries (i,j) being one
    if shapes orig[j] and dest[i] are intersecting, and zero otherwise.

    Note that the polygons must be in the same crs.

    Parameters
    ----------
    orig : Collection of shapely polygons
        Origin polygons.
    dest : Collection of shapely polygons
        Destination polygons.
    orig_crs : int or CRS, default 4326
        CRS of the origin polygons.
    dest_crs : int or CRS, default 4326
        CRS of the destination polygons.

    Returns
    -------
    I : sp.sparse.lil_matrix
      Intersectionmatrix

    """
    orig = orig.geometry if isinstance(orig, gpd.GeoDataFrame) else orig
    dest = dest.geometry if isinstance(dest, gpd.GeoDataFrame) else dest
    dest = reproject_shapes(dest, dest_crs, orig_crs)
    orig_list: list[Any] | pd.Series = (
        list(orig) if not isinstance(orig, pd.Series) else orig
    )
    dest_list: list[Any] | pd.Series = (
        list(dest) if not isinstance(dest, pd.Series) else dest
    )
    intersection = sp.sparse.lil_matrix((len(dest_list), len(orig_list)), dtype=float)
    tree = STRtree(orig_list)
    idx = {hash(o.wkt): i for i, o in enumerate(orig_list)}

    for i, d in enumerate(dest_list):
        for o in tree.query(d):
            # STRtree query returns a list of indices for shapely >= v2.0
            if isinstance(o, (int | np.integer)):
                o = orig_list[o]
            j = idx[hash(o.wkt)]
            intersection[i, j] = o.intersects(d)

    return intersection


def padded_transform_and_shape(
    bounds: tuple[float, float, float, float], res: float
) -> tuple[rio.Affine, tuple[int, int]]:
    """
    Return a padded raster transform and shape.

    Parameters
    ----------
    bounds : tuple
        Bounding box as ``(left, bottom, right, top)``.
    res : float
        Raster resolution.

    Returns
    -------
    tuple
        Affine transform and raster shape covering the padded bounds.
    """
    left, bottom = ((b // res) * res for b in bounds[:2])
    right, top = ((b // res + 1) * res for b in bounds[2:])
    shape = int((top - bottom) / res), int((right - left) / res)
    return rio.Affine(res, 0, left, 0, -res, top), shape


def projected_mask(
    raster: rio.DatasetReader,
    geom: GeoSeries,
    transform: rio.Affine | None = None,
    shape: tuple[int, int] | None = None,
    crs: CrsLike = None,
    allow_no_overlap: bool = False,
    **kwargs: Any,
) -> tuple[NDArray, rio.Affine]:
    """
    Load a raster mask and optionally reproject it.

    Parameters
    ----------
    raster : rasterio.DatasetReader
        Raster source used to build the mask.
    geom : geopandas.GeoSeries
        Geometry used for masking.
    transform : rasterio.Affine, optional
        Target transform.
    shape : tuple, optional
        Target array shape.
    crs : any, optional
        Target coordinate reference system.
    allow_no_overlap : bool, optional
        Whether to return a nodata mask when geometry and raster do not overlap.
    **kwargs
        Additional keyword arguments passed to ``rasterio.mask.mask``.

    Returns
    -------
    tuple
        Masked array and its affine transform.
    """
    nodata = kwargs.get("nodata", 255)
    kwargs.setdefault("indexes", 1)
    if geom.crs != raster.crs:
        geom = geom.to_crs(raster.crs)

    if allow_no_overlap:
        try:
            masked, transform_ = mask(raster, geom, crop=True, **kwargs)
        except ValueError:
            res = raster.res[0]
            transform_, shape_ = padded_transform_and_shape(geom.total_bounds, res)
            masked = np.full(shape_, nodata)
    else:
        masked, transform_ = mask(raster, geom, crop=True, **kwargs)

    if transform is None or (transform_ == transform and masked.shape == shape):
        return masked, transform_

    assert shape is not None and crs is not None
    return rio.warp.reproject(  # type: ignore[no-any-return]
        masked,
        empty(shape),
        src_crs=raster.crs,
        dst_crs=crs,
        src_transform=transform_,
        dst_transform=transform,
        dst_nodata=nodata,
    )


def pad_extent(
    src: NDArray,
    src_transform: rio.Affine,
    dst_transform: rio.Affine,
    src_crs: CrsLike,
    dst_crs: CrsLike,
    **kwargs: Any,
) -> tuple[NDArray, rio.Affine]:
    """
    Pad an array before reprojection.

    Parameters
    ----------
    src : numpy.ndarray
        Source array with spatial axes in the last two dimensions.
    src_transform : rasterio.Affine
        Transform of the source array.
    dst_transform : rasterio.Affine
        Transform of the destination raster.
    src_crs : any
        Source coordinate reference system.
    dst_crs : any
        Destination coordinate reference system.
    **kwargs
        Keyword arguments passed to ``numpy.pad``.

    Returns
    -------
    tuple
        Padded array and updated affine transform.
    """
    if src.size == 0:
        return src, src_transform

    left, top = src_transform * (0, 0)
    right, bottom = src_transform * (1, 1)
    covered = transform_bounds(src_crs, dst_crs, left, bottom, right, top)
    covered_res = min(abs(covered[2] - covered[0]), abs(covered[3] - covered[1]))
    pad = int(dst_transform[0] // covered_res * 1.1)

    kwargs.setdefault("mode", "constant")

    if src.ndim == 2:
        return rio.pad(src, src_transform, pad, **kwargs)  # type: ignore[no-any-return]

    npad = ((0, 0),) * (src.ndim - 2) + ((pad, pad), (pad, pad))
    padded = np.pad(src, npad, **kwargs)
    transform = list(src_transform)
    transform[2] -= pad * transform[0]
    transform[5] -= pad * transform[4]
    return padded, rio.Affine(*transform[:6])


def shape_availability(
    geometry: GeoSeries, excluder: ExclusionContainer
) -> tuple[NDArray, rio.Affine]:
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

    return ~exclusions, transform


def shape_availability_reprojected(
    geometry: GeoSeries,
    excluder: ExclusionContainer,
    dst_transform: rio.Affine,
    dst_crs: CrsLike,
    dst_shape: tuple[int, int],
) -> tuple[NDArray, rio.Affine]:
    """
    Compute availability and reproject it to a target raster.

    Parameters
    ----------
    geometry : geopandas.GeoSeries
        Geometry for which availability is computed.
    excluder : atlite.gis.ExclusionContainer
        Exclusion container defining masked areas.
    dst_transform : rasterio.Affine
        Transform of the target raster.
    dst_crs : any
        Coordinate reference system of the target raster.
    dst_shape : tuple
        Shape of the target raster.

    Returns
    -------
    tuple
        Reprojected availability array and destination transform.
    """
    masked, transform = shape_availability(geometry, excluder)
    masked, transform = pad_extent(
        masked, transform, dst_transform, excluder.crs, dst_crs
    )
    return rio.warp.reproject(  # type: ignore[no-any-return]
        masked.astype(np.uint8),
        empty(dst_shape),
        resampling=rio.warp.Resampling.average,
        src_transform=transform,
        dst_transform=dst_transform,
        src_crs=excluder.crs,
        dst_crs=dst_crs,
    )


class ExclusionContainer:
    """
    Container for exclusion objects and meta data.
    """

    def __init__(self, crs: CrsLike = 3035, res: float = 100) -> None:
        """
        Initialize a container for excluded areas.

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
        self.rasters: list[dict[str, Any]] = []
        self.geometries: list[dict[str, Any]] = []
        self.crs: CrsLike = crs
        self.res: float = res

    def add_raster(
        self,
        raster: PathLike | rio.DatasetReader,
        codes: int
        | list[int]
        | Sequence[int]
        | Callable[[NDArray], NDArray]
        | None = None,
        buffer: float = 0,
        invert: bool = False,
        nodata: int = 255,
        allow_no_overlap: bool = False,
        crs: CrsLike = None,
    ) -> None:
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
        nodata : int, optional
            Value to use for nodata pixels. The default is 255.
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
        d: dict[str, Any] = {
            "raster": raster,
            "codes": codes,
            "buffer": buffer,
            "invert": invert,
            "nodata": nodata,
            "allow_no_overlap": allow_no_overlap,
            "crs": crs,
        }
        self.rasters.append(d)

    def add_geometry(
        self,
        geometry: PathLike | GeoDataFrame | GeoSeries,
        buffer: float = 0,
        invert: bool = False,
    ) -> None:
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
        d: dict[str, Any] = {"geometry": geometry, "buffer": buffer, "invert": invert}
        self.geometries.append(d)

    def open_files(self) -> None:
        """
        Open rasters and load geometries.

        Raises
        ------
        ValueError
            If a raster has an invalid CRS and none is provided.
        """
        for d in self.rasters:
            raster = d["raster"]
            if isinstance(raster, (str | Path)):
                raster = rio.open(raster)
            else:
                assert isinstance(raster, rio.DatasetReader)

            # Check if the raster has a valid CRS
            if not raster.crs:
                if d["crs"]:
                    raster._crs = CRS(d["crs"])
                else:
                    raise ValueError(f"CRS of {raster} is invalid, please provide it.")
            d["raster"] = raster

        for d in self.geometries:
            geometry = d["geometry"]
            if isinstance(geometry, (str | Path)):
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
    def all_closed(self) -> bool:
        """
        Check whether all files in the raster container are closed.
        """
        return all(isinstance(d["raster"], (str | Path)) for d in self.rasters) and all(
            isinstance(d["geometry"], (str | Path)) for d in self.geometries
        )

    @property
    def all_open(self) -> bool:
        """
        Check whether all files in the raster container are open.
        """
        return all(
            isinstance(d["raster"], rio.DatasetReader) for d in self.rasters
        ) and all(isinstance(d["geometry"], gpd.GeoSeries) for d in self.geometries)

    def __repr__(self) -> str:
        return (
            f"Exclusion Container"
            f"\n registered rasters: {len(self.rasters)} "
            f"\n registered geometry collections: {len(self.geometries)}"
            f"\n CRS: {self.crs} - Resolution: {self.res}"
        )

    def compute_shape_availability(
        self,
        geometry: GeoDataFrame | GeoSeries,
        dst_transform: rio.Affine | None = None,
        dst_crs: CrsLike = None,
        dst_shape: tuple[int, int] | None = None,
    ) -> tuple[NDArray, rio.Affine]:
        """
        Compute the eligible area in one or more geometries and optionally
        reproject.

        Parameters
        ----------
        geometry : geopandas.Series
            Geometry of which the eligible area is computed. If the series contains
            more than one geometry, the eligble area of the combined geometries is
            computed.
        dst_transform : rasterio.Affine
            Transform of the target raster. Define if the availability
            should be reprojected. Defaults to None.
        dst_crs : rasterio.CRS/proj.CRS
            CRS of the target raster. Define if the availability
            should be reprojected. Defaults to None.
        dst_shape : tuple
            Shape of the target raster. Define if the availability
            should be reprojected. Defaults to None.

        Returns
        -------
        masked : np.array
            Mask whith eligible raster cells indicated by 1 and excluded cells by 0.
        transform : rasterion.Affine
            Affine transform of the mask.

        Raises
        ------
        ValueError
            If only some of ``dst_transform``, ``dst_crs``, ``dst_shape`` are given.

        """
        if isinstance(geometry, gpd.GeoDataFrame):
            geometry = geometry.geometry
        geometry = geometry.to_crs(self.crs)

        dst_args_not_none = [
            arg is not None for arg in [dst_transform, dst_crs, dst_shape]
        ]
        if any(dst_args_not_none):
            # if any is not None, require that all are not None
            if not all(dst_args_not_none):
                raise ValueError(
                    "Arguments dst_transform, dst_crs, dst_shape "
                    "should be all None or all defined."
                )
            assert (
                dst_transform is not None
                and dst_crs is not None
                and dst_shape is not None
            )
            return shape_availability_reprojected(
                geometry, self, dst_transform, dst_crs, dst_shape
            )
        return shape_availability(geometry, self)

    def plot_shape_availability(
        self,
        geometry: GeoDataFrame | GeoSeries,
        ax: Axes | None = None,
        set_title: bool = True,
        dst_transform: rio.Affine | None = None,
        dst_crs: CrsLike = None,
        dst_shape: tuple[int, int] | None = None,
        show_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
    ) -> Axes:
        """
        Plot the eligible area for one or more geometries and optionally
        reproject.

        This function uses its own default values for ``rasterio.plot.show`` and
        ``geopandas.GeoSeries.plot``. Therefore eligible land is drawn in green
        Note that this funtion will likely fail if another CRS than the one of the
        ExclusionContainer is used in the axis (e.g. cartopy projections).

        Parameters
        ----------
        geometry : geopandas.Series
            Geometry of which the eligible area is computed. If the series contains
            more than one geometry, the eligble area of the combined geometries is
            computed.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        set_title : boolean, optional
            Whether to set the title with additional information on the share of
            eligible land.
        dst_transform : rasterio.Affine
            Transform of the target raster. Define if the availability
            should be reprojected. Defaults to None.
        dst_crs : rasterio.CRS/proj.CRS
            CRS of the target raster. Define if the availability
            should be reprojected. Defaults to None.
        dst_shape : tuple
            Shape of the target raster. Define if the availability
            should be reprojected. Defaults to None.
        show_kwargs : dict, optional
            Keyword arguments passed to ``rasterio.plot.show``, by default {}
        plot_kwargs: dict, optional
            Keyword arguments passed to ``geopandas.GeoSeries.plot``, by default {}

        Returns
        -------
        _type_
            _description_

        """
        if show_kwargs is None:
            show_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        import matplotlib.pyplot as plt

        if isinstance(geometry, gpd.GeoDataFrame):
            geometry = geometry.geometry
        geometry = geometry.to_crs(self.crs)

        masked, transform = self.compute_shape_availability(
            geometry, dst_transform, dst_crs, dst_shape
        )

        if ax is None:
            ax = plt.gca()

        show_kwargs.setdefault("cmap", "Greens")
        ax = show(masked, transform=transform, ax=ax, **show_kwargs)
        plot_kwargs.setdefault("edgecolor", "k")
        plot_kwargs.setdefault("color", "None")
        geometry.plot(ax=ax, **plot_kwargs)

        if set_title:
            eligible_share = masked.sum() * self.res**2 / geometry.area.sum()
            ax.set_title(f"Eligible area (green) {eligible_share:.2%}")

        return ax


_mp_shapes: GeoSeries
_mp_excluder: ExclusionContainer
_mp_dst_transform: rio.Affine
_mp_dst_crs: CrsLike
_mp_dst_shapes: tuple[int, int]


def _init_process(
    shapes_: GeoSeries,
    excluder_: ExclusionContainer,
    dst_transform_: rio.Affine,
    dst_crs_: CrsLike,
    dst_shapes_: tuple[int, int],
) -> None:
    global _mp_shapes, _mp_excluder, _mp_dst_transform, _mp_dst_crs, _mp_dst_shapes
    _mp_shapes, _mp_excluder = shapes_, excluder_
    _mp_dst_transform, _mp_dst_crs, _mp_dst_shapes = (
        dst_transform_,
        dst_crs_,
        dst_shapes_,
    )


def _process_func(i: Any) -> NDArray:
    args = (_mp_excluder, _mp_dst_transform, _mp_dst_crs, _mp_dst_shapes)
    with catch_warnings():
        simplefilter("ignore")
        return shape_availability_reprojected(_mp_shapes.loc[[i]], *args)[0]


def compute_availabilitymatrix(
    cutout: Any,
    shapes: GeoDataFrame | GeoSeries,
    excluder: ExclusionContainer,
    nprocesses: int | None = None,
    disable_progressbar: bool = True,
) -> DataArray:
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
    shapes = shapes.geometry if isinstance(shapes, gpd.GeoDataFrame) else shapes
    shapes = shapes.to_crs(excluder.crs)

    args = (excluder, cutout.transform_r, cutout.crs, cutout.shape)
    tqdm_kwargs = {
        "ascii": False,
        "unit": " gridcells",
        "total": len(shapes),
        "desc": "Compute availability matrix",
    }

    if nprocesses is None:
        if not disable_progressbar:
            iterator = tqdm(shapes.index, **tqdm_kwargs)
        else:
            iterator = shapes.index
        with catch_warnings():
            simplefilter("ignore")
            availability = []
            for i in iterator:
                _ = shape_availability_reprojected(shapes.loc[[i]], *args)[0]
                availability.append(_)
    else:
        assert excluder.all_closed, (
            "For parallelization all raster files in excluder must be closed"
        )
        with mp.get_context("spawn").Pool(
            processes=nprocesses,
            initializer=_init_process,
            initargs=(shapes, *args),
            maxtasksperchild=20,
        ) as pool:
            if disable_progressbar:
                availability = list(pool.map(_process_func, shapes.index))
            else:
                availability = list(
                    tqdm(pool.imap(_process_func, shapes.index), **tqdm_kwargs)
                )

    availability_arr = np.stack(availability)[:, ::-1]  # flip axis, see Notes
    if availability_arr.ndim == 4:
        availability_arr = availability_arr.squeeze(axis=1)
    coords = [(shapes.index), ("y", cutout.data.y.data), ("x", cutout.data.x.data)]
    return xr.DataArray(availability_arr, coords=coords)


def maybe_swap_spatial_dims(
    ds: Dataset | DataArray, namex: str = "x", namey: str = "y"
) -> Dataset | DataArray:
    """
    Ensure spatial coordinates follow atlite's axis ordering.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Object with spatial coordinates.
    namex : str, optional
        Name of the x dimension.
    namey : str, optional
        Name of the y dimension.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Input object with spatial dimensions reversed if needed.
    """
    swaps = {}
    lx, rx = ds.indexes[namex][[0, -1]]
    ly, uy = ds.indexes[namey][[0, -1]]

    if lx > rx:
        swaps[namex] = slice(None, None, -1)
    if uy < ly:
        swaps[namey] = slice(None, None, -1)

    return ds.isel(swaps) if swaps else ds


def _as_transform(x: pd.Index, y: pd.Index) -> rio.Affine:
    lx, rx = x[[0, -1]]
    ly, uy = y[[0, -1]]

    dx = float(rx - lx) / float(len(x) - 1)
    dy = float(uy - ly) / float(len(y) - 1)

    return rio.Affine(dx, 0, lx - dx / 2, 0, dy, ly - dy / 2)


def regrid(
    ds: Dataset | DataArray,
    dimx: pd.Index,
    dimy: pd.Index,
    **kwargs: Any,
) -> Dataset | DataArray:
    """
    Reproject data to a new spatial grid.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data on a spatial grid.
    dimx : pandas.Index
        Target x coordinates. ``dimx.name`` must match the source x dimension.
    dimy : pandas.Index
        Target y coordinates. ``dimy.name`` must match the source y dimension.
    **kwargs
        Keyword arguments passed to ``rasterio.warp.reproject``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Regridded object on the target coordinates.
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

    def _reproject(src: NDArray, **kwargs: Any) -> NDArray:
        shape = src.shape[:-2] + dst_shape
        src, trans = pad_extent(
            src,
            src_transform,
            dst_transform,
            kwargs["src_crs"],
            kwargs["dst_crs"],
            mode="edge",
        )

        reprojected = rio.warp.reproject(
            src, empty(shape), src_transform=trans, **kwargs
        )[0]

        if reprojected.ndim != src.ndim:
            reprojected = reprojected.squeeze(axis=0)
        return cast("NDArray", reprojected)

    data_vars = ds.data_vars.values() if isinstance(ds, xr.Dataset) else (ds,)
    dtypes = {da.dtype for da in data_vars}
    assert len(dtypes) == 1, "regrid can only reproject datasets with homogeneous dtype"

    return cast(
        "Dataset | DataArray",
        (
            xr
            .apply_ufunc(
                _reproject,
                ds,
                input_core_dims=[[namey, namex]],
                output_core_dims=[["yout", "xout"]],
                output_dtypes=[dtypes.pop()],
                dask_gufunc_kwargs={
                    "output_sizes": {"yout": dst_shape[0], "xout": dst_shape[1]}
                },
                dask="parallelized",
                kwargs=kwargs,
            )
            .rename({"yout": namey, "xout": namex})
            .assign_coords(**{
                namey: (namey, dimy.data, ds.coords[namey].attrs),
                namex: (namex, dimx.data, ds.coords[namex].attrs),
            })
            .assign_attrs(**ds.attrs)
        ),
    )
