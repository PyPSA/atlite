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
from warnings import warn
from pyproj import CRS, Transformer
import geopandas as gpd
from shapely.ops import transform
import rasterio as rio
import rasterio.warp
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

    ds = xr.Dataset({'x': np.arange(-180, 180, dx),
                     'y': np.arange(-90, 90, dy),
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
    Compute the indicatormatrix

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
