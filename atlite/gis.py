# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Renewable Energy Atlas Lite (Atlite)

Light-weight version of Aarhus RE Atlas for converting weather data to power
systems data
"""

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy as sp
import scipy.sparse
from collections import OrderedDict
from warnings import warn
from itertools import product
from functools import partial
import pyproj
from shapely.prepared import prep
from shapely.ops import transform
from shapely.geometry import box
import rasterio as rio
import rasterio.warp
from rasterio.warp import Resampling
from rtree.index import Index
from shapely.strtree import STRtree

import logging
logger = logging.getLogger(__name__)


def get_coords(**cutoutparams):
    if {"x", "y", "time"}.difference(cutoutparams):
        raise RuntimeError("Arguments `x`, `y` and `time` need to "
                           "be specified (or `bounds` instead of `x` and `y`)")

    x = cutoutparams['x']
    y = cutoutparams['y']
    time = cutoutparams['time']

    x = slice(*sorted([x.start, x.stop]))
    y = slice(*sorted([y.start, y.stop]))

    dx = cutoutparams.get("dx", 0.25)
    dy = cutoutparams.get("dy", 0.25)
    dt = cutoutparams.get("dt", 'h')

    ds = xr.Dataset({'x': np.r_[-180:180:dx],
                     'y': np.r_[-90:90:dy],
                     'time': pd.date_range(start="1979", end="now", freq=dt)})
    ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    ds = ds.sel(x=x, y=y, time=time)
    return ds


def spdiag(v):
    N = len(v)
    inds = np.arange(N + 1, dtype=np.int32)
    return sp.sparse.csr_matrix((v, inds[:-1], inds), (N, N))


class RotProj(pyproj.Proj):
    def __call__(self, x, y, inverse=False, **kw):
        if inverse:
            gx, gy = super(RotProj, self).__call__(x, y,
                                                   inverse=False, **kw)
            return np.rad2deg(gx), np.rad2deg(gy)
        else:
            return super(RotProj, self).__call__(np.deg2rad(x),
                                                 np.deg2rad(y),
                                                 inverse=True, **kw)


def as_projection(p):
    if isinstance(p, pyproj.Proj):
        return p
    elif isinstance(p, str):
        return pyproj.Proj(dict(proj=p))
    else:
        return pyproj.Proj(p)


def reproject_shapes(shapes, p1, p2):
    """
    Project a collection of `shapes` from one projection `p1` to
    another projection `p2`

    Projections can be given as strings or instances of pyproj.Proj.
    Special care is taken for the case where the final projection is
    of type rotated pole as handled by RotProj.
    """

    if p1 == p2:
        return shapes

    if isinstance(p1, RotProj):
        if p2 == 'latlong':
            def reproject_points(x, y): return p1(x, y, inverse=True)
        else:
            raise NotImplementedError("`p1` can only be a RotProj if `p2` is "
                                      "latlong!")

    if isinstance(p2, RotProj):
        shapes = reproject(shapes, p1, 'latlong')
        reproject_points = p2
    else:
        reproject_points = partial(pyproj.transform, as_projection(p1),
                                   as_projection(p2))

    def _reproject_shape(shape):
        return transform(reproject_points, shape)

    if isinstance(shapes, pd.Series):
        return shapes.map(_reproject_shape)
    elif isinstance(shapes, dict):
        return OrderedDict((k, _reproject_shape(v)) for k, v in shapes.items())
    else:
        return list(map(_reproject_shape, shapes))


def reproject(shapes, p1, p2):
    warn("reproject has been renamed to reproject_shapes", DeprecationWarning)
    return reproject_shapes(shapes, p1, p2)


reproject.__doc__ = reproject_shapes.__doc__


def compute_indicatormatrix(
        orig,
        dest,
        orig_proj='latlong',
        dest_proj='latlong'):
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

    dest = reproject_shapes(dest, dest_proj, orig_proj)
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
      - src_crs, dst_crs define the different crs (default: EPSG:4326)
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
    kwargs.setdefault("src_crs", dict(init='EPSG:4326'))
    kwargs.setdefault("dst_crs", dict(init='EPSG:4326'))

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
                           output_sizes={'yout': dst_shape[0],
                                         'xout': dst_shape[1]},
                           dask='parallelized',
                           kwargs=kwargs)
            .rename({'yout': namey, 'xout': namex})
            .assign_coords(**{namey: (namey, dimy, ds.coords[namey].attrs),
                              namex: (namex, dimx, ds.coords[namex].attrs)})
            .assign_attrs(**ds.attrs))
