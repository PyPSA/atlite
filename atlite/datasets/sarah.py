# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module containing specific operations for creating cutouts from the SARAH2 dataset.
"""

from ..gis import regrid, Resampling, maybe_swap_spatial_dims
from .era5 import get_data as get_era5_data
import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from dask import delayed
from functools import partial

import logging
logger = logging.getLogger(__name__)


# def as_slice(zs, pad=True):
#     if not isinstance(zs, slice):
#         first, second, last = np.asarray(zs)[[0, 1, -1]]
#         dz = 0.1 * (second - first) if pad else 0.
#         zs = slice(first - dz, last + dz)
#     return zs


# Model, Projection and Resolution Settings
projection = 'latlong'
dx = 0.05
dy = 0.05
dt = '30m'
features = {'influx': ['influx_direct', 'influx_diffuse',]}
static_features = {}


def _rename_and_clean_coords(ds, add_lon_lat=True):
    ds = ds.rename({'lon': 'x', 'lat': 'y'})
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds


def get_filenames(sarah_dir, coords):

    def _filenames_starting_with(name):
        pattern = os.path.join(sarah_dir, "**", f"{name}*.nc")
        files = pd.Series(glob.glob(pattern, recursive=True))
        assert not files.empty, (f"No files found at {pattern}. Make sure "
                                 "sarah_dir points to the correct directory!")

        files.index = pd.to_datetime(files.str.extract(r"SI.in(\d{8})",
                                                       expand=False))
        return files.sort_index()
    files = pd.concat(dict(sis=_filenames_starting_with("SIS"),
                           sid=_filenames_starting_with("SID")),
                      join="inner", axis=1)

    start = coords['time'].to_index()[0]
    end = coords['time'].to_index()[-1]

    if (start < files.index[0]) or (end > files.index[-1]):
        logger.error(f"Files in {sarah_dir} do not cover the whole time span "
                     f" {start}-{end}")

    return files.loc[(files.index >= start) & (files.index <= end)].sort_index()



def interpolate(ds, dim='time'):
    '''
    Interpolate NaNs in a dataset along a chunked dimension.

    This function is similar to similar to xr.Dataset.interpolate_na but can
    be used for interpolating along an chunked dimensions (default 'time'').
    As the sarah data has mulitple nan in the areas of dawn and nightfall
    and the data is per default chunked along the time axis, use this function
    to interpolate those nans.
    '''
    def _interpolate1d(y):
        nan = np.isnan(y)
        if nan.all() or not nan.any():
            return y

        def x(z): return z.nonzero()[0]
        y = np.array(y)
        y[nan] = np.interp(x(nan), x(~nan), y[~nan])
        return y

    def _interpolate(a):
        return a.map_blocks(partial(np.apply_along_axis, _interpolate1d, -1),
                            dtype=a.dtype)

    data_vars = ds.data_vars.values() if isinstance(ds, xr.Dataset) else (ds,)
    dtypes = {da.dtype for da in data_vars}
    assert len(dtypes) == 1, \
        "interpolate only supports datasets with homogeneous dtype"

    return xr.apply_ufunc(_interpolate, ds,
                          input_core_dims=[[dim]],
                          output_core_dims=[[dim]],
                          output_dtypes=[dtypes.pop()],
                          output_sizes={dim: len(ds.indexes[dim])},
                          dask='allowed',
                          keep_attrs=True)


def get_data(cutout, feature, tmpdir, **creation_parameters):

    sarah_dir = creation_parameters['sarah_dir']

    files = get_filenames(sarah_dir, cutout.coords)
    # we only chunk on 'time' as the reprojection below requires the whole grid
    chunks = creation_parameters.get('chunks', {'time': 12})

    ds_sis = xr.open_mfdataset(files.sis, combine='by_coords', chunks=chunks)
    ds_sid = xr.open_mfdataset(files.sid, combine='by_coords', chunks=chunks)
    ds = xr.merge([ds_sis, ds_sid])

    ds = _rename_and_clean_coords(ds, add_lon_lat=False)
    ds = ds.sel(x=cutout.coords['x'], y=cutout.coords['y'])

    if creation_parameters.pop('interpolate', False):
        ds = interpolate(ds)
    else:
        ds = ds.fillna(0)

    ds = ds.resample(time='h').mean()

    ds['influx_diffuse'] = (
        (ds['SIS'] -
         ds['SID']) .assign_attrs(
            long_name='Surface Diffuse Shortwave Flux',
            units='W m-2'))
    ds = ds.rename({'SID': 'influx_direct'}).drop_vars('SIS')

    # if res is not None:
    #     ds = regrid(ds, coords['x'], coords['y'],
    #                 resampling=Resampling.average)

    ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])

    return ds

