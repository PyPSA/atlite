# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module containing specific operations for creating cutouts from the SARAH2 dataset.
"""

import os, glob
import pandas as pd
import numpy as np
import xarray as xr
from dask import delayed
from functools import partial

import logging
logger = logging.getLogger(__name__)


def as_slice(zs, pad=True):
    if not isinstance(zs, slice):
        first, second, last = np.asarray(zs)[[0,1,-1]]
        dz = 0.1 * (second - first) if pad else 0.
        zs = slice(first - dz, last + dz)
    return zs

from ..gis import regrid, Resampling, maybe_swap_spatial_dims

from .era5 import get_data as get_era5_data

# Model and Projection Settings
projection = 'latlong'
resolution = 0.2

features = {
    'influx': ['influx_toa', 'influx_direct', 'influx_diffuse', 'influx', 'albedo'],
    'temperature': ['temperature', 'soil_temperature'],
}

static_features = {}

def _rename_and_clean_coords(ds, add_lon_lat=True):
    ds = ds.rename({'lon': 'x', 'lat': 'y'})
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def _get_filenames(sarah_dir, period):
    def _filenames_starting_with(name):
        pattern = os.path.join(sarah_dir, "**", f"{name}*.nc")
        files = pd.Series(glob.glob(pattern, recursive=True))
        assert not files.empty, \
            f"No files found at {pattern}. Make sure sarah_dir points to the correct directory!"

        files.index = pd.to_datetime(files.str.extract(r"SI.in(\d{8})", expand=False))
        return files.sort_index()
    files = pd.concat(dict(sis=_filenames_starting_with("SIS"),
                           sid=_filenames_starting_with("SID")),
                      join="inner", axis=1)

    if isinstance(period, (str, slice)):
        selector = period
    elif isinstance(period, pd.Period):
        selector = str(period)
    elif isinstance(period, pd.Timestamp):
        # Files are daily, anyway
        selector = period.strftime("%Y-%m-%d")
    else:
        raise TypeError(f"{period} should be one of pd.Timestamp or pd.Period")

    files = files.loc[selector]
    if isinstance(files, pd.Series):
        files = pd.DataFrame([files], index=[pd.Timestamp(selector)])

    assert not files.empty, f"Files have not been found in {sarah_dir} for {period}"

    return files.sort_index()

def get_coords(time, x, y, **creation_parameters):
    files = _get_filenames(creation_parameters['sarah_dir'], time)

    res = creation_parameters.get('resolution', resolution)

    coords = {'time': pd.date_range(start=files.index[0],
                                    end=files.index[-1] + pd.offsets.DateOffset(days=1),
                                    closed="left", freq="h")}
    if res is not None:
        coords.update({'lon': np.r_[-65. + (65. % res):65.01:res],
                       'lat': np.r_[-65. + (65. % res):65.01:res]})
    else:
        coords.update({'lon': np.r_[-65.:65.01:res],
                       'lat': np.r_[-65.:65.01:res]})

    ds = xr.Dataset(coords)
    ds = _rename_and_clean_coords(ds)
    ds = ds.sel(x=x, y=y, time=time)

    return ds

def get_data_era5(coords, period, feature, sanitize=True, tmpdir=None, **creation_parameters):
    x = coords.indexes['x']
    y = coords.indexes['y']
    xs = slice(*x[[0,-1]])
    ys = slice(*y[[0,-1]])

    lx, rx = x[[0, -1]]
    ly, uy = y[[0, -1]]
    dx = float(rx - lx)/float(len(x)-1)
    dy = float(uy - ly)/float(len(y)-1)

    del creation_parameters['x'], creation_parameters['y'], creation_parameters['sarah_dir']

    ds = get_era5_data(coords, period, feature, sanitize=sanitize, tmpdir=tmpdir,
                       x=xs, y=ys, dx=dx, dy=dy, **creation_parameters)

    if feature == 'influx':
        ds = ds[['influx_toa', 'albedo']]
    elif feature == 'temperature':
        ds = ds[['temperature']]
    else:
        raise NotImplementedError("Support only 'influx' and 'temperature' from ERA5")

    ds = ds.assign_coords(x=x, y=y)
    return ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])

def _get_data_sarah(coords, period, sarah_dir, **creation_parameters):
    files = _get_filenames(sarah_dir, period)
    res = creation_parameters.get('resolution', resolution)
    chunks = creation_parameters.get('chunks', {'time': 12})

    ds_sis = xr.open_mfdataset(files.sis, combine='by_coords', chunks=chunks)
    ds_sid = xr.open_mfdataset(files.sid, combine='by_coords', chunks=chunks)
    ds = xr.merge([ds_sis, ds_sid])

    ds = _rename_and_clean_coords(ds, add_lon_lat=False)
    ds = ds.sel(x=as_slice(coords['x']), y=as_slice(coords['y']))

    def interpolate(ds, dim='time'):
        def _interpolate1d(y):
            nan = np.isnan(y)
            if nan.all() or not nan.any(): return y
            x = lambda z: z.nonzero()[0]
            y = np.array(y)
            y[nan] = np.interp(x(nan), x(~nan), y[~nan])
            return y

        def _interpolate(a):
            return a.map_blocks(partial(np.apply_along_axis, _interpolate1d, -1), dtype=a.dtype)

        data_vars = ds.data_vars.values() if isinstance(ds, xr.Dataset) else (ds,)
        dtypes = {da.dtype for da in data_vars}
        assert len(dtypes) == 1, "interpolate only supports datasets with homogeneous dtype"

        return xr.apply_ufunc(_interpolate, ds,
                              input_core_dims=[[dim]],
                              output_core_dims=[[dim]],
                              output_dtypes=[dtypes.pop()],
                              output_sizes={dim: len(ds.indexes[dim])},
                              dask='allowed',
                              keep_attrs=True)

    # We can't use interpolate_na, since it only works along non-chunked
    # dimensions, and we can't chunk on spatial dimensions, since the
    # reprojection below requires the whole grid

    # ds.interpolate_na(dim='time')
    ds = interpolate(ds)

    def hourly_mean(ds):
        ds1 = ds.isel(time=slice(None, None, 2))
        ds2 = ds.isel(time=slice(1, None, 2))
        ds2 = ds2.assign_coords(time=ds2.indexes['time'] - pd.Timedelta(30, 'm'))
        ds = ((ds1 + ds2)/2)
        ds.attrs = ds1.attrs
        for v in ds.variables:
            ds[v].attrs = ds1[v].attrs
        return ds

    ds = hourly_mean(ds)

    ds['influx_diffuse'] = ((ds['SIS'] - ds['SID'])
                            .assign_attrs(long_name='Surface Diffuse Shortwave Flux',
                                            units='W m-2'))
    ds = ds.rename({'SID': 'influx_direct'}).drop('SIS')

    if res is not None:
        ds = regrid(ds, coords['x'], coords['y'], resampling=Resampling.average)

    ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])

    return ds

def get_data_sarah(coords, period, **creation_parameters):
    return delayed(_get_data_sarah)(coords, period, **creation_parameters)

def get_data(coords, period, feature, sanitize=True, tmpdir=None, **creation_parameters):
    if feature == 'influx':
        ds = delayed(xr.merge)([
            get_data_era5(coords, period, feature, sanitize=sanitize, tmpdir=tmpdir, **creation_parameters),
            get_data_sarah(coords, period, **creation_parameters)
        ])
    elif feature == 'temperature':
        ds = get_data_era5(coords, period, feature, sanitize=sanitize, tmpdir=tmpdir, **creation_parameters)
    else:
        raise NotImplementedError(f"Feature '{feature}' has not been implemented for dataset sarah")

    return ds
