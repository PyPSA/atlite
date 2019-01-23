## Copyright 2016-2017 Jonas Hoersch (FIAS), Tom Brown (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Renewable Energy Atlas Lite (Atlite)

Light-weight version of Aarhus RE Atlas for converting weather data to power systems data
"""

from __future__ import absolute_import

import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
import pyproj
from six import iteritems
from itertools import groupby
from operator import itemgetter
import os
import glob
import logging
logger = logging.getLogger(__name__)

from collections import deque
from contextlib import contextmanager

@contextmanager
def receive(it):
    yield next(it)
    for i in it: pass

def as_slice(zs, pad=True):
    if not isinstance(zs, slice):
        first, second, last = np.asarray(zs)[[0,1,-1]]
        dz = 0.1 * (second - first) if pad else 0.
        zs = slice(first - dz, last + dz)
    return zs

from ..config import sarah_dir
from ..gis import regrid, Resampling, maybe_swap_spatial_dims
from .era5 import prepare_for_sarah

# Model and Projection Settings
projection = 'latlong'
resolution = None

def _rename_and_clean_coords(ds, add_lon_lat=True):
    ds = ds.rename({'lon': 'x', 'lat': 'y'})
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def prepare_meta_sarah(xs, ys, year, month, template_sis, template_sid, module, resolution=resolution):
    fns = [next(glob.iglob(t.format(year=year, month=month)))
           for t in (template_sis, template_sid)]

    with xr.open_mfdataset(fns, compat='identical') as ds:
        ds = _rename_and_clean_coords(ds)
        ds = ds.coords.to_dataset()

        t = pd.Timestamp(year=year, month=month, day=1)
        ds['time'] = pd.date_range(t, t + pd.DateOffset(months=1),
                                   freq='1h', closed='left')

        if resolution is not None:
            def p(s):
                s += 0.1*resolution
                return s - (s % resolution)
            xs = np.arange(p(xs.start), p(xs.stop) + 1.1*resolution, resolution)
            ys = np.arange(p(ys.start), p(ys.stop) - 0.1*resolution, - resolution)
            ds = ds.sel(x=xs, y=ys, method='nearest')
        else:
            ds = ds.sel(x=as_slice(xs), y=as_slice(ys))

        return ds.load()

def prepare_month_sarah(era5_func, xs, ys, year, month, template_sis, template_sid, resolution):
    with xr.open_mfdataset(template_sis.format(year=year, month=month)) as ds_sis, \
         xr.open_mfdataset(template_sid.format(year=year, month=month)) as ds_sid:
        ds = xr.merge([ds_sis, ds_sid])

        ds = _rename_and_clean_coords(ds, add_lon_lat=False)
        ds = ds.sel(x=as_slice(xs), y=as_slice(ys))

        def interpolate(ds, dim='time'):
            def _interpolate1d(y):
                nan = np.isnan(y)
                if nan.all(): return y
                x = lambda z: z.nonzero()[0]
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

        if resolution is not None:
            ds = regrid(ds, xs, ys, resampling=Resampling.average)

        x = ds.indexes['x']
        y = ds.indexes['y']
        xs = slice(*x[[0,-1]])
        ys = slice(*y[[0,-1]])

        lx, rx = x[[0, -1]]
        uy, ly = y[[0, -1]]
        dx = float(rx - lx)/float(len(x)-1)
        dy = float(uy - ly)/float(len(y)-1)

        logger.debug("Getting ERA5 data")
        with receive(era5_func(year, month, xs, ys, dx, dy, chunks=dict(time=24))) as ds_era:
            logger.debug("Merging SARAH and ERA5 data")
            ds_era = ds_era.assign_coords(x=ds.indexes['x'], y=ds.indexes['y'])
            ds = xr.merge([ds, ds_era]).assign_attrs(ds.attrs)
            ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])

            yield ((year, month), ds)

def tasks_monthly_sarah(xs, ys, yearmonths, prepare_func, era5_func, template_sis, template_sid, meta_attrs):
    resolution = meta_attrs.get('resolution', None)

    return [dict(prepare_func=prepare_func,
                 era5_func=era5_func,
                 template_sis=template_sis, template_sid=template_sid,
                 xs=xs, ys=ys, year=year, month=month,
                 resolution=resolution)
            for year, month in yearmonths]

weather_data_config = {
    '_': dict(tasks_func=tasks_monthly_sarah,
              prepare_func=prepare_month_sarah,
              era5_func=prepare_for_sarah,
              template_sid=os.path.join(sarah_dir, 'sid', 'SIDin{year}{month:02}*.nc'),
              template_sis=os.path.join(sarah_dir, 'sis', 'SISin{year}{month:02}*.nc'))
}

meta_data_config = dict(prepare_func=prepare_meta_sarah,
                        template_sid=weather_data_config['_']['template_sid'],
                        template_sis=weather_data_config['_']['template_sis'])
