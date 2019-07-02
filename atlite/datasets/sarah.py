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
import glob
import logging
logger = logging.getLogger(__name__)


def as_slice(zs, pad=True):
    if not isinstance(zs, slice):
        first, second, last = np.asarray(zs)[[0,1,-1]]
        dz = 0.1 * (second - first) if pad else 0.
        zs = slice(first - dz, last + dz)
    return zs

from ..utils import timeindex_from_slice, receive
from ..gis import regrid, Resampling, maybe_swap_spatial_dims

from .era5 import get_data as get_era5_data

# Model and Projection Settings
projection = 'latlong'
resolution = None

def _rename_and_clean_coords(ds, add_lon_lat=True):
    ds = ds.rename({'lon': 'x', 'lat': 'y'})
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def _get_filenames(sarah_dir, date):
    year = date.year
    month = date.month

    return (os.path.join(sarah_dir, 'sis', f'SISin{year}{month:02}*.nc'),
            os.path.join(sarah_dir, 'sid', f'SIDin{year}{month:02}*.nc'))

def get_coords(time, x, y, **creation_parameters):
    timeindex = timeindex_from_slice(time)

    fns = _get_filenames(creation_parameters.get('sarah_dir', config.sarah_dir), timeindex[0])
    res = creation_parameters.get('resolution', resolution)

    with xr.open_mfdataset(fns, compat='identical') as ds:
        ds = _rename_and_clean_coords(ds)
        ds = ds.coords.to_dataset()

        if res is not None:
            def p(s):
                s += 0.1*res
                return s - (s % res)
            x = np.arange(p(x.start), p(x.stop) + 1.1*res, res)
            y = np.arange(p(y.start), p(y.stop) - 0.1*res, - res)
            ds = ds.sel(x=x, y=y, method='nearest')
        else:
            ds = ds.sel(x=x, y=y)

        ds['time'] = timeindex

        return ds.load()

def get_data(coords, date, feature, x, y, **creation_parameters):
    sis_fn, sid_fn = _get_filenames(creation_parameters.get('sarah_dir', sarah_dir), date)
    res = creation_parameters.get('resolution', resolution)

    with xr.open_mfdataset(sis_fn) as ds_sis, \
         xr.open_mfdataset(sid_fn) as ds_sid:
        ds = xr.merge([ds_sis, ds_sid])

        ds = _rename_and_clean_coords(ds, add_lon_lat=False)
        ds = ds.sel(x=as_slice(coords['x']), y=as_slice(coords['y']))

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

        if res is not None:
            ds = regrid(ds, coords['x'], coords['y'], resampling=Resampling.average)

        x = ds.indexes['x']
        y = ds.indexes['y']
        xs = slice(*x[[0,-1]])
        ys = slice(*y[[0,-1]])

        lx, rx = x[[0, -1]]
        uy, ly = y[[0, -1]]
        dx = float(rx - lx)/float(len(x)-1)
        dy = float(uy - ly)/float(len(y)-1)

        logger.debug("Getting ERA5 data")
        with receive(get_era5_data(coords, date, 'influx', xs, ys, dx, dy, chunks=dict(time=24))) as ds_era_influx, \
             receive(get_era5_data(coords, data, 'temperature', xs, ys, dx, dy, chunks=dict(time=24))) as ds_era_temp:
            logger.debug("Merging SARAH and ERA5 data")
            ds_era_influx = ds_era_influx.assign_coords(x=x, y=y)
            ds_era_temp = ds_era_temp.assign_coords(x=x, y=y)
            ds = xr.merge([ds,
                           ds_era_influx[['influx_toa', 'albedo']],
                           ds_era_temp[['temperature']]]).assign_attrs(ds.attrs)
            ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])

            yield ds
