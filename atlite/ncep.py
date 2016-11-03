## Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)

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
from six import iteritems
import os
import glob

from .config import ncep_dir

engine = None
projection = 'latlong'

def convert_lons_lats_ncep(ds, lons, lats):
    if not isinstance(lons, slice):
        first, second, last = lons.values[[0,1,-1]]
        lons = slice(first - 0.1*(second - first), last + 0.1*(second - first))
    if not isinstance(lats, slice):
        first, second, last = lats.values[[0,1,-1]]
        lats = slice(first - 0.1*(second - first), last + 0.1*(second - first))

    ds = ds.sel(lat_0=lats)

    # Lons should go from -180. to +180.
    ds = xr.concat([ds.sel(lon_0=slice(lons.start + 360., lons.stop + 360.)),
                    ds.sel(lon_0=lons)],
                   dim="lon_0")
    ds = ds.assign_coords(lon_0=np.where(ds.coords['lon_0'].values <= 180,
                                         ds.coords['lon_0'].values,
                                         ds.coords['lon_0'].values - 360.))

    ds = ds.rename({'lon_0': 'lon', 'lat_0': 'lat'})
    return ds

def convert_time_hourly_ncep(ds, drop_time_vars=True):
    # Combine initial_time0 and forecast_time0
    ds = (ds
          .stack(time=("initial_time0_hours", "forecast_time0"))
          .assign_coords(time=np.ravel(ds.coords['initial_time0_hours'] + ds.coords['forecast_time0'])))
    if drop_time_vars:
        ds = ds.drop(['initial_time0', 'initial_time0_encoded'])
    return ds

def convert_unaverage_ncep(ds):
    # the fields ending in _avg contain averages which have to be unaveraged by using
    # \begin{equation}
    # \tilde x_1 = x_1 \quad \tilde x_i = i \cdot x_i - (i - 1) \cdot x_{i-1} \quad \forall i > 1
    # \end{equation}

    def unaverage(da, dim='forecast_time0'):
        coords = da.coords[dim]
        y = da * xr.DataArray(np.arange(1, len(coords)+1), dims=[dim], coords={dim: coords})
        return y - y.shift(**{dim: 1}).fillna(0.)
    for k, da in iteritems(ds):
        if k.endswith('_avg'):
            ds[k[:-len('_avg')]] = unaverage(da)
            ds = ds.drop(k)

    return ds

def prepare_wnd10m_ncep(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
        ds = convert_lons_lats_ncep(ds, lons, lats)
        ds = convert_time_hourly_ncep(ds)
        ds['wnd10m'] = np.sqrt(ds['VGRD_P0_L103_GGA0']**2 + ds['UGRD_P0_L103_GGA0']**2)
        ds = ds.drop(['VGRD_P0_L103_GGA0', 'UGRD_P0_L103_GGA0'])
        return [(yearmonth, ds.load())]

def prepare_influx_ncep(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
        ds = convert_lons_lats_ncep(ds, lons, lats)
        ds = convert_unaverage_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({'DSWRF_P8_L1_GGA0': 'influx'})
        return [(yearmonth, ds.load())]

def prepare_outflux_ncep(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
        ds = convert_lons_lats_ncep(ds, lons, lats)
        ds = convert_unaverage_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({'USWRF_P8_L1_GGA0': 'outflux'})
        return [(yearmonth, ds.load())]

def prepare_temperature_ncep(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
        ds = convert_lons_lats_ncep(ds, lons, lats)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({'TMP_P0_L103_GGA0': 'temperature'})
        return [(yearmonth, ds.load())]

def prepare_roughness_ncep(fn, lons, lats, yearmonths):
    with xr.open_dataset(fn, engine="pynio") as ds:
        # there are 3 different grids in the dataset, the one in use since 2011 is in lon_2, lat_2
        ds = ds.drop(['lon_0', 'lat_0', 'initial_time0_hours', 'lon_1', 'lat_1', 'initial_time1_hours',
                      'initial_time2_encoded', 'initial_time2'])
        ds = ds.rename({'initial_time2_hours': 'time', 'lon_2': 'lon_0', 'lat_2': 'lat_0'})
        ds = convert_lons_lats_ncep(ds, lons, lats)
        # roughness does not come on exactly the same grid as the
        # other data, so we interpolate with nearest grid point
        # selection
        ds = (ds.sel(lon=lons, lat=lats, method='nearest')
                .assign_coords(lon=lons, lat=lats))
        ds = ds.rename({'SFCR_P8_L1_GGA2': 'roughness'})
        # split time into months
        dt = pd.to_datetime(ds.coords['time'].values)
        ds = (ds.assign_coords(time=pd.MultiIndex.from_arrays([dt.year, dt.month], names=['year', 'month']))
                .unstack('time'))
        return [(ym, ds.sel(year=ym[0], month=ym[1]).load())
                for ym in yearmonths]

def prepare_meta_ncep(lons, lats, year, month, template, module):
    fn = next(glob.iglob(template.format(year=year, month=month)))
    with xr.open_dataset(fn, engine="pynio") as ds:
        ds = ds.coords.to_dataset()
        ds = convert_lons_lats_ncep(ds, lons, lats)
        ds = convert_time_hourly_ncep(ds, drop_time_vars=False)
        return ds.load()

def tasks_monthly_ncep(lons, lats, yearmonths, prepare_func, template, meta_attrs):
    return [dict(prepare_func=prepare_func,
                 lons=lons, lats=lats,
                 fn=next(glob.iglob(template.format(year=ym[0], month=ym[1]))),
                 yearmonth=ym)
            for ym in yearmonths]

def tasks_roughness_ncep(lons, lats, yearmonths, prepare_func, template, meta_attrs):
    return [dict(prepare_func=prepare_func,
                 lons=lons, lats=lats, yearmonths=yearmonths, fn=template)]

weather_data_config = {
    'influx': dict(tasks_func=tasks_monthly_ncep,
                   prepare_func=prepare_influx_ncep,
                   template=os.path.join(ncep_dir, '{year}{month:0>2}/dswsfc.*.grb2')),
    'outflux': dict(tasks_func=tasks_monthly_ncep,
                    prepare_func=prepare_outflux_ncep,
                    template=os.path.join(ncep_dir, '{year}{month:0>2}/uswsfc.*.grb2')),
    'temperature': dict(tasks_func=tasks_monthly_ncep,
                        prepare_func=prepare_temperature_ncep,
                        template=os.path.join(ncep_dir, '{year}{month:0>2}/tmp2m.*.grb2')),
    'wnd10m': dict(tasks_func=tasks_monthly_ncep,
                   prepare_func=prepare_wnd10m_ncep,
                   template=os.path.join(ncep_dir, '{year}{month:0>2}/wnd10m.*.grb2')),
    'roughness': dict(tasks_func=tasks_roughness_ncep,
                      prepare_func=prepare_roughness_ncep,
                      template=os.path.join(ncep_dir, 'roughness/flxf01.gdas.SFC_R.SFC.grb2'))
}

meta_data_config = dict(prepare_func=prepare_meta_ncep,
                        template=os.path.join(ncep_dir, '{year}{month:0>2}/tmp2m.*.grb2'))
