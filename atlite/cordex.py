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

from .config import cordex_dir, name

def rename_coords(ds):
	ds = ds.rename({'lon': 'glon'})
	ds = ds.rename({'lat': 'glat'})
	ds = ds.rename({'rlon': 'lon'})
	ds = ds.rename({'rlat': 'lat'})
	return ds

def prepare_wnd10m_cordex(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
	ds = rename_coords(ds)
        ds = ds.rename({'sfcWind': 'wnd10m'})
        return [(yearmonth, ds.load())]

def prepare_influx_cordex(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
	ds = rename_coords(ds)
        ds = ds.rename({'rsds': 'influx'})
        return [(yearmonth, ds.load())]

def prepare_outflux_cordex(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
	ds = rename_coords(ds)
        ds = ds.rename({'rsus': 'outflux'})
        return [(yearmonth, ds.load())]

def prepare_temperature_cordex(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
	ds = rename_coords(ds)
        ds = ds.rename({'tas': 'temperature'})
        return [(yearmonth, ds.load())]

def prepare_roughness_cordex(fn, yearmonth, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
	ds = rename_coords(ds)
        ds = ds.rename({'': 'roughness'})
        return [(yearmonth, ds.load())]

def prepare_meta_cordex(lons, lats, year, month, template):
    fn = next(glob.iglob(template.format(year=year, month=month)))
    with xr.open_dataset(fn, engine="pynio") as ds:
	ds = rename_coords(ds)
        ds = ds.coords.to_dataset()
        return ds.load()

def tasks_yearly_cordex(lons, lats, yearmonths, prepare_func, template):
    return [dict(prepare_func=prepare_func,
                 lons=lons, lats=lats,
                 fn=next(glob.iglob(template.format(year=ym)))
                 yearmonth=ym)
            for ym in yearmonths]

cordex_dir = os.path.join(cordex_dir, name)

weather_data_config = {
    'influx': dict(tasks_func=tasks_yearly_cordex,
                   prepare_func=prepare_influx_cordex,
                   template=os.path.join(cordex_dir, 'influx/rsds_*_{year}*.nc')),
    # Not yet available
    #'outflux': dict(tasks_func=tasks_yearly_cordex,
                    #prepare_func=prepare_outflux_cordex,
                    #template=os.path.join(cordex_dir, 'outflux/rsus_*_{year}*.nc')),
    'temperature': dict(tasks_func=tasks_yearly_cordex,
                        prepare_func=prepare_temperature_cordex,
                        template=os.path.join(cordex_dir, 'temperature/tas_*_{year}*.nc')),
    'wnd10m': dict(tasks_func=tasks_yearly_cordex,
                   prepare_func=prepare_wnd10m_cordex,
                   template=os.path.join(cordex_dir, 'wind/sfcWind_*_{year}*.nc')),
    # Not yet available
    #'roughness': dict(tasks_func=tasks_yearly_cordex,
                      #prepare_func=prepare_roughness_cordex,
                      #template=os.path.join(cordex_dir, 'roughness/'))
}

meta_data_config = dict(prepare_func=prepare_meta_cordex,
                        template=os.path.join(cordex_dir, 'temperature/tas_*_{year}*.nc'))
