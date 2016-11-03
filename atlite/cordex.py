## Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas
## Hoersch (FIAS), Tom Brown (FIAS), Markus Schlott (FIAS)

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
import pyproj
from six import iteritems
from itertools import groupby
from operator import itemgetter
import os
import glob

from .config import cordex_dir
from .shapes import RotProj

# Model and Projection Settings
model = 'CNRM-CERFACS_CNRM_CM5'
projection = RotProj(dict(proj='ob_tran', o_proj='longlat', lon_0=180,
                          o_lon_p=-162, o_lat_p=39.25))
def rename_and_clean_coords(ds):
    ds = ds.rename({'lon': 'glon', 'lat': 'glat'})
    ds = ds.rename({'rlon': 'lon', 'rlat': 'lat'})
    # drop some coordinates and variables we do not use
    ds = ds.drop((set(ds.coords) | set(ds.data_vars))
                 & {'bnds', 'height', 'rotated_pole'})
    return ds

def prepare_data_cordex(fn, year, months, oldname, newname, lons, lats):
    with xr.open_dataset(fn, engine="pynio") as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.rename({oldname: newname})
        return [((year, m), ds.sel(time="{}-{}".format(year, m)).load())
                for m in months]

def prepare_meta_cordex(lons, lats, year, month, template, module, model=model):
    fn = next(glob.iglob(template.format(year=year, model=model)))
    with xr.open_dataset(fn, engine="pynio") as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.coords.to_dataset()
        return ds.sel(time="{}-{}".format(year, month)).load()

def tasks_yearly_cordex(lons, lats, yearmonths, prepare_func, template, oldname, newname, meta_attrs):
    model = meta_attrs['model']
    return [dict(prepare_func=prepare_func,
                 lons=lons, lats=lats, oldname=oldname, newname=newname,
                 fn=next(glob.iglob(template.format(year=year, model=model))),
                 year=year, months=list(map(itemgetter(1), yearmonths)))
            for year, yearmonths in groupby(yearmonths, itemgetter(0))]

weather_data_config = {
    'influx': dict(tasks_func=tasks_yearly_cordex,
                   prepare_func=prepare_data_cordex,
                   oldname='rsds', newname='influx',
                   template=os.path.join(cordex_dir, '{model}', 'influx', 'rsds_*_{year}*.nc')),
    # Not yet available
    #'outflux': dict(tasks_func=tasks_yearly_cordex,
                    #prepare_func=prepare_outflux_cordex,
                    #template=os.path.join(cordex_dir, 'outflux/rsus_*_{year}*.nc')),
    'temperature': dict(tasks_func=tasks_yearly_cordex,
                        prepare_func=prepare_data_cordex,
                        oldname='tas', newname='temperature',
                        template=os.path.join(cordex_dir, '{model}', 'temperature', 'tas_*_{year}*.nc')),
    'wnd10m': dict(tasks_func=tasks_yearly_cordex,
                   prepare_func=prepare_data_cordex,
                   oldname='sfcWind', newname='wnd10m',
                   template=os.path.join(cordex_dir, '{model}', 'wind', 'sfcWind_*_{year}*.nc')),
    # Not yet available
    #'roughness': dict(tasks_func=tasks_yearly_cordex,
                      #prepare_func=prepare_roughness_cordex,
                      #template=os.path.join(cordex_dir, 'roughness/'))
}

meta_data_config = dict(prepare_func=prepare_meta_cordex,
                        template=os.path.join(cordex_dir, '{model}', 'temperature', 'tas_*_{year}*.nc'))
