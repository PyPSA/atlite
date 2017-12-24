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
import pyproj
from six import iteritems
from itertools import groupby
from operator import itemgetter
import os
import glob

from ..config import sarah_dir
from .era5 import prepare_for_sarah

# Model and Projection Settings
projection = 'latlong'

def _rename_and_clean_coords(ds):
    ds = ds.rename({'lon': 'x', 'lat': 'y'})
    ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def prepare_meta_sarah(xs, ys, year, month, template_sis, template_side, module):
    fns = [next(glob.iglob(t.format(year=year, month=month)))
           for t in (template_sis, template_sid)]

    with xr.open_mfdataset(fn, compat='identical') as ds:
        ds = _rename_and_clean_coords(ds)
        ds = ds.coords.to_dataset()

        t = pd.Timestamp(year=year, month=month, day=1)
        ds['time'] = pd.date_range(t, t + pd.offsets.MonthOffset(),
                                   freq='1h', closed='left')

        return ds.sel(x=xs, y=ys).load()

def prepare_month_sarah(era5_func, xs, ys, year, month, template_sis, template_sid):
    fns = sum([glob.glob(t.format(year=year, model=model))
               for t in (template_sis, template_sid)], [])

    with xr.open_mfdataset(fns) as ds:
        ds = _rename_and_clean_coords(ds)
        ds = ds.sel(x=xs, y=ys)

        ds = ds.rename({'SIS': influx})

        x = ds.indexes['x']
        y = ds.indexes['y']

        lx, rx = x[[0, -1]]
        uy, ly = y[[0, -1]]
        dx = float(rx - lx)/float(len(x)-1)
        dy = float(uy - ly)/float(len(y)-1)

        ds_era = era5_func(year, month, xs, ys, dx, dy)
        for v in ds_era.data_vars:
            ds[v] = ds_era[v]

        ds = ds.load()

    return ((year, month), ds)

def tasks_monthly_sarah(xs, ys, yearmonths, prepare_func, era5_func, template_sis, template_sid, meta_attrs):
    model = meta_attrs['model']

    if not isinstance(xs, slice):
        xs = slice(*xs.values[[0, -1]])
    if not isinstance(ys, slice):
        ys = slice(*ys.values[[0, -1]])

    return [dict(prepare_func=prepare_func,
                 era5_func=era5_func,
                 template_sis=template_sis, template_sid=template_sid,
                 xs=xs, ys=ys, year=year, month=month)
            for year, month in yearmonths]

weather_data_config = {
    '_': dict(tasks_func=tasks_monthly_sarah,
              prepare_func=prepare_month_sarah,
              era5_func=prepare_for_sarah,
              template_sid=os.path.join(sarah_dir, 'sid_eur', 'SIDin{year}{month}*.nc'),
              template_sis=os.path.join(sarah_dir, 'sis_eur', 'SISin{year}{month}*.nc'))
}

meta_data_config = dict(prepare_func=prepare_meta_sarah,
                        template_sid=weather_data_config['_']['template_sid'],
                        template_sis=weather_data_config['_']['template_sis'])
