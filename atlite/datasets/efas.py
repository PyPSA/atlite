## Copyright 2019 Jonas Hoersch

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

import os
import pandas as pd
import numpy as np
import xarray as xr
import shutil
from six.moves import range
from contextlib import contextmanager
from tempfile import mkstemp
import logging
logger = logging.getLogger(__name__)

try:
    import cdsapi
    has_cdsapi = True
except ImportError:
    has_cdsapi = False

# Model and Projection Settings
projection = 'latlong'

from .era5 import _get_data, _area, _rename_and_clean_coords

def prepare_meta_efas(xs, ys, year, month, module):
    # TODO
    # Reference of the quantities
    # https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
    # Geopotential is aka Orography in the CDS:
    # https://confluence.ecmwf.int/pages/viewpage.action?pageId=78296105
    #
    # (shortName) | (name)                        | (paramId)
    # z           | Geopotential (CDS: Orography) | 129

    # TODO function is used to return a preview of the coordinates, that is
    # going to be stored in the meta.nc file. The code below did query the
    # variable orography and then made up a date_range, today i'd prefer
    # something similar to the function on the v0.2 branch:
    # https://github.com/FRESNA/atlite/blob/8c9faa8fa7ad8ff73e3424f86b3d8ada33b7dbbf/atlite/datasets/era5.py#L128

    with _get_data(variable='orography',
                   year=year, month=month, day=1,
                   area=_area(xs, ys)) as ds:
        ds = _rename_and_clean_coords(ds)
        ds = _add_height(ds)

        t = pd.Timestamp(year=year, month=month, day=1)
        ds['time'] = pd.date_range(t, t + pd.DateOffset(months=1),
                                   freq='1h', closed='left')

        return ds.load()

def prepare_month_efas(year, month, xs, ys):
    area = _area(xs, ys)

    # Reference of the quantities
    # https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
    # TODO needs an update

    with _get_data(product="TODO-efas-product-name",
                   area=area, year=year, month=month,
                   variable=[

                   ]) as ds:

        ds = _rename_and_clean_coords(ds)

        # TODO
        # - which variables should be extracted and saved in which atlite fields
        # - do we need to get additional data from era5, maybe using a similar
        #   function as prepare_for_sarah in era5.py?

        yield (year, month), ds

def tasks_monthly_efas(xs, ys, yearmonths, prepare_func, meta_attrs):
    if not isinstance(xs, slice):
        xs = slice(*xs.values[[0, -1]])
    if not isinstance(ys, slice):
        ys = slice(*ys.values[[0, -1]])

    return [dict(prepare_func=prepare_func, xs=xs, ys=ys, year=year, month=month)
            for year, month in yearmonths]

weather_data_config = {
    '_': dict(tasks_func=tasks_monthly_efas,
              prepare_func=prepare_month_efas)
}

meta_data_config = dict(prepare_func=prepare_meta_efas)
