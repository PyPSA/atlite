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

import xarray as xr
import pandas as pd
import numpy as np
import os, sys, shutil
import filelock
from six import itervalues
from six.moves import map
from multiprocessing import Pool

import logging
logger = logging.getLogger(__name__)

from . import config, ncep, cordex
from .convert import convert_and_aggregate, heat_demand, wind
from .aggregate import aggregate_sum, aggregate_matrix
from .shapes import spdiag, compute_indicatormatrix
from .preparation import (cutout_do_task, cutout_prepare,
                          cutout_produce_specific_dataseries, cutout_get_meta)

class Cutout(object):
    def __init__(self, name=None, nprocesses=None, weather_dataset=None,
                 cutout_dir=config.cutout_dir, **cutoutparams):
        self.name = name
        self.nprocesses = nprocesses

        self.cutout_dir = os.path.join(cutout_dir, name)
        self.prepared = False

        if os.path.isdir(self.cutout_dir):
            self.meta = meta = xr.open_dataset(self.datasetfn()).stack(**{'year-month': ('year', 'month')})
            # check datasets very rudimentarily, series and coordinates should be checked as well
            if all(os.path.isfile(self.datasetfn(ym)) for ym in meta.coords['year-month'].to_index()):
                self.prepared = True
            else:
                assert False
            module = meta.attrs['atlite_module']

        if weather_dataset is None:
            d = config.weather_dataset.copy()
            d.update(cutoutparams)
            cutoutparams = d
            module = cutoutparams['atlite_module']

        self.dataset_module = sys.modules['atlite.' + module]

        if not self.prepared:
            if {"lons", "lats", "years"}.difference(cutoutparams):
                raise TypeError("Arguments `lons`, `lats` and `years` need to be specified")
            self.meta = self.get_meta(**cutoutparams)

    def datasetfn(self, *args):
        dataset = None

        if len(args) == 2:
            dataset = args
        elif len(args) == 1:
            dataset = args[0]
        else:
            dataset = None
        return os.path.join(self.cutout_dir, ("meta.nc"
                                              if dataset is None
                                              else "{}{:0>2}.nc".format(*dataset)))

    @property
    def meta_data_config(self):
        self.dataset_module.meta_data_config

    @property
    def weather_data_config(self):
        self.dataset_module.weather_data_config

    @property
    def projection(self):
        self.dataset_module.projection

    @property
    def coords(self):
        return self.meta.coords

    @property
    def shape(self):
        return len(self.coords["lon"]), len(self.coords["lat"])

    @property
    def extent(self):
        return (list(self.coords["lon"].values[[0, -1]]) +
                list(self.coords["lat"].values[[-1, 0]]))


    def grid_coordinates(self):
        lats, lons = np.meshgrid(self.coords["lat"], self.coords["lon"])
        return np.asarray((np.ravel(lons), np.ravel(lats))).T

    def grid_cells(self):
        from shapely.geometry import box
        coords = self.grid_coordinates()
        span = (coords[self.shape[1]+1] - coords[0]) / 2
        return [box(*c) for c in np.hstack((coords - span, coords + span))]

    def __repr__(self):
        yearmonths = self.coords['year-month'].to_index()
        return ('<Cutout {} lon={:.2f}-{:.2f} lat={:.2f}-{:.2f} time={}/{}-{}/{} {}prepared>'
                .format(self.name,
                        self.coords['lon'].values[0], self.coords['lon'].values[-1],
                        self.coords['lat'].values[0], self.coords['lat'].values[-1],
                        yearmonths[0][0],  yearmonths[0][1],
                        yearmonths[-1][0], yearmonths[-1][1],
                        "" if self.prepared else "UN"))

    ## Preparation functions

    get_meta = cutout_get_meta

    prepare = cutout_prepare

    produce_specific_dataseries = cutout_produce_specific_dataseries

    ## Conversion and aggregation functions

    convert_and_aggregate = convert_and_aggregate

    heat_demand = heat_demand

    wind = wind
