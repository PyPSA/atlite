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
import numpy as np
import os, sys
from six import string_types

import logging
logger = logging.getLogger(__name__)

from . import config, datasets

from .convert import (convert_and_aggregate, heat_demand, hydro, temperature,
                      wind, pv, runoff, solar_thermal, soil_temperature)
from .preparation import (cutout_do_task, cutout_prepare,
                          cutout_produce_specific_dataseries,
                          cutout_get_meta, cutout_get_meta_view)
from .gis import compute_indicatormatrix

class Cutout(object):
    def __init__(self, name=None, cutout_dir=config.cutout_dir, **cutoutparams):
        self.name = name

        self.cutout_dir = os.path.join(cutout_dir, name)
        self.prepared = False

        if 'bounds' in cutoutparams:
            x1, y1, x2, y2 = cutoutparams.pop('bounds')
            cutoutparams.update(xs=slice(x1, x2),
                                ys=slice(y2, y1))

        if os.path.isdir(self.cutout_dir):
            self.meta = meta = xr.open_dataset(self.datasetfn()).stack(**{'year-month': ('year', 'month')})
            # check datasets very rudimentarily, series and coordinates should be checked as well
            if all(os.path.isfile(self.datasetfn(ym)) for ym in meta.coords['year-month'].to_index()):
                self.prepared = True
            else:
                assert False

            if 'module' in meta.attrs:
                cutoutparams['module'] = meta.attrs['module']
            else:
                logger.warning('module not given in meta file of cutout, assuming it is NCEP')
                cutoutparams['module'] = 'ncep'

            if {"xs", "ys", "years", "months"}.intersection(cutoutparams):
                # Assuming the user is interested in a subview into
                # the data, update meta in place for the time
                # dimension and save the xs, ys slices, separately
                self.meta = meta = self.get_meta_view(**cutoutparams)
                logger.info("Assuming a view into the prepared cutout: %s", self)

        else:
            logger.info("Cutout %s not found in directory %s, building new one", name, cutout_dir)

            if 'module' not in cutoutparams:
                d = config.weather_dataset.copy()
                d.update(cutoutparams)
                cutoutparams = d

        self.dataset_module = sys.modules['atlite.datasets.' + cutoutparams['module']]

        if not self.prepared:
            if {"xs", "ys", "years"}.difference(cutoutparams):
                raise TypeError("Arguments `xs`, `ys` and `years` need to be specified")
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
        return self.dataset_module.meta_data_config

    @property
    def weather_data_config(self):
        return self.dataset_module.weather_data_config

    @property
    def projection(self):
        return self.dataset_module.projection

    @property
    def coords(self):
        return self.meta.coords

    @property
    def shape(self):
        return len(self.coords["y"]), len(self.coords["x"])

    @property
    def extent(self):
        return (list(self.coords["x"].values[[0, -1]]) +
                list(self.coords["y"].values[[-1, 0]]))


    def grid_coordinates(self):
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        return np.asarray((np.ravel(xs), np.ravel(ys))).T

    def grid_cells(self):
        from shapely.geometry import box
        coords = self.grid_coordinates()
        span = (coords[self.shape[1]+1] - coords[0]) / 2
        return [box(*c) for c in np.hstack((coords - span, coords + span))]

    def __repr__(self):
        yearmonths = self.coords['year-month'].to_index()
        return ('<Cutout {} x={:.2f}-{:.2f} y={:.2f}-{:.2f} time={}/{}-{}/{} {}prepared>'
                .format(self.name,
                        self.coords['x'].values[0], self.coords['x'].values[-1],
                        self.coords['y'].values[0], self.coords['y'].values[-1],
                        yearmonths[0][0],  yearmonths[0][1],
                        yearmonths[-1][0], yearmonths[-1][1],
                        "" if self.prepared else "UN"))

    def indicatormatrix(self, shapes, shapes_proj='latlong'):
        return compute_indicatormatrix(self.grid_cells(), shapes, self.projection, shapes_proj)

    ## Preparation functions

    get_meta = cutout_get_meta

    get_meta_view = cutout_get_meta_view

    prepare = cutout_prepare

    produce_specific_dataseries = cutout_produce_specific_dataseries

    ## Conversion and aggregation functions

    convert_and_aggregate = convert_and_aggregate

    heat_demand = heat_demand

    temperature = temperature

    soil_temperature = soil_temperature

    solar_thermal = solar_thermal

    wind = wind

    pv = pv

    runoff = runoff

    hydro = hydro
