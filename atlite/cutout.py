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
from warnings import warn

import logging
logger = logging.getLogger(__name__)


from . import datasets, utils

from .convert import (convert_and_aggregate, heat_demand, hydro, temperature,
                      wind, pv, runoff, solar_thermal, soil_temperature)
from .gis import compute_indicatormatrix
from .data import requires_coords, requires_windowed, cutout_prepare

class Cutout(object):
    dataset_module = None

    def __init__(self, name=None, data=None, cutout_dir=".", **cutoutparams):
        if isinstance(name, xr.Dataset):
            data = name
            name = data.attrs.get("name", "unnamed")

        if '/' in name:
            cutout_dir, name = os.path.split(name)

        self.name = name
        self.cutout_dir = cutout_dir

        if 'bounds' in cutoutparams:
            x1, y1, x2, y2 = cutoutparams.pop('bounds')
            cutoutparams.update(x=slice(x1, x2),
                                y=slice(y2, y1))

        if {'xs', 'ys'}.intersection(cutoutparams):
            warn("The arguments `xs` and `ys` have been deprecated in favour of `x` and `y`", DeprecationWarning)
            if 'xs' in cutoutparams: cutoutparams['x'] = cutoutparams.pop('xs')
            if 'ys' in cutoutparams: cutoutparams['y'] = cutoutparams.pop('ys')

        if {'years', 'months'}.intersection(cutoutparams):
            warn("The arguments `years` and `months` have been deprecated in favour of `time`", DeprecationWarning)
            assert 'years' in cutoutparams
            months = cutoutparams.pop("months", slice(1, 12))
            years = cutoutparams.pop("years")
            cutoutparams["time"] = slice("{}-{}".format(years.start, months.start),
                                         "{}-{}".format(years.stop, months.stop))

        if data is None:
            self.is_view = False

            if os.path.isfile(self.cutout_fn):
                data = xr.open_dataset(self.cutout_fn)
                prepared_features = data.attrs.get('prepared_features')
                assert prepared_features is not None, \
                    f"{self.cutout_fn} does not have the required attribute `prepared_features`"
                if not isinstance(prepared_features, list):
                    data.attrs['prepared_features'] = [prepared_features]
            elif os.path.isdir(os.path.join(self.cutout_dir, self.name)):
                data = utils.migrate_from_cutout_directory(os.path.join(self.cutout_dir, self.name),
                                                           self.name, self.cutout_fn, cutoutparams)
                self.is_view = True
            else:
                logger.info(f"Cutout {self.name} not found in directory {self.cutout_dir}, building new one")

                if {"x", "y", "time"}.difference(cutoutparams):
                    raise RuntimeError("Arguments `x`, `y` and `time` need to be specified (or `bounds` instead of `x` and `y`)")

                if 'module' not in cutoutparams:
                    logger.warning("`module` was not specified, falling back to 'era5'")

                data = xr.Dataset(attrs={'module': cutoutparams.pop('module', 'era5'),
                                         'prepared_features': [],
                                         'creation_parameters': str(cutoutparams)})

                # Remove `x`, `y` and `time` from cutoutparams, so we can use them for indicating a view
                cutoutparams = {k: v for k, v in cutoutparams.items() if k not in {"x", "y", "time"}}
        else:
            # User-provided dataset
            # TODO needs to be checked, sanitized and marked as immutable (is_view)
            self.is_view = True

        if 'module' in cutoutparams:
            module = cutoutparams.pop('module')
            if module != data.attrs.get('module'):
                logger.warning("Selected module '{}' disagrees with specification in dataset '{}'. Taking your choice."
                               .format(module, data.attrs.get('module')))
                data.attrs['module'] = module
        elif 'module' not in data.attrs:
            logger.warning("No module given as argument nor in the dataset. Falling back to 'era5'.")
            data.attrs['module'] = 'era5'

        if {"x", "y", "time"}.intersection(cutoutparams):
            # Assuming the user is interested in a subview into the data
            data = data.sel(**cutoutparams)
            self.is_view = True
            logger.info("Assuming a view into the cutout: {}".format(cutoutparams))

        self.data = data
        self.dataset_module = sys.modules['atlite.datasets.' + self.data.attrs['module']]

    @property
    def cutout_fn(self):
        return os.path.join(self.cutout_dir, self.name + ".nc")

    @property
    def projection(self):
        return self.data.attrs.get('projection', self.dataset_module.projection)

    @property
    @requires_coords
    def coords(self):
        return self.data.coords

    @property
    def meta(self):
        warn("The `meta` attribute is deprecated in favour of direct access to `data`", DeprecationWarning)
        return xr.Dataset(self.coords, attrs=self.data.attrs)

    @property
    def shape(self):
        return len(self.coords["y"]), len(self.coords["x"])

    @property
    def extent(self):
        return (list(self.coords["x"].values[[0, -1]]) +
                list(self.coords["y"].values[[-1, 0]]))

    @property
    def prepared(self):
        warn("The `prepared` attribute is deprecated in favour of the fine-grained `prepared_features` list", DeprecationWarning)
        return self.prepared_features == set(self.dataset_module.features)

    @property
    def prepared_features(self):
        return set(self.data.attrs["prepared_features"])

    def grid_coordinates(self):
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        return np.asarray((np.ravel(xs), np.ravel(ys))).T

    def grid_cells(self):
        from shapely.geometry import box
        coords = self.grid_coordinates()
        span = (coords[self.shape[1]+1] - coords[0]) / 2
        return [box(*c) for c in np.hstack((coords - span, coords + span))]

    def __repr__(self):
        return ('<Cutout {} x={:.2f}-{:.2f} y={:.2f}-{:.2f} time={}-{} prepared_features={} is_view={}>'
                .format(self.name,
                        self.coords['x'].values[0], self.coords['x'].values[-1],
                        self.coords['y'].values[0], self.coords['y'].values[-1],
                        self.coords['time'].values[0], self.coords['time'].values[-1],
                        list(self.prepared_features),
                        self.is_view))

    def indicatormatrix(self, shapes, shapes_proj='latlong'):
        return compute_indicatormatrix(self.grid_cells(), shapes, self.projection, shapes_proj)

    ## Preparation functions

    prepare = cutout_prepare

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
