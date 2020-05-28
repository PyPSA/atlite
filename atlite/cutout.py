# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Base class for Atlite.
"""

# There is a binary incompatibility between the pip wheels of netCDF4 and
# rasterio, which leads to the first one to work correctly while the second
# loaded one fails by loading netCDF4 first, we ensure that most of atlite's
# functionality works fine, even when the pip wheels have been used, only for
# resampling the sarah dataset it is important to use conda.
# Refer to
# https://github.com/pydata/xarray/issues/2535,
# https://github.com/rasterio/rasterio-wheels/issues/12
from .utils import CachedAttribute
from .data import requires_windowed, cutout_prepare
from .gis import get_coords, compute_indicatormatrix
from .convert import (convert_and_aggregate, heat_demand, hydro, temperature,
                      wind, pv, runoff, solar_thermal, soil_temperature)
from . import datasets, utils
import netCDF4

import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
from warnings import warn
from shapely.geometry import box
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


class Cutout:

    dataset_module = None

    def __init__(self, path="unnamed.nc", data=None, **cutoutparams):
        """
        Provide an Atlite cutout object.

        Create a cutout object to use atlite operations on it.
        Based on the provided parameters, atlite first checks
        whether this cutout already exists on disk and if yes,
        loads this cutout.

        If the cutout does not yet exist on disk, then atlite
        creates an "unprepared" cutout object containing all the
        necessary information for creating the requested cutout,
        but does not yet create ("prepare") it fully.
        The process of preparing has to be manually started by
        `cutout.prepare()`.

        Parameters
        ----------
        path : str | path-like
            NetCDF from which to load or where to store the cutout
        data : (opt.) xr.Dataset
            Specify the NetCDF data directly
        module : ["era5","ncep","cordex","sarah"]
            The dataset which works as a basis for the cutout
            creation.
        time : str | slice
            Time range to include in the cutout, e.g. "2011" or
            ("2011-01-05", "2011-01-25")
        bounds : (opt.) GeoSeries.bounds | DataFrame
            The outer bounds of the cutout or as a DataFrame
            containing (min.long, min.lat, max.long, max.lat).
        x : (opt.) slice
            Outer longitudinal bounds for the cutout (west, east).
        y : (opt.) slice
            Outer latitudinal bounds for the cutout (south, north).

        """

        # Deprecate arguments 'name' and 'cutout_dir'. To be removed in later versions
        # For now prefer them over new argument 'path' (only if specified).
        name = cutoutparams.get("name", None)
        cutout_dir = cutoutparams.get("cutout_dir", None)
        if cutout_dir or name:
            warn(
                "The argument `cutout_dir` and `name` have been deprecated in "
                "favour of `path`.",
                DeprecationWarning)
            path = Path(cutout_dir if cutout_dir else ".") / \
                name if name else path
        elif isinstance(path, xr.Dataset):
            data = path
        else:
            path = Path(path)

        if not path.suffix and not path.is_dir():
            path = path.with_suffix(".nc")

        if 'bounds' in cutoutparams:
            x1, y1, x2, y2 = cutoutparams.pop('bounds')
            cutoutparams.update(x=slice(x1, x2), y=slice(y1, y2))

        if {'xs', 'ys'}.intersection(cutoutparams):
            warn(
                "The arguments `xs` and `ys` have been deprecated in favour of "
                "`x` and `y`", DeprecationWarning)
            if 'xs' in cutoutparams:
                cutoutparams['x'] = cutoutparams.pop('xs')
            if 'ys' in cutoutparams:
                cutoutparams['y'] = cutoutparams.pop('ys')

        if {'years', 'months'}.intersection(cutoutparams):
            warn("The arguments `years` and `months` have been deprecated in "
                 "favour of `time`", DeprecationWarning)
            assert 'years' in cutoutparams
            months = cutoutparams.pop("months", slice(1, 12))
            years = cutoutparams.pop("years")
            cutoutparams["time"] = slice("{}-{}".format(years.start, months.start),
                                         "{}-{}".format(years.stop, months.stop))

        if data is None:
            self.is_view = False

            if path.is_file():
                data = xr.open_dataset(str(path), cache=False)
                prepared_features = data.attrs.get('prepared_features')
                assert prepared_features is not None, (
                    f"{self.name} does not" " have the required attribute `prepared_features`")
                if not isinstance(prepared_features, list):
                    data.attrs['prepared_features'] = [prepared_features]
            elif path.is_dir():
                data = utils.migrate_from_cutout_directory(path)
                self.is_view = True
            else:
                logger.info(f"Cutout {path} not found, building new one")

                coords = get_coords(**cutoutparams)

                if 'module' not in cutoutparams:
                    logger.warning("`module` was not specified, falling back "
                                   "to 'era5'")
                module = cutoutparams.pop('module', 'era5')
                # TODO: check for dx, dy, x, y fine with module data

                data = xr.Dataset(
                    coords=coords,
                    attrs={
                        'module': module,
                        'prepared_features': [],
                        'creation_parameters': str(cutoutparams)})
        else:
            # User-provided dataset
            # TODO needs to be checked, sanitized and marked as immutable
            # (is_view)
            self.is_view = True

        if 'module' in cutoutparams:
            module = cutoutparams.pop('module')
            if module != data.attrs.get('module'):
                logger.warning(
                    f"Selected module '{module}' disagrees with "
                    f"specification in dataset '{data.attrs.get('module')}'. "
                    "Taking your choice.")
                data.attrs['module'] = module
        elif 'module' not in data.attrs:
            logger.warning("No module given as argument nor in the dataset. "
                           "Falling back to 'era5'.")
            data.attrs['module'] = 'era5'

        self.path = path
        self.data = data
        self.dataset_module = sys.modules['atlite.datasets.' +
                                          self.data.attrs['module']]

    @property
    def name(self):
        return self.path.stem

    @property
    def projection(self):
        return self.data.attrs.get(
            'projection', self.dataset_module.projection)

    @property
    def available_features(self):
        if self.dataset_module and not self.is_view:
            return set(self.dataset_module.features)
        else:
            return set()

    @property
    def coords(self):
        return self.data.coords

    @property
    def meta(self):
        warn("The `meta` attribute is deprecated in favour of direct "
             "access to `data`", DeprecationWarning)
        return xr.Dataset(self.coords, attrs=self.data.attrs)

    @property
    def shape(self):
        return len(self.coords["y"]), len(self.coords["x"])

    @property
    def extent(self):
        return (list(self.coords["x"].values[[0, -1]]) +
                list(self.coords["y"].values[[0, -1]]))

    @property
    def dx(self):
        return (self.coords['x'][1] - self.coords['x'][0]).item()

    @property
    def dy(self):
        return (self.coords['y'][1] - self.coords['y'][0]).item()

    @property
    def dt(self):
        return pd.infer_freq(self.coords['time'].to_index())

    @property
    def prepared(self):
        warn("The `prepared` attribute is deprecated in favour of the "
             "fine-grained `prepared_features` list", DeprecationWarning)
        return self.prepared_features == self.available_features

    @property
    def prepared_features(self):
        return set(self.data.attrs.get("prepared_features", []))

    def grid_coordinates(self):
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        return np.asarray((np.ravel(xs), np.ravel(ys))).T

    @CachedAttribute
    def grid_cells(self):
        coords = self.grid_coordinates()
        span = (coords[self.shape[1] + 1] - coords[0]) / 2
        grid_cells = [box(*c)
                      for c in np.hstack((coords - span, coords + span))]
        return grid_cells

    def sel(self, **kwargs):
        if 'bounds' in kwargs:
            bounds = kwargs.pop('bounds')
            buffer = kwargs.pop('buffer', 0)
            if buffer > 0:
                bounds = box(*bounds).buffer(buffer).bounds
            x1, y1, x2, y2 = bounds
            kwargs.update(x=slice(x1, x2), y=slice(y1, y2))
        data = self.data.sel(**kwargs)
        return Cutout(self.path.name, data)

    def __repr__(self):
        start = np.datetime_as_string(self.coords['time'].values[0], unit='D')
        end = np.datetime_as_string(self.coords['time'].values[-1], unit='D')
        return ('Cutout "{}" \n'
                'x = {:.2f} ⟷ {:.2f}, dx = {:.2f}\n'
                'y = {:.2f} ⟷ {:.2f}, dy = {:.2f}\n'
                'time = {} ⟷ {}, dt = {}\n'
                'prepared_features = {}\nis_view = {}'
                .format(self.name,
                        self.coords['x'].values[0],
                        self.coords['x'].values[-1],
                        self.dx,
                        self.coords['y'].values[0],
                        self.coords['y'].values[-1],
                        self.dy,
                        start,
                        end,
                        self.dt,
                        list(self.prepared_features),
                        self.is_view))


    def indicatormatrix(self, shapes, shapes_proj='latlong'):
        return compute_indicatormatrix(self.grid_cells, shapes,
                                       self.projection, shapes_proj)

    # Preparation functions

    prepare = cutout_prepare

    # Conversion and aggregation functions

    convert_and_aggregate = convert_and_aggregate

    heat_demand = heat_demand

    temperature = temperature

    soil_temperature = soil_temperature

    solar_thermal = solar_thermal

    wind = wind

    pv = pv

    runoff = runoff

    hydro = hydro
