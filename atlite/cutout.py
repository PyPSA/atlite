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
import netCDF4

import xarray as xr
import numpy as np
import os, sys
from warnings import warn
from shapely.geometry import box
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from . import datasets, utils

from .convert import (convert_and_aggregate, heat_demand, hydro, temperature,
                      wind, pv, runoff, solar_thermal, soil_temperature)
from .gis import GridCells
from .data import requires_coords, requires_windowed, cutout_prepare

class Cutout:

    dataset_module = None

    def __init__(self, path=None, data=None, cutout_dir=None, **cutoutparams):
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
        time : (opt.) slice

        """
        if cutout_dir is not None:
            warn("The argument `cutout_dir` has been deprecated in favour of "
                 "just `path`", DeprecationWarning)
            path = Path(cutout_dir) / path
        elif isinstance(path, xr.Dataset):
            data = path
            path = Path("unnamed.nc")
        else:
            path = Path(path)

        if not path.suffix and not path.is_dir():
            path = path.with_suffix(".nc")

        if 'bounds' in cutoutparams:
            x1, y1, x2, y2 = cutoutparams.pop('bounds')
            cutoutparams.update(x=slice(x1, x2), y=slice(y1, y2))

        if {'xs', 'ys'}.intersection(cutoutparams):
            warn("The arguments `xs` and `ys` have been deprecated in favour of "
                 "`x` and `y`", DeprecationWarning)
            if 'xs' in cutoutparams: cutoutparams['x'] = cutoutparams.pop('xs')
            if 'ys' in cutoutparams: cutoutparams['y'] = cutoutparams.pop('ys')


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
                assert prepared_features is not None, (f"{self.name} does not"
                            " have the required attribute `prepared_features`")
                if not isinstance(prepared_features, list):
                    data.attrs['prepared_features'] = [prepared_features]
            elif path.is_dir():
                data = utils.migrate_from_cutout_directory(path)
                self.is_view = True
            else:
                logger.info(f"Cutout {path} not found, building new one")

                if {"x", "y", "time"}.difference(cutoutparams):
                    raise RuntimeError("Arguments `x`, `y` and `time` need to "
                            "be specified (or `bounds` instead of `x` and `y`)")

                # Ensure correct ordering of slices
                x = cutoutparams['x']
                y = cutoutparams['y']
                cutoutparams['x'] = slice(*sorted([x.start, x.stop]))
                cutoutparams['y'] = slice(*sorted([y.start, y.stop]))

                if 'module' not in cutoutparams:
                    logger.warning("`module` was not specified, falling back "
                                   "to 'era5'")

                data = xr.Dataset(attrs=
                                  {'module': cutoutparams.pop('module', 'era5'),
                                   'prepared_features': [],
                                   'creation_parameters': str(cutoutparams)})
        else:
            # User-provided dataset
            # TODO needs to be checked, sanitized and marked as immutable (is_view)
            self.is_view = True

        if 'module' in cutoutparams:
            module = cutoutparams.pop('module')
            if module != data.attrs.get('module'):
                logger.warning(f"Selected module '{module}' disagrees with "
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
        return self.data.attrs.get('projection', self.dataset_module.projection)

    @property
    def available_features(self):
        if self.dataset_module and not self.is_view:
            return set(self.dataset_module.features)
        else:
            return set()

    @property
    @requires_coords
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

    _grid_cells_cache = None
    @property
    def _grid_cells(self):
        if self._grid_cells_cache is not None:
            return self._grid_cells_cache

        sindex_fn = self.path.with_suffix(".sindex.pickle")
        grid_cells = None
        if not self.is_view and sindex_fn.exists():
            try:
                grid_cells = GridCells.from_file(sindex_fn)
            except (EOFError, OSError): logger.warning(f"Couldn't read "
                   f"GridCells from cache {sindex_fn.name}. Reconstructing ...")

        if grid_cells is None:
            grid_cells = GridCells.from_cutout(self)

        # Cache
        self._grid_cells_cache = grid_cells

        return grid_cells

    @property
    def grid_cells(self):
        return self._grid_cells.grid_cells

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
        return ('<Cutout {} x={:.2f}-{:.2f} y={:.2f}-{:.2f} time={}-{} '
                'prepared_features={} is_view={}>'
                .format(self.name,
                        self.coords['x'].values[0], self.coords['x'].values[-1],
                        self.coords['y'].values[0], self.coords['y'].values[-1],
                        np.datetime_as_string(self.coords['time'].values[0], unit='D'),
                        np.datetime_as_string(self.coords['time'].values[-1], unit='D'),
                        list(self.prepared_features),
                        self.is_view))

    def indicatormatrix(self, shapes, shapes_proj='latlong'):
        return self._grid_cells.indicatormatrix(shapes, shapes_proj)

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
