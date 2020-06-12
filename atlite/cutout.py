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

import xarray as xr
import pandas as pd
import numpy as np
from tempfile import mktemp
from numpy import atleast_1d
from warnings import warn
from shapely.geometry import box
from pathlib import Path

from .utils import CachedAttribute
from .data import cutout_prepare, available_features
from .gis import get_coords, compute_indicatormatrix
from .convert import (convert_and_aggregate, heat_demand, hydro, temperature,
                      wind, pv, runoff, solar_thermal, soil_temperature)
from .datasets import modules as datamodules

import logging
logger = logging.getLogger(__name__)


class Cutout:
    """
    Cutout base class.

    This class builds the starting point for most atlite functionalities.
    """

    def __init__(self, path, **cutoutparams):
        """
        Provide an Atlite cutout object.

        Create a cutout object to use atlite operations on it. Based on the
        provided parameters, atlite first checks whether this cutout already
        exists on disk and if yes, loads this cutout.

        If the cutout does not yet exist on disk, then atlite creates an
        "unprepared" cutout object. This does not yet contain the full data.
        The process of preparing (loading the data) can then be started with
        `cutout.prepare()`.

        Parameters
        ----------
        path : str | path-like
            NetCDF from which to load or where to store the cutout.
        module : str or list
            The dataset(s) which works as a basis for the cutout. Available
            modules are "era5", "sarah" and "gebco".
            This is necessary when building a new cutout.
            If more than one module is given, their order determines how atlite
            fills up missing features when preparing the cutout with
            `Cutout.prepare()`. For example `influx_diffuse` is provided by
            the `sarah` and the `era5` module. Prioritizing sarah and setting
            module=['sarah', 'era5'] will load `influx_diffuse` from the sarah
            module and ignoring the era5 'influx_diffuse' data.
        time : str | slice
            Time range to include in the cutout, e.g. "2011" or
            ("2011-01-05", "2011-01-25")
            This is necessary when building a new cutout.
        bounds : GeoSeries.bounds | DataFrame, optional
            The outer bounds of the cutout or as a DataFrame
            containing (min.long, min.lat, max.long, max.lat).
        x : slice, optional
            Outer longitudinal bounds for the cutout (west, east).
        y : slice, optional
            Outer latitudinal bounds for the cutout (south, north).
        dx : float, optional
            Step size of the x coordinate. The default is 0.25.
        dy : float, optional
            Step size of the y coordinate. The default is 0.25.
        dt : str, optional
            Frequency of the time coordinate. The default is 'h'. Valid are all
            pandas offset aliases.
        chunks : dict
            Chunks when opening netcdf files. For cutout preparation recommand
            to chunk only along the time dimension. Defaults to {'time': 20}
        data : xr.Dataset
            User provided cutout data.

        Other Parameters
        ----------------
        sanitize : bool, default True
            Whether to sanitize the data when preparing the cutout. Takes
            effect for 'era5' data loading.
        sarah_dir : str, Path
            Directory of on-disk sarah data. This must be given when using the
            sarah module.
        sarah_interpolate : bool, default True
            Whether to interpolate NaN's in the SARAH data. This takes effect for
            sarah data which has missing data for areas where dawn and
            nightfall happens (ca. 30 min gap).
        gebco_path: str
            Path to find the gebco netcdf file. Only necessary when including
            the gebco module.
        parallel : bool, default False
            Whether to open dataset in parallel mode. Take effect for all
            xr.open_mfdataset usages.

        """
        name = cutoutparams.get("name", None)
        cutout_dir = cutoutparams.get("cutout_dir", None)
        if cutout_dir or name or Path(path).is_dir():
            raise ValueError(
                "Old style format not supported. You can migrate the old "
                "cutout directory using the function "
                "`atlite.utils.migrate_from_cutout_directory()`. The argument "
                "`cutout_dir` and `name` have been deprecated in favour of `path`.")

        path = Path(path).with_suffix(".nc")
        chunks = cutoutparams.pop('chunks', {'time': 100})
        storable_chunks = {f'chunksize_{k}': v for k, v in (chunks or {}).items()}

        # Backward compatibility for xs, ys, months and years
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
            cutoutparams["time"] = slice(f"{years.start}-{months.start}",
                                         f"{years.stop}-{months.stop}")

        # Three cases. First, cutout exists -> take the data.
        # Second, data is given -> take it. Third, else -> build a new cutout
        if path.is_file():
            data = xr.open_dataset(str(path), chunks=chunks)
            data.attrs.update(storable_chunks)
            if 'module' in cutoutparams:
                module = cutoutparams.pop('module')
                if module != data.attrs.get('module'):
                    logger.warning(
                        f"Overwriting dataset module "
                        f"{data.attrs.get('module')} with module {module}.")
                data.attrs['module'] = module
        elif 'data' in cutoutparams:
            data = cutoutparams.pop('data')
        else:
            logger.info(f"Building new cutout {path}")

            if 'bounds' in cutoutparams:
                x1, y1, x2, y2 = cutoutparams.pop('bounds')
                cutoutparams.update(x=slice(x1, x2), y=slice(y1, y2))

            try:
                x = cutoutparams.pop('x')
                y = cutoutparams.pop('y')
                time = cutoutparams.pop('time')
                module = cutoutparams.pop('module')
            except KeyError as exc:
                raise TypeError("Arguments 'time' and 'module' must be "
                                "specified. Spatial bounds must either be "
                                "passed via argument 'bounds' or 'x' and 'y'.") from exc

            # TODO: check for dx, dy, x, y fine with module requirements
            coords = get_coords(x, y, time, **cutoutparams)

            attrs = {'module': module, 'prepared_features': [],
                     **storable_chunks, **cutoutparams}
            data = xr.Dataset(coords=coords, attrs=attrs)

        # Check projections
        modules = atleast_1d(data.attrs.get('module'))
        projection = set(datamodules[m].projection for m in modules)
        assert len(projection) == 1, f'Projections of {module} not compatible'

        self.path = path
        self.data = data


    @property
    def name(self):
        return self.path.stem

    @property
    def module(self):
        return self.data.attrs.get('module')

    @property
    def projection(self):
        return datamodules[atleast_1d(self.module)[0]].projection

    @property
    def available_features(self):
        return available_features(self.module)

    @property
    def chunks(self):
        chunks = {k.lstrip('chunksize_'): v for k, v in self.data.attrs.items()
                  if k.startswith('chunksize_')}
        return None if chunks == {} else chunks

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
        x = self.coords['x']
        return (x[-1] - x[0]).item() / (x.size - 1)

    @property
    def dy(self):
        y = self.coords['y']
        return (y[-1] - y[0]).item() / (y.size - 1)

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
        features = atleast_1d(self.data.attrs.get("prepared_features", []))
        return self.available_features.loc[:, features].drop_duplicates()

    def grid_coordinates(self):
        logger.warning("The order of elements returned by `grid_coordinates` changed. "
                       "Check the output of your workflow for correctness.")
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        return np.asarray((np.ravel(xs), np.ravel(ys))).T

    @CachedAttribute
    def grid_cells(self):
        logger.warning("The order of elements in `grid_cells` changed. "
                       "Check the output of your workflow for correctness.")
        coords = self.grid_coordinates()
        span = (coords[self.shape[1] + 1] - coords[0]) / 2
        grid_cells = [box(*c)
                      for c in np.hstack((coords - span, coords + span))]
        return grid_cells

    def sel(self, path=None, bounds=None, buffer=0, **kwargs):
        if path is None:
            path = mktemp(prefix=f"{self.path.stem}-", suffix=self.path.suffix,
                          dir=self.path.parent)

        if bounds is not None:
            if buffer > 0:
                bounds = box(*bounds).buffer(buffer).bounds
            x1, y1, x2, y2 = bounds
            kwargs.update(x=slice(x1, x2), y=slice(y1, y2))
        data = self.data.sel(**kwargs)
        return Cutout(path, data=data)

    def __repr__(self):
        start = np.datetime_as_string(self.coords['time'].values[0], unit='D')
        end = np.datetime_as_string(self.coords['time'].values[-1], unit='D')
        return ('<Cutout "{}">\n'
                ' x = {:.2f} ⟷ {:.2f}, dx = {:.2f}\n'
                ' y = {:.2f} ⟷ {:.2f}, dy = {:.2f}\n'
                ' time = {} ⟷ {}, dt = {}\n'
                ' prepared_features = {}'
                .format(self.name, self.coords['x'].values[0],
                        self.coords['x'].values[-1], self.dx,
                        self.coords['y'].values[0],
                        self.coords['y'].values[-1], self.dy,
                        start, end, self.dt,
                        list(self.prepared_features.index.unique('feature'))))


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
