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
import dask
import rasterio as rio
import geopandas as gpd
from tempfile import mktemp
from numpy import atleast_1d, append
from warnings import warn
from shapely.geometry import box
from pathlib import Path
from pyproj import CRS


from .utils import CachedAttribute
from .data import cutout_prepare, available_features
from .gis import (
    get_coords,
    compute_indicatormatrix,
    compute_availabilitymatrix,
    compute_intersectionmatrix,
)
from .convert import (
    convert_and_aggregate,
    heat_demand,
    hydro,
    temperature,
    wind,
    pv,
    csp,
    runoff,
    solar_thermal,
    soil_temperature,
    coefficient_of_performance,
    line_rating,
)
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
            User provided cutout data. Save the cutout using `Cutout.to_file()`
            afterwards.

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
                "`cutout_dir` and `name` have been deprecated in favour of `path`."
            )

        path = Path(path).with_suffix(".nc")
        chunks = cutoutparams.pop("chunks", {"time": 100})
        storable_chunks = {f"chunksize_{k}": v for k, v in (chunks or {}).items()}

        # Backward compatibility for xs, ys, months and years
        if {"xs", "ys"}.intersection(cutoutparams):
            warn(
                "The arguments `xs` and `ys` have been deprecated in favour of "
                "`x` and `y`",
                DeprecationWarning,
            )
            if "xs" in cutoutparams:
                cutoutparams["x"] = cutoutparams.pop("xs")
            if "ys" in cutoutparams:
                cutoutparams["y"] = cutoutparams.pop("ys")

        if {"years", "months"}.intersection(cutoutparams):
            warn(
                "The arguments `years` and `months` have been deprecated in "
                "favour of `time`",
                DeprecationWarning,
            )
            assert "years" in cutoutparams
            months = cutoutparams.pop("months", slice(1, 12))
            years = cutoutparams.pop("years")
            cutoutparams["time"] = slice(
                f"{years.start}-{months.start}", f"{years.stop}-{months.stop}"
            )

        # Three cases. First, cutout exists -> take the data.
        # Second, data is given -> take it. Third, else -> build a new cutout
        if path.is_file():
            data = xr.open_dataset(str(path), chunks=chunks)
            data.attrs.update(storable_chunks)
            if cutoutparams:
                warn(
                    f'Arguments {", ".join(cutoutparams)} are ignored, since '
                    "cutout is already built."
                )
        elif "data" in cutoutparams:
            data = cutoutparams.pop("data")
        else:
            logger.info(f"Building new cutout {path}")

            if "bounds" in cutoutparams:
                x1, y1, x2, y2 = cutoutparams.pop("bounds")
                cutoutparams.update(x=slice(x1, x2), y=slice(y1, y2))

            try:
                x = cutoutparams.pop("x")
                y = cutoutparams.pop("y")
                time = cutoutparams.pop("time")
                module = cutoutparams.pop("module")
            except KeyError as exc:
                raise TypeError(
                    "Arguments 'time' and 'module' must be "
                    "specified. Spatial bounds must either be "
                    "passed via argument 'bounds' or 'x' and 'y'."
                ) from exc

            # TODO: check for dx, dy, x, y fine with module requirements
            coords = get_coords(x, y, time, **cutoutparams)

            attrs = {
                "module": module,
                "prepared_features": [],
                **storable_chunks,
                **cutoutparams,
            }
            data = xr.Dataset(coords=coords, attrs=attrs)

        # Check compatibility of CRS
        modules = atleast_1d(data.attrs.get("module"))
        crs = set(CRS(datamodules[m].crs) for m in modules)
        assert len(crs) == 1, f"CRS of {module} not compatible"

        self.path = path
        self.data = data

    @property
    def name(self):
        """Name of the cutout."""
        return self.path.stem

    @property
    def module(self):
        """Data module of the cutout."""
        return self.data.attrs.get("module")

    @property
    def crs(self):
        """Coordinate Reference System of the cutout."""
        return CRS(datamodules[atleast_1d(self.module)[0]].crs)

    @property
    def available_features(self):
        """List of available weather data features for the cutout."""
        return available_features(self.module)

    @property
    def chunks(self):
        """Chunking of the cutout data used by dask."""
        chunks = {
            k.lstrip("chunksize_"): v
            for k, v in self.data.attrs.items()
            if k.startswith("chunksize_")
        }
        return None if chunks == {} else chunks

    @property
    def coords(self):
        """Geographic coordinates of the cutout."""
        return self.data.coords

    @property
    def meta(self):
        """Metadata of the cutout. Deprecated since v0.2."""
        warn(
            "The `meta` attribute is deprecated in favour of direct "
            "access to `data`",
            DeprecationWarning,
        )
        return xr.Dataset(self.coords, attrs=self.data.attrs)

    @property
    def shape(self):
        """Size of spatial dimensions (y, x) of the cutout data."""
        return len(self.coords["y"]), len(self.coords["x"])

    @property
    def extent(self):
        """Total extent of the area covered by the cutout (x, X, y, Y)."""
        xs, ys = self.coords["x"].values, self.coords["y"].values
        dx, dy = self.dx, self.dy
        return np.array(
            [xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2]
        )

    @property
    def bounds(self):
        """Total bounds of the area covered by the cutout (x, y, X, Y)."""
        return self.extent[[0, 2, 1, 3]]

    @property
    def transform(self):
        """Get the affine transform of the cutout."""
        return rio.Affine(
            self.dx,
            0,
            self.coords["x"].values[0] - self.dx / 2,
            0,
            self.dy,
            self.coords["y"].values[0] - self.dy / 2,
        )

    @property
    def transform_r(self):
        """Get the affine transform of the cutout with reverse y-order."""
        return rio.Affine(
            self.dx,
            0,
            self.coords["x"].values[0] - self.dx / 2,
            0,
            -self.dy,
            self.coords["y"].values[-1] + self.dy / 2,
        )

    @property
    def dx(self):
        """Spatial resolution on the x coordinates."""
        x = self.coords["x"]
        return round((x[-1] - x[0]).item() / (x.size - 1), 8)

    @property
    def dy(self):
        """Spatial resolution on the y coordinates."""
        y = self.coords["y"]
        return round((y[-1] - y[0]).item() / (y.size - 1), 8)

    @property
    def dt(self):
        """Time resolution of the cutout."""
        return pd.infer_freq(self.coords["time"].to_index())

    @property
    def prepared(self):
        """Boolean indicating whether all available features are prepared."""
        return self.prepared_features.sort_index().equals(
            self.available_features.sort_index()
        )

    @property
    def prepared_features(self):
        """Get the list of prepared features in the cutout."""
        index = [
            (self.data[v].attrs["module"], self.data[v].attrs["feature"])
            for v in self.data
        ]
        index = pd.MultiIndex.from_tuples(index, names=["module", "feature"])
        return pd.Series(list(self.data), index, dtype=object)

    def grid_coordinates(self):
        """Array of grid coordinates, deprecated since v0.2.1."""
        warn(
            "The function `grid_coordinates` has been deprecated in favour of "
            "`grid`",
            DeprecationWarning,
        )
        logger.warning(
            "The order of elements returned by `grid_coordinates` changed. "
            "Check the output of your workflow for correctness."
        )
        return self.grid[["x", "y"]].values

    def grid_cells(self):
        """List of grid cells, deprecated since v0.2.1."""
        warn(
            "The function `grid_cells` has been deprecated in favour of `grid`",
            DeprecationWarning,
        )
        logger.warning(
            "The order of elements in `grid_cells` changed. "
            "Check the output of your workflow for correctness."
        )
        return self.grid.geometry.to_list()

    @CachedAttribute
    def grid(self):
        """
        Cutout grid with coordinates and geometries.

        The coordinates represent the centers of the grid cells.

        Returns
        -------
        geopandas.GeoDataFrame
            Frame with coordinate columns 'x' and 'y', and geometries of the
            corresponding grid cells.

        """
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        coords = np.asarray((np.ravel(xs), np.ravel(ys))).T
        span = (coords[self.shape[1] + 1] - coords[0]) / 2
        cells = [box(*c) for c in np.hstack((coords - span, coords + span))]
        return gpd.GeoDataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "geometry": cells}, crs=self.crs
        )

    def sel(self, path=None, bounds=None, buffer=0, **kwargs):
        """
        Select parts of the cutout.

        Parameters
        ----------
        path : str | path-like
            File where to store the sub-cutout. Defaults to a temporary file.
        bounds : GeoSeries.bounds | DataFrame, optional
            The outer bounds of the cutout or as a DataFrame
            containing (min.long, min.lat, max.long, max.lat).
        buffer : float, optional
            Buffer around the bounds. The default is 0.
        **kwargs :
            Passed to `xr.Dataset.sel` for data selection.

        Returns
        -------
        selected : Cutout
            Selected cutout.

        """
        if path is None:
            path = mktemp(
                prefix=f"{self.path.stem}-",
                suffix=self.path.suffix,
                dir=self.path.parent,
            )

        if bounds is not None:
            if buffer > 0:
                bounds = box(*bounds).buffer(buffer).bounds
            x1, y1, x2, y2 = bounds
            kwargs.update(x=slice(x1, x2), y=slice(y1, y2))
        data = self.data.sel(**kwargs)
        return Cutout(path, data=data)

    def merge(self, other, path=None, **kwargs):
        """
        Merge two cutouts into a single cutout.

        Parameters
        ----------
        other : atlite.Cutout
            Other cutout to merge.
        path : str | path-like
            File where to store the merged cutout. Defaults to a temporary file.
        **kwargs
            Keyword arguments passed to `xarray.merge()`.

        Returns
        -------
        merged : Cutout
            Merged cutout.

        """
        assert isinstance(other, Cutout)

        if path is None:
            path = mktemp(
                prefix=f"{self.path.stem}-",
                suffix=self.path.suffix,
                dir=self.path.parent,
            )

        attrs = {**self.data.attrs, **other.data.attrs}
        attrs["module"] = list(set(append(*atleast_1d(self.module, other.module))))
        features = self.prepared_features.index.unique("feature")
        otherfeatures = other.prepared_features.index.unique("feature")
        attrs["prepared_features"] = list(features.union(otherfeatures))

        data = self.data.merge(other.data, **kwargs).assign_attrs(**attrs)

        return Cutout(path, data=data)

    def to_file(self, fn=None):
        """
        Save cutout to a netcdf file.

        Parameters
        ----------
        fn : str | path-like
            File name where to store the cutout, defaults to `cutout.path`.

        """
        if fn is None:
            fn = self.path
        self.data.to_netcdf(fn)

    def __repr__(self):
        start = np.datetime_as_string(self.coords["time"].values[0], unit="D")
        end = np.datetime_as_string(self.coords["time"].values[-1], unit="D")
        return (
            '<Cutout "{}">\n'
            " x = {:.2f} ⟷ {:.2f}, dx = {:.2f}\n"
            " y = {:.2f} ⟷ {:.2f}, dy = {:.2f}\n"
            " time = {} ⟷ {}, dt = {}\n"
            " module = {}\n"
            " prepared_features = {}".format(
                self.name,
                self.coords["x"].values[0],
                self.coords["x"].values[-1],
                self.dx,
                self.coords["y"].values[0],
                self.coords["y"].values[-1],
                self.dy,
                start,
                end,
                self.dt,
                self.module,
                list(self.prepared_features.index.unique("feature")),
            )
        )

    def indicatormatrix(self, shapes, shapes_crs=4326):
        """
        Compute the indicatormatrix.

        The indicatormatrix I[i,j] is a sparse representation of the ratio
        of the area in orig[j] lying in dest[i], where orig and dest are
        collections of polygons, i.e.

        A value of I[i,j] = 1 indicates that the shape orig[j] is fully
        contained in shape dest[j].

        Note that the polygons must be in the same crs.

        Parameters
        ----------
        shapes : Collection of shapely polygons

        Returns
        -------
        I : sp.sparse.lil_matrix
          Indicatormatrix
        """
        return compute_indicatormatrix(self.grid, shapes, self.crs, shapes_crs)

    def intersectionmatrix(self, shapes, shapes_crs=4326):
        """
        Compute the intersectionmatrix.

        The intersectionmatrix is a sparse matrix with entries (i,j) being one
        if shapes orig[j] and dest[i] are intersecting, and zero otherwise.

        Note that the polygons must be in the same crs.

        Parameters
        ----------
        orig : Collection of shapely polygons
        dest : Collection of shapely polygons

        Returns
        -------
        I : sp.sparse.lil_matrix
          Intersectionmatrix
        """
        return compute_intersectionmatrix(self.grid, shapes, self.crs, shapes_crs)

    def uniform_layout(self):
        """Get a uniform capacity layout for all grid cells."""
        return xr.DataArray(1, [self.coords["y"], self.coords["x"]])

    def layout_from_capacity_list(self, data, col="Capacity"):
        """
        Get a capacity layout aligned to the cutout based on a capacity list.

        Parameters
        ----------
        data : pandas.DataFrame
            Capacity list with columns 'x', 'y' and col. Each capacity entry
            is added to the grid cell intersecting with the coordinate (x,y).
        col : str, optional
            Name of the column with capacity values. The default is 'Capacity'.

        Returns
        -------
        xr.DataArray
            Capacity layout with dimensions 'x' and 'y' indicating the total
            capacity placed within one grid cell.

        Example
        -------
        >>> import atlite
        >>> import powerplantmatching as pm

        >>> data = pm.data.OPSD_VRE_country('DE')
        >>> data = (data.query('Fueltype == "Solar"')
                    .rename(columns={'lon':'x', 'lat':'y'}))

        >>> cutout = atlite.Cutout('Germany', x = slice(-5, 15), y = slice(40, 55),
                       time='2013-06-01', module='era5')
        >>> cutout.prepare(features=['influx', 'temperature'])
        >>> layout = cutout.layout_from_capacity_list(data)
        >>> pv = cutout.pv('CdTe', 'latitude_optimal', layout=layout)
        >>> pv.plot()

        """
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            nearest = (
                self.uniform_layout()
                .chunk()
                .sel({"x": data.x.values, "y": data.y.values}, "nearest")
            )

        data = (
            data.assign(x=nearest.x.data, y=nearest.y.data)
            .groupby(["y", "x"])[col]
            .sum()
        )
        return data.to_xarray().reindex_like(self.data).fillna(0)

    availabilitymatrix = compute_availabilitymatrix

    # Preparation functions

    prepare = cutout_prepare

    # Conversion and aggregation functions

    convert_and_aggregate = convert_and_aggregate

    heat_demand = heat_demand

    temperature = temperature

    soil_temperature = soil_temperature

    coefficient_of_performance = coefficient_of_performance

    solar_thermal = solar_thermal

    wind = wind

    pv = pv

    csp = csp

    runoff = runoff

    hydro = hydro

    line_rating = line_rating
