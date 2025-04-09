# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module containing specific operations for creating cutouts from the SARAH2
dataset.
"""

import glob
import logging
import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.warp import Resampling

from atlite.gis import regrid
from atlite.pv.solar_position import SolarPosition

logger = logging.getLogger(__name__)


# Model, CRS and Resolution Settings
crs = 4326
dx = 0.05
dy = 0.05
dt = "30min"
features = {
    "influx": [
        "influx_direct",
        "influx_diffuse",
        "solar_altitude",
        "solar_azimuth",
    ],
}
static_features = {}


requirements = {
    "x": slice(-65.0, 65.0, 0.05),
    "y": slice(-65.0, 65.0, 0.05),
    "offset": pd.Timestamp("1983-01-01"),
    "forecast": pd.Timestamp("2017-12-31 23:00"),
    "dt": pd.Timedelta(hours=0.5),
    "parallel": True,
}


def _checkModuleRequirements(x, y, time, time_now, **kwargs):
    """
    Load and check the data requirements for a given module.

    Parameters
    ----------
    x (slice): Defines the start, stop, and step values for the x-dimension.
    y (slice): Defines the start, stop, and step values for the y-dimension.
    time (slice): Defines the start, stop, and step values for the time dimension.
    **kwargs: Additional optional parameters.
    """

    # Extract start, stop, and step values for x
    x_start, x_stop, x_step = x.start, x.stop, x.step

    # Adjust x range based on module requirements
    if requirements["x"].start > x.start:
        x_start = requirements["x"].start
    if requirements["x"].stop < x.stop:
        x_stop = requirements["x"].stop
    if requirements["x"].step > x.step:
        x_step = requirements["x"].step

    x = slice(x_start, x_stop, x_step)

    # Extract start, stop, and step values for y
    y_start, y_stop, y_step = y.start, y.stop, y.step

    # Adjust y range based on module requirements
    if requirements["y"].start > y.start:
        y_start = requirements["y"].start
    if requirements["y"].stop < y.stop:
        y_stop = requirements["y"].stop
    if requirements["y"].step > y.step:
        y_step = requirements["y"].step

    y = slice(y_start, y_stop, y_step)

    # Extract time range parameters
    time_start = time.start
    time_stop = time.stop
    time_step = time.step

    # Check forecast feasibility
    feasible_start = time_now + requirements["offset"]
    feasible_end = time_now + requirements["forecast"]

    # Ensure time_start is within feasible bounds
    if time_start < feasible_start:
        logger.error(
            f"The required forecast start time {time_start} exceeds the model requirements."
        )
        logger.error(
            f"The minimum start time of the forecast for {time_now} is {feasible_start}."
        )
        logger.error(
            f"The maximum historical offset of the forecast is {requirements['offset']}."
        )
        raise ValueError(
            f"Invalid forecast start time: {time_start}. Must be >= {feasible_start}."
        )

    if time_start >= feasible_end:
        logger.error(
            f"The required forecast start time {time_start} exceeds the model requirements."
        )
        logger.error(
            f"The maximum start time of the forecast for {time_now} needs to be smaller than {feasible_end}."
        )
        raise ValueError(
            f"Invalid forecast start time: {time_start}. Must be < {feasible_end}."
        )

    # Ensure time_stop is greater than time_start
    if time_stop <= time_start:
        logger.error(
            f"The required forecast end time {time_stop} exceeds the model requirements."
        )
        logger.error(
            f"The minimum end time of the forecast for {time_now} needs to be larger than {time_start}."
        )
        raise ValueError(
            f"Invalid forecast end time: {time_stop}. Must be > {time_start}."
        )

    # Ensure time_stop is greater than time_start
    if time_stop > feasible_end:
        logger.error(
            f"The required forecast end time {time_stop} exceeds the model requirements."
        )
        logger.error(
            f"The maximum end time of the forecast for {time_now} is {feasible_end}."
        )
        logger.error(f"The maximum forecast horizon is {requirements['forecast']}.")
        raise ValueError(
            f"Invalid forecast end time: {time_stop}. Must be <= {feasible_end}."
        )

    # Ensure time step is within required limits
    if (time_step is pd.Timedelta(None)) or (time.step < requirements["dt"]):
        logger.warning(
            f"The required temporal forecast resolution {time_step} exceeds the model requirements."
        )
        logger.warning(
            f"The minimum temporal resolution of the forecast is {requirements['dt']}."
        )
        logger.info(
            f"Set the temporal forecast resolution to the minimum: {requirements['dt']}."
        )
        time_step = requirements["dt"]

    time = slice(time_start, time_stop, time_step)

    # Retrieve parallel processing setting from requirements
    parallel = requirements["parallel"]

    return x, y, time, parallel


def get_filenames(sarah_dir, coords):
    """
    Get all files in directory `sarah_dir` relevent for coordinates `coords`.

    This function parses all files in the sarah directory which lay in the time
    span of the coordinates.

    Parameters
    ----------
    sarah_dir : str
    coords : atlite.Cutout.coords

    Returns
    -------
    pd.DataFrame with two columns `sis` and `sid` for and timeindex for all
    relevant files.

    """

    def _filenames_starting_with(name):
        pattern = os.path.join(sarah_dir, "**", f"{name}*.nc")
        files = pd.Series(glob.glob(pattern, recursive=True))
        assert not files.empty, (
            f"No files found at {pattern}. Make sure "
            f"sarah_dir points to the correct directory!"
        )

        files.index = pd.to_datetime(files.str.extract(r"SI.in(\d{8})", expand=False))
        return files.sort_index()

    files = pd.concat(
        dict(sis=_filenames_starting_with("SIS"), sid=_filenames_starting_with("SID")),
        join="inner",
        axis=1,
    )

    # SARAH files are named based on day, need to .floor("D") to compare correctly
    start = coords["time"].to_index()[0].floor("D")
    end = coords["time"].to_index()[-1].floor("D")

    if (start < files.index[0]) or (end > files.index[-1]):
        logger.error(
            f"Files in {sarah_dir} do not cover the whole time span:"
            f"\t{start} until {end}"
        )

    return files.loc[(files.index >= start) & (files.index <= end)].sort_index()


def interpolate(ds, dim="time"):
    """
    Interpolate NaNs in a dataset along a chunked dimension.

    This function is similar to xr.Dataset.interpolate_na but can be
    used for interpolating along a chunked dimensions (default 'time'').
    As the sarah data has mulitple NaN's in the areas of dawn and
    nightfall and the data is per default chunked along the time axis,
    use this function to interpolate.
    """

    def _interpolate1d(y):
        nan = np.isnan(y)
        if nan.all() or not nan.any():
            return y

        def x(z):
            return z.nonzero()[0]

        y = np.array(y)
        y[nan] = np.interp(x(nan), x(~nan), y[~nan])
        return y

    def _interpolate(a):
        return a.map_blocks(
            partial(np.apply_along_axis, _interpolate1d, -1), dtype=a.dtype
        )

    data_vars = ds.data_vars.values() if isinstance(ds, xr.Dataset) else (ds,)
    dtypes = {da.dtype for da in data_vars}
    assert len(dtypes) == 1, "interpolate only supports datasets with homogeneous dtype"

    return xr.apply_ufunc(
        _interpolate,
        ds,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        output_dtypes=[dtypes.pop()],
        output_sizes={dim: len(ds.indexes[dim])},
        dask="allowed",
        keep_attrs=True,
    )


def as_slice(bounds, pad=True):
    """
    Convert coordinate bounds to slice and pad by 0.01.
    """
    if not isinstance(bounds, slice):
        bounds = bounds + (-0.01, 0.01)
        bounds = slice(*bounds)
    return bounds


def hourly_mean(ds):
    """
    Resample time data to one hour frequency.

    In contrast to the standard xarray resample function this preserves
    chunks sizes along the time dimension.
    """
    ds1 = ds.isel(time=slice(None, None, 2))
    ds2 = ds.isel(time=slice(1, None, 2))
    ds2 = ds2.assign_coords(time=ds2.indexes["time"] - pd.Timedelta(30, "m"))
    ds = (ds1 + ds2) / 2
    ds.attrs = ds1.attrs
    for v in ds.variables:
        ds[v].attrs = ds1[v].attrs
    return ds


def get_data(
    cutout, feature, tmpdir, lock=None, monthly_requests=False, **creation_parameters
):
    """
    Load stored SARAH data and reformat to matching the given cutout.

    This function loads and resamples the stored SARAH data for a given
    `atlite.Cutout`.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.sarah.features`
    monthly_requests : bool
        Takes no effect, only here for consistency with other dataset modules.
    concurrent_requests : bool
        Takes no effect, only here for consistency with other dataset modules.
    **creation_parameters :
        Mandatory arguments are:
            * 'sarah_dir', str. Directory of the stored SARAH data.
        Possible arguments are:
            * 'parallel', bool. Whether to load stored files in parallel
            mode. Default is False.
            * 'sarah_interpolate', bool. Whether to interpolate areas of dawn
            and nightfall. This might slow down the loading process if only
            a few cores are available. Default is True.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

    """
    assert cutout.dt in ("30min", "30T", "h", "1h")

    coords = cutout.coords
    chunks = cutout.chunks

    sarah_dir = creation_parameters["sarah_dir"]
    creation_parameters.setdefault("parallel", False)
    creation_parameters.setdefault("sarah_interpolate", True)

    files = get_filenames(sarah_dir, coords)
    open_kwargs = dict(chunks=chunks, parallel=creation_parameters["parallel"])
    ds_sis = xr.open_mfdataset(files.sis, combine="by_coords", **open_kwargs)[["SIS"]]
    ds_sid = xr.open_mfdataset(files.sid, combine="by_coords", **open_kwargs)[["SID"]]
    ds = xr.merge([ds_sis, ds_sid])
    ds = ds.sel(lon=as_slice(cutout.extent[:2]), lat=as_slice(cutout.extent[2:]))
    # fix float (im)precission
    ds = ds.assign_coords(
        lon=ds.lon.astype(float).round(4), lat=ds.lat.astype(float).round(4)
    )

    # Interpolate, resample and possible regrid
    if creation_parameters["sarah_interpolate"]:
        ds = interpolate(ds)
    else:
        ds = ds.fillna(0)

    if cutout.dt not in ["30min", "30T"]:
        ds = hourly_mean(ds)

    if (cutout.dx != dx) or (cutout.dy != dy):
        ds = regrid(ds, coords["lon"], coords["lat"], resampling=Resampling.average)

    dif_attrs = dict(long_name="Surface Diffuse Shortwave Flux", units="W m-2")
    ds["influx_diffuse"] = (ds["SIS"] - ds["SID"]).assign_attrs(**dif_attrs)
    ds = ds.rename({"SID": "influx_direct"}).drop_vars("SIS")
    ds = ds.assign_coords(x=ds.coords["lon"], y=ds.coords["lat"])

    ds = ds.swap_dims({"lon": "x", "lat": "y"})

    # Do not show DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift="0H")
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp])

    return ds
