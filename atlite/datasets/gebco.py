#!/usr/bin/env python3

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for loading gebco data.
"""

import logging

import pandas as pd
import rasterio as rio
import xarray as xr
from pandas import to_numeric
from rasterio.warp import Resampling

logger = logging.getLogger(__name__)

crs = 4326
features = {"height": ["height"]}


requirements = {
    "x": slice(-90, 90, 0.15),
    "y": slice(-90, 90, 0.15),
    "offset": pd.Timestamp("1940-01-01"),
    "forecast": pd.Timestamp("2050-01-01"),
    "dt": pd.Timedelta(hours=1),
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


def get_data_gebco_height(xs, ys, gebco_path):
    x, X = xs.data[[0, -1]]
    y, Y = ys.data[[0, -1]]

    dx = (X - x) / (len(xs) - 1)
    dy = (Y - y) / (len(ys) - 1)

    with rio.open(gebco_path) as dataset:
        window = dataset.window(x - dx / 2, y - dy / 2, X + dx / 2, Y + dy / 2)
        gebco = dataset.read(
            indexes=1,
            window=window,
            out_shape=(len(ys), len(xs)),
            resampling=Resampling.average,
        )
        gebco = gebco[::-1]  # change inversed y-axis
        tags = dataset.tags(bidx=1)
        tags = {k: to_numeric(v, errors="ignore") for k, v in tags.items()}

    return xr.DataArray(
        gebco, coords=[("y", ys.data), ("x", xs.data)], name="height", attrs=tags
    )


def get_data(
    cutout,
    feature,
    tmpdir,
    monthly_requests=False,
    concurrent_requests=False,
    **creation_parameters,
):
    """
    Get the gebco height data.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Takes no effect, only here for consistency with other dataset modules.
    tmpdir : str
        Takes no effect, only here for consistency with other dataset modules.
    monthly_requests : bool
        Takes no effect, only here for consistency with other dataset modules.
    concurrent_requests : bool
        Takes no effect, only here for consistency with other dataset modules.
    **creation_parameters :
        Must include `gebco_path`.

    Returns
    -------
    xr.Dataset

    """
    if "gebco_path" not in creation_parameters:
        logger.error('Argument "gebco_path" not defined')
    path = creation_parameters["gebco_path"]

    coords = cutout.coords
    # assign time dimension even if not used
    return (
        get_data_gebco_height(coords["x"], coords["y"], path)
        .to_dataset()
        .assign_coords(cutout.coords)
    )
