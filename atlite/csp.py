# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Functions for use in conjunction with csp data generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import xarray as xr
from dask.array import radians, sin

from atlite.pv.solar_position import SolarPosition

if TYPE_CHECKING:
    from dask.array import Array

logger = logging.getLogger(__name__)

NDArray: TypeAlias = np.ndarray
DataArray: TypeAlias = xr.DataArray
Dataset: TypeAlias = xr.Dataset
CSPTechnology = Literal["parabolic trough", "solar tower"]
FieldOrientation = Literal["horizontal", "tilted", "single-axis", "two-axis"]


def calculate_dni(
    ds: Dataset,
    solar_position: Dataset | None = None,
    altitude_threshold: float = 3.75,
) -> DataArray:
    """
    Calculate DNI on a perpendicular plane.

    Calculate the Direct Normal Irradiance (DNI) on a plane perpendicular to the solar
    irradiance based on solar altitude and direct solar influx on a horizontal plane.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the direct influx (influx_direct) into a horizontal plane.
    solar_position : xr.Dataset | None
        Dataset containing solar altitude (in rad, 0 to pi/2) for the input dataset.
        Calculated using atlite.pv.SolarPosition if not provided.
    altitude_threshold : float
        Threshold for solar altitude in degrees. Values in range (0, altitude_threshold]
        are set to altitude_threshold to prevent numerical issues when dividing by
        the sine of very low solar altitude. Default: 3.75 degrees corresponds to
        approximately 15 minutes of solar movement at 60 deg maximum altitude.

    Returns
    -------
    xr.DataArray
        Direct Normal Irradiance (DNI) in W/m^2 on a plane perpendicular to solar rays.

    """
    if solar_position is None:
        solar_position = SolarPosition(ds)

    altitude_threshold_rad: Array = radians(altitude_threshold)

    altitude: DataArray = solar_position["altitude"]
    altitude = altitude.where(lambda x: x > 0, np.nan)
    altitude = altitude.where(
        lambda x: x > altitude_threshold_rad, altitude_threshold_rad
    )

    dni: DataArray = ds["influx_direct"] / sin(altitude)

    return dni
