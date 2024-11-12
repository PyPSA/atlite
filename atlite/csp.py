# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Functions for use in conjunction with csp data generation.
"""

import logging

import numpy as np
from dask.array import radians, sin

from atlite.pv.solar_position import SolarPosition

logger = logging.getLogger(__name__)


def calculate_dni(ds, solar_position=None, altitude_threshold=3.75):
    """
    Calculate DNI on a perpendicular plane.

    Calculate the Direct Normal Irradiance (DNI) on a plane perpendicular to the solar
    irradiance based on solar altitude and direct solar influx on a horizontal plane.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the direct influx (influx_direct) into a horizontal plane.
    solar_position : xarray.Dataset (optional)
        solar_position containing a solar 'altitude' (in rad, 0 to pi/2) for the 'ds' dataset.
        Is calculated using atlite.pv.SolarPosition if omitted.
    altitude_threshold : float (default: 3.75 degrees)
        Threshold for solar altitude in degrees. Values in range (0, altitude_threshold]
        will be set to the altitude_threshold to avoid numerical issues when dividing by
        the sine of very low solar altitude.
        The default values '3.75 deg' corresponds to
        the solar altitude traversed by the sun within about 15 minutes in a location with
        maximum solar altitude of 60 deg and 10h day time.

    """
    if solar_position is None:
        solar_position = SolarPosition(ds)

    # solar altitude expected in rad, convert degrees (easier to specifcy) to match
    altitude_threshold = radians(altitude_threshold)

    # Sanitation of altitude values:
    # Prevent high calculated DNI values during low solar altitudes (sunset / dawn)
    # where sin(<low altitude>) results in a very low denominator in the DNI calculation
    altitude = solar_position["altitude"]
    altitude = altitude.where(lambda x: x > 0, np.nan)
    altitude = altitude.where(lambda x: x > altitude_threshold, altitude_threshold)

    # Calculate DNI and remove NaNs introduced during altitude sanitation
    # DNI is determined either by dividing by cos(azimuth) or sin(altitude)
    dni = ds["influx_direct"] / sin(altitude)

    return dni
