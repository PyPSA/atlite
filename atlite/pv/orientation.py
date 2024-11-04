# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

import sys

import numpy as np
import xarray as xr
from dask.array import arccos, arcsin, arctan, cos, logical_and, radians, sin
from numpy import pi


def get_orientation(name, **params):
    """
    Definitions:
    -`slope` is the angle between ground and panel.
    -`azimuth` is the clockwise angle from North.
        i.e. azimuth = 180 faces exactly South
    """
    if isinstance(name, dict):
        params = name
        name = params.pop("name", "constant")
    return getattr(sys.modules[__name__], f"make_{name}")(**params)


def make_latitude_optimal():
    """
    Returns an optimal tilt angle for the given ``lat``, assuming that the
    panel is facing towards the equator, using a simple method from [1].

    This method only works for latitudes between 0 and 50. For higher
    latitudes, a static 40 degree angle is returned.

    These results should be used with caution, but there is some
    evidence that tilt angle may not be that important [2].

    Function and documentation has been adapted from gsee [3].

    [1] http://www.solarpaneltilt.com/#fixed
    [2] http://dx.doi.org/10.1016/j.solener.2010.12.014
    [3] https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py

    Parameters
    ----------
    lat : float
        Latitude in degrees.

    """

    def latitude_optimal(lon, lat, solar_position):
        slope = np.empty_like(lat.values)

        below_25 = np.abs(lat.values) <= np.radians(25)
        below_50 = np.abs(lat.values) <= np.radians(50)

        slope[below_25] = 0.87 * np.abs(lat.values[below_25])
        slope[~below_25 & below_50] = 0.76 * np.abs(
            lat.values[~below_25 & below_50]
        ) + np.radians(0.31)
        slope[~below_50] = np.radians(40.0)

        # South orientation for panels on northern hemisphere and vice versa
        azimuth = np.where(lat.values < 0, 0, pi)
        return dict(
            slope=xr.DataArray(slope, coords=lat.coords),
            azimuth=xr.DataArray(azimuth, coords=lat.coords),
        )

    return latitude_optimal


def make_constant(slope, azimuth):
    slope = radians(slope)
    azimuth = radians(azimuth)

    def constant(lon, lat, solar_position):
        return dict(slope=slope, azimuth=azimuth)

    return constant


def make_latitude(azimuth=180):
    azimuth = radians(azimuth)

    def latitude(lon, lat, solar_position):
        return dict(slope=lat, azimuth=azimuth)

    return latitude


def SurfaceOrientation(ds, solar_position, orientation, tracking=None):
    """
    Compute cos(incidence) for slope and panel azimuth.

    References
    ----------
    [1] Sproul, A. B., Derivation of the solar geometric relationships using
    vector analysis, Renewable Energy, 32(7), 1187–1205 (2007).
    [2] Marion, William F., and Aron P. Dobos. Rotation angle for the optimum
    tracking of one-axis trackers. No. NREL/TP-6A20-58891. National Renewable
    Energy Lab.(NREL), Golden, CO (United States), 2013.

    """
    lon = radians(ds["lon"])
    lat = radians(ds["lat"])

    orientation = orientation(lon, lat, solar_position)
    surface_slope = orientation["slope"]
    surface_azimuth = orientation["azimuth"]

    sun_altitude = solar_position["altitude"]
    sun_azimuth = solar_position["azimuth"]

    if tracking is None:
        cosincidence = sin(surface_slope) * cos(sun_altitude) * cos(
            surface_azimuth - sun_azimuth
        ) + cos(surface_slope) * sin(sun_altitude)

    elif tracking == "horizontal":  # horizontal tracking with horizontal axis
        axis_azimuth = orientation[
            "azimuth"
        ]  # here orientation['azimuth'] refers to the azimuth of the tracker axis.
        rotation = arctan(
            (cos(sun_altitude) / sin(sun_altitude)) * sin(sun_azimuth - axis_azimuth)
        )
        surface_slope = abs(rotation)
        surface_azimuth = axis_azimuth + arcsin(
            sin(rotation / sin(surface_slope))
        )  # the 2nd part yields +/-1 and determines if the panel is facing east or west
        cosincidence = cos(surface_slope) * sin(sun_altitude) + sin(
            surface_slope
        ) * cos(sun_altitude) * cos(sun_azimuth - surface_azimuth)

    elif tracking == "tilted_horizontal":  # horizontal tracking with tilted axis'
        axis_tilt = orientation[
            "slope"
        ]  # here orientation['slope'] refers to the tilt of the tracker axis.

        rotation = arctan(
            (cos(sun_altitude) * sin(sun_azimuth - surface_azimuth))
            / (
                cos(sun_altitude) * cos(sun_azimuth - surface_azimuth) * sin(axis_tilt)
                + sin(sun_altitude) * cos(axis_tilt)
            )
        )

        surface_slope = arccos(cos(rotation) * cos(axis_tilt))

        azimuth_difference = sun_azimuth - surface_azimuth
        azimuth_difference = np.where(
            azimuth_difference > pi, 2 * pi - azimuth_difference, azimuth_difference
        )
        azimuth_difference = np.where(
            azimuth_difference < -pi, 2 * pi + azimuth_difference, azimuth_difference
        )
        rotation = np.where(
            logical_and(rotation < 0, azimuth_difference > 0),
            rotation + pi,
            rotation,
        )
        rotation = np.where(
            logical_and(rotation > 0, azimuth_difference < 0),
            rotation - pi,
            rotation,
        )

        cosincidence = cos(rotation) * (
            sin(axis_tilt) * cos(sun_altitude) * cos(sun_azimuth - surface_azimuth)
            + cos(axis_tilt) * sin(sun_altitude)
        ) + sin(rotation) * cos(sun_altitude) * sin(sun_azimuth - surface_azimuth)

    elif tracking == "vertical":  # vertical tracking, surface azimuth = sun_azimuth
        cosincidence = sin(surface_slope) * cos(sun_altitude) + cos(
            surface_slope
        ) * sin(sun_altitude)
    elif tracking == "dual":  # both vertical and horizontal tracking
        cosincidence = np.float64(1.0)
    else:
        assert False, (
            "Values describing tracking system must be None for no tracking,"
            + "'horizontal' for 1-axis horizontal tracking,"
            + "tilted_horizontal' for 1-axis horizontal tracking of tilted panle,"
            + "vertical' for 1-axis vertical tracking, or 'dual' for 2-axis tracking"
        )

    # fixup incidence angle: if the panel is badly oriented and the sun shines
    # on the back of the panel (incidence angle > 90degree), the irradiation
    # would be negative instead of 0; this is prevented here.
    cosincidence = cosincidence.clip(min=0)

    return xr.Dataset(
        {
            "cosincidence": cosincidence,
            "slope": surface_slope,
            "azimuth": surface_azimuth,
        }
    )
