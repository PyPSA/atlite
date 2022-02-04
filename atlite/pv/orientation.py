# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import numpy as np
import xarray as xr
from numpy import sin, cos, deg2rad, pi


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
    return getattr(sys.modules[__name__], "make_{}".format(name))(**params)


def make_latitude_optimal():
    """
    Returns an optimal tilt angle for the given ``lat``, assuming that
    the panel is facing towards the equator, using a simple method from [1].

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

        below_25 = np.abs(lat.values) <= deg2rad(25)
        below_50 = np.abs(lat.values) <= deg2rad(50)

        slope[below_25] = 0.87 * np.abs(lat.values[below_25])
        slope[~below_25 & below_50] = 0.76 * np.abs(
            lat.values[~below_25 & below_50]
        ) + deg2rad(0.31)
        slope[~below_50] = np.deg2rad(40.0)

        # South orientation for panels on northern hemisphere and vice versa
        azimuth = np.where(lat.values < 0, 0, pi)

        return dict(
            slope=xr.DataArray(slope, coords=lat.coords),
            azimuth=xr.DataArray(azimuth, coords=lat.coords),
        )

    return latitude_optimal


def make_constant(slope, azimuth):
    slope = deg2rad(slope)
    azimuth = deg2rad(azimuth)

    def constant(lon, lat, solar_position):
        return dict(slope=slope, azimuth=azimuth)

    return constant


def make_latitude(azimuth=180):
    azimuth = deg2rad(azimuth)

    def latitude(lon, lat, solar_position):
        return dict(slope=lat, azimuth=azimuth)

    return latitude


def SurfaceOrientation(ds, solar_position, orientation):
    """
    Compute cos(incidence) for slope and panel azimuth

    References
    ----------
    [1] Sproul, A. B., Derivation of the solar geometric relationships using
    vector analysis, Renewable Energy, 32(7), 1187–1205 (2007).
    """

    lon = deg2rad(ds["lon"])
    lat = deg2rad(ds["lat"])

    orientation = orientation(lon, lat, solar_position)
    surface_slope = orientation["slope"]
    surface_azimuth = orientation["azimuth"]

    sun_altitude = solar_position["altitude"]
    sun_azimuth = solar_position["azimuth"]

    cosincidence = sin(surface_slope) * cos(sun_altitude) * cos(
        surface_azimuth - sun_azimuth
    ) + cos(surface_slope) * sin(sun_altitude)

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
