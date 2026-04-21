# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Panel orientation and tilt angle utilities."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from dask.array import arccos, arcsin, arctan, cos, logical_and, radians, sin
from numpy import pi

if TYPE_CHECKING:
    from collections.abc import Callable

    from atlite._types import OrientationName, TrackingType


def get_orientation(
    name: OrientationName | dict[str, Any], **params: Any
) -> Callable[[xr.DataArray, xr.DataArray, xr.Dataset], dict[str, xr.DataArray]]:
    """
    Return an orientation factory by name.

    Conventions:
    - ``slope`` is the angle between ground and panel.
    - ``azimuth`` is the clockwise angle from North (i.e. azimuth=180 faces South).

    Parameters
    ----------
    name : str or dict
        Orientation name or parameter dictionary containing ``name``.
    **params
        Parameters passed to the orientation factory.

    Returns
    -------
    callable
        Orientation function returning ``slope`` and ``azimuth``.
    """
    if isinstance(name, dict):
        params = name
        name = params.pop("name", "constant")
    result: Callable[
        [xr.DataArray, xr.DataArray, xr.Dataset], dict[str, xr.DataArray]
    ] = getattr(sys.modules[__name__], f"make_{name}")(**params)
    return result


def make_latitude_optimal() -> Callable[
    [xr.DataArray, xr.DataArray, xr.Dataset], dict[str, xr.DataArray]
]:
    """
    Return an optimal tilt angle assuming the panel faces the equator.

    This method only works for latitudes between 0 and 50. For higher
    latitudes, a static 40 degree angle is returned.

    These results should be used with caution, but there is some
    evidence that tilt angle may not be that important [2].

    Function and documentation has been adapted from gsee [3].

    [1] http://www.solarpaneltilt.com/#fixed
    [2] http://dx.doi.org/10.1016/j.solener.2010.12.014
    [3] https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py

    Returns
    -------
    callable
        Orientation function returning latitude-optimal ``slope`` and ``azimuth``.
    """

    def latitude_optimal(
        lon: xr.DataArray, lat: xr.DataArray, solar_position: xr.Dataset
    ) -> dict[str, xr.DataArray]:
        """
        Build an orientation with latitude-dependent optimal tilt.

        Parameters
        ----------
        lon : xarray.DataArray
            Longitudes in radians.
        lat : xarray.DataArray
            Latitudes in radians.
        solar_position : xarray.Dataset
            Solar position dataset.

        Returns
        -------
        dict
            Mapping with ``slope`` and ``azimuth``.
        """
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
        return {
            "slope": xr.DataArray(slope, coords=lat.coords),
            "azimuth": xr.DataArray(azimuth, coords=lat.coords),
        }

    return latitude_optimal


def make_constant(
    slope: float, azimuth: float
) -> Callable[[xr.DataArray, xr.DataArray, xr.Dataset], dict[str, xr.DataArray]]:
    """
    Create an orientation function with constant slope and azimuth.

    Parameters
    ----------
    slope : float
        Surface slope in degrees.
    azimuth : float
        Surface azimuth in degrees clockwise from north.

    Returns
    -------
    callable
        Orientation function returning constant ``slope`` and ``azimuth``.
    """
    slope_rad = radians(slope)
    azimuth_rad = radians(azimuth)

    def constant(
        lon: xr.DataArray, lat: xr.DataArray, solar_position: xr.Dataset
    ) -> dict[str, xr.DataArray]:
        """
        Return the configured constant panel orientation.

        Parameters
        ----------
        lon : xarray.DataArray
            Longitudes in radians.
        lat : xarray.DataArray
            Latitudes in radians.
        solar_position : xarray.Dataset
            Solar position dataset.

        Returns
        -------
        dict
            Mapping with constant ``slope`` and ``azimuth``.
        """
        return {"slope": slope_rad, "azimuth": azimuth_rad}

    return constant


def make_latitude(
    azimuth: float = 180,
) -> Callable[[xr.DataArray, xr.DataArray, xr.Dataset], dict[str, xr.DataArray]]:
    """
    Create an orientation function with slope equal to latitude.

    Parameters
    ----------
    azimuth : float, default 180
        Surface azimuth in degrees clockwise from north.

    Returns
    -------
    callable
        Orientation function using latitude as slope.
    """
    azimuth_rad = radians(azimuth)

    def latitude(
        lon: xr.DataArray, lat: xr.DataArray, solar_position: xr.Dataset
    ) -> dict[str, xr.DataArray]:
        """
        Return an orientation with slope equal to latitude.

        Parameters
        ----------
        lon : xarray.DataArray
            Longitudes in radians.
        lat : xarray.DataArray
            Latitudes in radians.
        solar_position : xarray.Dataset
            Solar position dataset.

        Returns
        -------
        dict
            Mapping with latitude-based ``slope`` and constant ``azimuth``.
        """
        return {"slope": lat, "azimuth": azimuth_rad}

    return latitude


def SurfaceOrientation(
    ds: xr.Dataset,
    solar_position: xr.Dataset,
    orientation: Callable[
        [xr.DataArray, xr.DataArray, xr.Dataset], dict[str, xr.DataArray]
    ],
    tracking: TrackingType | None = None,
) -> xr.Dataset:
    """
    Compute cos(incidence) for slope and panel azimuth.

    Parameters
    ----------
    ds : xarray.Dataset
        Weather dataset containing ``lon`` and ``lat`` coordinates in degrees.
    solar_position : xarray.Dataset
        Dataset with solar position variables ``altitude`` and ``azimuth``
        in radians.
    orientation : callable
        Function returning a dict with ``slope`` and ``azimuth`` (in radians)
        given ``(lon, lat, solar_position)``. Typically produced by
        :func:`get_orientation`.
    tracking : {None, 'horizontal', 'tilted_horizontal', 'vertical', 'dual'}, optional
        Tracking type. ``None`` for fixed panels, ``'horizontal'`` for 1-axis
        horizontal tracking, ``'tilted_horizontal'`` for 1-axis horizontal
        tracking of a tilted panel, ``'vertical'`` for 1-axis vertical
        tracking, or ``'dual'`` for 2-axis tracking.

    Returns
    -------
    xarray.Dataset
        Dataset with ``cosincidence``, ``slope``, and ``azimuth``.

    Raises
    ------
    AssertionError
        If ``tracking`` is not a recognized tracking type.

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

    orientation_dict = orientation(lon, lat, solar_position)
    surface_slope = orientation_dict["slope"]
    surface_azimuth = orientation_dict["azimuth"]

    sun_altitude = solar_position["altitude"]
    sun_azimuth = solar_position["azimuth"]

    if tracking is None:
        cosincidence = sin(surface_slope) * cos(sun_altitude) * cos(
            surface_azimuth - sun_azimuth
        ) + cos(surface_slope) * sin(sun_altitude)

    elif tracking == "horizontal":  # horizontal tracking with horizontal axis
        # orientation_dict['azimuth'] refers here to the azimuth of the tracker axis
        axis_azimuth = orientation_dict["azimuth"]
        rotation = arctan(
            (cos(sun_altitude) / sin(sun_altitude)) * sin(sun_azimuth - axis_azimuth)
        )
        surface_slope = abs(rotation)
        # the 2nd part yields +/-1 and determines if the panel is facing east or west
        surface_azimuth = axis_azimuth + arcsin(sin(rotation) / sin(surface_slope))
        cosincidence = cos(surface_slope) * sin(sun_altitude) + sin(
            surface_slope
        ) * cos(sun_altitude) * cos(sun_azimuth - surface_azimuth)

    elif tracking == "tilted_horizontal":  # horizontal tracking with tilted axis
        # orientation_dict['slope'] refers here to the tilt of the tracker axis
        axis_tilt = orientation_dict["slope"]

        rotation = arctan(
            (cos(sun_altitude) * sin(sun_azimuth - surface_azimuth))
            / (
                cos(sun_altitude) * cos(sun_azimuth - surface_azimuth) * sin(axis_tilt)
                + sin(sun_altitude) * cos(axis_tilt)
            )
        )

        surface_slope = arccos(cos(rotation) * cos(axis_tilt))

        azimuth_difference = sun_azimuth - surface_azimuth
        azimuth_difference = xr.where(
            azimuth_difference > pi, azimuth_difference - 2 * pi, azimuth_difference
        )
        azimuth_difference = xr.where(
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

    elif tracking == "vertical":
        cosincidence = sin(surface_slope) * cos(sun_altitude) + cos(
            surface_slope
        ) * sin(sun_altitude)
    elif tracking == "dual":
        cosincidence = np.float64(1.0)
    else:
        msg = (
            "Values describing tracking system must be None for no tracking,"
            + "'horizontal' for 1-axis horizontal tracking,"
            + "'tilted_horizontal' for 1-axis horizontal tracking of tilted panel,"
            + "'vertical' for 1-axis vertical tracking, or 'dual' for 2-axis tracking"
        )
        raise AssertionError(msg)

    # fixup incidence angle: if the panel is badly oriented and the sun shines
    # on the back of the panel (incidence > 90 deg), the irradiation would be
    # negative instead of 0; this is prevented here.
    cosincidence = cosincidence.clip(min=0)

    return xr.Dataset({
        "cosincidence": cosincidence,
        "slope": surface_slope,
        "azimuth": surface_azimuth,
    })
