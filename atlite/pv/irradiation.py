# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

import logging

import numpy as np
from dask.array import cos, fmax, fmin, radians, sin, sqrt

logger = logging.getLogger(__name__)


def DiffuseHorizontalIrrad(ds, solar_position, clearsky_model, influx):
    # Clearsky model from Reindl 1990 to split downward radiation into direct
    # and diffuse contributions. Should switch to more up-to-date model, f.ex.
    # Ridley et al. (2010) http://dx.doi.org/10.1016/j.renene.2009.07.018 ,
    # Lauret et al. (2013):http://dx.doi.org/10.1016/j.renene.2012.01.049

    sinaltitude = sin(solar_position["altitude"])
    influx_toa = ds["influx_toa"]

    if clearsky_model is None:
        clearsky_model = (
            "enhanced" if "temperature" in ds and "humidity" in ds else "simple"
        )

    # Reindl 1990 clearsky model

    k = influx / influx_toa  # clearsky index
    # k.values[k.values > 1.0] = 1.0
    # k = k.rename('clearsky index')

    if clearsky_model == "simple":
        # Simple Reindl model without ambient air temperature and
        # relative humidity
        fraction = (
            ((k > 0.0) & (k <= 0.3))
            * fmin(1.0, 1.020 - 0.254 * k + 0.0123 * sinaltitude)
            + ((k > 0.3) & (k < 0.78))
            * fmin(0.97, fmax(0.1, 1.400 - 1.749 * k + 0.177 * sinaltitude))
            + (k >= 0.78) * fmax(0.1, 0.486 * k - 0.182 * sinaltitude)
        )
    elif clearsky_model == "enhanced":
        # Enhanced Reindl model with ambient air temperature and relative
        # humidity
        T = ds["temperature"]
        rh = ds["humidity"]

        fraction = (
            ((k > 0.0) & (k <= 0.3))
            * fmin(
                1.0,
                1.000 - 0.232 * k + 0.0239 * sinaltitude - 0.000682 * T + 0.0195 * rh,
            )
            + ((k > 0.3) & (k < 0.78))
            * fmin(
                0.97,
                fmax(
                    0.1,
                    1.329 - 1.716 * k + 0.267 * sinaltitude - 0.00357 * T + 0.106 * rh,
                ),
            )
            + (k >= 0.78)
            * fmax(0.1, 0.426 * k - 0.256 * sinaltitude + 0.00349 * T + 0.0734 * rh)
        )
    else:
        raise KeyError(
            "`clearsky model` must be chosen from 'simple' and " "'enhanced'"
        )

    # Set diffuse fraction to one when the sun isn't up
    # fraction = fraction.where(sinaltitude >= sin(radians(threshold))).fillna(1.0)
    # fraction = fraction.rename('fraction index')

    return (influx * fraction).rename("diffuse horizontal")


def TiltedDiffuseIrrad(ds, solar_position, surface_orientation, direct, diffuse):
    # Hay-Davies Model

    sinaltitude = sin(solar_position["altitude"])
    influx_toa = ds["influx_toa"]

    cosincidence = surface_orientation["cosincidence"]
    surface_slope = surface_orientation["slope"]

    influx = direct + diffuse

    with np.errstate(divide="ignore", invalid="ignore"):
        # brightening factor
        f = sqrt(direct / influx).fillna(0.0)

        # anisotropy factor
        A = direct / influx_toa

    # geometric factor
    R_b = cosincidence / sinaltitude

    diffuse_t = (
        (1.0 - A)
        * ((1 + cos(surface_slope)) / 2.0)
        * (1.0 + f * sin(surface_slope / 2.0) ** 3)
        + A * R_b
    ) * diffuse

    # fixup: clip all negative values (unclear why it gets negative)
    # note: REatlas does not do the fixup
    if logger.isEnabledFor(logging.WARNING):
        if ((diffuse_t < 0.0) & (sinaltitude > sin(radians(1.0)))).any():
            logger.warning(
                "diffuse_t exhibits negative values above altitude threshold."
            )

    with np.errstate(invalid="ignore"):
        diffuse_t = diffuse_t.clip(min=0).fillna(0)

    return diffuse_t.rename("diffuse tilted")


def TiltedDirectIrrad(solar_position, surface_orientation, direct):
    sinaltitude = sin(solar_position["altitude"])
    cosincidence = surface_orientation["cosincidence"]

    # geometric factor
    R_b = cosincidence / sinaltitude

    return (R_b * direct).rename("direct tilted")


def _albedo(ds, influx):
    if "albedo" in ds:
        albedo = ds["albedo"]
    elif "outflux" in ds:
        albedo = (ds["outflux"] / influx.where(influx != 0)).fillna(0).clip(max=1)
    else:
        raise AssertionError(
            "Need either albedo or outflux as a variable in the dataset. "
            "Check your cutout and dataset module."
        )

    return albedo


def TiltedGroundIrrad(ds, solar_position, surface_orientation, influx):
    surface_slope = surface_orientation["slope"]
    ground_t = influx * _albedo(ds, influx) * (1.0 - cos(surface_slope)) / 2.0
    return ground_t.rename("ground tilted")


def TiltedIrradiation(
    ds,
    solar_position,
    surface_orientation,
    trigon_model,
    clearsky_model,
    tracking=0,
    altitude_threshold=1.0,
    irradiation="total",
):
    """
    Calculate the irradiation on a tilted surface.

    Parameters
    ----------
    ds : xarray.Dataset
        Cutout data used for calculating the irradiation on a tilted surface.
    solar_position : xarray.Dataset
        Solar position calculated using atlite.pv.SolarPosition,
        containing a solar 'altitude' (in rad, 0 to pi/2) for the 'ds' dataset.
    surface_orientation : xarray.Dataset
        Surface orientation calculated using atlite.orientation.SurfaceOrientation.
    trigon_model : str
        Type of trigonometry model. Defaults to 'simple'if used via `convert_irradiation`.
    clearsky_model : str or None
        Either the 'simple' or the 'enhanced' Reindl clearsky
        model. The default choice of None will choose dependending on
        data availability, since the 'enhanced' model also
        incorporates ambient air temperature and relative humidity.
        NOTE: this option is only used if the used climate dataset
        doesn't provide direct and diffuse irradiation separately!
    altitude_threshold : float
        Threshold for solar altitude in degrees. Values in range (0, altitude_threshold]
        will be set to zero. Default value equals 1.0 degrees.
    irradiation : str
        The irradiation quantity to be returned. Defaults to "total" for total
        combined irradiation. Other options include "direct" for direct irradiation,
        "diffuse" for diffuse irradation, and "ground" for irradiation reflected
        by the ground via albedo. NOTE: "ground" irradiation is not calculated
        by all `trigon_model` options in the `convert_irradiation` method,
        so use with caution!

    Returns
    -------
    result : xarray.DataArray
        The desired irradiation quantity on the tilted surface.

    """
    influx_toa = ds["influx_toa"]

    def clip(influx, influx_max):
        # use .data in clip due to dask-xarray incompatibilities
        return influx.clip(min=0, max=influx_max.transpose(*influx.dims).data)

    if "influx" in ds:
        influx = clip(ds["influx"], influx_toa)
        diffuse = DiffuseHorizontalIrrad(ds, solar_position, clearsky_model, influx)
        direct = influx - diffuse
    elif "influx_direct" in ds and "influx_diffuse" in ds:
        direct = clip(ds["influx_direct"], influx_toa)
        diffuse = clip(ds["influx_diffuse"], influx_toa - direct)
    else:
        raise AssertionError(
            "Need either influx or influx_direct and influx_diffuse in the "
            "dataset. Check your cutout and dataset module."
        )
    if trigon_model == "simple":
        k = surface_orientation["cosincidence"] / sin(solar_position["altitude"])
        if tracking != "dual":
            cos_surface_slope = cos(surface_orientation["slope"])
        elif tracking == "dual":
            cos_surface_slope = sin(solar_position["altitude"])

        influx = direct + diffuse
        direct_t = k * direct
        diffuse_t = (1.0 + cos_surface_slope) / 2.0 * diffuse
        ground_t = _albedo(ds, influx) * influx * ((1.0 - cos_surface_slope) / 2.0)

        total_t = direct_t.fillna(0.0) + diffuse_t.fillna(0.0) + ground_t.fillna(0.0)
    else:
        diffuse_t = TiltedDiffuseIrrad(
            ds, solar_position, surface_orientation, direct, diffuse
        )
        direct_t = TiltedDirectIrrad(solar_position, surface_orientation, direct)
        ground_t = TiltedGroundIrrad(
            ds, solar_position, surface_orientation, direct + diffuse
        )

        total_t = direct_t + diffuse_t + ground_t

    if irradiation == "total":
        result = total_t.rename("total tilted")
    elif irradiation == "direct":
        result = direct_t.rename("direct tilted")
    elif irradiation == "diffuse":
        result = diffuse_t.rename("diffuse tilted")
    elif irradiation == "ground":
        result = ground_t.rename("ground tilted")

    # The solar_position algorithms have a high error for small solar altitude
    # values, leading to big overall errors from the 1/sinaltitude factor.
    # => Suppress irradiation below solar altitudes of 1 deg.

    cap_alt = solar_position["altitude"] < radians(altitude_threshold)
    result = result.where(~(cap_alt | (direct + diffuse <= 0.01)), 0)
    result.attrs["units"] = "W m**-2"

    return result
