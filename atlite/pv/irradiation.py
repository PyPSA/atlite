# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Solar irradiation decomposition and transposition models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from dask.array import cos, fmax, fmin, radians, sin, sqrt

if TYPE_CHECKING:
    import xarray as xr

    from atlite._types import (
        ClearskyModel,
        IrradiationType,
        TrackingType,
        TrigonModel,
    )

logger = logging.getLogger(__name__)


def DiffuseHorizontalIrrad(
    ds: xr.Dataset,
    solar_position: xr.Dataset,
    clearsky_model: ClearskyModel | None,
    influx: xr.DataArray,
) -> xr.DataArray:
    """
    Estimate diffuse horizontal irradiation from total horizontal irradiation.

    Clearsky model from Reindl 1990 to split downward radiation into direct
    and diffuse contributions. Should switch to more up-to-date model, e.g.
    Ridley et al. (2010) http://dx.doi.org/10.1016/j.renene.2009.07.018 ,
    Lauret et al. (2013): http://dx.doi.org/10.1016/j.renene.2012.01.049

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing top-of-atmosphere irradiation and, for the enhanced
        model, temperature and humidity.
    solar_position : xarray.Dataset
        Solar position with an ``altitude`` variable in radians.
    clearsky_model : str or None
        Reindl clearsky model to use, either ``"simple"`` or ``"enhanced"``.
        If None, the model is chosen from the available data.
    influx : xarray.DataArray
        Total horizontal irradiation.

    Returns
    -------
    xarray.DataArray
        Diffuse horizontal irradiation.

    Raises
    ------
    KeyError
        If ``clearsky_model`` is not ``'simple'`` or ``'enhanced'``.
    """
    sinaltitude = sin(solar_position["altitude"])
    influx_toa = ds["influx_toa"]

    if clearsky_model is None:
        clearsky_model = (
            "enhanced" if "temperature" in ds and "humidity" in ds else "simple"
        )

    # Reindl 1990 clearsky model
    k = influx / influx_toa  # clearsky index

    if clearsky_model == "simple":
        fraction = (
            ((k > 0.0) & (k <= 0.3))
            * fmin(1.0, 1.020 - 0.254 * k + 0.0123 * sinaltitude)
            + ((k > 0.3) & (k < 0.78))
            * fmin(0.97, fmax(0.1, 1.400 - 1.749 * k + 0.177 * sinaltitude))
            + (k >= 0.78) * fmax(0.1, 0.486 * k - 0.182 * sinaltitude)
        )
    elif clearsky_model == "enhanced":
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
        raise KeyError("`clearsky model` must be chosen from 'simple' and 'enhanced'")

    result: xr.DataArray = (influx * fraction).rename("diffuse horizontal")
    return result


def TiltedDiffuseIrrad(
    ds: xr.Dataset,
    solar_position: xr.Dataset,
    surface_orientation: xr.Dataset,
    direct: xr.DataArray,
    diffuse: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate diffuse irradiation on a tilted surface (Hay-Davies model).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing top-of-atmosphere irradiation.
    solar_position : xarray.Dataset
        Solar position with an ``altitude`` variable in radians.
    surface_orientation : xarray.Dataset
        Surface orientation including ``cosincidence`` and ``slope``.
    direct : xarray.DataArray
        Direct horizontal irradiation.
    diffuse : xarray.DataArray
        Diffuse horizontal irradiation.

    Returns
    -------
    xarray.DataArray
        Diffuse tilted irradiation.
    """
    sinaltitude = sin(solar_position["altitude"])
    influx_toa = ds["influx_toa"]

    cosincidence = surface_orientation["cosincidence"]
    surface_slope = surface_orientation["slope"]

    influx = direct + diffuse

    with np.errstate(divide="ignore", invalid="ignore"):
        f = sqrt(direct / influx).fillna(0.0)  # brightening factor
        A = direct / influx_toa  # anisotropy factor

    R_b = cosincidence / sinaltitude

    diffuse_t = (
        (1.0 - A)
        * ((1 + cos(surface_slope)) / 2.0)
        * (1.0 + f * sin(surface_slope / 2.0) ** 3)
        + A * R_b
    ) * diffuse

    if (
        logger.isEnabledFor(logging.WARNING)
        and ((diffuse_t < 0.0) & (sinaltitude > sin(radians(1.0)))).any()
    ):
        logger.warning("diffuse_t exhibits negative values above altitude threshold.")

    # fixup: clip all negative values (unclear why it gets negative)
    # note: REatlas does not do the fixup
    with np.errstate(invalid="ignore"):
        diffuse_t = diffuse_t.clip(min=0).fillna(0)

    result: xr.DataArray = diffuse_t.rename("diffuse tilted")
    return result


def TiltedDirectIrrad(
    solar_position: xr.Dataset, surface_orientation: xr.Dataset, direct: xr.DataArray
) -> xr.DataArray:
    """
    Calculate direct irradiation on a tilted surface.

    Parameters
    ----------
    solar_position : xarray.Dataset
        Solar position with an ``altitude`` variable in radians.
    surface_orientation : xarray.Dataset
        Surface orientation including ``cosincidence``.
    direct : xarray.DataArray
        Direct horizontal irradiation.

    Returns
    -------
    xarray.DataArray
        Direct tilted irradiation.
    """
    sinaltitude = sin(solar_position["altitude"])
    cosincidence = surface_orientation["cosincidence"]

    R_b = cosincidence / sinaltitude

    result: xr.DataArray = (R_b * direct).rename("direct tilted")
    return result


def _albedo(ds: xr.Dataset, influx: xr.DataArray) -> xr.DataArray:
    """
    Retrieve or derive surface albedo from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing either ``albedo`` or ``outflux``.
    influx : xarray.DataArray
        Downward surface irradiation used when deriving albedo from outflux.

    Returns
    -------
    xarray.DataArray
        Surface albedo.

    Raises
    ------
    AssertionError
        If the dataset lacks both ``albedo`` and ``outflux`` variables.
    """
    if "albedo" in ds:
        return ds["albedo"]
    if "outflux" in ds:
        return (ds["outflux"] / influx.where(influx != 0)).fillna(0).clip(max=1)
    raise AssertionError(
        "Need either albedo or outflux as a variable in the dataset. "
        "Check your cutout and dataset module."
    )


def TiltedGroundIrrad(
    ds: xr.Dataset,
    solar_position: xr.Dataset,
    surface_orientation: xr.Dataset,
    influx: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate ground-reflected irradiation on a tilted surface.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing albedo information or reflected outflux.
    solar_position : xarray.Dataset
        Solar position dataset.
    surface_orientation : xarray.Dataset
        Surface orientation including ``slope``.
    influx : xarray.DataArray
        Total horizontal irradiation.

    Returns
    -------
    xarray.DataArray
        Ground-reflected tilted irradiation.
    """
    surface_slope = surface_orientation["slope"]
    ground_t = influx * _albedo(ds, influx) * (1.0 - cos(surface_slope)) / 2.0
    result: xr.DataArray = ground_t.rename("ground tilted")
    return result


def TiltedIrradiation(
    ds: xr.Dataset,
    solar_position: xr.Dataset,
    surface_orientation: xr.Dataset,
    trigon_model: TrigonModel,
    clearsky_model: ClearskyModel | None,
    tracking: TrackingType | int | None = 0,
    altitude_threshold: float = 1.0,
    irradiation: IrradiationType = "total",
) -> xr.DataArray:
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
        Type of trigonometry model. Defaults to 'simple' if used via
        `convert_irradiation`.
    clearsky_model : str or None
        Either the 'simple' or the 'enhanced' Reindl clearsky
        model. The default choice of None will choose dependending on
        data availability, since the 'enhanced' model also
        incorporates ambient air temperature and relative humidity.
        NOTE: this option is only used if the used climate dataset
        doesn't provide direct and diffuse irradiation separately!
    tracking : int or str, default 0
        Type of solar tracking. 0 for fixed, other values for tracking modes.
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

    Raises
    ------
    AssertionError
        If the dataset lacks required irradiation variables.
    ValueError
        If ``irradiation`` is not a recognized type.
    """
    influx_toa = ds["influx_toa"]

    def clip(influx: xr.DataArray, influx_max: xr.DataArray) -> xr.DataArray:
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

    result: xr.DataArray
    if irradiation == "total":
        result = total_t.rename("total tilted")
    elif irradiation == "direct":
        result = direct_t.rename("direct tilted")
    elif irradiation == "diffuse":
        result = diffuse_t.rename("diffuse tilted")
    elif irradiation == "ground":
        result = ground_t.rename("ground tilted")
    else:
        msg = f"Unknown irradiation type: {irradiation}"
        raise ValueError(msg)

    # The solar_position algorithms have a high error for small solar altitude
    # values, leading to big overall errors from the 1/sinaltitude factor.
    # => Suppress irradiation below altitude_threshold.
    cap_alt = solar_position["altitude"] < radians(altitude_threshold)
    result = result.where(~(cap_alt | (direct + diffuse <= 0.01)), 0)
    result.attrs["units"] = "W m**-2"

    return result
