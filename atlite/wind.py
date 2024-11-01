# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Functions for use in conjunction with wind data generation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio as rio
import xarray as xr

from . import datasets

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import Literal


def extrapolate_wind_speed(
    ds: xr.Dataset,
    to_height: int | float,
    from_height: int | None = None,
    method: Literal["logarithmic", "power"] = "logarithmic",
) -> xr.DataArray:
    """
    Extrapolate the wind speed from a given height above ground to another.

    If ds already contains a key refering to wind speeds at the desired to_height,
    no conversion is done and the wind speeds are directly returned.

    Extrapolation of the wind speed can either use the "logarithmic" law as
    described in [1]_ or the "power" law as described in [2]_. See also discussion
    in GH issue: https://github.com/PyPSA/atlite/issues/231 .

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the wind speed time-series at 'from_height' with key
        'wnd{height:d}m' and the surface orography with key 'roughness' at the
        geographic locations of the wind speeds.
    from_height : int, optional
        Height (m) from which the wind speeds are interpolated to 'to_height'.
        If not provided, the closest height to 'to_height' is selected.
    to_height : int|float
        Height (m) to which the wind speeds are extrapolated to.
    from_height : int, optional
        Height (m) from which the wind speeds are interpolated to 'to_height'.
        If not provided, the closest height to 'to_height' is selected.
    method : {"logarithmic", "power"}
        Method to use for extra/interpolating wind speeds

    Returns
    -------
    da : xarray.DataArray
        DataArray containing the extrapolated wind speeds. Name of the DataArray
        is 'wnd{to_height:d}'.

    Raises
    ------
    RuntimeError
        If the cutout is missing the data for the chosen `method`

    References
    ----------
    .. [1] Equation (2) in Andresen, G. et al (2015): 'Validation of Danish
       wind time series from a new global renewable energy atlas for energy
       system analysis'.

    .. [2] Gualtieri, G. (2021): 'Reliability of ERA5 Reanalysis Data for
       Wind Resource Assessment: A Comparison against Tall Towers'
       https://doi.org/10.3390/en14144169 .
    """
    if from_height is None:
        # Determine closest height to to_name
        heights = np.asarray(
            [int(m.group(1)) for s in ds if (m := re.match(r"wnd(\d+)m", s))]
        )

        if len(heights) == 0:
            raise AssertionError("Wind speed is not in dataset")

        from_height = heights[np.argmin(np.abs(heights - to_height))]

    from_name = f"wnd{int(from_height):0d}m"

    # Fast lane
    if from_height == to_height:
        return ds[from_name]

    if method == "logarithmic":
        try:
            roughness = ds["roughness"]
        except KeyError:
            raise RuntimeError(
                "The logarithmic interpolation method requires surface roughness (roughness);\n"
                "make sure you choose a compatible dataset like ERA5"
            )
        wnd_spd = ds[from_name] * (
            np.log(to_height / roughness) / np.log(from_height / roughness)
        )
        method_desc = "logarithmic method with roughness"
    elif method == "power":
        try:
            wnd_shear_exp = ds["wnd_shear_exp"]
        except KeyError:
            raise RuntimeError(
                "The power law interpolation method requires a wind shear exponent (wnd_shear_exp);\n"
                "make sure you choose a compatible dataset like ERA5 and update your cutout"
            )
        wnd_spd = ds[from_name] * (to_height / from_height) ** wnd_shear_exp
        method_desc = "power method with wind shear exponent"
    else:
        raise ValueError(
            f"Interpolation method must be 'logarithmic' or 'power',  but is: {method}"
        )

    return wnd_spd.assign_attrs(
        {
            "long name": (
                f"extrapolated {to_height} m wind speed using {method_desc} "
                f" and {from_height} m wind speed"
            ),
            "units": "m s**-1",
        }
    ).rename(f"wnd{to_height}m")


def calculate_windspeed_bias_correction(
    cutout, real_average: str | rio.DatasetReader, height: int = 100
):
    """
    Derive a bias correction factor for windspeed at lra_height

    Regrids the raster dataset in real_average to cutout grid, retrieves the average
    windspeed from the first dataset that offers
    :py:func:`retrieve_longrunaverage_windspeed` (only ERA5, currently).

    Parameters
    ----------
    cutout : Cutout
        Atlite cutout
    real_average : Path or rasterio.Dataset
        Raster dataset with wind speeds to bias correct average wind speeds
    height : int
        Height in meters at which average windspeeds are provided

    Returns
    -------
    DataArray
        Ratio between windspeeds in `real_average` to those of average windspeeds in
        dataset.
    """
    if isinstance(real_average, str | Path):
        real_average = rio.open(real_average)

    if isinstance(real_average, rio.DatasetReader):
        real_average = rio.band(real_average, 1)

    if isinstance(real_average, rio.Band):
        real_average, transform = rio.warp.reproject(
            real_average,
            np.empty(cutout.shape),
            dst_crs=cutout.crs,
            dst_transform=cutout.transform,
            dst_nodata=np.nan,
            resampling=rio.enums.Resampling.average,
        )

        real_average = xr.DataArray(
            real_average, [cutout.coords["y"], cutout.coords["x"]]
        )

    for module in np.atleast_1d(cutout.module):
        retrieve_windspeed_average = getattr(
            getattr(datasets, module), "retrieve_windspeed_average"
        )
        if retrieve_windspeed_average is not None:
            break
    else:
        raise AssertionError(
            "None of the datasets modules define retrieve_windspeed_average"
        )

    logger.info("Retrieving average windspeeds at %d from module %s", height, module)
    data_average = retrieve_windspeed_average(cutout, height)

    return real_average / data_average
