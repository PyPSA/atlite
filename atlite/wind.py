# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Functions for use in conjunction with wind data generation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

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
    # Fast lane
    to_name = f"wnd{int(to_height):0d}m"
    if to_name in ds:
        return ds[to_name]

    if from_height is None:
        # Determine closest height to to_name
        heights = np.asarray([int(s[3:-1]) for s in ds if re.match(r"wnd\d+m", s)])

        if len(heights) == 0:
            raise AssertionError("Wind speed is not in dataset")

        from_height = heights[np.argmin(np.abs(heights - to_height))]

    from_name = f"wnd{int(from_height):0d}m"

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
            f"Interpolation method must be 'logarithmic' or 'power', "
            f" but is: {method}"
        )

    wnd_spd.attrs.update(
        {
            "long name": (
                f"extrapolated {to_height} m wind speed using {method_desc} "
                f" and {from_height} m wind speed"
            ),
            "units": "m s**-1",
        }
    )

    return wnd_spd.rename(to_name)
