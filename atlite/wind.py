# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT
"""
Functions for use in conjunction with wind data generation.
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


def extrapolate_wind_speed(
    ds, to_height, from_height=None, method: str = "logarithmic"
):
    """
    Extrapolate the wind speed from a given height above ground to another.

    If ds already contains a key refering to wind speeds at the desired to_height,
    no conversion is done and the wind speeds are directly returned.

    Extrapolation of the wind speed can either use the "logarithmic" law as
    described in [1] or the "power" law as described in [2]. See also discussion
    in GH issue: https://github.com/PyPSA/atlite/issues/231 .

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the wind speed time-series at 'from_height' with key
        'wnd{height:d}m' and the surface orography with key 'roughness' at the
        geographic locations of the wind speeds.
    from_height : int
        (Optional)
        Height (m) from which the wind speeds are interpolated to 'to_height'.
        If not provided, the closest height to 'to_height' is selected.
    to_height : int|float
        Height (m) to which the wind speeds are extrapolated to.
    method : {"logarithmic", "power"}
        Method to use for extra/interpolating wind speeds

    Returns
    -------
    da : xarray.DataArray
        DataArray containing the extrapolated wind speeds. Name of the DataArray
        is 'wnd{to_height:d}'.

    References
    ----------
    [1] Equation (2) in Andresen, G. et al (2015): 'Validation of Danish wind
    time series from a new global renewable energy atlas for energy system
    analysis'.

    [2] Gualtieri, G. (2021): 'Reliability of ERA5 Reanalysis Data for Wind
    Resource Assessment: A Comparison against Tall Towers'
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
        wnd_spd = ds[from_name] * (
            np.log(to_height / ds["roughness"]) / np.log(from_height / ds["roughness"])
        )
        method_desc = "logarithmic method with roughness"
    elif method == "power":
        wnd_spd = ds[from_name] * (to_height / from_height) ** ds["wnd_shear_exp"]
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
