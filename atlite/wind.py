# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Functions for use in conjunction with wind data generation.
"""

import numpy as np

import logging

logger = logging.getLogger(__name__)


def extrapolate_wind_speed(ds, to_height, from_height=None):
    """Extrapolate the wind speed from a given height above ground to another.

    If ds already contains a key refering to wind speeds at the desired to_height,
    no conversion is done and the wind speeds are directly returned.

    Extrapolation of the wind speed follows the logarithmic law as desribed in [1].

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

    [2] https://en.wikipedia.org/w/index.php?title=Roughness_length&oldid=862127433,
    Retrieved 2019-02-15.
    """

    # Fast lane
    to_name = "wnd{h:0d}m".format(h=int(to_height))
    if to_name in ds:
        return ds[to_name]

    if from_height is None:
        # Determine closest height to to_name
        heights = np.asarray([int(s[3:-1]) for s in ds if s.startswith("wnd")])

        if len(heights) == 0:
            raise AssertionError("Wind speed is not in dataset")

        from_height = heights[np.argmin(np.abs(heights - to_height))]

    from_name = "wnd{h:0d}m".format(h=int(from_height))

    # Wind speed extrapolation
    wnd_spd = ds[from_name] * (
        np.log(to_height / ds["roughness"]) / np.log(from_height / ds["roughness"])
    )

    wnd_spd.attrs.update(
        {
            "long name": "extrapolated {ht} m wind speed using logarithmic "
            "method with roughness and {hf} m wind speed"
            "".format(ht=to_height, hf=from_height),
            "units": "m s**-1",
        }
    )

    return wnd_spd.rename(to_name)
