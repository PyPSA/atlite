# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT
"""
Functions for use in conjunction with wind data generation.
"""

import logging
import re
from pathlib import Path

import numpy as np
import rasterio as rio
import xarray as xr

from .gis import _as_transform

logger = logging.getLogger(__name__)


def extrapolate_wind_speed(ds, to_height, from_height=None):
    """
    Extrapolate the wind speed from a given height above ground to another.

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

    # Wind speed extrapolation
    wnd_spd = ds[from_name] * (
        np.log(to_height / ds["roughness"]) / np.log(from_height / ds["roughness"])
    )

    return wnd_spd.assign_attrs(
        {
            "long name": f"extrapolated {to_height} m wind speed using logarithmic "
            f"method with roughness and {from_height} m wind speed",
            "units": "m s**-1",
        }
    ).rename(f"wnd{to_height}m")


def calculate_windspeed_bias_correction(ds, real_lra, lra_height, crs):
    data_lra = ds[f"wnd{lra_height}m_lra"]

    if isinstance(real_lra, (str, Path)):
        real_lra = rio.open(real_lra)

    if isinstance(real_lra, rio.DatasetReader):
        real_lra = rio.band(real_lra, 1)

    if isinstance(real_lra, rio.Band):
        real_lra, transform = rio.warp.reproject(
            real_lra,
            np.empty(data_lra.shape),
            dst_crs=crs,
            dst_transform=_as_transform(
                x=data_lra.indexes["x"], y=data_lra.indexes["y"]
            ),
            resampling=rio.enums.Resampling.average,
        )

        real_lra = xr.DataArray(
            real_lra, [data_lra.indexes["y"], data_lra.indexes["x"]]
        )

    return real_lra / data_lra
