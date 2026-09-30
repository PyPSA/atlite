# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Tests for the units and names of derived ERA5 variables (#509)."""

import numpy as np
import pandas as pd
import xarray as xr

from atlite.datasets.era5 import _add_height, _process_influx, _process_wind


def _field(value: float, units: str, long_name: str) -> xr.DataArray:
    return xr.DataArray(
        np.full((2, 2, 2), value),
        coords={
            "time": pd.date_range("2013-01-01", periods=2, freq="h"),
            "y": [56.0, 56.25],
            "x": [0.0, 0.25],
        },
        dims=("time", "y", "x"),
        attrs={"units": units, "long_name": long_name},
    )


def test_height_attrs():
    ds = _add_height(xr.Dataset({"z": _field(100.0, "m**2 s**-2", "Geopotential")}))
    assert ds["height"].attrs == {"units": "m", "long_name": "Height"}


def test_wind_azimuth_attrs():
    raw = xr.Dataset({
        name: _field(value, "m s**-1", f"{name} wind component")
        for name, value in [("u10", 1.0), ("v10", 2.0), ("u100", 3.0), ("v100", 4.0)]
    })
    raw["fsr"] = _field(0.1, "m", "Forecast surface roughness")
    ds = _process_wind(raw)
    assert ds["wnd_azimuth"].attrs == {
        "units": "rad",
        "long_name": "100 metre wind azimuth",
    }


def test_solar_position_attrs():
    raw = xr.Dataset({
        name: _field(value, "J m**-2", name)
        for name, value in [
            ("ssrd", 3600.0),
            ("ssr", 1800.0),
            ("fdir", 1800.0),
            ("tisr", 7200.0),
        ]
    })
    raw = raw.assign_coords(
        lon=raw.x.assign_attrs(long_name="longitude"),
        lat=raw.y.assign_attrs(units="degrees_north", long_name="latitude"),
    )
    ds = _process_influx(raw)
    assert ds["solar_altitude"].attrs["long_name"] == "solar altitude"
    assert ds["solar_azimuth"].attrs["long_name"] == "solar azimuth"
    assert ds["solar_altitude"].attrs["units"] == "rad"
