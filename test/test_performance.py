# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Tests that conversions stay lazy and cutouts read data efficiently."""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from atlite import Cutout
from atlite.convert import convert_csp, convert_pv
from atlite.pv.orientation import get_orientation
from atlite.resource import get_cspinstallationconfig, get_solarpanelconfig


def _forbid_compute(*args, **kwargs):
    raise RuntimeError("Unexpected eager computation.")


@pytest.fixture
def ds():
    rng = np.random.default_rng(0)
    time = pd.date_range("2013-06-01", periods=48, freq="h")
    coords = {"time": time, "y": [50.0, 51.0], "x": [5.0, 6.0, 7.0]}
    shape = (len(time), 2, 3)

    def var(low, high):
        return (("time", "y", "x"), rng.uniform(low, high, shape))

    ds = xr.Dataset(
        {
            "influx_toa": var(800, 1000),
            "influx_direct": var(0, 400),
            "influx_diffuse": var(0, 200),
            "albedo": var(0, 0.3),
            "temperature": var(270, 300),
            "solar_altitude": var(-0.5, 1.2),
            "solar_azimuth": var(0, 2 * np.pi),
        },
        coords=coords,
    )
    ds = ds.assign_coords(lon=("x", coords["x"]), lat=("y", coords["y"]))
    return ds.chunk({"time": 24})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tracking": None, "trigon_model": "other"},
        {"tracking": "tilted_horizontal"},
    ],
)
def test_convert_pv_is_lazy(ds, kwargs):
    panel = get_solarpanelconfig("CSi")
    orientation = get_orientation({"slope": 30.0, "azimuth": 180.0})
    with dask.config.set(scheduler=_forbid_compute):
        da = convert_pv(ds, panel, orientation, **kwargs)
    assert isinstance(da.data, dask.array.Array)


def test_convert_csp_keeps_chunks_and_coords(ds):
    installation = get_cspinstallationconfig("SAM_solar_tower")
    da = convert_csp(ds, installation)
    assert da.chunks == ds["influx_direct"].chunks
    assert set(da.coords) <= set(ds.coords)


def test_cutout_chunks_align_with_file_chunks(tmp_path):
    path = tmp_path / "cutout.nc"
    time = pd.date_range("2013-01-01", periods=300, freq="h")
    ds = xr.Dataset(
        {"temperature": (("time", "y", "x"), np.zeros((300, 2, 3)))},
        coords={"time": time, "y": [50.0, 51.0], "x": [5.0, 6.0, 7.0]},
        attrs={"module": "era5"},
    )
    encoding = {"temperature": {"chunksizes": (150, 2, 3), "zlib": True}}
    ds.to_netcdf(path, encoding=encoding)

    chunks = Cutout(path).data["temperature"].chunks
    assert all(c % 150 == 0 for c in chunks[0])
