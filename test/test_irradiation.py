# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Tests for the irradiation on tilted surfaces."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from atlite.convert import convert_irradiation
from atlite.pv.orientation import get_orientation


@pytest.mark.parametrize("trigon_model", ["simple", "other"])
def test_total_irradiation_is_zero_where_data_is_missing(trigon_model):
    time = pd.date_range("2013-06-01 10:00", periods=3, freq="h")
    coords = {"time": time, "y": [50.0, 70.0], "x": [5.0]}

    def var(values):
        return (("time", "y", "x"), np.broadcast_to(values, (3, 2, 1)).copy())

    missing = np.array([[1.0], [np.nan]])
    ds = xr.Dataset(
        {
            "influx_toa": var(1000.0),
            "influx_direct": var(300.0 * missing),
            "influx_diffuse": var(100.0 * missing),
            "albedo": var(0.2),
            "solar_altitude": var(0.8 * missing),
            "solar_azimuth": var(3.0 * missing),
        },
        coords=coords,
    )
    ds = ds.assign_coords(lon=("x", coords["x"]), lat=("y", coords["y"]))
    orientation = get_orientation({"slope": 30.0, "azimuth": 180.0})

    res = convert_irradiation(ds, orientation, trigon_model=trigon_model)

    assert (res.sel(y=50.0) > 0).all()
    assert (res.sel(y=70.0) == 0).all()
