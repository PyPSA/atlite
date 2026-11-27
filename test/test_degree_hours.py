# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Tests for sub-daily heating and cooling degree days (degree hours)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from atlite.convert import (
    convert_and_aggregate,
    convert_cooling_demand,
    convert_heat_demand,
    cooling_demand,
    heat_demand,
)


class MockCutout:
    def __init__(self, data: xr.Dataset) -> None:
        self.data = data
        grid_coords = np.array([(x, y) for y in data.y.values for x in data.x.values])
        self.grid = pd.DataFrame(grid_coords, columns=["x", "y"])

    def convert_and_aggregate(self, **kwargs: Any) -> Any:
        return convert_and_aggregate(self, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def temperature() -> xr.Dataset:
    # two days of hourly temperature (in K) on a 1 x 2 grid: a daily cycle
    # between 10 °C and 30 °C in the first cell, constant 20 °C in the second
    times = pd.date_range("2020-07-01", periods=48, freq="h")
    cycle = 20 + 10 * np.sin(2 * np.pi * (np.arange(48) - 9) / 24)
    values = np.stack([cycle, np.full(48, 20.0)], axis=-1)[:, None, :] + 273.15
    return xr.Dataset({
        "temperature": xr.DataArray(
            values,
            dims=["time", "y", "x"],
            coords={"time": times, "y": [50.0], "x": [5.0, 6.0]},
        )
    })


def celsius(ds: xr.Dataset) -> xr.DataArray:
    return ds["temperature"] - 273.15


def test_default_is_daily(temperature: xr.Dataset) -> None:
    kwargs: dict[str, Any] = {
        "threshold": 15.0,
        "a": 1.0,
        "constant": 0.0,
        "hour_shift": 0.0,
    }
    default = convert_heat_demand(temperature, **kwargs)
    explicit = convert_heat_demand(temperature, freq="1D", **kwargs)
    xr.testing.assert_identical(default, explicit)
    assert default.sizes["time"] == 2


def test_cooling_degree_hours(temperature: xr.Dataset) -> None:
    cdh = convert_cooling_demand(
        temperature, threshold=23.0, a=1.0, constant=0.0, hour_shift=0.0, freq=None
    )
    expected = (celsius(temperature) - 23.0).clip(min=0.0)
    np.testing.assert_allclose(cdh.values, expected.values)
    assert cdh.sizes["time"] == 48


def test_heat_degree_hours(temperature: xr.Dataset) -> None:
    hdh = convert_heat_demand(
        temperature, threshold=15.0, a=2.0, constant=1.0, hour_shift=3.0, freq=None
    )
    expected = 2.0 * (15.0 - celsius(temperature)).clip(min=0.0) + 1.0
    np.testing.assert_allclose(hdh.values, expected.values)
    # hour_shift has no effect without averaging
    xr.testing.assert_equal(hdh.time, temperature.time)


def test_intermediate_frequency(temperature: xr.Dataset) -> None:
    cdd = convert_cooling_demand(
        temperature, threshold=23.0, a=1.0, constant=0.0, hour_shift=0.0, freq="3h"
    )
    expected = (celsius(temperature).resample(time="3h").mean() - 23.0).clip(min=0)
    assert cdd.sizes["time"] == 16
    np.testing.assert_allclose(cdd.values, expected.values)


def test_hourly_captures_warm_afternoons(temperature: xr.Dataset) -> None:
    # the daily mean of the first cell is 20 °C, below the threshold of 23 °C
    kwargs: dict[str, Any] = {
        "threshold": 23.0,
        "a": 1.0,
        "constant": 0.0,
        "hour_shift": 0.0,
    }
    daily = convert_cooling_demand(temperature, **kwargs)
    hourly = convert_cooling_demand(temperature, freq=None, **kwargs)
    np.testing.assert_allclose(daily.isel(x=0).values, 0.0, atol=1e-9)
    assert float(hourly.isel(x=0).sum()) > 0
    assert float(hourly.isel(x=1).sum()) == 0


@pytest.mark.parametrize("func", [heat_demand, cooling_demand])
def test_aggregation_with_freq(temperature: xr.Dataset, func: Any) -> None:
    cutout = MockCutout(temperature)
    index = pd.Index(["region"], name="region")
    result = func(
        cutout,
        freq=None,
        matrix=np.array([[1.0, 1.0]]),
        index=index,
        aggregate_time=None,
    )
    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"time", "region"}
    assert result.sizes["time"] == 48


@pytest.mark.parametrize(("freq", "expected_steps"), [("W", 1), ("MS", 1)])
def test_coarser_than_daily(
    temperature: xr.Dataset, freq: str, expected_steps: int
) -> None:
    hdd = convert_heat_demand(
        temperature, threshold=25.0, a=1.0, constant=0.0, hour_shift=0.0, freq=freq
    )
    assert hdd.sizes["time"] == expected_steps
    expected = (25.0 - celsius(temperature).mean("time")).clip(min=0.0)
    np.testing.assert_allclose(hdd.isel(time=0).values, expected.values)


@pytest.mark.parametrize("freq", ["30min", "1min"])
@pytest.mark.parametrize("func", [convert_heat_demand, convert_cooling_demand])
def test_freq_finer_than_resolution_raises(
    temperature: xr.Dataset, func: Any, freq: str
) -> None:
    with pytest.raises(ValueError, match="finer than the temporal resolution"):
        func(
            temperature,
            threshold=15.0,
            a=1.0,
            constant=0.0,
            hour_shift=0.0,
            freq=freq,
        )
