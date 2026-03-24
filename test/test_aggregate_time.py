# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from atlite.convert import convert_and_aggregate


class MockCutout:
    def __init__(self, data):
        self.data = data
        grid_coords = np.array([(x, y) for y in data.y.values for x in data.x.values])
        self.grid = pd.DataFrame(grid_coords, columns=["x", "y"])


def identity_convert(ds, **kwargs):
    return ds["var"]


@pytest.fixture
def cutout():
    np.random.seed(42)
    times = xr.date_range("2020-01-01", periods=24, freq="h")
    data = xr.Dataset(
        {
            "var": xr.DataArray(
                np.random.rand(24, 3, 4),
                dims=["time", "y", "x"],
                coords={
                    "time": times,
                    "y": [50.0, 51.0, 52.0],
                    "x": [5.0, 6.0, 7.0, 8.0],
                },
            )
        }
    )
    return MockCutout(data)


class TestAggregateTimeNoSpatial:
    def test_aggregate_time_none_returns_timeseries(self, cutout):
        result = convert_and_aggregate(cutout, identity_convert, aggregate_time=None)
        assert "time" in result.dims

    def test_aggregate_time_mean(self, cutout):
        result = convert_and_aggregate(cutout, identity_convert, aggregate_time="mean")
        assert "time" not in result.dims
        expected = cutout.data["var"].mean("time")
        np.testing.assert_allclose(result.values, expected.values)

    def test_aggregate_time_sum(self, cutout):
        result = convert_and_aggregate(cutout, identity_convert, aggregate_time="sum")
        assert "time" not in result.dims
        expected = cutout.data["var"].sum("time")
        np.testing.assert_allclose(result.values, expected.values)

    def test_legacy_default_no_spatial_sums_over_time(self, cutout):
        with pytest.warns(FutureWarning, match="aggregate_time='legacy'"):
            result = convert_and_aggregate(cutout, identity_convert)
        expected = cutout.data["var"].sum("time")
        assert "time" not in result.dims
        xr.testing.assert_identical(result, expected)


@pytest.fixture
def layout(cutout):
    return xr.DataArray(
        np.ones((3, 4)),
        dims=["y", "x"],
        coords={"y": cutout.data.y, "x": cutout.data.x},
    )


@pytest.fixture
def result_ts(cutout, layout):
    return convert_and_aggregate(
        cutout, identity_convert, layout=layout, aggregate_time=None
    )


class TestAggregateTimeWithSpatial:
    def test_aggregate_time_mean_with_layout(self, cutout, layout, result_ts):
        result_mean = convert_and_aggregate(
            cutout, identity_convert, layout=layout, aggregate_time="mean"
        )
        assert "time" in result_ts.dims
        assert "time" not in result_mean.dims
        np.testing.assert_allclose(result_mean.values, result_ts.mean("time").values)

    def test_aggregate_time_sum_with_layout(self, cutout, layout, result_ts):
        result_sum = convert_and_aggregate(
            cutout, identity_convert, layout=layout, aggregate_time="sum"
        )
        assert "time" not in result_sum.dims
        np.testing.assert_allclose(result_sum.values, result_ts.sum("time").values)

    def test_legacy_default_with_layout_returns_timeseries(self, cutout, layout):
        with pytest.warns(FutureWarning, match="aggregate_time='legacy'"):
            result = convert_and_aggregate(cutout, identity_convert, layout=layout)
        assert "time" in result.dims

    def test_aggregate_time_with_per_unit(self, cutout):
        layout = xr.DataArray(
            np.ones((3, 4)) * 2.0,
            dims=["y", "x"],
            coords={"y": cutout.data.y, "x": cutout.data.x},
        )
        result_pu = convert_and_aggregate(
            cutout,
            identity_convert,
            layout=layout,
            per_unit=True,
            aggregate_time="mean",
        )
        assert "time" not in result_pu.dims

        result_pu_ts = convert_and_aggregate(
            cutout,
            identity_convert,
            layout=layout,
            per_unit=True,
            aggregate_time=None,
        )
        np.testing.assert_allclose(result_pu.values, result_pu_ts.mean("time").values)


class TestDeprecatedParams:
    def test_capacity_factor_warns(self, cutout):
        with pytest.warns(FutureWarning, match="capacity_factor is deprecated"):
            result = convert_and_aggregate(
                cutout, identity_convert, capacity_factor=True
            )
        assert "time" not in result.dims

    def test_capacity_factor_timeseries_warns(self, cutout):
        with pytest.warns(
            FutureWarning, match="capacity_factor_timeseries is deprecated"
        ):
            result = convert_and_aggregate(
                cutout, identity_convert, capacity_factor_timeseries=True
            )
        assert "time" in result.dims

    def test_capacity_factor_with_aggregate_time_raises(self, cutout):
        with pytest.raises(ValueError, match="Cannot use"):
            convert_and_aggregate(
                cutout,
                identity_convert,
                capacity_factor=True,
                aggregate_time="mean",
            )


class TestInvalidArgs:
    def test_invalid_aggregate_time_value(self, cutout):
        with pytest.raises(ValueError, match="aggregate_time must be"):
            convert_and_aggregate(cutout, identity_convert, aggregate_time="invalid")

    def test_aggregate_time_false_raises(self, cutout):
        with pytest.raises(ValueError, match="aggregate_time must be"):
            convert_and_aggregate(cutout, identity_convert, aggregate_time=False)

    def test_aggregate_time_true_raises(self, cutout):
        with pytest.raises(ValueError, match="aggregate_time must be"):
            convert_and_aggregate(cutout, identity_convert, aggregate_time=True)
