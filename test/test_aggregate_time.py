# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
import pytest
import xarray as xr

from atlite.convert import convert_and_aggregate


class MockCutout:
    def __init__(self, data):
        self.data = data
        grid_coords = np.array([(x, y) for y in data.y.values for x in data.x.values])
        import pandas as pd

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
    def test_aggregate_time_false_returns_timeseries(self, cutout):
        result = convert_and_aggregate(cutout, identity_convert, aggregate_time=False)
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

    def test_default_no_spatial_aggregates_over_time(self, cutout):
        result = convert_and_aggregate(cutout, identity_convert)
        expected = cutout.data["var"].sum("time")
        assert "time" not in result.dims
        xr.testing.assert_identical(result, expected)


class TestAggregateTimeWithSpatial:
    def test_aggregate_time_mean_with_layout(self, cutout):
        layout = xr.DataArray(
            np.ones((3, 4)),
            dims=["y", "x"],
            coords={"y": cutout.data.y, "x": cutout.data.x},
        )
        result_ts = convert_and_aggregate(
            cutout,
            identity_convert,
            layout=layout,
            aggregate_time=False,
        )
        result_mean = convert_and_aggregate(
            cutout,
            identity_convert,
            layout=layout,
            aggregate_time="mean",
        )
        assert "time" in result_ts.dims
        assert "time" not in result_mean.dims
        np.testing.assert_allclose(result_mean.values, result_ts.mean("time").values)

    def test_aggregate_time_sum_with_layout(self, cutout):
        layout = xr.DataArray(
            np.ones((3, 4)),
            dims=["y", "x"],
            coords={"y": cutout.data.y, "x": cutout.data.x},
        )
        result_ts = convert_and_aggregate(
            cutout,
            identity_convert,
            layout=layout,
            aggregate_time=False,
        )
        result_sum = convert_and_aggregate(
            cutout,
            identity_convert,
            layout=layout,
            aggregate_time="sum",
        )
        assert "time" not in result_sum.dims
        np.testing.assert_allclose(result_sum.values, result_ts.sum("time").values)

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
            aggregate_time=False,
        )
        np.testing.assert_allclose(result_pu.values, result_pu_ts.mean("time").values)


class TestDeprecatedParams:
    def test_capacity_factor_warns(self, cutout):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_and_aggregate(
                cutout, identity_convert, capacity_factor=True
            )
            assert any(
                "capacity_factor is deprecated" in str(warning.message) for warning in w
            )
        assert "time" not in result.dims

    def test_capacity_factor_timeseries_warns(self, cutout):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_and_aggregate(
                cutout, identity_convert, capacity_factor_timeseries=True
            )
            assert any(
                "capacity_factor_timeseries is deprecated" in str(warning.message)
                for warning in w
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

    def test_aggregate_time_true_raises(self, cutout):
        with pytest.raises(ValueError, match="aggregate_time must be"):
            convert_and_aggregate(cutout, identity_convert, aggregate_time=True)
