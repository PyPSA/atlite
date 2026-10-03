# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Tests for spatial aggregation with a sparse matrix."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xarray as xr

from atlite.aggregate import aggregate_matrix
from atlite.utils import ensure_coords

BUSES = pd.Index(["a", "b", "c"], name="bus")
REGIONS = pd.MultiIndex.from_tuples(
    [("DE", 1), ("DE", 2), ("FR", 1)], names=["country", "n"]
)


@pytest.fixture
def da():
    rng = np.random.default_rng(42)
    return xr.DataArray(
        rng.random((5, 3, 4)),
        dims=["time", "y", "x"],
        coords={
            "time": pd.date_range("2020-01-01", periods=5, freq="h"),
            "y": [50.0, 51.0, 52.0],
            "x": [5.0, 6.0, 7.0, 8.0],
        },
    )


@pytest.mark.parametrize("chunked", [False, True], ids=["numpy", "dask"])
@pytest.mark.parametrize(
    "index",
    [BUSES, REGIONS, ensure_coords(REGIONS)],
    ids=["index", "multiindex", "coords"],
)
def test_aggregate_matrix_keeps_index(da, index, chunked):
    matrix = sp.csr_matrix(np.random.default_rng(0).random((3, 12)))
    data = da.chunk({"time": 2}) if chunked else da

    result = aggregate_matrix(data, matrix, index).transpose(..., "time")

    expected = matrix @ da.stack(spatial=("y", "x")).transpose("spatial", "time").values
    np.testing.assert_allclose(result.values, expected)
    xr.testing.assert_identical(result.coords["time"], da.coords["time"])
    dim = result.dims[0]
    assert result.indexes[dim].equals(index if isinstance(index, pd.Index) else REGIONS)
