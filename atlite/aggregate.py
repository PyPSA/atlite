# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Functions for aggregating results."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dask
import xarray as xr

if TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import spmatrix

    from atlite._types import DataArray


def aggregate_matrix(
    da: DataArray,
    matrix: spmatrix,
    index: pd.Index,
) -> DataArray:
    """
    Aggregate spatial data with a sparse matrix.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with spatial dimensions ``y`` and ``x``.
    matrix : scipy.sparse.spmatrix
        Aggregation matrix mapping flattened spatial cells to ``index``.
    index : pandas.Index
        Index defining the aggregated dimension.

    Returns
    -------
    xarray.DataArray
        Aggregated data indexed by ``index`` and, if present, time.
    """
    if index.name is None:
        index = index.rename("dim_0")
    if isinstance(da.data, dask.array.core.Array):
        da = da.stack(spatial=("y", "x"))
        da = da.chunk({"spatial": -1})
        result = xr.apply_ufunc(
            lambda da: da * matrix.T,
            da,
            input_core_dims=[["spatial"]],
            output_core_dims=[[index.name]],
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={"output_sizes": {index.name: index.size}},
        ).assign_coords(**{index.name: index})
        return cast("DataArray", result)
    da = da.stack(spatial=("y", "x")).transpose("spatial", "time")
    return xr.DataArray(matrix * da, [index, da.coords["time"]])
