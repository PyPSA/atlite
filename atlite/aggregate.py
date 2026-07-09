# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Functions for aggregating results."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dask
import xarray as xr

from atlite.utils import ensure_coords

if TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import spmatrix


def aggregate_matrix(
    da: xr.DataArray,
    matrix: spmatrix,
    index: xr.Coordinates | pd.Index,
) -> xr.DataArray:
    """
    Aggregate spatial data with a sparse matrix.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with spatial dimensions ``y`` and ``x``.
    matrix : scipy.sparse.spmatrix
        Aggregation matrix mapping flattened spatial cells to ``index``.
    index : xarray.Coordinates | pandas.Index
        Index defining the aggregated dimension.

    Returns
    -------
    xarray.DataArray
        Aggregated data indexed by ``index`` and, if present, time.
    """
    coords = ensure_coords(index)

    if isinstance(da.data, dask.array.core.Array):
        da = da.stack(spatial=("y", "x"))
        da = da.chunk({"spatial": -1})
        result = xr.apply_ufunc(
            lambda da: da * matrix.T,
            da,
            input_core_dims=[["spatial"]],
            output_core_dims=[list(coords.dims)],
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={"output_sizes": coords.sizes},
        ).assign_coords(coords)
        return cast("xr.DataArray", result)
    da = da.stack(spatial=("y", "x")).transpose("spatial", "time")
    return xr.DataArray(matrix * da, [index, da.coords["time"]])
