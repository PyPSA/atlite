# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Functions for aggregating and resolving spatial/temporal results."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix

from atlite.gis import spdiag

if TYPE_CHECKING:
    import dask.array

    from scipy.sparse import spmatrix

    from atlite._types import DataArray
    from atlite.cutout import Cutout


def ensure_index_name(index: pd.Index) -> pd.Index:
    """Return *index* with name ``"dim_0"`` when it has no name."""
    if index.name is None:
        return index.rename("dim_0")
    return index


def resolve_matrix(
    cutout: Cutout,
    matrix: Any,
    index: Any,
    shapes: Any,
    shapes_crs: int,
    layout: Any,
) -> tuple[csr_matrix | None, pd.Index | None]:
    """Resolve *matrix*, *shapes* and *layout* into a sparse matrix and index.

    Validates the inputs, builds an indicator matrix from *shapes* when
    needed, and folds *layout* capacities into the matrix.  Returns
    ``(None, None)`` when no spatial aggregation is requested.
    """
    if matrix is not None:
        if shapes is not None:
            raise ValueError(
                "Passing matrix and shapes is ambiguous. Pass only one of them."
            )

        if isinstance(matrix, xr.DataArray):
            coords = matrix.indexes.get(matrix.dims[1]).to_frame(index=False)
            if not np.array_equal(coords[["x", "y"]], cutout.grid[["x", "y"]]):
                raise ValueError(
                    "Matrix spatial coordinates not aligned with cutout spatial "
                    "coordinates."
                )
            if index is None:
                index = matrix

        if not matrix.ndim == 2:
            raise ValueError("Matrix not 2-dimensional.")

        matrix = csr_matrix(matrix)

    if shapes is not None:
        geoseries_like = (pd.Series, gpd.GeoDataFrame, gpd.GeoSeries)
        if isinstance(shapes, geoseries_like) and index is None:
            index = shapes.index
        matrix = cutout.indicatormatrix(shapes, shapes_crs)

    if layout is not None:
        assert isinstance(layout, xr.DataArray)
        layout = layout.reindex_like(cutout.data).stack(spatial=["y", "x"])
        if matrix is None:
            matrix = csr_matrix(layout.expand_dims("new"))
        else:
            matrix = csr_matrix(matrix) * spdiag(layout)

    if matrix is not None and index is None:
        index = pd.RangeIndex(matrix.shape[0])

    return matrix, index


def normalize_aggregate_time(
    aggregate_time: Literal["sum", "mean", "legacy"] | None,
    no_spatial: bool,
    capacity_factor: bool = False,
    capacity_factor_timeseries: bool = False,
) -> Literal["sum", "mean"] | None:
    """Normalise the *aggregate_time* parameter to ``"sum"``, ``"mean"`` or ``None``.

    Handles the deprecated ``"legacy"`` value and the deprecated
    *capacity_factor* / *capacity_factor_timeseries* flags, emitting
    :class:`FutureWarning` where appropriate.
    """
    if aggregate_time not in ("sum", "mean", "legacy", None):
        raise ValueError(
            f"aggregate_time must be 'sum', 'mean', 'legacy', or None, "
            f"got {aggregate_time!r}"
        )

    if aggregate_time == "legacy":
        warnings.warn(
            "aggregate_time='legacy' is deprecated and will be removed in a "
            "future release. Pass 'sum', 'mean', or None explicitly.",
            FutureWarning,
            stacklevel=3,
        )

    if capacity_factor or capacity_factor_timeseries:
        if aggregate_time != "legacy":
            raise ValueError(
                "Cannot use 'aggregate_time' together with deprecated "
                "'capacity_factor' or 'capacity_factor_timeseries'."
            )
        if capacity_factor:
            warnings.warn(
                "capacity_factor is deprecated. Use aggregate_time='mean' instead.",
                FutureWarning,
                stacklevel=3,
            )
            aggregate_time = "mean"
        if capacity_factor_timeseries:
            warnings.warn(
                "capacity_factor_timeseries is deprecated. "
                "Use aggregate_time=None instead.",
                FutureWarning,
                stacklevel=3,
            )
            aggregate_time = None

    if aggregate_time == "legacy":
        return "sum" if no_spatial else None
    return aggregate_time


def reduce_time(
    da: xr.DataArray, method: Literal["sum", "mean"] | None
) -> xr.DataArray:
    """Reduce *da* along the ``time`` dimension using *method*.

    Returns *da* unchanged when *method* is ``None``.
    """
    if method == "sum":
        return da.sum("time", keep_attrs=True)
    if method == "mean":
        return da.mean("time", keep_attrs=True)
    return da


def build_capacity_array(matrix: Any, index: pd.Index) -> xr.DataArray:
    """Sum *matrix* columns to obtain the installed capacity per bus."""
    capacity = xr.DataArray(np.asarray(matrix.sum(-1)).flatten(), [index])
    capacity.attrs["units"] = "MW"
    return capacity


def wrap_matrix_result(
    data: np.ndarray,
    time: xr.DataArray,
    index: pd.Index,
) -> DataArray:
    """Wrap a ``(time, n_regions)`` numpy array into a labelled DataArray."""
    index = ensure_index_name(index)
    return xr.DataArray(
        data,
        dims=("time", index.name),
        coords={"time": time, index.name: index},
    )


def finalize_aggregated_result(
    result: xr.DataArray,
    matrix: Any,
    index: pd.Index,
    per_unit: bool,
    return_capacity: bool,
    aggregate_time_method: Literal["sum", "mean"] | None,
) -> DataArray | tuple[DataArray, DataArray]:
    """Apply per-unit normalisation, time aggregation and capacity extraction.

    Returns either the finalised DataArray or a ``(result, capacity)`` tuple
    when *return_capacity* is ``True``.
    """
    capacity = None
    if per_unit or return_capacity:
        capacity = build_capacity_array(matrix, index)

    if per_unit:
        result = (result / capacity.where(capacity != 0)).fillna(0.0)
        result.attrs["units"] = "p.u."
    else:
        result.attrs["units"] = "MW"

    result = reduce_time(result, aggregate_time_method)

    if return_capacity:
        return result, capacity
    return result


def aggregate_matrix(
    da: DataArray,
    matrix: spmatrix,
    index: pd.Index,
) -> DataArray:
    """Aggregate spatial data with a sparse *matrix*.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with spatial dimensions ``y`` and ``x``.
    matrix : scipy.sparse.spmatrix
        Aggregation matrix mapping flattened spatial cells to *index*.
    index : pandas.Index
        Index defining the aggregated dimension.

    Returns
    -------
    xarray.DataArray
        Aggregated data indexed by *index* and, if present, time.
    """
    import dask as _dask

    index = ensure_index_name(index)
    if isinstance(da.data, _dask.array.core.Array):
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
