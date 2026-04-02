# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Streaming conversion backend with chunk-aligned I/O.

Processes weather-to-energy conversions one time-chunk at a time so that
the full ``(time, y, x)`` grid is never materialised in memory.  Only
cases that actually benefit from streaming (matrix aggregation or
temporal reduction) are handled; all other cases fall back to the
dask-backed path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from atlite.aggregate import (
    finalize_aggregated_result,
    reduce_time,
    wrap_matrix_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from scipy.sparse import spmatrix

    from atlite._types import DataArray
    from atlite.cutout import Cutout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamSpec:
    """Spatial / temporal metadata needed by the streaming loop."""

    time: xr.DataArray
    y: xr.DataArray
    x: xr.DataArray
    n_spatial: int
    out_ny: int
    out_nx: int
    units: str

    @property
    def n_time(self) -> int:
        return len(self.time)


def optimal_chunk_size(ds: xr.Dataset) -> int:
    """Derive the time-chunk size from the on-disk encoding of *ds*.

    Falls back to 2190 (≈ one quarter of hourly year data) when no
    ``chunksizes`` metadata is found.
    """
    time_sizes: set[int] = set()
    for var in ds.data_vars:
        chunksizes = ds[var].encoding.get("chunksizes")
        if chunksizes and len(chunksizes) > 0:
            time_sizes.add(chunksizes[0])
    return max(time_sizes) if time_sizes else 2190


def can_stream(cutout: Cutout) -> bool:
    """Return ``True`` when *cutout* is backed by an on-disk NetCDF file."""
    source = cutout.data.encoding.get("source")
    return source is not None and cutout.path.is_file()


def has_streaming_benefit(
    matrix: spmatrix | None,
    aggregate_time_method: Literal["sum", "mean"] | None,
) -> bool:
    """Return ``True`` when streaming can reduce peak memory over dask.

    Streaming helps when either a sparse-matrix aggregation collapses the
    spatial grid or a temporal reduction (sum/mean) allows an accumulator
    instead of a full output buffer.
    """
    return matrix is not None or aggregate_time_method in ("sum", "mean")


def validate_output(
    da_chunk: xr.DataArray,
    ds_chunk: xr.Dataset,
    convert_func: Callable[..., Any],
) -> bool:
    """Check that *convert_func* produced streamable output.

    Returns ``False`` (with a debug log message) when the output is
    missing a ``time`` dimension, has a different time length than the
    input chunk, or contains unexpected extra dimensions.
    """
    if "time" not in da_chunk.dims:
        logger.debug(
            "Streaming aborted: %s output has no 'time' dimension.",
            convert_func.__name__,
        )
        return False

    if len(da_chunk["time"]) != len(ds_chunk["time"]):
        logger.debug(
            "Streaming aborted: %s changed time dimension (%d → %d).",
            convert_func.__name__,
            len(ds_chunk["time"]),
            len(da_chunk["time"]),
        )
        return False

    extra_dims = set(da_chunk.dims) - {"time", "y", "x"}
    if extra_dims:
        logger.debug(
            "Streaming aborted: %s has unexpected dims %s.",
            convert_func.__name__,
            extra_dims,
        )
        return False

    return True


def build_stream_spec(
    cutout: Cutout,
    da_chunk: xr.DataArray,
) -> StreamSpec:
    """Build a :class:`StreamSpec` from the cutout and a sample converted chunk."""
    y = cutout.data["y"]
    x = cutout.data["x"]
    out_ny = da_chunk.sizes.get("y", len(y))
    out_nx = da_chunk.sizes.get("x", len(x))
    return StreamSpec(
        time=cutout.data["time"],
        y=y,
        x=x,
        n_spatial=out_ny * out_nx,
        out_ny=out_ny,
        out_nx=out_nx,
        units=da_chunk.attrs.get("units", ""),
    )


def init_buffers(
    spec: StreamSpec,
    matrix: spmatrix | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Allocate the output buffer(s) for the streaming loop.

    Returns ``(result_data, accum)`` where exactly one is non-``None``:
    *result_data* for matrix aggregation, *accum* for temporal reduction.
    """
    if matrix is not None:
        return np.empty((spec.n_time, matrix.shape[0]), dtype=np.float64), None
    return None, np.zeros((spec.out_ny, spec.out_nx), dtype=np.float64)


def finalize_matrix(
    spec: StreamSpec,
    result_data: np.ndarray,
    matrix: spmatrix,
    index: pd.Index,
    per_unit: bool,
    return_capacity: bool,
    aggregate_time_method: Literal["sum", "mean"] | None,
) -> DataArray | tuple[DataArray, DataArray]:
    """Wrap the matrix-aggregated buffer and apply per-unit / time aggregation."""
    result = wrap_matrix_result(result_data, spec.time, index)
    return finalize_aggregated_result(
        result, matrix, index, per_unit, return_capacity, aggregate_time_method
    )


def finalize_grid(
    spec: StreamSpec,
    accum: np.ndarray,
    aggregate_time_method: Literal["sum", "mean"] | None,
) -> DataArray:
    """Build a spatially-indexed DataArray from the temporal accumulator."""
    coords = {"y": spec.y, "x": spec.x}
    values = accum if aggregate_time_method == "sum" else accum / spec.n_time
    result = xr.DataArray(values, dims=("y", "x"), coords=coords)
    if spec.units:
        result.attrs["units"] = spec.units
    return result


def stream_conversion(
    cutout: Cutout,
    convert_func: Callable[..., Any],
    matrix: spmatrix | None,
    index: pd.Index | None,
    per_unit: bool,
    return_capacity: bool,
    aggregate_time: Literal["sum", "mean"] | None,
    show_progress: bool,
    convert_kwds: dict[str, Any],
) -> DataArray | tuple[DataArray, DataArray] | None:
    """Execute *convert_func* on *cutout* one time-chunk at a time.

    Returns ``None`` when streaming offers no memory benefit for the
    given arguments or when the conversion output is not streamable, so
    the caller can fall back to the dask-backed path.
    """
    if not has_streaming_benefit(matrix, aggregate_time):
        return None

    source = cutout.data.encoding.get("source")
    ds_eager = xr.open_dataset(source, chunks=None)
    try:
        return stream_inner(
            ds_eager,
            cutout,
            convert_func,
            matrix,
            index,
            per_unit,
            return_capacity,
            aggregate_time,
            show_progress,
            convert_kwds,
        )
    finally:
        ds_eager.close()


def stream_inner(
    ds_eager: xr.Dataset,
    cutout: Cutout,
    convert_func: Callable[..., Any],
    matrix: spmatrix | None,
    index: pd.Index | None,
    per_unit: bool,
    return_capacity: bool,
    aggregate_time_method: Literal["sum", "mean"] | None,
    show_progress: bool,
    convert_kwds: dict[str, Any],
) -> DataArray | tuple[DataArray, DataArray] | None:
    """Core streaming loop over time-chunks of *ds_eager*.

    Reads one storage-aligned chunk at a time, runs *convert_func*
    eagerly, and either multiplies by the sparse *matrix* or accumulates
    into a temporal reducer.  Returns ``None`` when the first chunk
    reveals that the conversion is not streamable.
    """
    chunk_size = optimal_chunk_size(ds_eager)
    time_idx = cutout.data["time"]
    n_time = len(time_idx)

    src_times = ds_eager["time"].values
    target_times = time_idx.values

    matrix_t = matrix.T.tocsc() if matrix is not None else None
    result_data = None
    accum = None
    spec: StreamSpec | None = None

    chunks_iter = list(range(0, n_time, chunk_size))
    n_chunks = len(chunks_iter)

    for i_chunk, start in enumerate(chunks_iter):
        end = min(start + chunk_size, n_time)
        chunk_times = target_times[start:end]
        mask = np.isin(src_times, chunk_times)
        idx = np.nonzero(mask)[0]
        sl_src = slice(int(idx[0]), int(idx[-1]) + 1)

        ds_chunk = ds_eager.isel(time=sl_src)
        da_chunk = convert_func(ds_chunk, **convert_kwds)

        if spec is None:
            if not validate_output(da_chunk, ds_chunk, convert_func):
                return None
            spec = build_stream_spec(cutout, da_chunk)
            result_data, accum = init_buffers(spec, matrix)

        chunk_values = da_chunk.values
        sl_out = slice(start, end)

        if matrix is not None:
            flat = chunk_values.reshape(chunk_values.shape[0], spec.n_spatial)
            result_data[sl_out] = flat @ matrix_t
        else:
            accum += chunk_values.sum(axis=0)

        if show_progress:
            logger.info(
                "Streaming %s: chunk %d/%d",
                convert_func.__name__,
                i_chunk + 1,
                n_chunks,
            )

    if matrix is not None:
        return finalize_matrix(
            spec, result_data, matrix, index,
            per_unit, return_capacity, aggregate_time_method,
        )
    return finalize_grid(spec, accum, aggregate_time_method)
