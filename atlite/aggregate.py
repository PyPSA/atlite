# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Functions for aggregating results.
"""

import dask
import xarray as xr


def aggregate_matrix(da, matrix, index):
    if index.name is None:
        index = index.rename("dim_0")
    if isinstance(da.data, dask.array.core.Array):
        da = da.stack(spatial=("y", "x"))
        da = da.chunk(dict(spatial=-1))
        return xr.apply_ufunc(
            lambda da: da * matrix.T,
            da,
            input_core_dims=[["spatial"]],
            output_core_dims=[[index.name]],
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs=dict(output_sizes={index.name: index.size}),
        ).assign_coords(**{index.name: index})
    else:
        da = da.stack(spatial=("y", "x")).transpose("spatial", "time")
        return xr.DataArray(matrix * da, [index, da.coords["time"]])


def aggregate_gridcells(da, matrix, index, freq=None, agg="sum"):
    """
    Resample and aggregate the data array `da` based on the specified frequency 
    and aggregation method using a matrix for spatial aggregation.

    Parameters:
    -----------
    da : xarray.DataArray
        The data array to process, typically containing time-series data 
        with spatial dimensions ("y", "x").
    matrix : sparse matrix or equivalent
        Sparse matrix used to map spatial grid cells to aggregated regions 
        (e.g., buses or zones).
    index : pandas.Index or xarray.Index
        Index corresponding to the aggregated regions (e.g., bus IDs).
    freq : str, optional
        Resampling frequency string (e.g., 'D' for daily, 'M' for monthly). 
        If None, no resampling is applied.
    agg : str, optional
        Aggregation method to apply. Options are "mean" or "sum" 
        (default is "sum").

    Returns:
    --------
    da : xarray.DataArray
        The aggregated data array after spatial and temporal processing.
    layout : xarray.DataArray
        A DataArray representing the layout or weight of each grid cell in the 
        aggregation process, aligned with the output dimensions.
    """
    # Ensure the `index` has a name; if not, assign a default name.
    if index.name is None:
        index = index.rename("dim_0")
    
    # Check if the input data uses Dask for lazy evaluation.
    if isinstance(da.data, dask.array.core.Array):
        # Stack the spatial dimensions ("y", "x") into a single "spatial" dimension for efficient processing.
        da = da.stack(spatial=("y", "x"))
        da = da.chunk({"spatial": -1})  # Optimize chunking along the "spatial" dimension.

        # Convert the sparse matrix into a dense DataArray for compatibility with xarray operations.
        layout = xr.DataArray(
            matrix.toarray(),
            dims=(index.name, "spatial"),
            coords={index.name: index, "spatial": da["spatial"]},
        )
        
        # Transpose the layout so that the "spatial" dimension comes first for alignment with `da`.
        layout = layout.transpose("spatial", index.name)
        
        # Expand `da` to include the `index.name` dimension for matrix multiplication.
        da = da.expand_dims({index.name: layout[index.name]})
        
        # Perform element-wise multiplication of `da` and `layout` for spatial aggregation.
        da = da * layout
    else:
        # For non-Dask arrays, follow a similar process without lazy evaluation.
        da = da.stack(spatial=("y", "x")).transpose("spatial", "time")
        
        layout = xr.DataArray(
            matrix.toarray(),
            dims=(index.name, "spatial"),
            coords={index.name: index, "spatial": da["spatial"]},
        )
        layout = layout.transpose("spatial", index.name)
        
        # Element-wise multiplication for spatial aggregation.
        da = xr.DataArray(da * layout)

    # Unstack the "spatial" dimension back into the original "y" and "x" dimensions.
    da = da.unstack("spatial")
    layout = layout.unstack("spatial")
    
    # Resample the data if a frequency is provided.
    if freq is not None:
        da = da.resample(time=freq)  # Resample along the "time" dimension.

        # Apply the specified aggregation method during resampling.
        if agg == "mean":
            da = da.mean("time", keep_attrs=True)
        elif agg == "sum":
            da = da.sum("time", keep_attrs=True)
        else:
            # Raise an error if an invalid aggregation method is specified.
            raise ValueError(
                f"Invalid aggregation method '{agg}' for frequency '{freq}'. "
                "Use 'mean' or 'sum' instead."
            )
    else:
        # If no frequency is provided, apply aggregation directly over time.
        if agg == "mean":
            da = da.mean("time", keep_attrs=True)
        elif agg == "sum":
            da = da.sum("time", keep_attrs=True)
        # If `agg` is None or unsupported, leave the data unmodified.

    return da, layout