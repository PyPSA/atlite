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
