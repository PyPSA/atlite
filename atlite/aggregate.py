# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Functions for aggregating results.
"""

import scipy.sparse as sp
import xarray as xr
from dask.array.core import Array


def aggregate_matrix(
    da: xr.DataArray, matrix: sp.csr_matrix, coords: xr.Coordinates
) -> xr.DataArray:
    if isinstance(da.data, Array):
        da = da.stack(spatial=("y", "x"))
        da = da.chunk(dict(spatial=-1))
        return xr.apply_ufunc(
            lambda da: da * matrix.T,
            da,
            input_core_dims=[["spatial"]],
            output_core_dims=[list(coords.dims)],
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs=dict(output_sizes=coords.sizes),
        ).assign_coords(coords)
    else:
        da = da.stack(spatial=("y", "x")).transpose("spatial", "time")
        return xr.DataArray(matrix * da, coords.assign(time=da.coords["time"]))
