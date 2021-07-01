# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Functions for aggregating results.
"""

import xarray as xr
import dask


def aggregate_matrix(da, matrix, index):
    if index.name is None:
        index = index.rename("dim_0")
    if isinstance(da.data, dask.array.core.Array):
        da = da.stack(spatial=("y", "x"))
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
