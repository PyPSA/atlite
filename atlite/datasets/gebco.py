#!/usr/bin/env python3

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Module for loading gebco data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import rasterio as rio
import xarray as xr
from pandas import to_numeric
from rasterio.warp import Resampling

if TYPE_CHECKING:
    from atlite._types import PathLike

logger = logging.getLogger(__name__)

crs = 4326
features = {"height": ["height"]}


def get_data_gebco_height(
    xs: xr.DataArray, ys: xr.DataArray, gebco_path: PathLike
) -> xr.DataArray:
    """Read and resample GEBCO bathymetry/elevation to the target grid.

    Parameters
    ----------
    xs : xr.DataArray
        Target x (longitude) coordinates.
    ys : xr.DataArray
        Target y (latitude) coordinates.
    gebco_path : PathLike
        Path to the GEBCO GeoTIFF or NetCDF file.

    Returns
    -------
    xr.DataArray
        Height data resampled to the target grid with dimensions ``(y, x)``.
    """
    x, X = xs.data[[0, -1]]
    y, Y = ys.data[[0, -1]]

    dx = (X - x) / (len(xs) - 1)
    dy = (Y - y) / (len(ys) - 1)

    with rio.open(gebco_path) as dataset:
        window = dataset.window(x - dx / 2, y - dy / 2, X + dx / 2, Y + dy / 2)
        gebco = dataset.read(
            indexes=1,
            window=window,
            out_shape=(len(ys), len(xs)),
            resampling=Resampling.average,
        )
        gebco = gebco[::-1]  # change inversed y-axis
        tags = dataset.tags(bidx=1)
        tags = {k: to_numeric(v, errors="ignore") for k, v in tags.items()}

    return xr.DataArray(
        gebco, coords=[("y", ys.data), ("x", xs.data)], name="height", attrs=tags
    )


def get_data(
    cutout: Any,
    feature: str,
    tmpdir: PathLike,
    monthly_requests: bool = False,
    concurrent_requests: bool = False,
    **creation_parameters: Any,
) -> xr.Dataset:
    """Retrieve GEBCO height data for a cutout.

    Parameters
    ----------
    cutout : Any
        Cutout instance providing target coordinates.
    feature : str
        Feature name (expected ``"height"``).
    tmpdir : PathLike
        Temporary directory (unused, kept for interface consistency).
    monthly_requests : bool, optional
        Unused, kept for interface consistency.
    concurrent_requests : bool, optional
        Unused, kept for interface consistency.
    **creation_parameters : Any
        Must include ``"gebco_path"`` pointing to the GEBCO data file.

    Returns
    -------
    xr.Dataset
        Dataset with height variable on the cutout grid.
    """
    if "gebco_path" not in creation_parameters:
        logger.error('Argument "gebco_path" not defined')
    path = creation_parameters["gebco_path"]

    coords = cutout.coords
    return (
        get_data_gebco_height(coords["x"], coords["y"], path)
        .to_dataset()
        .assign_coords(cutout.coords)
    )
