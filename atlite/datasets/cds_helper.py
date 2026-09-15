# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Shared helpers for datasets downloaded via the Climate Data Store (CDS).

Used by both the era5 and glofas dataset modules.
"""

from __future__ import annotations

import logging
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import xarray as xr

if TYPE_CHECKING:
    from atlite._types import PathLike

logger = logging.getLogger(__name__)


def _area(coords: dict[str, xr.DataArray]) -> list[float]:
    """
    Extract CDS API bounding box from coordinates.

    Parameters
    ----------
    coords : dict[str, xr.DataArray]
        Coordinate arrays with 'x' (longitude) and 'y' (latitude).

    Returns
    -------
    list[float]
        Bounding box as [north, west, south, east].
    """
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    return [y1, x0, y0, x1]


def noisy_unlink(path: PathLike) -> None:
    """
    Remove a file with debug logging, handling PermissionError gracefully.

    Parameters
    ----------
    path : PathLike
        Path to the file to delete.
    """
    logger.debug("Deleting file %s", path)
    try:
        Path(path).unlink()
    except PermissionError:
        logger.error("Unable to delete file %s, as it is still in use.", path)


def add_finalizer(ds: xr.Dataset, target: PathLike) -> None:
    """
    Register a weak-reference callback to delete a temp file on garbage collection.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose lifetime controls the temp file.
    target : PathLike
        Path to the temporary file to clean up.
    """
    logger.debug("Adding finalizer for %s", target)
    assert ds._close is not None
    weakref.finalize(cast("Any", ds._close).__self__.ds, noisy_unlink, target)


def sanitize_chunks(chunks: Any, **dim_mapping: str) -> Any:
    """
    Remap internal dimension names to CDS dimension names in chunk specs.

    Translates atlite dimension names (time, x, y) to the corresponding
    CDS names (valid_time, longitude, latitude).

    Parameters
    ----------
    chunks : Any
        Chunk specification. If not a dict, returned as-is.
    **dim_mapping : str
        Additional or override dimension name mappings.

    Returns
    -------
    Any
        Remapped chunk dict, or original value if not a dict.
    """
    dim_mapping = {
        "time": "valid_time",
        "x": "longitude",
        "y": "latitude",
    } | dim_mapping
    if not isinstance(chunks, dict):
        return chunks

    return {
        extname: chunks[intname]
        for intname, extname in dim_mapping.items()
        if intname in chunks
    }


def open_with_grib_conventions(
    grib_file: PathLike,
    chunks: dict[str, int] | None = None,
    tmpdir: PathLike | None = None,
) -> xr.Dataset:
    """
    Open a GRIB file using cfgrib with standardized coordinate conventions.

    Performs the same conversion as the CDS backend, but locally.
    Based on the documentation at
    https://confluence.ecmwf.int/display/CKB/GRIB+to+netCDF+conversion+on+new+CDS+and+ADS+systems

    Parameters
    ----------
    grib_file : PathLike
        Path to the GRIB file.
    chunks : dict[str, int] or None, optional
        Dask chunk specification for lazy loading.
    tmpdir : PathLike or None, optional
        If set, the file is kept (managed externally).

    Returns
    -------
    xr.Dataset
        Opened dataset with standardized dimensions.
    """
    # Open grib file as dataset.
    # Options below normalize different grib variants into consistent
    # netCDF-compatible hypercubes. Options relevant only to e.g. wave-model
    # data have been removed to keep this routine focused on the products we use.
    ds = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        time_dims=["valid_time"],
        ignore_keys=["edition"],
        coords_as_attributes=[
            "surface",
            "depthBelowLandLayer",
            "entireAtmosphere",
            "heightAboveGround",
            "meanSea",
        ],
        chunks=sanitize_chunks(chunks),
    )
    if tmpdir is None:
        add_finalizer(ds, grib_file)

    def safely_expand_dims(dataset: xr.Dataset, expand_dims: list[str]) -> xr.Dataset:
        """Expand missing dimensions while preserving their original order.

        Returns
        -------
        xr.Dataset
            Dataset with the requested dimensions present.
        """
        dims_required = [
            c for c in dataset.coords if c in expand_dims + list(dataset.dims)
        ]
        dims_missing = [
            (c, i) for i, c in enumerate(dims_required) if c not in dataset.dims
        ]
        dataset = dataset.expand_dims(
            dim=[x[0] for x in dims_missing], axis=[x[1] for x in dims_missing]
        )
        return dataset

    logger.debug("Converting grib file to netcdf format")
    rename_vars = {
        "time": "forecast_reference_time",
        "step": "forecast_period",
        "isobaricInhPa": "pressure_level",
        "hybrid": "model_level",
    }
    rename_vars = {k: v for k, v in rename_vars.items() if k in ds}
    ds = ds.rename(rename_vars)

    ds = safely_expand_dims(ds, ["valid_time", "pressure_level", "model_level"])

    return ds
