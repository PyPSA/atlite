# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Shared helpers for datasets downloaded via the Climate Data Store (CDS).

Used by both the era5 and glofas dataset modules.
"""

import logging
import os
import weakref
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


def noisy_unlink(path):
    """
    Delete file at given path.
    """
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    return [y1, x0, y0, x1]


def sanitize_chunks(chunks, **dim_mapping):
    dim_mapping = dict(time="valid_time", x="longitude", y="latitude") | dim_mapping
    if not isinstance(chunks, dict):
        # preserve "auto" or None
        return chunks

    return {
        extname: chunks[intname]
        for intname, extname in dim_mapping.items()
        if intname in chunks
    }


def add_finalizer(ds: xr.Dataset, target: str | Path):
    logger.debug(f"Adding finalizer for {target}")
    weakref.finalize(ds._close.__self__.ds, noisy_unlink, target)


def open_with_grib_conventions(
    grib_file: str | Path, chunks=None, tmpdir: str | Path | None = None
) -> xr.Dataset:
    """
    Convert a grib file downloaded from the CDS to netcdf conventions locally.

    The function does the same thing as the CDS backend does, but locally.
    This is needed, as the grib file is the recommended download file type for CDS, with conversion to netcdf locally.
    The routine is a reduced version based on the documentation here:
    https://confluence.ecmwf.int/display/CKB/GRIB+to+netCDF+conversion+on+new+CDS+and+ADS+systems#GRIBtonetCDFconversiononnewCDSandADSsystems-jupiternotebook

    Parameters
    ----------
    grib_file : str | Path
        Path to the grib file to be converted.
    chunks
        Chunks
    tmpdir : Path, optional
        If None adds a finalizer to the dataset object

    Returns
    -------
    xr.Dataset
    """
    #
    # Open grib file as dataset
    # Options to open different datasets into a datasets of consistent hypercubes which are compatible netCDF
    # There are options that might be relevant for e.g. for wave model data, that have been removed here
    # to keep the code cleaner and shorter
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
        """
        Expand dimensions in an xarray dataset, ensuring that the new dimensions are not already in the dataset
        and that the order of dimensions is preserved.
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
    # Variables and dimensions to rename if they exist in the dataset
    rename_vars = {
        "time": "forecast_reference_time",
        "step": "forecast_period",
        "isobaricInhPa": "pressure_level",
        "hybrid": "model_level",
    }
    rename_vars = {k: v for k, v in rename_vars.items() if k in ds}
    ds = ds.rename(rename_vars)

    # safely expand dimensions in an xarray dataset to ensure that data for the new dimensions are in the dataset
    ds = safely_expand_dims(ds, ["valid_time", "pressure_level", "model_level"])

    return ds
