# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading data from the C3S Energy bias-corrected ERA5 reanalysis.

The data is retrieved from the Climate Data Store dataset
``sis-energy-global-reanalysis``, see
https://cds.climate.copernicus.eu/datasets/sis-energy-global-reanalysis

Only the 100 m wind speed is available, in two variants which are both stored
in the cutout (feature ``wind_corrected``):

* ``wnd100m_corrected``: bias-corrected 100 m wind speed
  (CDS variable ``100m_wind_speed``)
* ``wnd100m_corrected_extrapolated``: wind speed extrapolated to 100 m with a
  power law from the bias-corrected lower level wind speed
  (CDS variable ``100m_wind_speed_extrapolated``)

Combine the module with ``era5`` to get the roughness needed for hub heights
other than 100 m. Afterwards choose which 100 m wind speed is used by
:py:meth:`atlite.Cutout.wind` by assigning it to ``wnd100m``.

Example
-------
>>> cutout = atlite.Cutout(
...     "corrected.nc",
...     module=["era5", "era5-corrected"],
...     x=slice(-12, 19),
...     y=slice(36, 52),
...     time="2010-01-01",
... )
>>> cutout.prepare()
>>> cutout.data["wnd100m"] = cutout.data["wnd100m_corrected_extrapolated"]
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed

from atlite.datasets.cds_helper import _area, sanitize_chunks
from atlite.gis import maybe_swap_spatial_dims

if TYPE_CHECKING:
    from dask.utils import SerializableLock

    from atlite._types import PathLike

logger = logging.getLogger(__name__)

# Model and CRS Settings
crs = 4326

features = {"wind_corrected": ["wnd100m_corrected", "wnd100m_corrected_extrapolated"]}

DATASET = "sis-energy-global-reanalysis"
# CDS variable -> (tag in the delivered file name, name in the cutout)
WIND_VARIABLES = {
    "100m_wind_speed": ("_mbc_", "wnd100m_corrected"),
    "100m_wind_speed_extrapolated": ("_pwl_", "wnd100m_corrected_extrapolated"),
}


def _rename_and_clean_coords(ds: xr.Dataset, add_lon_lat: bool = True) -> xr.Dataset:
    """
    Standardize coordinates to atlite conventions.

    The dataset uses longitudes in [0, 360), which are shifted to [-180, 180)
    and sorted, and are renamed to x/y.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with 'longitude' and 'latitude' coordinates.
    add_lon_lat : bool, optional
        Whether to add 'lon'/'lat' as coordinate aliases. Default True.

    Returns
    -------
    xr.Dataset
        Dataset with standardized coordinates.
    """
    ds = ds.rename({"longitude": "x", "latitude": "y"})
    x = ((ds.x.astype(float) + 180) % 360) - 180
    ds = ds.assign_coords(x=np.round(x, 5), y=np.round(ds.y.astype(float), 5))
    ds = ds.sortby("x")
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds


def retrieve_data(
    chunks: dict[str, int] | None = None,
    tmpdir: PathLike | None = None,
    lock: SerializableLock | None = None,
    **updates: Any,
) -> xr.Dataset:
    """
    Download data from the CDS API and return it as an xarray Dataset.

    If you want to track the state of your request go to
    https://cds.climate.copernicus.eu/requests?tab=all

    Note that the CDS delivers complete months, irrespective of the requested
    days.

    Parameters
    ----------
    chunks : dict[str, int] or None, optional
        Dask chunk specification of the returned dataset.
    tmpdir : PathLike or None, optional
        Directory for temporary download files.
    lock : SerializableLock or None, optional
        Lock for thread-safe file creation.
    **updates : Any
        Additional CDS API request parameters. Must include at least
        'variable', 'year', 'month' and 'day'.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """
    import cdsapi

    request: dict[str, Any] = {
        "spatial_coverage": "global",
        "spatial_resolution": ["0_25_degree"],
        "temporal_resolution": ["1_hour"],
        "version": "1_00",
        "file_format": "netcdf_4",
    }
    request.update(updates)

    assert {"year", "month", "day", "variable"}.issubset(request), (
        "Need to specify at least 'variable', 'year', 'month' and 'day'"
    )

    logger.debug("Requesting %s with API request: %s", DATASET, request)

    client = cdsapi.Client(
        info_callback=logger.debug, debug=logging.root.level <= logging.DEBUG
    )
    result = client.retrieve(DATASET, request)

    cm: Any = nullcontext() if lock is None else lock
    with cm:
        fd, archive = tempfile.mkstemp(suffix=".zip", dir=tmpdir)
        os.close(fd)

        year_str = (
            ",".join(request["year"])
            if isinstance(request["year"], list)
            else str(request["year"])
        )
        month_str = (
            ",".join(request["month"])
            if isinstance(request["month"], list)
            else str(request["month"])
        )
        timestr = f"{year_str}-{month_str}"
        varstr = "\n\t".join(f"{v} ({timestr})" for v in request["variable"])
        logger.info("CDS: Downloading variables\n\t%s\n", varstr)
        result.download(archive)

    # The files are read into memory so that they can be removed right away,
    # which is required on Windows where open files cannot be deleted.
    # The zip can hold several files; the file name tags bias-corrected (mbc)
    # and power-law extrapolated (pwl) data. When multiple months are requested,
    # the zip contains one NetCDF file per month.
    try:
        datasets = []
        with zipfile.ZipFile(archive) as zf:
            names = list(dict.fromkeys(n for n in zf.namelist() if n.endswith(".nc")))
            for variable in request["variable"]:
                tag, new_name = WIND_VARIABLES[variable]
                members = sorted(n for n in names if tag in n)
                assert members, (
                    f"Expected at least one NetCDF file for {variable}, got none"
                )
                var_datasets = []
                for member in members:
                    fd, target = tempfile.mkstemp(suffix=".nc", dir=tmpdir)
                    try:
                        with os.fdopen(fd, "wb") as out:
                            out.write(zf.read(member))
                        with xr.open_dataset(target) as raw:
                            var_datasets.append(raw.load().rename({"ws100": new_name}))
                    finally:
                        os.remove(target)
                if len(var_datasets) == 1:
                    datasets.append(var_datasets[0])
                else:
                    combined = xr.concat(var_datasets, dim="time")
                    combined = combined.sortby("time")
                    datasets.append(combined)
    finally:
        os.remove(archive)

    ds = xr.merge(datasets)
    ds = _rename_and_clean_coords(ds)
    chunks = sanitize_chunks(chunks, time="time", x="x", y="y")
    return ds.chunk(chunks) if isinstance(chunks, dict) else ds


def get_data(
    cutout: Any,
    feature: str,
    tmpdir: PathLike,
    lock: SerializableLock | None = None,
    monthly_requests: bool = False,
    concurrent_requests: bool = False,
    **creation_parameters: Any,
) -> xr.Dataset:
    """
    Download bias-corrected ERA5 data for a given feature.

    Parameters
    ----------
    cutout : Cutout
        Cutout object defining the spatial and temporal extent.
    feature : str
        Feature to retrieve. Only 'wind_corrected' is supported.
    tmpdir : PathLike
        Directory for temporary download files.
    lock : SerializableLock or None, optional
        Lock for thread-safe file creation.
    monthly_requests : bool, optional
        If True, split API requests by month. Otherwise, request one year at a
        time. Default False.
    concurrent_requests : bool, optional
        If True, use dask.delayed for parallel downloads. Default False.
    **creation_parameters : Any
        Additional cutout parameters, ignored.

    Notes
    -----
    Timeframes longer than 12 months are rejected because the CDS API request
    would exceed its size limit. Each wind variable is downloaded separately.

    Returns
    -------
    xr.Dataset
        Data for the requested feature, aligned to cutout coordinates.
    """
    if feature not in features:
        raise NotImplementedError(f"Feature {feature!r} not supported by era5-corrected")
    if (cutout.dx, cutout.dy) != (0.25, 0.25):
        raise NotImplementedError(
            "era5-corrected only supports the native 0.25°×0.25° grid."
        )

    coords = cutout.coords
    retrieval_params: dict[str, Any] = {
        "area": _area(coords),
        "chunks": cutout.chunks,
        "tmpdir": tmpdir,
        "lock": lock,
    }

    logger.info("Requesting data for feature %s...", feature)

    # Keep requests to at most 12 months to stay within the CDS request limit.
    time = coords["time"].to_index()
    if len(time) and time[-1] > time[0] + pd.DateOffset(months=12):
        logger.warning(
            "The era5-corrected API supports cutouts of at most 12 months; "
            "the requested timeframe is longer."
        )
        raise ValueError("era5-corrected timeframe cannot exceed 12 months.")

    # The CDS delivers whole months regardless of 'day', so request a single
    # day per month. Group those months by year unless monthly requests were
    # explicitly requested.
    if monthly_requests:
        time_chunks = [
            {
                "year": [str(month.year)],
                "month": [month.strftime("%m")],
                "day": ["01"],
            }
            for month in time.to_period("M").unique()
        ]
    else:
        time_chunks = [
            {
                "year": [str(year)],
                "month": [
                    month.strftime("%m")
                    for month in time[time.year == year].to_period("M").unique()
                ],
                "day": ["01"],
            }
            for year in time.year.unique()
        ]

    def retrieve_once(t: dict[str, Any]) -> xr.Dataset:
        datasets = [
            retrieve_data(variable=[variable], **retrieval_params, **t)
            for variable in WIND_VARIABLES
        ]
        return xr.merge(datasets)

    if concurrent_requests:
        datasets = compute(*[delayed(retrieve_once)(c) for c in time_chunks])
    else:
        datasets = map(retrieve_once, time_chunks)

    result = xr.concat(datasets, dim="time").sel(time=coords["time"])
    assert isinstance(result, xr.Dataset)
    return result
