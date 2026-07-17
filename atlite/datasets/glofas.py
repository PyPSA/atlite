# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and curating data from ECMWFs GLOFAS dataset (via CDS).

For further reference see
https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical?tab=overview
"""

import logging
import os
import zipfile
from pathlib import Path
from tempfile import mkstemp
from typing import Any

import cdsapi
import numpy as np
import xarray as xr
from dask import compute, delayed
from dask.utils import SerializableLock
from numpy import atleast_1d

from atlite.datasets.cds_helper import (
    _area,
    add_finalizer,
    open_with_grib_conventions,
    sanitize_chunks,
)
from atlite.gis import maybe_swap_spatial_dims

# Null context for running a with statements wihout any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib

    @contextlib.contextmanager  # type: ignore[no-redef]
    def nullcontext():  # noqa: D103
        yield


logger = logging.getLogger(__name__)

# Model and CRS Settings
crs = 4326

dataset = "cems-glofas-historical"

features = {"discharge": ["discharge"]}


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and
    longitude columns as 'lat' and 'lon'.

    Returns
    -------
    xr.Dataset
        Dataset with standardized coordinates.
    """
    ds = ds.rename({
        "longitude": "x",
        "latitude": "y",
        "valid_time": "time",
        "dis24": "discharge",
    })
    # round coords since cds coords are float64 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds.drop_vars(["expver", "number"], errors="ignore")


def retrieve_data(
    product: str,
    chunks: dict[str, int] | None = None,
    tmpdir: str | Path | None = None,
    lock: SerializableLock | None = None,
    **updates: Any,
) -> xr.Dataset:
    """
    Download data like Glofas from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://ewds.climate.copernicus.eu/requests?tab=all

    Parameters
    ----------
    product : str
        Product name, e.g. 'cems-glofas-historical'.
    chunks : dict, optional
        Chunking for xarray dataset, e.g. {'time': 1, 'x': 100, 'y': 100}.
        Default is None.
    tmpdir : str, optional
        Directory where the downloaded data is temporarily stored.
        Default is None, which uses the system's temporary directory.
    lock : dask.utils.SerializableLock, optional
        Lock for thread-safe file writing. Default is None.
    updates : dict
        Additional parameters for the request.
        Must include 'hyear', 'hmonth', 'hday', and 'variable'.
        Can include e.g. 'data_format'.

    Returns
    -------
    xarray.Dataset
        Dataset with the retrieved variables.

    Examples
    --------
    >>> ds = retrieve_data(
    ...     product='cems-glofas-historical',
    ...     chunks={'time': 1, 'x': 100, 'y': 100},
    ...     tmpdir='/tmp',
    ...     lock=None,
    ...     hyear='2020',
    ...     hmonth='01',
    ...     variable=['river_discharge_in_the_last_24_hours'],
    ...     data_format='grib'
    ... )
    """
    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "data_format": "grib",
        "download_format": "zip",
    }
    request.update(updates)

    assert {"hyear", "hmonth", "variable"}.issubset(request), (
        "Need to specify at least 'variable', 'hyear' and 'hmonth'"
    )

    logger.debug("Requesting %s with API request: %s", product, request)
    # Url needs to be set manually here, overrides url from .cdsapirc (for use with multiple modules)
    client = cdsapi.Client(
        info_callback=logger.debug,
        debug=logging.root.level <= logging.DEBUG,
        url="https://ewds.climate.copernicus.eu/api",
    )
    result = client.retrieve(product, request)

    if lock is None:
        lock = nullcontext()

    suffix = f".{request['data_format']}"  # .netcdf or .grib
    with lock:
        fd, target = mkstemp(suffix=suffix, dir=tmpdir)
        os.close(fd)

        timestr = f"{request['hyear']}-{request['hmonth']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info("CDS: Downloading variables\n\t%s\n", varstr)
        result.download(target)

    if request.get("download_format") == "zip":
        extract_dir = Path(target).parent / Path(target).stem
        with zipfile.ZipFile(target, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        Path(target).unlink()
        target = str(extract_dir / f"data{suffix}")

    # Convert from grib to netcdf locally, same conversion as in CDS backend
    if request["data_format"] == "grib":
        ds = open_with_grib_conventions(target, chunks=chunks, tmpdir=tmpdir)
    else:
        ds = xr.open_dataset(target, chunks=sanitize_chunks(chunks))
        if tmpdir is None:
            add_finalizer(ds, target)
    return ds


def retrieval_times(coords, static=False, monthly_requests=False):
    """
    Get list of retrieval cdsapi arguments for time dimension in coordinates.

    If static is False, this function creates a query for each month and year
    in the time axis in coords. This ensures not running into size query limits
    of the cdsapi even with very (spatially) large cutouts.
    If static is True, the function return only one set of parameters
    for the very first time point.

    Parameters
    ----------
    coords : atlite.Cutout.coords
    static : bool, optional
    monthly_requests : bool, optional
        If True, the data is requested on a monthly basis. This is useful for
        large cutouts, where the data is requested in smaller chunks. The
        default is False

    Returns
    -------
    list of dicts witht retrieval arguments

    """
    time = coords["time"].to_index()
    if static:
        return {
            "hyear": time[0].strftime("%Y"),
            "hmonth": time[0].strftime("%m"),
            "hday": time[0].strftime("%d"),
        }

    # Prepare request for all months and years
    times = []
    for year in time.year.unique():
        t = time[time.year == year]
        if monthly_requests:
            for month in t.month.unique():
                query = {
                    "hyear": str(year),
                    "hmonth": list(t[t.month == month].strftime("%m").unique()),
                    "hday": list(t[t.month == month].strftime("%d").unique()),
                }
                times.append(query)
        else:
            query = {
                "hyear": str(year),
                "hmonth": list(t.strftime("%m").unique()),
                "hday": list(t.strftime("%d").unique()),
            }
            times.append(query)
    return times


def get_data(
    cutout,
    feature,
    tmpdir="tmp",
    lock=None,
    data_format="grib",
    monthly_requests=False,
    concurrent_requests=False,
    **creation_parameters,
):
    """
    Retrieve data from ECMWFs GLOFAS dataset (via CDS).

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.glofas.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    data_format : str, optional
        The format of the data to be downloaded. Can be either 'grib' or 'netcdf',
        'grib' highly recommended because CDSAPI limits request size for netcdf.
    concurrent_requests : bool, optional
        If True, the monthly data requests are posted concurrently.
        Only has an effect if `monthly_requests` is True.
    **creation_parameters :
        Additional keyword arguments:
        - 'sanitize' (default True): sets sanitization of the data on or off.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables at the native GLOFAS
        resolution (daily values on the native grid).

    """
    coords = cutout.coords

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "product": "cems-glofas-historical",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "tmpdir": tmpdir,
        "lock": lock,
        "data_format": data_format,
    }

    def retrieve_once(time):
        ds = retrieve_data(
            variable=["river_discharge_in_the_last_24_hours"],
            **retrieval_params,
            **time,
        )
        ds = _rename_and_clean_coords(ds)
        # dis24 is timestamped at the end of its 24h window; shift to the flow day
        ds = ds.assign_coords(time=ds["time"] - np.timedelta64(1, "D"))
        if sanitize:
            ds["discharge"] = ds["discharge"].clip(min=0.0).fillna(0.0)
        return ds

    time_chunks = retrieval_times(coords, monthly_requests=monthly_requests)
    if concurrent_requests:
        delayed_datasets = [delayed(retrieve_once)(chunk) for chunk in time_chunks]
        datasets = list(compute(*delayed_datasets))
    else:
        datasets = list(map(retrieve_once, time_chunks))

    # Keep discharge at its native GLOFAS resolution. Interpolation needs to happen
    # later to avoid interpolating the whole grid here.
    return xr.concat(datasets, dim="time").sortby("time")
