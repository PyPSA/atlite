# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and curating data from ECMWFs ERA5 dataset (via CDS).

For further reference see
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
"""

import hashlib
import logging
import os
import time
import warnings
import weakref
from tempfile import mkstemp

import cdsapi
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dask import compute, delayed
from dask.array import arctan2, sqrt
from numpy import atleast_1d

from atlite.gis import maybe_swap_spatial_dims
from atlite.pv.solar_position import SolarPosition

download_status = {}
file_aliases = {}
MAX_DISPLAY_FILES = 3

# Null context for running a with statements wihout any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib

    @contextlib.contextmanager
    def nullcontext():
        yield


logger = logging.getLogger(__name__)

# Model and CRS Settings
crs = 4326

features = {
    "height": ["height"],
    "wind": ["wnd100m", "wnd_azimuth", "roughness"],
    "influx": [
        "influx_toa",
        "influx_direct",
        "influx_diffuse",
        "albedo",
        "solar_altitude",
        "solar_azimuth",
    ],
    "temperature": ["temperature", "soil temperature", "dewpoint temperature"],
    "runoff": ["runoff"],
}

static_features = {"height"}


def _add_height(ds):
    """
    Convert geopotential 'z' to geopotential height following [1].

    References
    ----------
    [1] ERA5: surface elevation and orography, retrieved: 10.02.2019
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+surface+elevation+and+orography
    """
    g0 = 9.80665
    z = ds["z"]
    if "time" in z.coords:
        z = z.isel(time=0, drop=True)
    ds["height"] = z / g0
    ds = ds.drop_vars("z")
    return ds


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and
    longitude columns as 'lat' and 'lon'.
    """
    ds = ds.rename({"longitude": "x", "latitude": "y"})
    if "valid_time" in ds.sizes:
        ds = ds.rename({"valid_time": "time"}).unify_chunks()
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])

    # Combine ERA5 and ERA5T data into a single dimension.
    # See https://github.com/PyPSA/atlite/issues/190
    if "expver" in ds.coords:
        unique_expver = np.unique(ds["expver"].values)
        if len(unique_expver) > 1:
            expver_dim = xr.DataArray(
                unique_expver, dims=["expver"], coords={"expver": unique_expver}
            )
            ds = (
                ds.assign_coords({"expver_dim": expver_dim})
                .drop_vars("expver")
                .rename({"expver_dim": "expver"})
                .set_index(expver="expver")
            )
            for var in ds.data_vars:
                ds[var] = ds[var].expand_dims("expver")
            # expver=1 is ERA5 data, expver=5 is ERA5T data This combines both
            # by filling in NaNs from ERA5 data with values from ERA5T.
            ds = ds.sel(expver="0001").combine_first(ds.sel(expver="0005"))
    ds = ds.drop_vars(["expver", "number"], errors="ignore")

    return ds


def get_data_wind(retrieval_params):
    """
    Get wind data for given retrieval parameters.
    """
    ds = retrieve_data(
        variable=[
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "forecast_surface_roughness",
        ],
        **retrieval_params,
    )
    ds = _rename_and_clean_coords(ds)

    ds["wnd100m"] = sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
        units=ds["u100"].attrs["units"], long_name="100 metre wind speed"
    )
    # span the whole circle: 0 is north, π/2 is east, -π is south, 3π/2 is west
    azimuth = arctan2(ds["u100"], ds["v100"])
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)
    ds = ds.drop_vars(["u100", "v100"])
    ds = ds.rename({"fsr": "roughness"})

    return ds


def sanitize_wind(ds):
    """
    Sanitize retrieved wind data.
    """
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)
    return ds


def get_data_influx(retrieval_params):
    """
    Get influx data for given retrieval parameters.
    """
    ds = retrieve_data(
        variable=[
            "surface_net_solar_radiation",
            "surface_solar_radiation_downwards",
            "toa_incident_solar_radiation",
            "total_sky_direct_solar_radiation_at_surface",
        ],
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename({"fdir": "influx_direct", "tisr": "influx_toa"})
    ds["albedo"] = (
        ((ds["ssrd"] - ds["ssr"]) / ds["ssrd"].where(ds["ssrd"] != 0))
        .fillna(0.0)
        .assign_attrs(units="(0 - 1)", long_name="Albedo")
    )
    ds["influx_diffuse"] = (ds["ssrd"] - ds["influx_direct"]).assign_attrs(
        units="J m**-2", long_name="Surface diffuse solar radiation downwards"
    )
    ds = ds.drop_vars(["ssrd", "ssr"])

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a] / (60.0 * 60.0)
        ds[a].attrs["units"] = "W m**-2"

    # ERA5 variables are mean values for previous hour, i.e. 13:01 to 14:00 are labelled as "14:00"
    # account by calculating the SolarPosition for the center of the interval for aggregation happens
    # see https://github.com/PyPSA/atlite/issues/158
    # Do not show DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        time_shift = pd.to_timedelta("-30 minutes")
        sp = SolarPosition(ds, time_shift=time_shift)
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp])

    return ds


def sanitize_influx(ds):
    """
    Sanitize retrieved influx data.
    """
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def get_data_temperature(retrieval_params):
    """
    Get wind temperature for given retrieval parameters.
    """
    ds = retrieve_data(
        variable=[
            "2m_temperature",
            "soil_temperature_level_4",
            "2m_dewpoint_temperature",
        ],
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename(
        {
            "t2m": "temperature",
            "stl4": "soil temperature",
            "d2m": "dewpoint temperature",
        }
    )

    return ds


def get_data_runoff(retrieval_params):
    """
    Get runoff data for given retrieval parameters.
    """
    ds = retrieve_data(variable=["runoff"], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({"ro": "runoff"})

    return ds


def sanitize_runoff(ds):
    """
    Sanitize retrieved runoff data.
    """
    ds["runoff"] = ds["runoff"].clip(min=0.0)
    return ds


def get_data_height(retrieval_params):
    """
    Get height data for given retrieval parameters.
    """
    ds = retrieve_data(variable="geopotential", **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = _add_height(ds)

    return ds


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    return [y1, x0, y0, x1]


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
            "year": str(time[0].year),
            "month": str(time[0].month),
            "day": str(time[0].day),
            "time": time[0].strftime("%H:00"),
        }

    # Prepare request for all months and years
    times = []
    for year in time.year.unique():
        t = time[time.year == year]
        if monthly_requests:
            for month in t.month.unique():
                query = {
                    "year": str(year),
                    "month": str(month),
                    "day": list(t[t.month == month].day.unique()),
                    "time": ["%02d:00" % h for h in t[t.month == month].hour.unique()],
                }
                times.append(query)
        else:
            query = {
                "year": str(year),
                "month": list(t.month.unique()),
                "day": list(t.day.unique()),
                "time": ["%02d:00" % h for h in t.hour.unique()],
            }
            times.append(query)
    return times


def noisy_unlink(path):
    """
    Delete file at given path.
    """
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def get_cache_filename(request, cache_dir):
    """
    Generate a unique cache filename based on the request parameters.
    """
    # Serialize the request dictionary into a sorted string to ensure consistency
    request_str = "_".join(
        f"{key}-{sorted(value) if isinstance(value, list) else value}"
        for key, value in sorted(request.items())
    )
    # Generate a hash of the request string
    request_hash = hashlib.md5(request_str.encode("utf-8")).hexdigest()
    # Use the first 8 characters of the hash for brevity
    return f"{request_hash}.nc"


def custom_download(url, size, target, lock, filename):
    """
    Optimized download function that uses a simple progress bar and removes
    completed files from the display.
    """
    if target is None:
        target = url.split("/")[-1]

    # Assign a short alias for the filename (e.g. f1, f2, ...)
    file_number = len(file_aliases) + 1
    file_aliases[filename] = f"f{file_number}"

    logging.info(f"Downloading {filename} to {target} ({size} bytes)")
    start = time.time()

    mode = "wb"
    total = 0
    sleep = 10
    tries = 0
    headers = None

    while tries < 5:
        r = requests.get(url, stream=True, headers=headers)
        try:
            r.raise_for_status()

            with open(target, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
                        with lock:
                            download_status[filename] = total / size * 100
                            update_progress_bar()

        except requests.exceptions.ConnectionError as e:
            logging.error(f"Download interrupted: {e}")
            break
        finally:
            r.close()

        if total >= size:
            break

        logging.error(f"Download incomplete, downloaded {total} bytes out of {size}")
        logging.warning(f"Sleeping {sleep} seconds")
        time.sleep(sleep)
        mode = "ab"
        total = os.path.getsize(target)
        sleep *= 1.5
        headers = {"Range": f"bytes={total}-"}
        tries += 1

    if total != size:
        raise Exception(f"Download failed: downloaded {total} bytes out of {size}")

    elapsed = time.time() - start
    if elapsed:
        logging.info(f"Download rate {total / elapsed:.2f} bytes/s")

    return target


def update_progress_bar():
    """
    Update a progress bar that shows the percentage of all files being
    downloaded.

    Files that have reached 100% are removed from the display. Only
    short aliases are displayed.
    """
    completed_files = [
        file for file, progress in download_status.items() if progress >= 100
    ]

    # Remove completed files from the progress dictionary
    for file in completed_files:
        del download_status[file]
        del file_aliases[file]  # Remove alias as well

    if not download_status:
        # If no active downloads, clear the progress bar
        print("\r", end="")
        return

    # Only display the top N files to avoid multi-line output
    displayed_files = list(download_status.items())[:MAX_DISPLAY_FILES]

    # Create progress string using the short aliases
    progress = " | ".join(
        [
            f"{file_aliases[file]}: {int(progress)}%"
            for file, progress in displayed_files
        ]
    )

    # If there are more files, show a summary
    if len(download_status) > MAX_DISPLAY_FILES:
        progress += f" | ... and {len(download_status) - MAX_DISPLAY_FILES} more"

    # Use \r to overwrite the same line
    print(f"\r{progress}", end="")


def retrieve_data(product, chunks=None, tmpdir=None, lock=None, **updates):
    """
    Download data like ERA5 from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://cds-beta.climate.copernicus.eu/requests?tab=all
    """
    request = {"product_type": "reanalysis", "format": "netcdf"}
    request.update(updates)

    assert {"year", "month", "variable"}.issubset(
        request
    ), "Need to specify at least 'variable', 'year' and 'month'"

    # Use tmpdir for cache directory; if not provided, use current working directory
    if tmpdir is None:
        tmpdir = os.getcwd()
    cache_dir = tmpdir
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache filename based on request
    cache_filename = get_cache_filename(request, cache_dir)
    cache_filepath = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_filepath):
        logging.info(f"Using cached file for request: {cache_filename}")
        ds = xr.open_dataset(cache_filepath, chunks=chunks or {})
        return ds

    client = cdsapi.Client(
        info_callback=logger.debug, debug=logging.DEBUG >= logging.root.level
    )
    result = client.retrieve(product, request)

    if lock is None:
        lock = nullcontext()

    with lock:
        fd, target = mkstemp(suffix=".nc", dir=tmpdir)
        os.close(fd)

    # Inform user about data being downloaded as "* variable (year-month)"
    timestr = f"{request['year']}-{request['month']}"
    variables = atleast_1d(request["variable"])
    varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
    filename = f"{variables[0]}_{timestr}.nc"
    logger.info(f"CDS: Downloading variables\n\t{varstr}\n")
    custom_download(result.location, result.content_length, target, lock, filename)

    # Move the downloaded file to cache directory
    os.rename(target, cache_filepath)
    ds = xr.open_dataset(cache_filepath, chunks=chunks or {})
    if tmpdir is None:
        logger.debug(f"Adding finalizer for {cache_filepath}")
        weakref.finalize(ds._file_obj._manager, noisy_unlink, cache_filepath)

    # Remove default encoding we get from CDSAPI, which can lead to NaN values after loading with subsequent
    # saving due to how xarray handles netcdf compression (only float encoded as short int seem affected)
    # Fixes issue by keeping "float32" encoded as "float32" instead of internally saving as "short int", see:
    # https://stackoverflow.com/questions/75755441/why-does-saving-to-netcdf-without-encoding-change-some-values-to-nan
    # and hopefully fixed soon (could then remove), see https://github.com/pydata/xarray/issues/7691
    for v in ds.data_vars:
        if ds[v].encoding.get("dtype") == "int16":
            ds[v].encoding.clear()

    return ds


def get_data(
    cutout,
    feature,
    tmpdir,
    lock=None,
    monthly_requests=False,
    concurrent_requests=False,
    **creation_parameters,
):
    """
    Retrieve data from ECMWFs ERA5 dataset (via CDS).

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.era5.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    monthly_requests : bool, optional
        If True, the data is requested on a monthly basis in ERA5. This is useful for
        large cutouts, where the data is requested in smaller chunks. The
        default is False
    concurrent_requests : bool, optional
        If True, the monthly data requests are posted concurrently.
        Only has an effect if `monthly_requests` is True.
    **creation_parameters :
        Additional keyword arguments. The only effective argument is 'sanitize'
        (default True) which sets sanitization of the data on or off.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.
    """
    coords = cutout.coords

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "product": "reanalysis-era5-single-levels",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": [cutout.dx, cutout.dy],
        "tmpdir": tmpdir,
        "lock": lock,
    }

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    logger.info(f"Requesting data for feature {feature}...")

    def retrieve_once(time):
        ds = func({**retrieval_params, **time})
        if sanitize and sanitize_func is not None:
            ds = sanitize_func(ds)
        return ds

    if feature in static_features:
        return retrieve_once(retrieval_times(coords, static=True)).squeeze()

    time_chunks = retrieval_times(coords, monthly_requests=monthly_requests)
    if concurrent_requests:
        delayed_datasets = [delayed(retrieve_once)(chunk) for chunk in time_chunks]
        datasets = compute(*delayed_datasets)
    else:
        datasets = map(retrieve_once, time_chunks)

    return xr.concat(datasets, dim="time").sel(time=coords["time"])
