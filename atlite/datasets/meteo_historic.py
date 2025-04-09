# SPDX-FileCopyrightText: 2016-2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for downloading and curating historic data from ECMWFs ERA5 dataset (via Open-Meteo).

For further reference see
https://open-meteo.com/en/docs/historical-weather-api
"""

import datetime
import io
import logging
import os
import re
import time
import warnings
import weakref
import zipfile
from tempfile import mkstemp

import cdsapi
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import xarray as xr
from dask import compute, delayed
from numpy import atleast_1d
from retry_requests import retry

from ..gis import maybe_swap_spatial_dims
from ..pv.solar_position import SolarPosition

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

# Setup Open-Meteo client with 7-day cache and retry on failure
# Cache duration: -1 = never expire, 0 = no caching, timedelta = expire after set time
cache_window = datetime.timedelta(days=7)
cache_session = requests_cache.CachedSession(
    ".meteo.cache", backend="sqlite", expire_after=cache_window
)
retry_session = retry(cache_session, retries=5, backoff_factor=3)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Set url for data download, this allows to switch to different data
# sources more easily.
era5_url = "https://cds.climate.copernicus.eu/api"
meteo_url = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo request limits
MINUTE_LIMIT = 600
HOUR_LIMIT = 5000
DAY_LIMIT = 10000

# Delay of ERA5 data upload
# For Open-Meteo slow changing variables from ERA5 are always interpolated
# Starting from data of at least 7 days before nowtime
ERA5_DELAY = pd.Timedelta(hours=-6.0 * 24)

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
    "temperature": ["temperature", "soil temperature"],
}

static_features = {"height"}

requirements = {
    "x": slice(-90, 90, 0.1),
    "y": slice(-90, 90, 0.1),
    "offset": (
        pd.Timestamp("1940-01-01")
        - pd.Timestamp.utcnow().replace(tzinfo=None).floor("h")
    ),
    "forecast": (
        pd.Timedelta(hours=-1.0 * 24)
        + (
            pd.Timestamp.utcnow().replace(tzinfo=None).ceil("d")
            - pd.Timestamp.utcnow().replace(tzinfo=None)
        )
    ).floor("h"),
    "dt": pd.Timedelta(hours=1),
    "parallel": False,
}


def _checkModuleRequirements(x, y, time, time_now, **kwargs):
    """
    Load and check the data requirements for a given module.

    Parameters
    ----------
    x (slice): Defines the start, stop, and step values for the x-dimension.
    y (slice): Defines the start, stop, and step values for the y-dimension.
    time (slice): Defines the start, stop, and step values for the time dimension.
    **kwargs: Additional optional parameters.
    """

    # Extract start, stop, and step values for x
    x_start, x_stop, x_step = x.start, x.stop, x.step

    # Adjust x range based on module requirements
    if requirements["x"].start > x.start:
        x_start = requirements["x"].start
    if requirements["x"].stop < x.stop:
        x_stop = requirements["x"].stop
    if requirements["x"].step > x.step:
        x_step = requirements["x"].step

    x = slice(x_start, x_stop, x_step)

    # Extract start, stop, and step values for y
    y_start, y_stop, y_step = y.start, y.stop, y.step

    # Adjust y range based on module requirements
    if requirements["y"].start > y.start:
        y_start = requirements["y"].start
    if requirements["y"].stop < y.stop:
        y_stop = requirements["y"].stop
    if requirements["y"].step > y.step:
        y_step = requirements["y"].step

    y = slice(y_start, y_stop, y_step)

    # Extract time range parameters
    time_start = time.start
    time_stop = time.stop
    time_step = time.step

    # Check forecast feasibility
    feasible_start = time_now + requirements["offset"]
    feasible_end = time_now + requirements["forecast"]

    # Ensure time_start is within feasible bounds
    if time_start < feasible_start:
        logger.error(
            f"The required forecast start time {time_start} exceeds the model requirements."
        )
        logger.error(
            f"The minimum start time of the forecast for {time_now} is {feasible_start}."
        )
        logger.error(
            f"The maximum historical offset of the forecast is {requirements['offset']}."
        )
        raise ValueError(
            f"Invalid forecast start time: {time_start}. Must be >= {feasible_start}."
        )

    if time_start >= feasible_end:
        logger.error(
            f"The required forecast start time {time_start} exceeds the model requirements."
        )
        logger.error(
            f"The maximum start time of the forecast for {time_now} needs to be smaller than {feasible_end}."
        )
        raise ValueError(
            f"Invalid forecast start time: {time_start}. Must be < {feasible_end}."
        )

    # Ensure time_stop is greater than time_start
    if time_stop <= time_start:
        logger.error(
            f"The required forecast end time {time_stop} exceeds the model requirements."
        )
        logger.error(
            f"The minimum end time of the forecast for {time_now} needs to be larger than {time_start}."
        )
        raise ValueError(
            f"Invalid forecast end time: {time_stop}. Must be > {time_start}."
        )

    # Ensure time_stop is greater than time_start
    if time_stop > feasible_end:
        logger.error(
            f"The required forecast end time {time_stop} exceeds the model requirements."
        )
        logger.error(
            f"The maximum end time of the forecast for {time_now} is {feasible_end}."
        )
        logger.error(f"The maximum forecast horizon is {requirements['forecast']}.")
        raise ValueError(
            f"Invalid forecast end time: {time_stop}. Must be <= {feasible_end}."
        )

    # Ensure time step is within required limits
    if (time_step is pd.Timedelta(None)) or (time.step < requirements["dt"]):
        logger.warning(
            f"The required temporal forecast resolution {time_step} exceeds the model requirements."
        )
        logger.warning(
            f"The minimum temporal resolution of the forecast is {requirements['dt']}."
        )
        logger.info(
            f"Set the temporal forecast resolution to the minimum: {requirements['dt']}."
        )
        time_step = requirements["dt"]

    time = slice(time_start, time_stop, time_step)

    # Retrieve parallel processing setting from requirements
    parallel = requirements["parallel"]

    return x, y, time, parallel


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


def _interpolate(ds, ds_ref, static, interp_s, interp_t):
    # Interpolate data to specific latitude and longitude values given as input (due to specific model resolution)

    if not static:
        try:
            ds = ds.interp(
                time=ds_ref.time.values,
                method=interp_t,
                kwargs={"fill_value": "extrapolate"},
            )
        except ValueError:
            logger.info(
                f"Interpolation: Not enough supporting points for used interpolation method {interp_t}."
            )
            logger.info("Interpolation method is set to 'nearest' instead.")
            ds = ds.interp(
                time=ds_ref.time.values,
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )

    try:
        ds = ds.interp(
            x=ds_ref.x.values,
            y=ds_ref.y.values,
            method=interp_s,
            kwargs={"fill_value": "extrapolate"},
        )
    except ValueError:
        logger.info(
            f"Interpolation: Not enough supporting points for used interpolation method {interp_s}."
        )
        logger.info("Interpolation method is set to 'nearest' instead.")
        ds = ds.interp(
            x=ds_ref.x.values,
            y=ds_ref.y.values,
            method="nearest",
            kwargs={"fill_value": "extrapolate"},
        )

    return ds


def get_data_meteo_wind(retrieval_params):
    """Get all data from meteo API for given retrieval parameters at once to save requests and runtime."""

    ds = retrieve_meteo_data(
        url=meteo_url,
        variable=[
            "windspeed_100m",
            "winddirection_100m",
        ],
        **retrieval_params,
    )

    return ds


def get_data_meteo_influx(retrieval_params):
    """Get all data from meteo API for given retrieval parameters at once to save requests and runtime."""

    ds = retrieve_meteo_data(
        url=meteo_url,
        variable=[
            # "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            # "direct_normal_irradiance",
            # "terrestrial_radiation",
        ],
        **retrieval_params,
    )

    return ds


def get_data_meteo_temperature(retrieval_params):
    """Get all data from meteo API for given retrieval parameters at once to save requests and runtime."""

    ds = retrieve_meteo_data(
        url=meteo_url,
        variable=[
            "temperature_2m",
            "soil_temperature_54cm",
        ],
        **retrieval_params,
    )

    return ds


def get_data_era5_wind(retrieval_params):
    """Get wind data for given retrieval parameters."""

    ds = retrieve_era5_data(
        url=era5_url,
        variable=["forecast_surface_roughness"],
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)

    return ds


def combine_data_wind(ds_meteo, ds_era5, interp_s, interp_t):
    ds_era5 = _interpolate(
        ds=ds_era5,
        ds_ref=ds_meteo,
        static=False,
        interp_s=interp_s,
        interp_t=interp_t,
    )

    ds = xr.merge([ds_meteo, ds_era5])

    ds = ds.rename(
        {
            "windspeed_100m": "wnd100m",
            "winddirection_100m": "wnd_azimuth",
            "fsr": "roughness",
        }
    )

    ds.wnd100m.attrs.update(
        units="m s**-1", long_name="Wind speed at 100m above ground"
    )
    ds.wnd_azimuth.attrs.update(
        units="degree", long_name="Wind direction at 100m above ground"
    )

    # unify_chunks() is necessary to avoid a bug in xarray
    ds = ds.unify_chunks()

    return ds


def sanitize_wind(ds):
    """Sanitize retrieved wind data."""
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)
    return ds


def get_data_era5_influx(retrieval_params):
    """Get influx data for given retrieval parameters."""

    ds = retrieve_era5_data(
        url=era5_url,
        variable=["forecast_albedo", "toa_incident_solar_radiation"],
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)

    return ds


def combine_data_influx(ds_meteo, ds_era5, interp_s, interp_t):
    ds_era5 = _interpolate(
        ds=ds_era5,
        ds_ref=ds_meteo,
        static=False,
        interp_s=interp_s,
        interp_t=interp_t,
    )

    ds = xr.merge([ds_meteo, ds_era5])

    ds = ds.rename(
        {
            "direct_radiation": "influx_direct",
            "diffuse_radiation": "influx_diffuse",
            "tisr": "influx_toa",
            "fal": "albedo",
        }
    )

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    ds["influx_toa"] = ds["influx_toa"] / (60.0 * 60.0)

    ds.influx_direct.attrs.update(
        units="W m**-2", long_name="Surface direct solar radiation downwards"
    )
    ds.influx_diffuse.attrs.update(
        units="W m**-2", long_name="Surface diffuse solar radiation downwards"
    )
    ds.influx_toa.attrs.update(
        units="W m**-2", long_name="TOA incident solar radiation"
    )

    # unify_chunks() is necessary to avoid a bug in xarray
    ds = ds.unify_chunks()

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
    """Sanitize retrieved influx data."""
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def combine_data_temperature(ds_meteo, ds_era5, interp_s, interp_t):
    """Get wind temperature for given retrieval parameters."""
    ds = xr.merge([ds_meteo, ds_era5])

    ds = ds.rename(
        {"temperature_2m": "temperature", "soil_temperature_54cm": "soil temperature"}
    )

    # Convert from Celsius to Kelvin C -> K, by adding 273.15
    ds = ds + 273.15

    ds["temperature"].attrs.update(units="K", long_name="2 metre temperature")
    ds["soil temperature"].attrs.update(units="K", long_name="Soil temperature 54cm")

    # unify_chunks() is necessary to avoid a bug in xarray
    ds = ds.unify_chunks()

    return ds


def get_data_era5_height(retrieval_params):
    """Get height data for given retrieval parameters."""
    ds = retrieve_era5_data(url=era5_url, variable=["geopotential"], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = _add_height(ds)

    return ds


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    return [y1, x0, y0, x1]


def noisy_unlink(path):
    """Delete file at given path."""
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def retrieve_meteo_data(url, product, chunks=None, tmpdir=None, lock=None, **updates):
    """
    Download meteorological data (e.g., ERA5-style) from the Open-Meteo API.

    Parameters
    ----------
    url : str
        API endpoint (typically `meteo_url`).
    product : str
        Product identifier (not used here but reserved for compatibility).
    chunks : dict, optional
        Chunking configuration for final xarray dataset.
    tmpdir : str, optional
        Temporary storage path (not used here).
    lock : threading.Lock, optional
        Lock for parallel-safe writing (not used here).
    updates : dict
        Additional request parameters including 'coords', 'start', 'end', and 'variable'.

    Returns
    -------
    xarray.Dataset
        Weather data with time, latitude, and longitude dimensions, chunked as requested.
    """

    # Build request from base + user overrides
    request = {"product_type": "meteo_api", "format": "direct_download"}
    request.update(updates)

    # Generate list of (lon, lat) coordinate pairs
    grid = np.meshgrid(request["coords"]["x"], request["coords"]["y"])
    coords = pd.DataFrame(
        zip(grid[0].flatten(), grid[1].flatten()), columns=["longitude", "latitude"]
    )

    # Calculate time and variable counts
    start_date = request["start"].strftime("%Y-%m-%d")
    end_date = request["end"].strftime("%Y-%m-%d")
    nr_days = abs((pd.to_datetime(start_date) - pd.to_datetime(end_date)).days)
    nr_variables = len(request["variable"])
    nr_locations = len(coords)

    # Estimate request weight based on Open-Meteo's internal model
    weight_of_full_api_request = (
        max(nr_variables / 10, (nr_variables / 10) * (nr_days / 14)) * nr_locations
    )

    # Dynamically determine chunk size based on rate limit thresholds
    if weight_of_full_api_request <= MINUTE_LIMIT:
        chunk_size = MINUTE_LIMIT / 10
    elif weight_of_full_api_request <= HOUR_LIMIT:
        chunk_size = HOUR_LIMIT / 10
    else:
        chunk_size = DAY_LIMIT / 10

    chunk_size = int(max(1, chunk_size))  # Ensure chunk_size ≥ 1 and integer

    logger.info(f"Meteo-API: Downloading variables\n\t{request['variable']}\n")
    logger.info(
        f"Meteo-API: Expected request weight of full request: {weight_of_full_api_request}"
    )

    # Loop through coordinate grid in blocks and request data
    data = []
    for i in range(0, len(coords), chunk_size):
        coord_chunk = coords.iloc[i : i + chunk_size]

        # Prepare API parameters for the current chunk
        params = {
            "longitude": coord_chunk["longitude"].tolist(),
            "latitude": coord_chunk["latitude"].tolist(),
            "hourly": request["variable"],
            "wind_speed_unit": "ms",
            "start_date": start_date,
            "end_date": end_date,
        }

        try:
            responses = openmeteo.weather_api(url, params=params)
        except Exception as e:
            logger.info(f"{e}")
            try:
                # Extract and classify rate limiting error
                rate_limiting_error = list(e.args)[0]["reason"]
                match = re.search(
                    r"(Minutely|Hourly|Daily) API request limit", rate_limiting_error
                )
                if match:
                    apply_rate_limiting(error=match[0])
                else:
                    apply_rate_limiting(error=None)
                # Retry after delay
                responses = openmeteo.weather_api(url, params=params)
            except Exception as e:
                # Skip this chunk on repeated failure
                logger.error(
                    f"Meteo-API: Failed to fetch data for block starting at "
                    f"({coord_chunk.loc[0, 'longitude']}, {coord_chunk.loc[0, 'latitude']}): {e}"
                )
                continue

        # Parse chunked response and append results
        data.extend(parse_meteo_responses(responses, params))

    # Combine all parsed DataFrames into a single xarray dataset
    ds = pd.concat(data).to_xarray()
    ds = _rename_and_clean_coords(ds)
    ds = ds.chunk(chunks=chunks)

    return ds


def parse_meteo_responses(responses, params):
    """
    Parse raw Open-Meteo API responses into a list of DataFrames.

    Parameters
    ----------
    responses : list
        List of Open-Meteo response objects, one per coordinate.
    params : dict
        Parameters used in the API call, containing latitude, longitude, and variable info.

    Returns
    -------
    list of pd.DataFrame
        Each DataFrame contains one location’s weather data with time as index.
    """

    data = []
    for j, response in enumerate(responses):
        # Reconstruct time index based on interval
        range_start = pd.to_datetime(response.Hourly().Time(), unit="s")
        range_end = pd.to_datetime(response.Hourly().TimeEnd(), unit="s")
        date_range = pd.date_range(
            start=range_start,
            end=range_end,
            freq=pd.Timedelta(seconds=response.Hourly().Interval()),
            inclusive="left",
        )

        # Initialize empty DataFrame for the current location
        response_df = pd.DataFrame(columns=params["hourly"])
        response_df["time"] = date_range
        response_df["latitude"] = params["latitude"][j]
        response_df["longitude"] = params["longitude"][j]
        response_df = response_df.set_index(["time", "latitude", "longitude"])

        # Fill in variable values
        for i, param in enumerate(params["hourly"]):
            response_df[param] = response.Hourly().Variables(i).ValuesAsNumpy()

        data.append(response_df)

    return data


def apply_rate_limiting(error=None):
    """
    Apply appropriate sleep duration based on API rate-limiting error.

    Parameters
    ----------
    error : str or None
        One of 'Minutely API request limit', 'Hourly API request limit',
        'Daily API request limit', or None (fallback delay).

    Behavior
    --------
    - Sleeps 60s for minutely errors
    - Sleeps 1h for hourly errors
    - Sleeps until 00:05 next day for daily errors
    - Sleeps 2 minutes as fallback
    """

    now = datetime.datetime.now()
    midnight = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(
        days=1, minutes=5
    )
    time_until_midnight = (midnight - now).total_seconds()

    sleep_times = {
        None: 120,  # Fallback for unknown errors
        "Minutely API request limit": 60,
        "Hourly API request limit": 60 * 60,
        "Daily API request limit": time_until_midnight,
    }

    sleep_time = sleep_times[error]
    logger.info(f"Sleeping for {sleep_time / 60:.2f} minutes.")
    time.sleep(sleep_time)


def retrieve_era5_data(url, product, chunks=None, tmpdir=None, lock=None, **updates):
    """
    Download data like ERA5 from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://cds-beta.climate.copernicus.eu/requests?tab=all
    """
    request = {
        "product_type": ["reanalysis"],
        "data_format": "netcdf",
        "download_format": "zip",
    }

    request.update(updates)

    assert {"year", "month", "variable"}.issubset(request), (
        "Need to specify at least 'variable', 'year' and 'month'"
    )

    client = cdsapi.Client(
        url=url, info_callback=logger.debug, debug=logging.DEBUG >= logging.root.level
    )
    result = client.retrieve(product, request)

    if lock is None:
        lock = nullcontext()

    with lock:
        fd, target_zip = mkstemp(suffix=".zip", dir=tmpdir)
        os.close(fd)

        # Inform user about data being downloaded as "* variable (year-month)"
        timestr = f"{request['year']}-{request['month']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info(f"CDS: Downloading variables\n\t{varstr}\n")
        result.download(target_zip)

        # Open the .zip file in memory
        with zipfile.ZipFile(target_zip, "r") as zf:
            # Identify .nc files inside the .zip
            nc_files = [name for name in zf.namelist() if name.endswith(".nc")]

            if not nc_files:
                raise FileNotFoundError(
                    "No .nc files found in the downloaded .zip archive."
                )

            if len(nc_files) == 1:
                # If there's only one .nc file, read it into memory
                with zf.open(nc_files[0]) as nc_file:
                    # Pass the in-memory file-like object to Xarray
                    ds = xr.open_dataset(
                        io.BytesIO(nc_file.read()), chunks=chunks or {}
                    )

            else:
                # If multiple .nc files, combine them using Xarray
                datasets = []
                for nc_file in nc_files:
                    with zf.open(nc_file) as file:
                        dataset = xr.open_dataset(
                            io.BytesIO(file.read()), chunks=chunks or {}
                        )

                        if "expver" in dataset.variables:
                            dataset = dataset.drop_vars(
                                ["expver", "number"], errors="ignore"
                            )

                        datasets.append(dataset)

                ds = xr.merge(datasets)

    if tmpdir is None:
        logging.debug(f"Adding finalizer for {target_zip}")
        weakref.finalize(ds._file_obj._manager, noisy_unlink, target_zip)

    return ds


def retrieval_times_era5_forecast(
    coords, initialization_time, static=False, monthly_requests=False
):
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

    # Convert time coordinates to a pandas Index
    time = coords["time"].to_index()
    frequency = time.freq

    # Determine the latest available ERA5 data time based on initialization time and required delay
    latest_era5_time = pd.Timestamp(initialization_time) + ERA5_DELAY

    # Round up to the next full day and subtract 1 hour to align with ERA5 update frequency
    latest_era5_time = latest_era5_time.ceil("D") - pd.Timedelta(hours=1)

    # Define the minimum required time horizon for ERA5 downloads (last 24 hours)
    minimum_era5_time_horizon = pd.date_range(
        start=latest_era5_time
        - pd.Timedelta(days=1),  # One day before the latest available time
        end=latest_era5_time,  # Up to the latest available time
        freq=frequency,  # Maintain original time frequency
    )

    # Merge the existing time index with the minimum ERA5 time horizon, avoiding duplicates
    time = time.union(minimum_era5_time_horizon)

    # Ensure a continuous time index by filling missing values based on the determined frequency
    complete_time_range = pd.date_range(
        start=time.min(), end=time.max(), freq=frequency
    )

    # Keep only timestamps up to the latest available ERA5 time
    time = complete_time_range[complete_time_range <= latest_era5_time]

    if static:
        return {
            "year": [str(time[0].year)],
            "month": [str(time[0].month).zfill(2)],
            "day": [str(time[0].day).zfill(2)],
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
                    "month": [str(month).zfill(2)],
                    "day": list(
                        t[t.month == month].day.unique().astype(str).str.zfill(2)
                    ),
                    "time": ["%02d:00" % h for h in t[t.month == month].hour.unique()],
                }
                times.append(query)
        else:
            query = {
                "year": [str(year)],
                "month": list(t.month.unique().astype(str).str.zfill(2)),
                "day": list(t.day.unique().astype(str).str.zfill(2)),
                "time": ["%02d:00" % h for h in t.hour.unique()],
            }
            times.append(query)
    return times


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
    Retrieve data from Meteo API.

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
    initialization_time = creation_parameters["init_time"]

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params_meteo = {
        "product": "meteo_historic_forecast_api_data",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": [cutout.dx, cutout.dy],
        "tmpdir": tmpdir,
        "lock": lock,
    }

    retrieval_params_era5 = {
        "product": "reanalysis-era5-single-levels",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": [cutout.dx, cutout.dy],
        "tmpdir": tmpdir,
        "lock": lock,
    }

    # Get fast changing variabels from meteo forecast
    func_meteo = globals().get(f"get_data_meteo_{feature}")
    logger.info(f"Requesting data for feature {feature} from meteo...")

    if func_meteo is not None:
        datasets_meteo = func_meteo(
            {
                **retrieval_params_meteo,
                **{
                    "start": coords["time"].to_index()[0],
                    "end": coords["time"].to_index()[-1],
                    "coords": coords,
                },
            }
        )
    else:
        datasets_meteo = xr.Dataset()

    def retrieve_once(time):
        ds = func_era5({**retrieval_params_era5, **time})
        return ds

    # Get missing and slow changing variabels from era5 data and interpolation
    func_era5 = globals().get(f"get_data_era5_{feature}")

    if func_era5 is not None:
        logger.info(f"Requesting addtional data for feature {feature} from era5...")

        if feature in static_features:
            return retrieve_once(
                retrieval_times_era5_forecast(coords, initialization_time, static=True)
            ).squeeze()

        time_chunks = retrieval_times_era5_forecast(
            coords, initialization_time, monthly_requests=monthly_requests
        )
        if concurrent_requests:
            delayed_datasets = [delayed(retrieve_once)(chunk) for chunk in time_chunks]
            datasets_era5 = compute(*delayed_datasets)
        else:
            datasets_era5 = map(retrieve_once, time_chunks)

        datasets_era5 = xr.concat(datasets_era5, dim="time")

    else:
        datasets_era5 = xr.Dataset()

    # Combine datasets and calculate the required variables
    combine_func = globals().get(f"combine_data_{feature}")

    logger.info(f"Combine meteo and era5 datasets for feature {feature}...")

    if combine_func is not None:
        datasets = combine_func(
            datasets_meteo, datasets_era5, cutout.data.interp_s, cutout.data.interp_t
        )
    else:
        datasets = xr.merge([datasets_meteo, datasets_era5])

    sanitize_func = globals().get(f"sanitize_{feature}")
    if sanitize and sanitize_func is not None:
        # Sanitize the data after interpolation to remove residuals
        datasets = sanitize_func(datasets)

    return datasets.sel(time=coords["time"])
