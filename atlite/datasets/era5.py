# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and curating data from ECMWFs ERA5 dataset (via CDS).

For further reference see
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
"""

import logging
import os
import io
import zipfile
import warnings
import weakref
from tempfile import mkstemp

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import arctan2, sqrt
from numpy import atleast_1d

from atlite.gis import maybe_swap_spatial_dims
from atlite.pv.solar_position import SolarPosition

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

# Set url for data download, this allows to switch to different data 
# sources more easily.
era5_url = 'https://cds.climate.copernicus.eu/api'

# Model and CRS Settings
crs = 4326

features = {
    "height": ["height"],
    "wind": ["wnd100m", "wnd_shear_exp", "wnd_azimuth", "roughness"],
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

requirements = {'x': slice(-90, 90, 0.25),
                'y': slice(-90, 90, 0.25),
                'offset': (pd.Timestamp('1940-01-01')-pd.Timestamp.utcnow().replace(tzinfo=None).floor("h")),
                'forecast': pd.Timedelta(hours=-5*24),
                'dt': pd.Timedelta(hours=1),
                'parallel': True,
                }


def _checkModuleRequirements(x, y, time, time_now, **kwargs):
    """
    Load and check the data requirements for a given module.
    
    Parameters:
    x (slice): Defines the start, stop, and step values for the x-dimension.
    y (slice): Defines the start, stop, and step values for the y-dimension.
    time (slice): Defines the start, stop, and step values for the time dimension.
    **kwargs: Additional optional parameters.
    """
    
    # Extract start, stop, and step values for x
    x_start, x_stop, x_step = x.start, x.stop, x.step
    
    # Adjust x range based on module requirements
    if requirements['x'].start > x.start:
        x_start = requirements['x'].start 
    if requirements['x'].stop < x.stop:
        x_stop = requirements['x'].stop 
    if requirements['x'].step > x.step:
        x_step = requirements['x'].step
    
    x = slice(x_start, x_stop, x_step)
    
    # Extract start, stop, and step values for y
    y_start, y_stop, y_step = y.start, y.stop, y.step
    
    # Adjust y range based on module requirements
    if requirements['y'].start > y.start:
        y_start = requirements['y'].start 
    if requirements['y'].stop < y.stop:
        y_stop = requirements['y'].stop 
    if requirements['y'].step > y.step:
        y_step = requirements['y'].step
    
    y = slice(y_start, y_stop, y_step)
    
    
    # Extract time range parameters
    time_start = time.start
    time_stop = time.stop
    time_step = time.step
    
    # Check forecast feasibility
    feasible_start = time_now + requirements['offset']
    feasible_end = time_now + requirements['forecast']
    
    # Ensure time_start is within feasible bounds
    if time_start < feasible_start:
        logger.error(f"The required forecast start time {time_start} exceeds the model requirements.")
        logger.error(f"The minimum start time of the forecast for {time_now} is {feasible_start}.")
        logger.error(f"The maximum historical offset of the forecast is {requirements['offset']}.")
        raise ValueError(f"Invalid forecast start time: {time_start}. Must be >= {feasible_start}.")
        
    if time_start >= feasible_end:
        logger.error(f"The required forecast start time {time_start} exceeds the model requirements.")
        logger.error(f"The maximum start time of the forecast for {time_now} needs to be smaller than {feasible_end}.")
        raise ValueError(f"Invalid forecast start time: {time_start}. Must be < {feasible_end}.")

    # Ensure time_stop is greater than time_start
    if time_stop <= time_start:
        logger.error(f"The required forecast end time {time_stop} exceeds the model requirements.")
        logger.error(f"The minimum end time of the forecast for {time_now} needs to be larger than {time_start}.")
        raise ValueError(f"Invalid forecast end time: {time_stop}. Must be > {time_start}.")
          
    # Ensure time_stop is greater than time_start
    if time_stop > feasible_end:
        logger.error(f"The required forecast end time {time_stop} exceeds the model requirements.")
        logger.error(f"The maximum end time of the forecast for {time_now} is {feasible_end}.")
        logger.error(f"The maximum forecast horizon is {requirements['forecast']}.")
        raise ValueError(f"Invalid forecast end time: {time_stop}. Must be <= {feasible_end}.")  
         
    # Ensure time step is within required limits
    if (time_step is pd.Timedelta(None)) or (time.step < requirements['dt']):
        logger.warning(f"The required temporal forecast resolution {time_step} exceeds the model requirements.")
        logger.warning(f"The minimum temporal resolution of the forecast is {requirements['dt']}.")
        logger.info(f"Set the temporal forecast resolution to the minimum: {requirements['dt']}.") 
        time_step = requirements['dt']
        
    time = slice(time_start, time_stop, time_step)
    
    # Retrieve parallel processing setting from requirements
    parallel = requirements['parallel']
    
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
    ds = ds.drop_vars(["expver", "number"], errors="ignore")

    return ds


def get_data_wind(retrieval_params):
    """
    Get wind data for given retrieval parameters.
    """
    ds = retrieve_data(
        url=era5_url,
        variable=[
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "forecast_surface_roughness",
        ],
        **retrieval_params,
    )
    ds = _rename_and_clean_coords(ds)

    for h in [10, 100]:
        ds[f"wnd{h}m"] = sqrt(ds[f"u{h}"] ** 2 + ds[f"v{h}"] ** 2).assign_attrs(
            units=ds[f"u{h}"].attrs["units"], long_name=f"{h} metre wind speed"
        )
    ds["wnd_shear_exp"] = (
        np.log(ds["wnd10m"] / ds["wnd100m"]) / np.log(10 / 100)
    ).assign_attrs(units="", long_name="wind shear exponent")

    # span the whole circle: 0 is north, π/2 is east, -π is south, 3π/2 is west
    azimuth = arctan2(ds["u100"], ds["v100"])
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)

    ds = ds.drop_vars(["u100", "v100", "u10", "v10", "wnd10m"])
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
        url=era5_url,
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
        url=era5_url,
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
    ds = retrieve_data(
        url=era5_url,
        variable=["runoff"],
        **retrieval_params)

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
    ds = retrieve_data(
        url=era5_url,
        variable=["geopotential"],
        **retrieval_params)

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
                    "day": list(t[t.month == month].day.unique().astype(str).str.zfill(2)),
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


def noisy_unlink(path):
    """
    Delete file at given path.
    """
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def retrieve_data(url, product, chunks=None, tmpdir=None, lock=None, **updates):
    """
    Download data like ERA5 from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://cds-beta.climate.copernicus.eu/requests?tab=all
    """    
    request = {"product_type": ["reanalysis"],
               "data_format": "netcdf", 
               "download_format": "zip"}
    
    request.update(updates)

    assert {"year", "month", "variable"}.issubset(
        request
    ), "Need to specify at least 'variable', 'year' and 'month'"

    client = cdsapi.Client(
        url = url,
        info_callback=logger.debug, 
        debug=logging.DEBUG >= logging.root.level
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
                raise FileNotFoundError("No .nc files found in the downloaded .zip archive.")
     
            if len(nc_files) == 1:
                # If there's only one .nc file, read it into memory
                with zf.open(nc_files[0]) as nc_file:
                    # Pass the in-memory file-like object to Xarray
                    ds = xr.open_dataset(io.BytesIO(nc_file.read()), chunks=chunks or {})
                    
            else:
                # If multiple .nc files, combine them using Xarray
                datasets = []
                for nc_file in nc_files:
                    with zf.open(nc_file) as file:
                        datasets.append(xr.open_dataset(io.BytesIO(file.read()), chunks=chunks or {}))
                # Combine datasets along temporal dimension
                ds = xr.merge(datasets) 
        
    if tmpdir is None:
        logging.debug(f"Adding finalizer for {target_zip}")
        weakref.finalize(ds._file_obj._manager, noisy_unlink, target_zip)

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
