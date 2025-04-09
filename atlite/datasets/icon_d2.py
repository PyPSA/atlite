# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for downloading and curating data from DWD ICON-D2 dataset (via ODS).

For further reference see:
https://www.dwd.de/DE/leistungen/nwv_icon_d2_modelldokumentation/nwv_icon_d2_modelldokumentation.html
"""

import os
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import requests
import logging

from pathlib import Path
from retry import retry
from bz2 import decompress
from bs4 import BeautifulSoup

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

# URL for accessing Open Data from DWD (Deutscher Wetterdienst)
dwd_url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/"

# ICON-D2 model runs are available at fixed intervals: 00, 03, 06, 09, 12, 15, 18, 21 UTC
model_run_hours = np.array([0, 3, 6, 9, 12, 15, 18, 21])

# Averaging window of different model runs
averaging_window = 24 #hours

# Coordinate Reference System (CRS) used for geospatial data
crs = 4326

# Dictionary defining available meteorological features and their associated data fields
features = {
    "height": ["height"],  # Elevation data
    "wind": ["wnd100m", "wnd_azimuth", "roughness"],  # Wind speed, direction, and surface roughness
    "influx": [
        "influx_toa",  # Top-of-atmosphere solar radiation
        "influx_direct",  # Direct solar radiation
        "influx_diffuse",  # Diffuse solar radiation
        "albedo",  # Surface reflectivity
        "solar_altitude",  # Solar altitude angle
        "solar_azimuth",  # Solar azimuth angle
    ],
    "temperature": ["temperature", "soil temperature"],  # Air and soil temperature
    "runoff": ["runoff"],  # Surface water runoff
}

# Features that remain constant over time
static_features = {"height"}

# Model requirements specifying spatial and temporal constraints
requirements = {
    'x': slice(-3.84, 20.21, 0.02),  # Longitude range with resolution
    'y': slice(43.19, 57.63, 0.02),  # Latitude range with resolution
    'offset': pd.Timedelta(hours=-18),  # Time offset for forecast initialization
    'forecast': pd.Timedelta(hours=48),  # Maximum forecast range
    'dt': pd.Timedelta(hours=1),  # Temporal resolution of data
    'parallel': True,  # Flag for enabling parallel processing
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
        # logger.info(f"Set the start time to the minimum start time {feasible_start} and proceed.") 
        # time_start = time_now + requirements['offset']
        
    if time_start >= feasible_end:
        logger.error(f"The required forecast start time {time_start} exceeds the model requirements.")
        logger.error(f"The maximum start time of the forecast for {time_now} needs to be smaller than {feasible_end}.")
        raise ValueError(f"Invalid forecast start time: {time_start}. Must be < {feasible_end}.")
        # logger.info(f"Set the start time to the minimum start time {feasible_start} and proceed.") 
        # time_start = time_now + requirements['offset']

    # Ensure time_stop is greater than time_start
    if time_stop <= time_start:
        logger.error(f"The required forecast end time {time_stop} exceeds the model requirements.")
        logger.error(f"The minimum end time of the forecast for {time_now} needs to be larger than {time_start}.")
        raise ValueError(f"Invalid forecast end time: {time_stop}. Must be > {time_start}.")
        # logger.info(f"Set the end time to the maximum end time {feasible_end} and proceed.") 
        # time_stop = time_now + requirements['forecast']      
          
    # Ensure time_stop is greater than time_start
    if time_stop > feasible_end:
        logger.error(f"The required forecast end time {time_stop} exceeds the model requirements.")
        logger.error(f"The maximum end time of the forecast for {time_now} is {feasible_end}.")
        logger.error(f"The maximum forecast horizon is {requirements['forecast']}.")
        raise ValueError(f"Invalid forecast end time: {time_stop}. Must be <= {feasible_end}.")
        # logger.info(f"Set the end time to the maximum end time {feasible_end} and proceed.") 
        # time_stop = time_now + requirements['forecast']      
         
    # # Check if forecast hours exceed limits
    # forecastHours = (time_stop - time_now)
    # if forecastHours > requirements['forecast']:
    #     logger.error(f"The end time of the forecast {time_stop} exceedes the model requirements.")
    #     logger.error(f"The maximum end time of the forecast for {time_now} is {feasible_end}.")
    #     logger.error(f"The required forecast horizon {forecastHours} exceeds the maximum forecast horzion {requirements['forecast']}.")
    #     # logger.info(f"Set it to maximum forecast hours of {requirements['forecast']} hours.")   
    #     # forecastHours = requirements['forecast']
    #     # time_stop = time_now + forecastHours
        
    # # Check if offset is within required limits
    # offset = (time_start - time_now)    
    # if offset < requirements['offset']:
    #     logger.error(f"The start time of the forecast {time_start} exceeds model requirements.")
    #     logger.error(f"The minimum start time of the forecast for {time_now} is {feasible_start}.")
    #     logger.warning(f"Forecast offset of {offset} hours is below model requirements.")
    #     logger.info(f"Set it to minimum offset of {requirements['offset']} hours.")   
    #     offset = requirements['offset']
    #     time_start = time_now + offset
            
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


def _getCurrentRun(time):
    '''
    Determines the most recent available model run based on the current time.
    The latest run is fully available approximately 2 hours after initialization.
    To ensure the model run is successfully uploaded, the download delay time is set to 3 hours.

    This code was adapted from: https://github.com/prayer007/dwdGribExtractor/tree/main

    Parameters
    ----------
    time : datetime
        The current datetime in UTC.
        
    Returns
    -------
    datetime
        The timestamp of the most recent available model run, floored to the hour.
    '''
    download_delay = 3  # Delay in hours before the run is fully available
    
    # Adjust the current time by the delay to ensure availability
    adjusted_time = time - pd.Timedelta(hours=download_delay)

    # Find the most recent available run by flooring to the nearest model run hour
    run_hour = max(hour for hour in model_run_hours if hour <= adjusted_time.hour)
    
    # Construct the correct model run time
    run_time = adjusted_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)

    return run_time



def _createDownloadUrl(url, var, field, run, hours):
    '''
    Generates a list of download URLs for meteorological data from the DWD server.
    The function scrapes the available files for a given variable and model run,
    filtering based on field type, forecast hours, and model levels.
    
    This code was adopted from: https://github.com/prayer007/dwdGribExtractor/tree/main
    
    Parameters
    ----------
    url : string
        Base URL of the DWD data server.
    var : string
        The variable name, optionally including levels separated by '/'.
    field : string
        The field parameter: 'time-invariant' (static), 'soil-level' (162cm), 
        'model-level' (62;63), or 'single-level' (2D field).
    run : string
        Model run identifier.
    hours : int
        Maximum forecast hours to retrieve.
    
    Returns
    -------
    list
        List of filtered download URLs.
    '''   
        
    # Extract variable name and associated levels
    levels = pd.Series(var.split('/')[1:]).astype(int)  # Convert levels to integers
    var = var.split('/')[0]  # Extract variable name
    
    # Construct the data URL based on provided parameters
    data_url = "{url}{run}/{var}/".format(url=url, var=var, run=run)
    
    # Send an HTTP GET request to fetch available files
    response = requests.get(data_url)
    
    # Raise an error if the request fails
    response.raise_for_status()  
    
    # Parse the HTML content to extract links
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all anchor tags ('a') representing file links
    link_tags = soup.find_all('a')
    
    # Initialize an empty list to store the file URLs
    urls = []
    
    # Iterate through all link tags, extract URLs, and store them
    for tag in link_tags:
        link = tag.get('href')  # Extract the hyperlink reference
    
        if link:
            # Construct the full URL by appending the relative link to the base URL
            full_url = data_url + link
            urls.append(full_url)
    
    # Convert the list of URLs into a Pandas Series for easy filtering
    urls = pd.Series(urls)
    
    # Filter URLs to retain only those containing 'regular-lat-lon' grid format
    urls = urls[urls.str.contains('regular-lat-lon')]
   
    # Further filter URLs based on the specified model field
    urls = urls[urls.str.contains(field)]
    
    # Apply forecast time horizon filter (excluding 'time-invariant' fields)
    if field != 'time-invariant':
        urls = urls[urls.str.findall(r"\_(\d{3})\_").str[0].astype(int) <= hours]       
    
    # Filter URLs based on model levels, if specified
    if not levels.empty:
        url_mask = pd.Series(index=urls.index, data=False)  # Initialize boolean mask
        for level in levels:
            url_mask += urls.str.contains(f"_{level}_")  # Check if URL contains level
        urls = urls[url_mask]  # Apply filter    

    # Convert filtered URLs back to a list
    urls = list(urls)
        
    return urls


def _deaverage(da):
    '''
    Converts a temporally averaged data array into individual time-step values.
    Each time step's original value is reconstructed by reversing the cumulative averaging process.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array with a time dimension containing cumulative averages.
    
    Returns
    -------
    xarray.DataArray
        Data array with de-averaged values.
    '''
    # Create an integer index for time, matching da's shape
    time_index = xr.DataArray(np.arange(1, da.sizes["time"] + 1), dims="time", coords={"time": da.time})

    # Apply the reverse operation: Ψ_inst(t) = t * Ψ(t) - (t-1) * Ψ(t-1)
    da_instantaneous = (time_index * da - (time_index - 1) * da.shift(time=1, fill_value=0))
    
    # Fill the first timestep with NaNs, since it is always zero
    da_instantaneous = da_instantaneous.where(da_instantaneous.time != da_instantaneous.time[0], np.nan)

    return da_instantaneous  
    

def _deaccumulate(da):
    '''
    Converts accumulated data into time-step differences.
    This function takes an accumulated dataset and calculates the incremental
    values between consecutive time steps.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array with a time dimension containing accumulated values.
    
    Returns
    -------
    xarray.DataArray
        Data array with de-accumulated values (time-step differences).
    '''
    
    # Apply the reverse operation: Ψ_inst(t) = Ψ(t) - Ψ(t-1)
    da_instantaneous = da - da.shift(time=1, fill_value=0)
    
    # Fill the first timestep with NaNs, since it is always zero
    da_instantaneous = da_instantaneous.where(da_instantaneous.time != da_instantaneous.time[0], np.nan)

    return da_instantaneous  


def _average_duplicate_times(ds): 
    """
    Averages duplicate timestamps in an xarray Dataset.

    Given an xarray Dataset with duplicated timestamps (after concatenation),
    this function computes the mean over all datasets that share the same time index.
    Unique timestamps remain unchanged.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset that contains duplicated time indices.

    Returns
    -------
    xarray.Dataset
        A dataset where duplicate timestamps are averaged and unique timestamps are preserved.
    """

    # Step 1: Compute the mean for duplicate timestamps while keeping unique ones
    ds_mean = ds.groupby("time").mean(dim="time", keep_attrs=True)

    # Step 2: Preserve dataset and variable attributes from the original dataset
    ds_mean.attrs = ds.attrs  # Preserve global dataset attributes
    for var in ds_mean.data_vars:
        ds_mean[var].attrs = ds[var].attrs  # Preserve variable attributes

    return ds_mean


def _mainDataCollector(url, var, field, forecast, offset, coords, tmpdir):
    '''
    Downloads meteorological data for a given variable and processes it accordingly.
    
    This function retrieves data from the specified URL, processes it to rename and clean
    coordinates, and applies de-averaging or de-accumulation where necessary based on
    the GRIB step type.
    
    Parameters
    ----------
    url : string
        The base URL for downloading data.
    var : string
        The variable name to be downloaded.
    field : string
        The field type, e.g., 'time-invariant' (static), 'soil-level' (e.g., 162cm),
        'model-level' (e.g., 62;63), or 'single-level' (2D field).
    forecast : int
        The number of forecast hours to retrieve.
    offset : int
        The forecast offset time in hours.
    coords : atlite.Cutout.coords
        The spatial coordinates where data is required.
    tmpdir : string
        Path to the temporary directory where downloaded files are stored.
    
    Returns
    -------
    xarray.Dataset
        Processed dataset containing the collected meteorological data.
    '''        
    
    # Extract the most recent forecast run time
    latestRun = forecast[0]

    # Filter only the previous runs before the latest run
    previousRuns = offset[offset < latestRun]

    # Keep only entries that align with ICON model run hours
    previousRuns = previousRuns[previousRuns.hour.isin(model_run_hours)].sort_values(ascending=True)
       
    if len(previousRuns) > 0:
        # Get the hour of the earliest previous run
        first_prev_hour = previousRuns[0].hour
    
        # Find the previous index in `model_run_hours`
        prev_idx = np.where(model_run_hours == first_prev_hour)[0][0] - 1
    
        # Compute the adjusted previous run
        previousRun = pd.DatetimeIndex([previousRuns[0].replace(
            hour=model_run_hours[prev_idx], minute=0, second=0, microsecond=0
        )])
    
        # Add previousRun to previousRuns, ensuring uniqueness
        previousRuns = previousRuns.union(previousRun).sort_values(ascending=True)

    # Create a list of runs including the latest run and previous runs
    # Use an averaging_window of X hours for all previous runs to average the results
    runs = [(run.strftime("%H"), averaging_window) for run in previousRuns] + [(latestRun.strftime("%H"), len(forecast))]

    # # Generate download URLs for the specified variable and field
    # urls = []
    # for run, hours in runs:
    #     urls = urls + _createDownloadUrl(url, var, field, run, hours)
        
    # urls = pd.Series(urls).unique()
        
    ds_temps = []  # List to store temporary datasets
    
    for run, hours in runs:
        
        # Generate download URLs for the specified variable and field
        urls = _createDownloadUrl(url, var, field, run, hours)
        
        # Download and collect the main dataset for the given variable
        ds_temps.append(_download(urls, tmpdir)) 
    

    # Concatenate along the time dimension, keeping also duplicated timestamps
    ds = xr.concat(ds_temps, dim="time")
    
    # Average duplicates for smooth forecast transitioning
    ds = _average_duplicate_times(ds)
            
    # # Download and collect the main dataset for the given variable
    # ds_temp = _mainDataCollector(url, var, field, forecast, offset, tmpdir)   
    
    # Rename and clean coordinate labels for consistency
    # ds_temp = _rename_and_clean_coords(ds_temp)
    
    # # Iterate through all data variables in the dataset
    # for ds_var in list(ds_temp.data_vars):
    #     # If the variable is an averaged quantity, apply de-averaging
    #     if ds_temp[ds_var].attrs['GRIB_stepType'] == 'avg':
    #         ds_temp[ds_var] = _deaverage(ds_temp[ds_var])
                       
    #     # If the variable is an accumulated quantity, apply de-accumulation
    #     elif ds_temp[ds_var].attrs['GRIB_stepType'] == 'accum':
    #         ds_temp[ds_var] = _deaccumulate(ds_temp[ds_var])

    # Return the processed dataset
    return ds


def _interpolate(ds, static, coords, grid, interp_s, interp_t):
    '''
    Interpolates a dataset to match specific latitude and longitude coordinates.
    
    If the data is not static, it first interpolates temporally. Then, it applies
    spatial interpolation using binning to adjust the data to the grid resolution.
    
    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be interpolated.
    static : bool
        Whether the dataset contains static variables (i.e., no time dimension).
    coords : dict
        Dictionary containing target coordinate values for interpolation.
    grid : tuple
        Grid resolution in (x, y) directions for spatial binning.
    interp_s : string
        Spatial interpolation method (not used in the function but can be applied elsewhere).
    interp_t : string
        Temporal interpolation method to be used.
    
    Returns
    -------
    xarray.Dataset
        The interpolated dataset adjusted to the target spatial and temporal resolution.
    '''
    
    # Perform temporal interpolation if the data is not static
    if not static:
        try:
            ds = ds.interp(time=coords['time'].values, 
                           method=interp_t, 
                           kwargs={"fill_value": "extrapolate"})
        except ValueError:
            logger.info(f"Interpolation: Not enough supporting points for used interpolation method {interp_t}.")
            logger.info("Interpolation method is set to 'nearest' instead.")
            ds = ds.interp(time=coords['time'].values, 
                           method="nearest", 
                           kwargs={"fill_value": "extrapolate"})
    
    # Create bin edges and labels for x-coordinates
    x_bins = coords['x'].values
    x_bins = np.insert(x_bins, 0, np.round(x_bins[0] - grid[0], 8), axis=0)  # Extend bin range
    x_bins_label = np.round(x_bins[:-1] + grid[0], 8)  # Compute bin centers
    
    # Create bin edges and labels for y-coordinates
    y_bins = coords['y'].values
    y_bins = np.insert(y_bins, 0, np.round(y_bins[0] - grid[1], 8), axis=0)  # Extend bin range
    y_bins_label = np.round(y_bins[:-1] + grid[1], 8)  # Compute bin centers
    
    # Store original dataset attributes
    attrs = ds.attrs
    
    # Perform spatial binning by grouping data into bins along x and y dimensions and computing the mean
    ds = ds.groupby_bins("x", x_bins, labels=x_bins_label).mean(dim="x")
    ds = ds.groupby_bins("y", y_bins, labels=y_bins_label).mean(dim="y")
    
    # Rename bins to standard coordinate names
    ds = ds.rename({'y_bins': 'y', 'x_bins': 'x'})
    
    # Reassign original dataset attributes
    ds = ds.assign_attrs(attrs)
    
    return ds


@retry(tries=5, delay=5, backoff=2, logger=logger)
def _urlopen_with_retry(data_url, tmpfp, engine='cfgrib', **kwargs):
    '''
    Attempts to download and decompress a dataset file with automatic retry on failure.
    
    This function fetches data from a given URL, retries up to five times in case of failure,
    and decompresses the response content before saving it to a temporary file.
    
    Parameters
    ----------
    data_url : string
        The URL from which data should be downloaded.
    tmpfp : string
        The file path where the downloaded content will be temporarily stored.
    
    Returns
    -------
    tuple
        - resp (requests.Response): The HTTP response object from the request.
        - ds (xarray.Dataset): The dataset extracted from the downloaded file.
    '''
    
    # Send an HTTP GET request to the data URL with a timeout of 5 seconds
    resp = requests.get(data_url, timeout=5)

    # Check if the request was successful (HTTP 200 OK)
    if resp.status_code == 200:
        # Open the specified temporary file and write the decompressed response content
        with open(tmpfp, 'wb') as f:
            f.write(decompress(resp.content))
    else:
        # Raise an error if the response was unsuccessful
        raise ValueError(f"Error in response: {resp.reason}, status code: {resp.status_code}")
    
    # Load the downloaded file as an xarray dataset using the 'cfgrib' engine
    ds = xr.open_dataset(tmpfp, engine=engine)   
    
    # Return both the HTTP response object and the loaded dataset
    return resp, ds


def _download(urls, tmpdir=None):
    '''
    Collects meteorological data for all timesteps of a given variable.
    
    This function retrieves data files, processes them, and merges them into
    a single dataset. It determines the latest available runs, downloads
    the necessary files, and structures them according to the expected format.
    
    Parameters
    ----------
    url : string
        The base URL for downloading data.
    var : string
        The variable name to be downloaded.
    field : string
        The field type, e.g., 'time-invariant' (static), 'soil-level' (e.g., 162cm),
        'model-level' (e.g., 62;63), or 'single-level' (2D field).
    forecast : list of datetime
        List of forecast time steps.
    offset : numpy array
        Array representing the forecast offsets.
    tmpdir : string, optional
        Temporary directory for storing downloaded files.
    
    Returns
    -------
    xarray.Dataset
        Merged dataset containing the collected meteorological data.
    '''    
    
    # # Extract the most recent forecast run time
    # latestRun = forecast[0]

    # # Filter only the previous runs before the latest run
    # previousRuns = offset[offset < latestRun]

    # # Keep only entries that align with ICON model run hours
    # previousRuns = previousRuns[previousRuns.hour.isin(model_run_hours)]

    # # Create a list of runs including the latest run and previous runs
    # runs = [(latestRun.strftime("%H"), len(forecast))] + [(run.strftime("%H"), 3) for run in previousRuns]

    # # Generate download URLs for the specified variable and field
    # urls = []
    # for run, hours in runs:
    #     urls = urls + _createDownloadUrl(url, var, field, run, hours)
        
    # urls = pd.Series(urls).unique()
        
    ds_temps = []  # List to store temporary datasets
    
    # Iterate over generated URLs and process each file
    for data_url in urls:
        logger.info("ICON-D2 data -> Processing file: {f}".format(f=data_url))                
        
        # Extract filename from URL and construct temporary file path
        tmpfn = os.path.basename(data_url) 
        tmpfn = Path(tmpfn).with_suffix('')
        tmpfp = "{p}/{tmpfn}".format(tmpfn=tmpfn, p=tmpdir) 
        
        # Attempt to download and extract the dataset
        try:
            resp, ds_temp = _urlopen_with_retry(data_url, tmpfp)
        except Exception as err:
            logger.info("Could not get {url}: {err}".format(err=err, url=data_url))
            continue  # Skip to next URL if download fails
        
        # Check if the dataset contains a 'generalVerticalLayer' coordinate
        if 'generalVerticalLayer' in ds_temp.coords:
            ds_coords = list(ds_temp.coords)
            ds_coords_to_keep = ["valid_time", "longitude", "latitude", "generalVerticalLayer"] 
            ds_coords_to_drop = [ds_coord for ds_coord in ds_coords if ds_coord not in ds_coords_to_keep]
            
            # Expand dataset dimensions and remove unwanted coordinates
            ds_temp = ds_temp.expand_dims(dim=["valid_time", "generalVerticalLayer"]).drop_vars(ds_coords_to_drop)
            
            # Assign coordinate values back to dataset
            ds_temp = ds_temp.assign_coords({"valid_time": ds_temp.valid_time, 
                                             "latitude": ds_temp.latitude,
                                             "longitude": ds_temp.longitude,
                                             "generalVerticalLayer": ds_temp.generalVerticalLayer})
            ds_temps.append(ds_temp)
            
        else:
            ds_coords = list(ds_temp.coords)
            ds_coords_to_keep = ["valid_time", "longitude", "latitude"] 
            ds_coords_to_drop = [ds_coord for ds_coord in ds_coords if ds_coord not in ds_coords_to_keep]
            
            # Swap 'step' dimension with 'valid_time' if applicable
            if "step" in ds_temp.dims:
                ds_temp = ds_temp.swap_dims({"step": "valid_time"}).drop_vars(ds_coords_to_drop)
            else:
                ds_temp = ds_temp.expand_dims(dim="valid_time").drop_vars(ds_coords_to_drop)
            
            # Assign coordinate values back to dataset
            ds_temp = ds_temp.assign_coords({"valid_time": ds_temp.valid_time, 
                                             "latitude": ds_temp.latitude,
                                             "longitude": ds_temp.longitude})
            ds_temps.append(ds_temp)
       
    # Merge all collected datasets into a single dataset
    ds = xr.merge(ds_temps)   
    
    # Rename and clean coordinate labels for consistency
    ds = _rename_and_clean_coords(ds)
    
    # Iterate through all data variables in the dataset
    for ds_var in list(ds.data_vars):
        # If the variable is an averaged quantity, apply de-averaging
        if ds[ds_var].attrs['GRIB_stepType'] == 'avg':
            ds[ds_var] = _deaverage(ds[ds_var])
                                   
        # If the variable is an accumulated quantity, apply de-accumulation
        elif ds[ds_var].attrs['GRIB_stepType'] == 'accum':
            ds[ds_var] = _deaccumulate(ds[ds_var])

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

    return ds

def _interpolate_to_cutout_resolution(ds, retrieval_params, static):
    
    # Interpolate the data spatially and temporally to the wanted cutout resolution
    ds_temps = []
    for idx, var in enumerate(ds.data_vars):
        ds_temps.append(_interpolate(ds[var], static, 
                                     retrieval_params['coords'], 
                                     retrieval_params['grid'],
                                     retrieval_params['interp_s'],
                                     retrieval_params['interp_t'])
                        )
    
    ds = xr.merge(ds_temps)
    ds = ds.assign_coords(lon=("x", ds.x.values), lat=("y", ds.y.values))
    
    ds = ds.unify_chunks().chunk(chunks=retrieval_params['chunks'] or {})
    
    return ds


def get_data_wind(retrieval_params):
    '''
    Retrieves and processes wind data from the DWD server.
    
    The function collects wind speed and direction data at 100m above ground level,
    as well as surface roughness data. It then processes and interpolates this data
    to match the desired spatial and temporal resolution.
    
    Parameters
    ----------
    retrieval_params : dict
        Dictionary containing parameters for data retrieval, including coordinates,
        grid resolution, and interpolation methods.
    
    Returns
    -------
    xarray.Dataset
        Processed dataset containing wind speed, wind direction, and surface roughness.
    '''
    
    # Retrieve wind data from model levels 62 and 63
    retrieval_params['field'] = ['model-level', 'model-level']
    ds = retrieve_data(
        url=dwd_url,
        variable=[
            "u/62/63",  # Zonal (east-west) wind component at levels 62 and 63
            "v/62/63",  # Meridional (north-south) wind component at levels 62 and 63
        ],
        **retrieval_params,
    )
    
    # Compute the mean wind values across the general vertical layers
    ds["u"] = ds["u"].mean('generalVerticalLayer')
    ds["v"] = ds["v"].mean('generalVerticalLayer')
    ds = ds.drop_dims('generalVerticalLayer')  # Remove the dimension after averaging
    ds = ds.rename({"u": "u_100m", "v": "v_100m"})  # Rename variables for clarity
    
    
    # Retrieve surface roughness data from single-level data
    retrieval_params['field'] = ['single-level']
    ds2 = retrieve_data(
        url=dwd_url,
        variable=["z0"],  # Surface roughness length
        **retrieval_params,
    )
        
    # Merge wind data with roughness data into a single dataset
    ds = xr.merge([ds, ds2])
    
    # Rename roughness variable for clarity
    ds = ds.rename({"fsr": "roughness"})
    ds["roughness"] = ds["roughness"].assign_attrs(
        units="m",
        long_name="Surface roughness"
    )
    
    # Compute wind speed at 100m using the Pythagorean theorem
    ds["wnd100m"] = np.sqrt(ds["u_100m"] ** 2 + ds["v_100m"] ** 2).assign_attrs(
        units="m/s", long_name="100 metre wind speed"
    )
    
    # Compute wind direction azimuth (0 = North, π/2 = East, π = South, 3π/2 = West)
    azimuth = np.arctan2(ds["u_100m"], ds["v_100m"])
    
    # Ensure wind azimuth is within the 0 to 2π range
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)
    
    # Remove intermediate wind component variables after processing
    ds = ds.drop_vars(["u_100m", "v_100m"])
        
    return ds


def sanitize_wind(ds):
    """Sanitize retrieved wind data."""
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)
    return ds


def get_data_influx(retrieval_params):
    """Get influx data for given retrieval parameters."""
    # Retrieve single-level data
    retrieval_params['field'] = ['single-level', 'single-level', 'single-level', 'single-level']
    ds = retrieve_data(
        url=dwd_url,
        variable=[
            "asob_t",
            "aswdir_s",
            "aswdifd_s",
            "alb_rad",
        ],
        **retrieval_params,
    )

    ds = ds.rename({"avg_tnswrf": "influx_toa", 
                    "ASWDIR_S": "influx_direct", 
                    "ASWDIFD_S": "influx_diffuse", 
                    "al": "albedo"})
    
    ds["albedo"] = (ds["albedo"]/100).assign_attrs(units="(0 - 1)", long_name="Shortwave broadband albedo for diffuse radiation")
    ds["influx_diffuse"] = ds["influx_diffuse"].assign_attrs(units="W m**-2", long_name="Surface down solar diffuse radiation")
    ds["influx_direct"] = ds["influx_direct"].assign_attrs(units="W m**-2", long_name="Surface down solar direct radiation")
    ds["influx_toa"] = ds["influx_toa"].assign_attrs(units="W m**-2", long_name="Net short-wave radiation flux at top of atmosphere (TOA)")
            
    # # Interpolate the data spatially and temporally to the wanted cutout resolution
    # ds_temps = []
    # for idx, var in enumerate(ds):
    #     ds_temps.append(_interpolate(ds[var], False, 
    #                                  retrieval_params['coords'], 
    #                                  retrieval_params['grid'],
    #                                  retrieval_params['interp_s'],
    #                                  retrieval_params['interp_t'])
    #                     )
    
    # ds = xr.merge(ds_temps)
    # ds = ds.assign_coords(lon=("x", ds.x.values), lat=("y", ds.y.values))
        
    # ICON-D2 variables are mean values for previous hour, i.e. 13:01 to 14:00 are labelled as "14:00"
    # account by calculating the SolarPosition for the center of the interval for aggregation happens
    # see https://github.com/PyPSA/atlite/issues/158
    # Do not show DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # Convert dt / time frequency to timedelta and shift solar position by half
        # (freqs like ["H","30T"] do not work with pd.to_timedelta(...)
        time_shift = (
            -1
            / 2
            * pd.to_timedelta(
                pd.date_range(
                    "1970-01-01", periods=1, freq=pd.infer_freq(ds["time"])
                ).freq
            )
        )
        sp = SolarPosition(ds, time_shift=time_shift)
        
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp])
    
    # # Interpolate the data spatially and temporally to the wanted cutout resolution
    # ds_temps = []
    # for idx, var in enumerate(ds):
    #     ds_temps.append(_interpolate(ds[var], False, 
    #                                  retrieval_params['coords'], 
    #                                  retrieval_params['grid'],
    #                                  retrieval_params['interp_s'],
    #                                  retrieval_params['interp_t'])
    #                     )
    
    # ds = xr.merge(ds_temps)
    # ds = ds.assign_coords(lon=("x", ds.x.values), lat=("y", ds.y.values))
    
    # ds = ds.unify_chunks.chunk(chunks=retrieval_params['chunks'] or {})
    
    
    
    # ds = ds.drop_vars(['lon','lat'])
    # ds = ds.assign_coords(lon=("x", ds.x.values), lat=("y", ds.y.values))
        
    return ds


def sanitize_influx(ds):
    """Sanitize retrieved influx data."""
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def get_data_temperature(retrieval_params):
    """Get wind temperature for given retrieval parameters."""
    # Retrieve single-level data
    retrieval_params['field'] = ['single-level','soil-level']
    ds = retrieve_data(
        url=dwd_url,
        variable=["t_2m",
                  "t_so/162"],
        **retrieval_params
    )

    ds = ds.rename({"t2m": "temperature",
                    "T_SO": "soil temperature"})
    
    ds["temperature"] = ds["temperature"].assign_attrs(units="K", long_name="Temperature at 2m above ground")
    ds["soil temperature"] = ds["soil temperature"].assign_attrs(units="K", long_name="Soil temperature in 162 cm depth ")
    
    return ds


def get_data_runoff(retrieval_params):
    """Get runoff data for given retrieval parameters."""
    # Retrieve single-level data
    retrieval_params['field'] = ['single-level','single-level']
    ds = retrieve_data(url=dwd_url,
                       variable=["runoff_s", 
                                 "runoff_g"], 
                       **retrieval_params)
        
    ds["runoff"] = (ds["RUNOFF_S"] + ds["RUNOFF_G"]).assign_attrs(units="kg m**-2", long_name="Surface and Soil water runoff (accumulated since model start)")

    ds = ds.drop_vars(["RUNOFF_S", "RUNOFF_G"])
    
    return ds


def sanitize_runoff(ds):
    """Sanitize retrieved runoff data."""
    ds["runoff"] = ds["runoff"].clip(min=0.0)
    return ds


def get_data_height(retrieval_params):
    """Get height data for given retrieval parameters."""
    # Retrieve time-invariant data
    retrieval_params['field'] = ['time-invariant']
    ds = retrieve_data(url=dwd_url,
                       variable=["hsurf"],
                       **retrieval_params)
    
    ds = ds.rename({"HSURF": "height"})
    ds["height"] = ds["height"].assign_attrs(
        units="m",
        long_name="Geometric Height of the earths surface above sea level (2D field)"
        )

    return ds


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    return [y1, x0, y0, x1]


def retrieval_times(coords, tz, static=False):
    """
    Get retrieval time dimension of the forecast.

    Parameters
    ----------
    coords : atlite.Cutout.coords
        Coordinate object containing the time dimension.
    tz : timezone
        Timezone information of the input time and date (currently unused here).
    static : bool, optional (default=False)
        If True, return only the first forecast time step and an empty offset.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'forecast': pd.DatetimeIndex of forecast timestamps (including any filled gaps)
        - 'offset': pd.DatetimeIndex of past timestamps before current model run time
    """
    # Convert xarray time coordinate to pandas index
    time = coords["time"].to_index()

    # Get the current model run time (e.g., most recent 6-hourly forecast release)
    currentRunTime = _getCurrentRun(pd.Timestamp.utcnow().replace(tzinfo=None))

    # Split times into forecast (≥ currentRunTime) and offset (< currentRunTime)
    forecast_times = time[time >= currentRunTime]
    offset_times = time[time < currentRunTime]

    # If the forecast doesn't include currentRunTime explicitly, fill the missing range
    if len(forecast_times) > 0 and forecast_times[0] > currentRunTime:
        # Infer time resolution, fallback to difference if not directly available
        freq = time.freq or pd.infer_freq(time)
        if freq is None:
            freq = time[1] - time[0]  # fallback to timedelta if no freq is inferable

        # Fill the missing time steps from currentRunTime up to just before the first forecast time
        fill = pd.date_range(currentRunTime, forecast_times[0] - freq, freq=freq)

        # Prepend the filled range to the forecast times
        forecast_times = fill.append(forecast_times)

    # If static mode is requested, return only the first forecast time and no offset
    if static:
        forecast_times = forecast_times[:1]
        offset_times = pd.DatetimeIndex([])

    # Return dictionary with forecast and offset times
    return {
        "forecast": forecast_times,
        "offset": offset_times,
        }


def noisy_unlink(path):
    """Delete file at given path."""
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")

    
def retrieve_data(url, product, chunks=None, tmpdir=None, lock=None, **updates):

    """
    Download data from the ICON-D2 Model from the Open Data Server (ODS) of DWD.
    
    If you want to manually downolad the data go to:
    https://opendata.dwd.de/weather/nwp/icon-d2/grib/
    """
    
    request = {"product_type": "icon_d2", "format": "direct-download"}
    request.update(updates)

    ds_temps = []
    #Download data for each variable individually and then merge all in one xarray
    logger.info(f"open-dwd: Downloading variables\n\t{request['variable']}\n")
    for idx, var in enumerate(request['variable']):
        ds_temps.append(_mainDataCollector(url,
                                  var, 
                                  request['field'][idx],
                                  request['forecast'], 
                                  request['offset'],
                                  request['coords'],
                                  tmpdir)
                        ) 
          
    ds = xr.merge(ds_temps).chunk(chunks=chunks)
    
    return ds


def get_data(cutout, feature, tmpdir, 
             lock=None,     
             monthly_requests=False,
             concurrent_requests=False, 
             **creation_parameters):
    """
    Retrieve data from DWDs ICON-D2 Model dataset (via ODS).

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.icon_d2.features`
    monthly_requests : bool
        Takes no effect, only here for consistency with other dataset modules.
    concurrent_requests : bool
        Takes no effect, only here for consistency with other dataset modules.
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
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
        "product": "dwd_icon_d2",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": [cutout.dx, cutout.dy],
        "tmpdir": tmpdir,
        "lock": lock,
        "tz": cutout.data.tz,
        "interp_s": cutout.data.interp_s,
        "interp_t": cutout.data.interp_t,
        "coords": coords,
    }

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    logger.info(f"Requesting data for feature {feature} for ICON-D2 from open-dwd...")

    def retrieve_once(time, static=False):
        ds = func({**retrieval_params, **time})
        ds = _interpolate_to_cutout_resolution(ds, retrieval_params, static)
        # Sanitize the data after interpolation to remove residuals
        if sanitize and sanitize_func is not None:
            ds = sanitize_func(ds)
        return ds

    if feature in static_features:
        return retrieve_once(retrieval_times(coords, cutout.data.tz, True), True).squeeze()
    
    dataset = retrieve_once(retrieval_times(coords, cutout.data.tz, False), False)

    return dataset.sel(time=coords["time"])
