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
import weakref
import zipfile
from pathlib import Path
from tempfile import mkstemp

import cdsapi
import numpy as np
import xarray as xr
from dask import compute, delayed
from dask.utils import SerializableLock
from numpy import atleast_1d

from atlite.gis import maybe_swap_spatial_dims

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

dataset = "cems-glofas-historical"

features = {"discharge": ["discharge"]}


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


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and
    longitude columns as 'lat' and 'lon'.
    """
    ds = ds.rename(
        {
            "longitude": "x",
            "latitude": "y",
            "valid_time": "time",
            "dis24": "discharge",
        }
    )
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    ds = ds.drop_vars(["expver", "number"], errors="ignore")

    return ds


def open_with_grib_conventions(
    grib_file: str | Path, chunks=None, tmpdir: str | Path | None = None
) -> xr.Dataset:
    """
    Convert grib file of Glofas data from the CDS to netcdf file.

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


def retrieve_data(
    product: str,
    chunks: dict[str, int] | None = None,
    tmpdir: str | Path | None = None,
    lock: SerializableLock | None = None,
    **updates,
) -> xr.Dataset:
    """
    Download data like Glofas from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://cds-beta.climate.copernicus.eu/requests?tab=all

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
        Must include 'year', 'month', and 'variable'.
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
    ...     year='2020',
    ...     month='01',
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

    logger.debug(f"Requesting {product} with API request: {request}")
    # Url needs to be set manually here, overrides url from .cdsapirc (for use with multiple modules)
    client = cdsapi.Client(
        info_callback=logger.debug,
        debug=logging.DEBUG >= logging.root.level,
        url="https://ewds.climate.copernicus.eu/api",
    )
    result = client.retrieve(product, request)

    if lock is None:
        lock = nullcontext()

    suffix = f".{request['data_format']}"  # .netcdf or .grib
    with lock:
        fd, target = mkstemp(suffix=suffix, dir=tmpdir)
        os.close(fd)

        # Inform user about data being downloaded as "* variable (year-month)"
        timestr = f"{request['hyear']}-{request['hmonth']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info(f"CDS: Downloading variables\n\t{varstr}\n")
        result.download(target)

    # Extract data if downloaded as zip file
    if request.get("download_format") == "zip":
        with zipfile.ZipFile(target, "r") as zip_ref:
            zip_ref.extractall(Path(tmpdir) / Path(target).stem)
        os.unlink(target)  # delete the zip file after extraction
        target = Path(tmpdir) / Path(Path(target).stem) / Path("data" + suffix)

    # Convert from grib to netcdf locally, same conversion as in CDS backend
    if request["data_format"] == "grib":
        ds = open_with_grib_conventions(target, chunks=chunks, tmpdir=tmpdir)
    else:
        ds = xr.open_dataset(target, chunks=sanitize_chunks(chunks))
        if tmpdir is None:
            add_finalizer(target)
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
        # Make sure that day and month are in two digit format
        day = str(time[0].day) if time[0].day >= 10 else f"0{time[0].day}"
        month = str(time[0].month) if time[0].month >= 10 else f"0{time[0].month}"
        return {
            "hyear": str(time[0].year),
            "hmonth": month,
            "hday": day,
            "time": time[0].strftime("%H:00"),
        }

    # Prepare request for all months and years
    times = []
    for year in time.year.unique():
        t = time[time.year == year]
        if monthly_requests:
            for month in t.month.unique():
                # Make sure that day and month are in two digit format
                days = list(t[t.month == month].day.unique())
                days_str = [str(d) if d >= 10 else f"0{d}" for d in days]
                query = {
                    "hyear": str(year),
                    "hmonth": str(month) if month >= 10 else f"0{month}",
                    "hday": days_str,
                }
                times.append(query)
        else:
            # Make sure that day and month are in two digit format
            days = list(t.day.unique())
            days_str = [str(d) if d >= 10 else f"0{d}" for d in days]
            months = list(t.month.unique())
            months_str = [str(m) if m >= 10 else f"0{m}" for m in months]
            query = {"hyear": str(year), "hmonth": months_str, "hday": days_str}
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
                - 'time_fill_method' (default "nearest"): strategy used to align the
                    returned dataset to cutout time coordinates. Supported values are
                    xarray reindex methods (e.g. "nearest", "pad", "backfill") and
                    "interpolate" (or "interp") for time interpolation.
                - 'time_interp_method' (default "linear"): interpolation method passed
                    to xarray when 'time_fill_method' is "interpolate".

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

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
        if sanitize:
            ds["discharge"] = ds["discharge"].clip(min=0.0).fillna(0.0)
        return ds

    time_chunks = retrieval_times(coords, monthly_requests=monthly_requests)
    if concurrent_requests:
        delayed_datasets = [delayed(retrieve_once)(chunk) for chunk in time_chunks]
        datasets = list(compute(*delayed_datasets))
    else:
        datasets = list(map(retrieve_once, time_chunks))

    ds = xr.concat(datasets, dim="time").sortby("time")
    fill_method = creation_parameters.get("time_fill_method", "interpolate")
    if fill_method is None:
        return ds.reindex(time=coords["time"])
    if isinstance(fill_method, str) and fill_method.lower() in {
        "interpolate",
        "interp",
    }:
        interp_method = creation_parameters.get("time_interp_method", "linear")
        interpolated = ds.interp(time=coords["time"], method=interp_method)
        return interpolated.bfill("time").ffill("time")
    return ds.reindex(time=coords["time"], method=fill_method)
