# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and curating data from ECMWFs ERA5 dataset (via CDS).

For further reference see
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
"""

import datetime
import logging
import os
import warnings
import weakref
from pathlib import Path
from tempfile import mkstemp
from typing import Literal

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import arctan2, sqrt
from dask.utils import SerializableLock
from numpy import atleast_1d

from atlite.gis import maybe_swap_spatial_dims, rotate
from atlite.pv.solar_position import SolarPosition
from atlite.wind import calculate_windspeed_bias_correction

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
    "wind": ["wnd100m", "wnd_shear_exp", "wnd_azimuth", "roughness"],
    "windspeed_bias_correction": ["wnd_bias_correction"],
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
    ds = ds.rename({"longitude": "x", "latitude": "y", "valid_time": "time"})
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
                    "time": [f"{h:02d}:00" for h in t[t.month == month].hour.unique()],
                }
                times.append(query)
        else:
            query = {
                "year": str(year),
                "month": list(t.month.unique()),
                "day": list(t.day.unique()),
                "time": [f"{h:02d}:00" for h in t.hour.unique()],
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


def add_finalizer(ds: xr.Dataset, target: str | Path):
    logger.debug(f"Adding finalizer for {target}")
    weakref.finalize(ds._close.__self__.ds, noisy_unlink, target)


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


def open_with_grib_conventions(
    grib_file: str | Path, chunks=None, tmpdir: str | Path | None = None
) -> xr.Dataset:
    """
    Convert grib file of ERA5 data from the CDS to netcdf file.

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
        # extra_coords={"expver": "valid_time"},
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
    dataset: str,
    chunks: dict[str, int] | None = None,
    tmpdir: str | Path | None = None,
    lock: SerializableLock | None = None,
    data_format: Literal["grib", "netcdf"] = "grib",
    **updates,
) -> xr.Dataset:
    """
    Download data like ERA5 from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://cds-beta.climate.copernicus.eu/requests?tab=all

    Parameters
    ----------
    product : str
        Product name, e.g. 'reanalysis-era5-single-levels'.
    chunks : dict, optional
        Chunking for xarray dataset, e.g. {'time': 1, 'x': 100, 'y': 100}.
        Default is None.
    tmpdir : str, optional
        Directory where the downloaded data is temporarily stored.
        Default is None, which uses the system's temporary directory.
    lock : dask.utils.SerializableLock, optional
        Lock for thread-safe file writing. Default is None.
    data_format : {"grib", "netcdf"}
        Data format to use for retrieving from CDS.
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
    ...     dataset='reanalysis-era5-single-levels',
    ...     chunks={'time': 1, 'x': 100, 'y': 100},
    ...     tmpdir='/tmp',
    ...     lock=None,
    ...     year='2020',
    ...     month='01',
    ...     variable=['10m_u_component_of_wind', '10m_v_component_of_wind'],
    ...     data_format='netcdf'
    ... )
    """
    request = {"product_type": ["reanalysis"], "download_format": "unarchived"}
    request.update(updates, data_format=data_format)

    assert {"year", "month", "variable"}.issubset(request), (
        "Need to specify at least 'variable', 'year' and 'month'"
    )

    logger.debug(f"Requesting {dataset} with API request: {request}")

    client = cdsapi.Client(
        info_callback=logger.debug, debug=logging.DEBUG >= logging.root.level
    )
    result = client.retrieve(dataset, request)

    if lock is None:
        lock = nullcontext()

    suffix = f".{data_format}"  # .netcdf or .grib
    with lock:
        fd, target = mkstemp(suffix=suffix, dir=tmpdir)
        os.close(fd)

        # Inform user about data being downloaded as "* variable (year-month)"
        timestr = f"{request['year']}-{request['month']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info(f"CDS: Downloading variables\n\t{varstr}\n")
        result.download(target)

    # Convert from grib to netcdf locally, same conversion as in CDS backend
    if data_format == "grib":
        ds = open_with_grib_conventions(target, chunks=chunks, tmpdir=tmpdir)
    else:
        ds = xr.open_dataset(target, chunks=sanitize_chunks(chunks))
        if tmpdir is None:
            add_finalizer(target)

    return ds


def retrieve_windspeed_average(
    cutout=None,
    height: int = 100,
    first_year: int = 2008,
    last_year: int | None = 2017,
    data_format: Literal["grib", "netcdf"] = "grib",
    **retrieval_params,
):
    """
    Retrieve average windspeed from `first_year` to `last_year`

    The default time-period 2008-2017 was chosen to align with the simulation
    period of GWA3.

    Parameters
    ----------
    cutout : atlite.Cutout or None
        Cutout for which to retrieve windspeeds from CDS. If no cutout is
        specified the global means are retrieved at native resolution 0.25/0.25.
    height : int
        Height of windspeeds (ERA5 typically knows about 10m, 100m)
    first_year : int, defaults to 2008
        First year to take into account
    last_year : int, defaults to 2017
        Last year to take into account (if omitted takes the previous year)
    data_format : {"grib", "netcdf"}
        Data format to use for retrieving from CDS.
    **retrieval_params

    References
    ----------
    https://globalwindatlas.info/

    Returns
    -------
    DataArray
        Mean windspeed at cutout dimension
    """
    if last_year is None:
        last_year = datetime.date.today().year - 1

    retrieval_params = (
        {
            "dataset": "reanalysis-era5-single-levels",
            "product_type": "reanalysis",
        }
        | (
            {
                "area": _area(cutout.coords),
                "chunks": cutout.chunks,
                "grid": f"{cutout.dx}/{cutout.dy}",
            }
            if cutout is not None
            else {}
        )
        | retrieval_params
    )

    def retrieve_chunk(year):
        ds = retrieve_data(
            variable=[
                f"{height}m_u_component_of_wind",
                f"{height}m_v_component_of_wind",
            ],
            year=[year],
            month=[f"{m:02d}" for m in range(1, 12 + 1)],
            day=[f"{d:02d}" for d in range(1, 31 + 1)],
            time=[f"{h:02d}" for h in range(0, 23 + 1)],
            data_format=data_format,
            **retrieval_params,
        )
        ds = _rename_and_clean_coords(ds)

        if cutout is None:
            # the default longitude range of CDS is [0, 360], while [-180, 180] is standard
            ds = rotate(ds)

        return (
            sqrt(ds[f"u{height}"] ** 2 + ds[f"v{height}"] ** 2)
            .mean("time")
            .assign_attrs(
                units=ds[f"u{height}"].attrs["units"],
                long_name=f"{height} metre wind speed as long run average",
            )
        )

    years = range(first_year, last_year + 1)
    return xr.concat(
        compute(*(delayed(retrieve_chunk)(str(year)) for year in years)),
        dim=pd.Index(years, name="year"),
    ).mean("year")


def get_data_windspeed_bias_correction(cutout, retrieval_params, creation_parameters):
    """
    Get windspeed bias correction
    """
    real_average_path = creation_parameters.get("windspeed_real_average_path")
    if real_average_path is None:
        logger.warning(
            "Skipping feature windspeed_bias_correction, since windspeed_real_average_path was not provided.\n"
            "Download mean wind speeds from global wind atlas at https://globalwindatlas.info/ and add it\n"
            'to the cutout with `cutout.prepare(windspeed_real_average_path="path/to/gwa3_250_windspeed_100m.tif")`'
        )
        return None
    height = creation_parameters.get("windspeed_height", 100)
    data_average = retrieve_windspeed_average(cutout, height, **retrieval_params)

    bias_correction = calculate_windspeed_bias_correction(
        cutout, real_average_path, height=height, data_average=data_average
    )
    return bias_correction.to_dataset(name="wnd_bias_correction")


def get_data(
    cutout,
    feature,
    tmpdir,
    lock=None,
    data_format="grib",
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
    data_format : str, optional
        The format of the data to be downloaded. Can be either 'grib' or 'netcdf',
        'grib' highly recommended because CDSAPI limits request size for netcdf.
    concurrent_requests : bool, optional
        If True, the monthly data requests are posted concurrently.
        Only has an effect if `monthly_requests` is True.
    **creation_parameters :
        Additional keyword arguments.
        `sanitize` (default True) sets sanitization of the data on or off.
        `windspeed_real_average_path` and `windspeed_height` are used by the
        "windspeed_bias_correction" feature to calculate the correction factor.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

    """
    coords = cutout.coords

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "dataset": "reanalysis-era5-single-levels",
        "product_type": "reanalysis",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": f"{cutout.dx}/{cutout.dy}",
        "tmpdir": tmpdir,
        "lock": lock,
        "data_format": data_format,
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
    elif feature == "windspeed_bias_correction":
        return func(
            cutout,
            retrieval_params=retrieval_params,
            creation_parameters=creation_parameters,
        )

    time_chunks = retrieval_times(coords, monthly_requests=monthly_requests)
    if concurrent_requests:
        delayed_datasets = [delayed(retrieve_once)(chunk) for chunk in time_chunks]
        datasets = compute(*delayed_datasets)
    else:
        datasets = map(retrieve_once, time_chunks)

    return xr.concat(datasets, dim="time").sel(time=coords["time"])
