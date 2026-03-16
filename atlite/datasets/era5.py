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
import warnings
import weakref
from pathlib import Path
from tempfile import mkstemp

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import arctan2, sqrt
from dask.utils import SerializableLock
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
    Retrieve and derive wind variables from ERA5 data.

    Parameters
    ----------
    retrieval_params : dict
        Parameters passed to :func:`retrieve_data`.

    Returns
    -------
    xarray.Dataset
        Dataset containing wind speed, shear exponent, azimuth, and roughness.
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
    Sanitize wind data variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Retrieved wind dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with non-physical roughness values replaced.
    """
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)
    return ds


def get_data_influx(retrieval_params):
    """
    Retrieve and derive solar influx variables from ERA5 data.

    Parameters
    ----------
    retrieval_params : dict
        Parameters passed to :func:`retrieve_data`.

    Returns
    -------
    xarray.Dataset
        Dataset containing direct, diffuse, and top-of-atmosphere influx,
        albedo, and solar position variables.
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
    Sanitize solar influx data.

    Parameters
    ----------
    ds : xarray.Dataset
        Retrieved influx dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with negative influx values clipped to zero.
    """
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def get_data_temperature(retrieval_params):
    """
    Retrieve temperature-related ERA5 variables.

    Parameters
    ----------
    retrieval_params : dict
        Parameters passed to :func:`retrieve_data`.

    Returns
    -------
    xarray.Dataset
        Dataset containing air, soil, and dewpoint temperature variables.
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
    Retrieve runoff data from ERA5.

    Parameters
    ----------
    retrieval_params : dict
        Parameters passed to :func:`retrieve_data`.

    Returns
    -------
    xarray.Dataset
        Dataset containing runoff values.
    """
    ds = retrieve_data(variable=["runoff"], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({"ro": "runoff"})

    return ds


def sanitize_runoff(ds):
    """
    Sanitize runoff data.

    Parameters
    ----------
    ds : xarray.Dataset
        Retrieved runoff dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with negative runoff values clipped to zero.
    """
    ds["runoff"] = ds["runoff"].clip(min=0.0)
    return ds


def get_data_height(retrieval_params):
    """
    Retrieve geopotential height data from ERA5.

    Parameters
    ----------
    retrieval_params : dict
        Parameters passed to :func:`retrieve_data`.

    Returns
    -------
    xarray.Dataset
        Dataset containing surface height derived from geopotential.
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
            "year": [time[0].strftime("%Y")],
            "month": [time[0].strftime("%m")],
            "day": [time[0].strftime("%d")],
            "time": time[0].strftime("%H:00"),
        }

    # Prepare request for all months and years
    times = []
    for year in time.year.unique():
        t = time[time.year == year]
        if monthly_requests:
            for month in t.month.unique():
                query = {
                    "year": [str(year)],
                    "month": [t[t.month == month][0].strftime("%m")],
                    "day": list(t[t.month == month].strftime("%d").unique()),
                    "time": list(t[t.month == month].strftime("%H:00").unique()),
                }
                times.append(query)
        else:
            query = {
                "year": [str(year)],
                "month": list(t.strftime("%m").unique()),
                "day": list(t.strftime("%d").unique()),
                "time": list(t.strftime("%H:00").unique()),
            }
            times.append(query)
    return times


def noisy_unlink(path):
    """
    Delete a file and log failures.

    Parameters
    ----------
    path : str | Path
        File path to delete.

    Returns
    -------
    None
    """
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def add_finalizer(ds: xr.Dataset, target: str | Path):
    """
    Register deletion of a temporary file when a dataset is closed.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset associated with the temporary file.
    target : str | Path
        Path to the temporary file.

    Returns
    -------
    None
    """
    logger.debug(f"Adding finalizer for {target}")
    weakref.finalize(ds._close.__self__.ds, noisy_unlink, target)


def sanitize_chunks(chunks, **dim_mapping):
    """
    Map internal chunk dimension names to ERA5 dataset dimensions.

    Parameters
    ----------
    chunks : dict or Any
        Chunk specification passed to xarray.
    **dim_mapping
        Additional mappings from internal to external dimension names.

    Returns
    -------
    dict or Any
        Chunk mapping with renamed dimensions, or the original value if no
        mapping is needed.
    """
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
    Open a CDS GRIB file using ERA5-compatible conventions.

    Parameters
    ----------
    grib_file : str | Path
        Path to the GRIB file.
    chunks : dict, optional
        Chunk specification passed to xarray.
    tmpdir : str | Path, optional
        Temporary directory. If ``None``, the source file is removed when the
        dataset is closed.

    Returns
    -------
    xarray.Dataset
        Dataset with renamed variables and expanded dimensions matching CDS
        netCDF conventions.
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
        Expand missing dimensions while preserving dimension order.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to expand.
        expand_dims : list of str
            Dimensions that should exist in the dataset.

        Returns
        -------
        xarray.Dataset
            Dataset with missing dimensions inserted in a stable order.
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
    Retrieve ERA5 data from the Climate Data Store.

    Parameters
    ----------
    product : str
        CDS product name.
    chunks : dict, optional
        Chunk specification passed to xarray.
    tmpdir : str | Path, optional
        Directory used for temporary downloads.
    lock : dask.utils.SerializableLock, optional
        Lock used while writing downloaded files.
    **updates
        Additional CDS request parameters. Must include ``year``, ``month``,
        and ``variable``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the requested variables.
    """
    request = {"product_type": ["reanalysis"], "download_format": "unarchived"}
    request.update(updates)

    assert {"year", "month", "variable"}.issubset(request), (
        "Need to specify at least 'variable', 'year' and 'month'"
    )

    logger.debug(f"Requesting {product} with API request: {request}")

    client = cdsapi.Client(
        info_callback=logger.debug, debug=logging.DEBUG >= logging.root.level
    )
    result = client.retrieve(product, request)

    if lock is None:
        lock = nullcontext()

    suffix = f".{request['data_format']}"  # .netcdf or .grib
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
    if request["data_format"] == "grib":
        ds = open_with_grib_conventions(target, chunks=chunks, tmpdir=tmpdir)
    else:
        ds = xr.open_dataset(target, chunks=sanitize_chunks(chunks))
        if tmpdir is None:
            add_finalizer(target)

    return ds


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
    Retrieve and format ERA5 data for a cutout feature.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the requested spatiotemporal domain.
    feature : str
        Feature name defined in :data:`atlite.datasets.era5.features`.
    tmpdir : str | Path
        Directory used for temporary files.
    lock : dask.utils.SerializableLock, optional
        Lock used while writing downloaded files.
    data_format : str, optional
        Download format, typically ``"grib"`` or ``"netcdf"``.
    monthly_requests : bool, optional
        Whether to split requests by month.
    concurrent_requests : bool, optional
        Whether monthly requests should be submitted concurrently.
    **creation_parameters
        Additional creation options. Supports ``sanitize``.

    Returns
    -------
    xarray.Dataset
        Dataset containing the requested feature variables.
    """
    coords = cutout.coords

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "product": "reanalysis-era5-single-levels",
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
        """
        Retrieve and optionally sanitize one temporal ERA5 request.

        Parameters
        ----------
        time : dict
            Time selection arguments for a single CDS request.

        Returns
        -------
        xarray.Dataset
            Retrieved dataset for the requested time slice.
        """
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
