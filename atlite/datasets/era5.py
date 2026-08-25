# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and curating data from ECMWFs ERA5 dataset (via CDS).

For further reference see
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
"""

from __future__ import annotations

import logging
import os
import warnings
from tempfile import mkstemp
from typing import TYPE_CHECKING, Any, Literal, cast

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from numpy import atleast_1d

from atlite.datasets.cds_helper import (
    _area,
    add_finalizer,
    open_with_grib_conventions,
    sanitize_chunks,
)
from atlite.gis import maybe_swap_spatial_dims
from atlite.pv.solar_position import SolarPosition

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from dask.utils import SerializableLock

    from atlite._types import PathLike

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
    "wave": ["wave_height", "wave_period"],
}

static_features = {"height"}


def _add_height(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert geopotential to height and replace the 'z' variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing geopotential variable 'z'.

    Returns
    -------
    xr.Dataset
        Dataset with 'height' variable in meters, 'z' removed.

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
    return ds.drop_vars("z")


def _rename_and_clean_coords(ds: xr.Dataset, add_lon_lat: bool = True) -> xr.Dataset:
    """
    Standardize coordinate names and clean up auxiliary variables.

    Renames longitude/latitude/valid_time to x/y/time, rounds spatial
    coordinates, and drops 'expver'/'number' if present.

    Parameters
    ----------
    ds : xr.Dataset
        Raw ERA5 dataset with original coordinate names.
    add_lon_lat : bool, optional
        Whether to add 'lon'/'lat' as coordinate aliases. Default True.

    Returns
    -------
    xr.Dataset
        Dataset with standardized coordinates.
    """
    ds = ds.rename({"longitude": "x", "latitude": "y", "valid_time": "time"})
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = cast("xr.Dataset", maybe_swap_spatial_dims(ds))
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds.drop_vars(["expver", "number"], errors="ignore")


def _process_wind(ds: xr.Dataset, single_precision: bool = False) -> xr.Dataset:
    """
    Derive wind speed, shear exponent, azimuth and roughness from raw components.

    Shared by the CDS (:mod:`atlite.datasets.era5`) and EDH
    (:mod:`atlite.datasets.era5_edh`) backends. Operates on a dataset carrying
    the raw ``u10``/``v10``/``u100``/``v100``/``fsr`` variables.
    ``single_precision`` casts the float64-promoted shear and azimuth back to
    float32 (used by EDH to halve on-disk size).

    Returns
    -------
    xr.Dataset
        Dataset with variables: wnd100m, wnd_shear_exp, wnd_azimuth, roughness.
    """
    for h in (10, 100):
        units = ds[f"u{h}"].attrs.get("units", "m s**-1")
        ds[f"wnd{h}m"] = np.sqrt(ds[f"u{h}"] ** 2 + ds[f"v{h}"] ** 2).assign_attrs(
            units=units, long_name=f"{h} metre wind speed"
        )
    shear = (np.log(ds["wnd10m"] / ds["wnd100m"]) / np.log(10 / 100)).assign_attrs(
        units="", long_name="wind shear exponent"
    )
    ds["wnd_shear_exp"] = shear.astype(np.float32) if single_precision else shear

    # span the whole circle: 0 is north, π/2 is east, -π is south, 3π/2 is west
    azimuth = np.arctan2(ds["u100"], ds["v100"])
    azimuth = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)
    ds["wnd_azimuth"] = azimuth.astype(np.float32) if single_precision else azimuth

    ds = ds.drop_vars(["u100", "v100", "u10", "v10", "wnd10m"])
    return ds.rename({"fsr": "roughness"})


def get_data_wind(retrieval_params: dict[str, Any]) -> xr.Dataset:
    """
    Retrieve and compute wind speed variables from ERA5.

    Downloads u/v wind components at 10m and 100m, computes wind speed,
    shear exponent, azimuth angle, and surface roughness.

    Parameters
    ----------
    retrieval_params : dict[str, Any]
        CDS API retrieval parameters including area, time, and format.

    Returns
    -------
    xr.Dataset
        Dataset with variables: wnd100m, wnd_shear_exp, wnd_azimuth, roughness.
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
    return _process_wind(ds)


def sanitize_wind(ds: xr.Dataset) -> xr.Dataset:
    """
    Clip negative roughness values to a minimum of 2e-4.

    Parameters
    ----------
    ds : xr.Dataset
        Wind dataset containing 'roughness' variable.

    Returns
    -------
    xr.Dataset
        Dataset with corrected roughness values.
    """
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)
    return ds


def _process_influx(ds: xr.Dataset, single_precision: bool = False) -> xr.Dataset:
    """
    Derive influx variables and solar position from raw radiation fields.

    Shared by the CDS (:mod:`atlite.datasets.era5`) and EDH
    (:mod:`atlite.datasets.era5_edh`) backends. Operates on a dataset carrying
    the raw ``ssrd``/``ssr``/``fdir``/``tisr`` variables. ``single_precision``
    casts the solar-position fields back to float32 (used by EDH to halve
    on-disk size).

    Returns
    -------
    xr.Dataset
        Dataset with variables: influx_toa, influx_direct, influx_diffuse,
        albedo, solar_altitude, solar_azimuth.
    """
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

    # ERA5 variables are mean values for previous hour, i.e. 13:01 to 14:00
    # are labelled as "14:00". Account by calculating the SolarPosition for the
    # center of the interval for aggregation.
    # See https://github.com/PyPSA/atlite/issues/158
    # Suppress DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift=pd.to_timedelta("-30 minutes"))
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})
    if single_precision:
        sp = sp.astype(np.float32)

    return xr.merge([ds, sp])


def get_data_influx(retrieval_params: dict[str, Any]) -> xr.Dataset:
    """
    Retrieve and compute solar radiation variables from ERA5.

    Downloads radiation components, converts from J/m² to W/m², computes
    albedo, diffuse radiation, and solar position (altitude/azimuth).

    Parameters
    ----------
    retrieval_params : dict[str, Any]
        CDS API retrieval parameters including area, time, and format.

    Returns
    -------
    xr.Dataset
        Dataset with variables: influx_toa, influx_direct, influx_diffuse,
        albedo, solar_altitude, solar_azimuth.
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
    return _process_influx(ds)


def sanitize_influx(ds: xr.Dataset) -> xr.Dataset:
    """
    Clip negative radiation values to zero.

    Parameters
    ----------
    ds : xr.Dataset
        Influx dataset with influx_direct, influx_diffuse, influx_toa.

    Returns
    -------
    xr.Dataset
        Dataset with non-negative radiation values.
    """
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def get_data_temperature(retrieval_params: dict[str, Any]) -> xr.Dataset:
    """
    Retrieve temperature variables from ERA5.

    Downloads 2m temperature, soil temperature (level 4), and 2m dewpoint.

    Parameters
    ----------
    retrieval_params : dict[str, Any]
        CDS API retrieval parameters including area, time, and format.

    Returns
    -------
    xr.Dataset
        Dataset with variables: temperature, soil temperature, dewpoint temperature.
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
    return ds.rename({
        "t2m": "temperature",
        "stl4": "soil temperature",
        "d2m": "dewpoint temperature",
    })


def get_data_runoff(retrieval_params: dict[str, Any]) -> xr.Dataset:
    """
    Retrieve runoff data from ERA5.

    Parameters
    ----------
    retrieval_params : dict[str, Any]
        CDS API retrieval parameters including area, time, and format.

    Returns
    -------
    xr.Dataset
        Dataset with 'runoff' variable.
    """
    ds = retrieve_data(variable=["runoff"], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    return ds.rename({"ro": "runoff"})


def sanitize_runoff(ds: xr.Dataset) -> xr.Dataset:
    """
    Clip negative runoff values to zero.

    Parameters
    ----------
    ds : xr.Dataset
        Runoff dataset containing 'runoff' variable.

    Returns
    -------
    xr.Dataset
        Dataset with non-negative runoff values.
    """
    ds["runoff"] = ds["runoff"].clip(min=0.0)
    return ds


def get_data_wave_height(retrieval_params):
    """
    Get wave height data for given retrieval parameters.
    """
    ds = retrieve_data(
        variable=[
            "significant_height_of_combined_wind_waves_and_swell",
        ],
        **retrieval_params,
    )
    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({"swh": "wave_height"})

    return ds


def sanitize_wave_height(ds):
    """
    Sanitize retrieved wave height data.
    """
    ds["wave_height"] = ds["wave_height"].clip(min=0.0)
    return ds


def get_data_wave_period(retrieval_params):
    """
    Get wave period data for given retrieval parameters.
    """
    ds = retrieve_data(
        variable=["peak_wave_period"],
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({"pp1d": "wave_period"})

    return ds


def sanitize_wave_period(ds):
    """
    Sanitize retrieved wave period data.
    """
    ds["wave_period"] = ds["wave_period"].clip(min=0.0)
    return ds


def get_data_height(retrieval_params: dict[str, Any]) -> xr.Dataset:
    """
    Retrieve geopotential and convert to terrain height.

    Parameters
    ----------
    retrieval_params : dict[str, Any]
        CDS API retrieval parameters including area, time, and format.

    Returns
    -------
    xr.Dataset
        Dataset with 'height' variable in meters.
    """
    ds = retrieve_data(variable="geopotential", **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    return _add_height(ds)


def retrieval_times(
    coords: dict[str, xr.DataArray],
    static: bool = False,
    monthly_requests: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Generate time parameter chunks for CDS API requests.

    Splits the time coordinate into year-based (or year-month-based) chunks
    suitable for the CDS API query format.

    Parameters
    ----------
    coords : dict[str, xr.DataArray]
        Coordinate arrays with 'time' dimension.
    static : bool, optional
        If True, return a single time point (for time-invariant fields).
    monthly_requests : bool, optional
        If True, split requests by month within each year.

    Returns
    -------
    dict[str, Any] or list[dict[str, Any]]
        Single dict if static, otherwise list of dicts with
        'year', 'month', 'day', 'time' keys.
    """
    time = coords["time"].to_index()
    if static:
        return {
            "year": [time[0].strftime("%Y")],
            "month": [time[0].strftime("%m")],
            "day": [time[0].strftime("%d")],
            "time": time[0].strftime("%H:00"),
        }

    times: list[dict[str, Any]] = []
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


def retrieve_data(
    product: str,
    chunks: dict[str, int] | None = None,
    tmpdir: PathLike | None = None,
    lock: SerializableLock | None = None,
    **updates: Any,
) -> xr.Dataset:
    """
    Download ERA5 data from the CDS API and return as an xarray Dataset.

    If you want to track the state of your request go to
    https://cds.climate.copernicus.eu/requests?tab=all

    Parameters
    ----------
    product : str
        CDS product name (e.g. 'reanalysis-era5-single-levels').
    chunks : dict[str, int] or None, optional
        Dask chunk specification for lazy loading.
    tmpdir : PathLike or None, optional
        Directory for temporary download files. If None, files are
        cleaned up via finalizer on GC.
    lock : SerializableLock or None, optional
        Lock for thread-safe file creation.
    **updates : Any
        Additional CDS API request parameters. Must include at least
        'variable', 'year', and 'month'.

    Returns
    -------
    xr.Dataset
        Downloaded ERA5 data.

    Examples
    --------
    >>> ds = retrieve_data(
    ...     product='reanalysis-era5-single-levels',
    ...     chunks={'time': 1, 'x': 100, 'y': 100},
    ...     tmpdir='/tmp',
    ...     lock=None,
    ...     year='2020',
    ...     month='01',
    ...     variable=['10m_u_component_of_wind', '10m_v_component_of_wind'],
    ...     data_format='netcdf',
    ... )
    """
    request: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "download_format": "unarchived",
    }
    request.update(updates)

    assert {"year", "month", "variable"}.issubset(request), (
        "Need to specify at least 'variable', 'year' and 'month'"
    )

    logger.debug("Requesting %s with API request: %s", product, request)

    client = cdsapi.Client(
        info_callback=logger.debug, debug=logging.root.level <= logging.DEBUG
    )
    result = client.retrieve(product, request)

    cm: AbstractContextManager = nullcontext() if lock is None else lock

    suffix = f".{request['data_format']}"
    with cm:
        fd, target = mkstemp(suffix=suffix, dir=tmpdir)
        os.close(fd)

        timestr = f"{request['year']}-{request['month']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info("CDS: Downloading variables\n\t%s\n", varstr)
        result.download(target)

    # Convert from grib to netcdf locally, same conversion as in CDS backend
    if request["data_format"] == "grib":
        ds = open_with_grib_conventions(target, chunks=chunks, tmpdir=tmpdir)
    else:
        ds = xr.open_dataset(target, chunks=sanitize_chunks(chunks))
        if tmpdir is None:
            add_finalizer(ds, target)

    return ds


def get_data(
    cutout: Any,
    feature: str,
    tmpdir: PathLike,
    lock: SerializableLock | None = None,
    data_format: Literal["grib", "netcdf"] = "grib",
    monthly_requests: bool = False,
    concurrent_requests: bool = False,
    **creation_parameters: Any,
) -> xr.Dataset:
    """
    Download ERA5 data for a given feature.

    Dispatches to feature-specific ``get_data_{feature}`` functions,
    optionally applies ``sanitize_{feature}``, and concatenates time chunks.

    Parameters
    ----------
    cutout : Cutout
        Cutout object defining the spatial and temporal extent.
    feature : str
        Feature to retrieve (e.g. 'wind', 'influx', 'temperature',
        'runoff', 'height').
    tmpdir : PathLike
        Directory for temporary download files.
    lock : SerializableLock or None, optional
        Lock for thread-safe file creation.
    data_format : {{'grib', 'netcdf'}}, optional
        Download format. Default 'grib'; ``grib`` is recommended over
        ``netcdf`` because the CDSAPI limits request size for the latter.
    monthly_requests : bool, optional
        If True, split API requests by month. Default False.
    concurrent_requests : bool, optional
        If True, use dask.delayed for parallel downloads. Default False.
    **creation_parameters : Any
        Additional parameters; 'sanitize' (bool, default True) controls
        whether post-processing is applied.

    Returns
    -------
    xr.Dataset
        ERA5 data for the requested feature, aligned to cutout coordinates.
    """
    coords = cutout.coords

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params: dict[str, Any] = {
        "product": "reanalysis-era5-single-levels",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": f"{cutout.dx}/{cutout.dy}",
        "tmpdir": tmpdir,
        "lock": lock,
        "data_format": data_format,
    }

    func: Callable[[dict[str, Any]], xr.Dataset] | None = globals().get(
        f"get_data_{feature}"
    )
    sanitize_func: Callable[[xr.Dataset], xr.Dataset] | None = globals().get(
        f"sanitize_{feature}"
    )

    logger.info("Requesting data for feature %s...", feature)

    def retrieve_once(time: dict[str, Any]) -> xr.Dataset:
        ds = func({**retrieval_params, **time})  # type: ignore[misc]
        if sanitize and sanitize_func is not None:
            ds = sanitize_func(ds)
        return ds

    if feature in static_features:
        static_times = retrieval_times(coords, static=True)
        assert isinstance(static_times, dict)
        return retrieve_once(static_times).squeeze()

    time_chunks = retrieval_times(coords, monthly_requests=monthly_requests)
    assert isinstance(time_chunks, list)
    if concurrent_requests:
        delayed_datasets = [delayed(retrieve_once)(chunk) for chunk in time_chunks]
        datasets = compute(*delayed_datasets)
    else:
        datasets = map(retrieve_once, time_chunks)

    result = xr.concat(datasets, dim="time").sel(time=coords["time"])
    assert isinstance(result, xr.Dataset)
    return result
