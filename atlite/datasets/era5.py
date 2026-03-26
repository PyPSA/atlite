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
import weakref
from pathlib import Path
from tempfile import mkstemp
from typing import TYPE_CHECKING, Any, Literal

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import arctan2, sqrt
from numpy import atleast_1d

from atlite.gis import maybe_swap_spatial_dims
from atlite.pv.solar_position import SolarPosition

if TYPE_CHECKING:
    from collections.abc import Callable

    from dask.utils import SerializableLock

    from atlite._types import ERA5RetrievalParams, PathLike

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
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds.drop_vars(["expver", "number"], errors="ignore")


def get_data_wind(retrieval_params: ERA5RetrievalParams) -> xr.Dataset:
    """
    Retrieve and compute wind speed variables from ERA5.

    Downloads u/v wind components at 10m and 100m, computes wind speed,
    shear exponent, azimuth angle, and surface roughness.

    Parameters
    ----------
    retrieval_params : ERA5RetrievalParams
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

    for h in [10, 100]:
        ds[f"wnd{h}m"] = sqrt(ds[f"u{h}"] ** 2 + ds[f"v{h}"] ** 2).assign_attrs(
            units=ds[f"u{h}"].attrs["units"], long_name=f"{h} metre wind speed"
        )
    ds["wnd_shear_exp"] = (
        np.log(ds["wnd10m"] / ds["wnd100m"]) / np.log(10 / 100)
    ).assign_attrs(units="", long_name="wind shear exponent")

    azimuth = arctan2(ds["u100"], ds["v100"])
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)

    ds = ds.drop_vars(["u100", "v100", "u10", "v10", "wnd10m"])
    return ds.rename({"fsr": "roughness"})


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


def get_data_influx(retrieval_params: ERA5RetrievalParams) -> xr.Dataset:
    """
    Retrieve and compute solar radiation variables from ERA5.

    Downloads radiation components, converts from J/m² to W/m², computes
    albedo, diffuse radiation, and solar position (altitude/azimuth).

    Parameters
    ----------
    retrieval_params : ERA5RetrievalParams
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

    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a] / (60.0 * 60.0)
        ds[a].attrs["units"] = "W m**-2"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        time_shift = pd.to_timedelta("-30 minutes")
        sp = SolarPosition(ds, time_shift=time_shift)
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    return xr.merge([ds, sp])


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


def get_data_temperature(retrieval_params: ERA5RetrievalParams) -> xr.Dataset:
    """
    Retrieve temperature variables from ERA5.

    Downloads 2m temperature, soil temperature (level 4), and 2m dewpoint.

    Parameters
    ----------
    retrieval_params : ERA5RetrievalParams
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


def get_data_runoff(retrieval_params: ERA5RetrievalParams) -> xr.Dataset:
    """
    Retrieve runoff data from ERA5.

    Parameters
    ----------
    retrieval_params : ERA5RetrievalParams
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


def get_data_height(retrieval_params: ERA5RetrievalParams) -> xr.Dataset:
    """
    Retrieve geopotential and convert to terrain height.

    Parameters
    ----------
    retrieval_params : ERA5RetrievalParams
        CDS API retrieval parameters including area, time, and format.

    Returns
    -------
    xr.Dataset
        Dataset with 'height' variable in meters.
    """
    ds = retrieve_data(variable="geopotential", **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    return _add_height(ds)


def _area(coords: dict[str, xr.DataArray]) -> list[float]:
    """
    Extract CDS API bounding box from coordinates.

    Parameters
    ----------
    coords : dict[str, xr.DataArray]
        Coordinate arrays with 'x' (longitude) and 'y' (latitude).

    Returns
    -------
    list[float]
        Bounding box as [north, west, south, east].
    """
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    return [y1, x0, y0, x1]


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


def noisy_unlink(path: PathLike) -> None:
    """
    Remove a file with debug logging, handling PermissionError gracefully.

    Parameters
    ----------
    path : PathLike
        Path to the file to delete.
    """
    logger.debug("Deleting file %s", path)
    try:
        Path(path).unlink()
    except PermissionError:
        logger.error("Unable to delete file %s, as it is still in use.", path)


def add_finalizer(ds: xr.Dataset, target: PathLike) -> None:
    """
    Register a weak-reference callback to delete a temp file when the dataset
    is garbage collected.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose lifetime controls the temp file.
    target : PathLike
        Path to the temporary file to clean up.
    """
    logger.debug("Adding finalizer for %s", target)
    weakref.finalize(ds._close.__self__.ds, noisy_unlink, target)


def sanitize_chunks(chunks: Any, **dim_mapping: str) -> Any:
    """
    Remap internal dimension names to ERA5/CDS dimension names in chunk specs.

    Translates atlite dimension names (time, x, y) to the corresponding
    ERA5 names (valid_time, longitude, latitude).

    Parameters
    ----------
    chunks : Any
        Chunk specification. If not a dict, returned as-is.
    **dim_mapping : str
        Additional or override dimension name mappings.

    Returns
    -------
    Any
        Remapped chunk dict, or original value if not a dict.
    """
    dim_mapping = {
        "time": "valid_time",
        "x": "longitude",
        "y": "latitude",
    } | dim_mapping
    if not isinstance(chunks, dict):
        return chunks

    return {
        extname: chunks[intname]
        for intname, extname in dim_mapping.items()
        if intname in chunks
    }


def open_with_grib_conventions(
    grib_file: PathLike,
    chunks: dict[str, int] | None = None,
    tmpdir: PathLike | None = None,
) -> xr.Dataset:
    """
    Open a GRIB file using cfgrib with standardized coordinate conventions.

    Renames forecast/pressure/model dimensions and expands missing dimensions.
    If ``tmpdir`` is None, registers a finalizer to delete the file on GC.

    Parameters
    ----------
    grib_file : PathLike
        Path to the GRIB file.
    chunks : dict[str, int] or None, optional
        Dask chunk specification for lazy loading.
    tmpdir : PathLike or None, optional
        If set, the file is kept (managed externally).

    Returns
    -------
    xr.Dataset
        Opened dataset with standardized dimensions.
    """
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
    rename_vars = {
        "time": "forecast_reference_time",
        "step": "forecast_period",
        "isobaricInhPa": "pressure_level",
        "hybrid": "model_level",
    }
    rename_vars = {k: v for k, v in rename_vars.items() if k in ds}
    ds = ds.rename(rename_vars)

    ds = safely_expand_dims(ds, ["valid_time", "pressure_level", "model_level"])

    return ds


def retrieve_data(
    product: str,
    chunks: dict[str, int] | None = None,
    tmpdir: PathLike | None = None,
    lock: SerializableLock | None = None,
    **updates: Any,
) -> xr.Dataset:
    """
    Download ERA5 data from the CDS API and return as an xarray Dataset.

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

    if lock is None:
        lock = nullcontext()

    suffix = f".{request['data_format']}"
    with lock:
        fd, target = mkstemp(suffix=suffix, dir=tmpdir)
        os.close(fd)

        timestr = f"{request['year']}-{request['month']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info("CDS: Downloading variables\n\t%s\n", varstr)
        result.download(target)

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
    Main entry point for downloading ERA5 data for a given feature.

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
        Download format. Default 'grib'.
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

    retrieval_params: ERA5RetrievalParams = {
        "product": "reanalysis-era5-single-levels",
        "area": _area(coords),
        "chunks": cutout.chunks,
        "grid": f"{cutout.dx}/{cutout.dy}",
        "tmpdir": tmpdir,
        "lock": lock,
        "data_format": data_format,
    }

    func: Callable[[ERA5RetrievalParams], xr.Dataset] | None = globals().get(
        f"get_data_{feature}"
    )
    sanitize_func: Callable[[xr.Dataset], xr.Dataset] | None = globals().get(
        f"sanitize_{feature}"
    )

    logger.info("Requesting data for feature %s...", feature)

    def retrieve_once(time: dict[str, Any]) -> xr.Dataset:
        ds = func({**retrieval_params, **time})  # type: ignore[misc, typeddict-item]
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

    return xr.concat(datasets, dim="time").sel(time=coords["time"])
