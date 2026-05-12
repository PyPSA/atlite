# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Download ERA5 data from the NCAR THREDDS server"""

from __future__ import annotations

import hashlib
import logging
import math
import urllib.error
import warnings
from calendar import monthrange
from collections.abc import Callable
from datetime import date
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import dask
import numpy as np
import pandas as pd
import tenacity
import xarray as xr
from dask import compute, delayed
from dask.threaded import ContextAwareThreadPoolExecutor
from pydap.client import open_url

from atlite.datasets.era5 import (
    _add_height,
    sanitize_chunks,
    sanitize_influx,
    sanitize_runoff,
    sanitize_wind,
)
from atlite.gis import maybe_swap_spatial_dims
from atlite.pv.solar_position import SolarPosition

#  avoid circular imports during typecheck
if TYPE_CHECKING:
    from atlite.cutout import Cutout


TemporalSpec = dict[str, str | np.ndarray | None]
SpatialSpec = dict[str, int]
RawArrays = dict[str, xr.DataArray]
Handler = Callable[["Cutout", Path], xr.Dataset]
Sanitizer = Callable[[xr.Dataset], xr.Dataset]

# Custom dask thread pool to limit simultaneous THREDDS requests to avoid rate limits
_FETCH_POOL = ContextAwareThreadPoolExecutor(8, thread_name_prefix="ncar-fetch")

# zarr-python can use consolidated metadata, but releases a warning because some
# other implementations apparently can't. We ignore the warning
_ZARR_CONSOLIDATED_METADATA_WARNING = (
    r"Consolidated metadata is currently not part in the Zarr format 3 "
    r"specification\."
)
warnings.filterwarnings(
    "ignore",
    message=_ZARR_CONSOLIDATED_METADATA_WARNING,
    category=UserWarning,
)

logger = logging.getLogger(__name__)
logging.getLogger("pydap").setLevel(logging.WARNING)

# default chunk size to avoid OOM for large cutouts
_ZARR_DISK_CHUNKS = {"time": 100}

crs = 4326

features: dict[str, list[str]] = {
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

static_features: set[str] = {"height"}

_DODS_BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000"
_ERA5_RES = 0.25  # ERA5 is stored at 0.25/0.25 deg. grid resolution on NCAR THREDDS
_N_LON = int(round(360.0 / _ERA5_RES))
_BBOX_PAD = 0.5  # degrees added to each side for interpolation

# atlite name → (NCAR product dir, NCAR param code, DAP2 variable name)
# mapping of variables for creating download URLs
VARIABLES: dict[str, tuple[str, str, str]] = {
    # analysis surface (1D time)
    "u10": ("e5.oper.an.sfc", "128_165_10u", "VAR_10U"),
    "v10": ("e5.oper.an.sfc", "128_166_10v", "VAR_10V"),
    "u100": ("e5.oper.an.sfc", "228_246_100u", "VAR_100U"),
    "v100": ("e5.oper.an.sfc", "228_247_100v", "VAR_100V"),
    "fsr": ("e5.oper.an.sfc", "128_244_fsr", "FSR"),
    "t2m": ("e5.oper.an.sfc", "128_167_2t", "VAR_2T"),
    "d2m": ("e5.oper.an.sfc", "128_168_2d", "VAR_2D"),
    "stl4": ("e5.oper.an.sfc", "128_236_stl4", "STL4"),
    # forecast accumulation (2D time: forecast_initial_time × forecast_hour)
    "ssrd": ("e5.oper.fc.sfc.accumu", "128_169_ssrd", "SSRD"),
    "ssr": ("e5.oper.fc.sfc.accumu", "128_176_ssr", "SSR"),
    "fdir": ("e5.oper.fc.sfc.accumu", "228_021_fdir", "FDIR"),
    "tisr": ("e5.oper.fc.sfc.accumu", "128_212_tisr", "TISR"),
    "ro": ("e5.oper.fc.sfc.accumu", "128_205_ro", "RO"),
    # invariant (no time)
    "z": ("e5.oper.invariant", "128_129_z", "Z"),
}


# ---------------------------------------------------------------------------
# Functions for building download URLs
# ---------------------------------------------------------------------------


def _build_url(tspec: TemporalSpec, sspec: SpatialSpec, var_name: str) -> str:
    """
    Assemble a DAP2 download URL. The dataset is stored in monthly/biweekly NetCDF
    files for each variable. A single file covers the whole globe, but DAP constraint
    expressions allow us to download a spatial subset of that. So for a cutout that is
    limited in space and time, we:
    1) Determine which files cover the required time period (TemporalSpec)
    2) Determine which spatial slice of those files is required for the cutout (SpatialSpec)

    Parameters
    ----------
    tspec : dict
        Temporal file specification from :func:`_temporal_file_specs`.
    sspec : dict
        Spatial index specification from :func:`_spatial_specs`.
    var_name : str
        NCAR variable name.

    Returns
    -------
    str
        URL including DAP2 constraint expression.
    """
    spatial_ce = (
        f"[{sspec['lat_s']}:{sspec['lat_e']}][{sspec['lon_s']}:{sspec['lon_e']}]"
    )
    return f"{tspec['base_url']}?{var_name}{tspec['time_ce']}{spatial_ce}"


def _temporal_file_specs(
    product: str, param_code: str, start: date | None, end: date | None
) -> list[TemporalSpec]:
    """
    Build temporal file specifications for an ERA5 variable. This is a list of URLs
    of the required NetCDF files for this variable, plus the indices of time variables
    within those files. Indices are required, because a DAP2 constraint expression that
    we use for spatial subsetting must hardcode values for all variables, even if you
    want the whole time range (and different months have different numbers of hours, so
    we cannot hardcode them).

    All variables follow roughly the same format:
    {BASE_URL}/{PRODUCT_CODE}/{YYYYMM}/{PRODUCT_CODE}.{PARAMETER_CODE}.ll025sc.{YYYYMMDDHH_start}.{YYYYMMDDHH_end}.nc

    But invariant, analysis, and forecast variables are stored in different formats underneath, hence
    the three branches.

    Parameters
    ----------
    product : str
        NCAR product directory.
    param_code : str
        ERA5 parameter code used in the NCAR file name.
    start, end : datetime.date or None
        Inclusive requested date range. Invariant variables do not require a
        date range.

    Returns
    -------
    list of dict
        Dictionaries with ``base_url``, ``time_ce`` and ``time_coord`` entries.
    """
    specs: list[TemporalSpec] = []

    if product == "e5.oper.invariant":
        specs.append(
            {
                "base_url": (
                    f"{_DODS_BASE}/e5.oper.invariant/197901/"
                    "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
                ),
                "time_ce": "[0:0]",
                "time_coord": None,
            }
        )

    elif product == "e5.oper.an.sfc":
        if start is None or end is None:
            raise ValueError("Analysis variables require a time range.")
        for first_day, last_day in _month_bounds_in_range(start, end):
            ym = first_day.strftime("%Y%m")
            n_time = (last_day - first_day).days * 24 + 23  # inclusive stop index
            fname = (
                f"{product}.{param_code}.ll025sc."
                f"{first_day.strftime('%Y%m%d')}00_{last_day.strftime('%Y%m%d')}23.nc"
            )
            t0 = np.datetime64(first_day, "h")
            t_end = np.datetime64(last_day, "h") + np.timedelta64(23, "h")
            time_coord = np.arange(
                t0, t_end + np.timedelta64(1, "h"), np.timedelta64(1, "h")
            )
            specs.append(
                {
                    "base_url": f"{_DODS_BASE}/{product}/{ym}/{fname}",
                    "time_ce": f"[0:{n_time}]",
                    "time_coord": time_coord,
                }
            )

    elif product == "e5.oper.fc.sfc.accumu":
        if start is None or end is None:
            raise ValueError("Forecast variables require a time range.")
        for first_day, last_day in _halfmonth_bounds_in_range(start, end):
            ym = first_day.strftime("%Y%m")
            n_init = (last_day - first_day).days * 2 - 1  # inclusive stop index
            fname = (
                f"{product}.{param_code}.ll025sc."
                f"{first_day.strftime('%Y%m%d')}06_{last_day.strftime('%Y%m%d')}06.nc"
            )
            init_start = np.datetime64(first_day, "h") + np.timedelta64(6, "h")
            init_end = np.datetime64(last_day, "h") + np.timedelta64(6, "h")
            init_times = np.arange(init_start, init_end, np.timedelta64(12, "h"))
            fhr = np.arange(1, 13, dtype="int64")  # forecast hours 1..12 (12 steps)
            time_coord = (
                init_times[:, None] + fhr[None, :].astype("timedelta64[h]")
            ).ravel()
            specs.append(
                {
                    "base_url": f"{_DODS_BASE}/{product}/{ym}/{fname}",
                    "time_ce": f"[0:{n_init}][0:{11}]",
                    "time_coord": time_coord,
                }
            )

    else:
        raise ValueError(f"Unknown product: {product!r}")

    return specs


def _month_bounds_in_range(start: date, end: date) -> list[tuple[date, date]]:
    """
    Return monthly file periods covering a date range. Used for analysis variables,
    which are stored in monthly files.

    Parameters
    ----------
    start, end : datetime.date
        Inclusive date range.

    Returns
    -------
    list of tuple of datetime.date
        Month start and month end pairs.
    """
    if start > end:
        return []
    out: list[tuple[date, date]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        _, last = monthrange(cur.year, cur.month)
        out.append((date(cur.year, cur.month, 1), date(cur.year, cur.month, last)))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def _add_month(y: int, m: int, delta: int) -> tuple[int, int]:
    """
    Add months to a year-month pair.

    Parameters
    ----------
    y, m : int
        Year and month.
    delta : int
        Number of months to add.

    Returns
    -------
    tuple of int
        Updated ``(year, month)`` pair.
    """
    m += delta
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return y, m


def _halfmonth_bounds_in_range(start: date, end: date) -> list[tuple[date, date]]:
    """
    Return half-month file periods covering a date range. Used for forecast variables,
    which are stored in (roughly) 15-day files.

    Parameters
    ----------
    start, end : datetime.date
        Inclusive date range.

    Returns
    -------
    list of tuple of datetime.date
        Period start and next period start pairs.

    Notes
    -----
    Forecast files initialized at 06:00 UTC require the preceding half-month
    file when the requested range starts on the 1st or 16th.
    """
    if start > end:
        return []

    def cur_start(d: date) -> date:
        return date(d.year, d.month, 1) if d.day <= 15 else date(d.year, d.month, 16)

    def next_start(s: date) -> date:
        return (
            date(s.year, s.month, 16)
            if s.day == 1
            else date(*_add_month(s.year, s.month, 1), 1)
        )

    out: list[tuple[date, date]] = []
    cs = cur_start(start)

    if start.day in (1, 16):
        if cs.day == 1:
            py, pm = _add_month(cs.year, cs.month, -1)
            out.append((date(py, pm, 16), cs))
        else:
            out.append((date(cs.year, cs.month, 1), cs))

    cur = cs
    while cur <= end:
        nxt = next_start(cur)
        if nxt > start:
            out.append((cur, nxt))
        cur = nxt

    return out


def _spatial_specs(
    north: float, south: float, west: float, east: float
) -> list[SpatialSpec]:
    """
    Map atlite coords to ERA5 grid index segments.

    atlite coordinate system:
    - x: -180:180
    - y: -90:90

    ncar d633000 coordinate system:
    - x: 0:360
    - y: -90:90


    Parameters
    ----------
    north, south, west, east : float
        Bounding box edges in degrees.

    Returns
    -------
    list of dict
        One or two index segments with latitude and longitude start/stop
        entries. Two segments are returned when the box crosses the prime
        meridian in NCAR's 0-360 longitude grid.
    """
    lat_s = math.ceil((90 - north) / _ERA5_RES)
    lat_e = math.floor((90 - south) / _ERA5_RES)

    base: SpatialSpec = {"lat_s": lat_s, "lat_e": lat_e}
    # handle whole world slices
    if east - west >= 360.0 - 1e-9:
        return [{**base, "lon_s": 0, "lon_e": _N_LON - 1}]

    west_360 = west % 360
    east_360 = east % 360
    j_west = math.ceil(west_360 / _ERA5_RES) % _N_LON
    j_east = math.floor(east_360 / _ERA5_RES) % _N_LON

    if j_west <= j_east:
        return [{**base, "lon_s": j_west, "lon_e": j_east}]
    else:
        return [
            {**base, "lon_s": j_west, "lon_e": _N_LON - 1},
            {**base, "lon_s": 0, "lon_e": j_east},
        ]


# ---------------------------------------------------------------------------
# Download/retrieval machinery
# ---------------------------------------------------------------------------


def _fetch_vars(atlite_names: list[str], cutout: Cutout, tmpdir: Path) -> RawArrays:
    """
    Get raw ERA5 data relevant to a list of atlite variables.

    Parameters
    ----------
    atlite_names : list of str
        Variable names used by atlite.
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.

    Returns
    -------
    dict
        Mapping from variable name to native-grid :class:`xarray.DataArray`.
    """
    coords = cutout.coords
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()
    area = {
        "north": min(90.0, y1 + _BBOX_PAD),
        "south": max(-90.0, y0 - _BBOX_PAD),
        "west": x0 - _BBOX_PAD,
        "east": x1 + _BBOX_PAD,
    }
    if "time" in cutout.coords:
        time_index = cutout.coords["time"].to_index()
        start, end = time_index[0].date(), time_index[-1].date()
    else:
        start, end = None, None
    chunks = cutout.chunks or {"time": 100}
    return {
        name: _load_var(
            name, **area, start=start, end=end, tmpdir=tmpdir, chunks=chunks
        )
        for name in atlite_names
    }


def _load_var(
    atlite_name: str,
    north: float,
    south: float,
    west: float,
    east: float,
    start: date | None,
    end: date | None,
    tmpdir: Path,
    chunks: dict[str, int],
) -> xr.DataArray:
    """
    Load one NCAR variable for the requested spatial extent and time range.

    Parameters
    ----------
    atlite_name : str
        Variable name used by atlite.
    north, south, west, east : float
        Requested bounding box in degrees.
    start, end : datetime.date or None
        Inclusive requested date range.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.
    chunks : dict
        Dask chunks passed to :func:`xarray.open_mfdataset`.

    Returns
    -------
    xarray.DataArray
        Lazy array on the native ERA5 latitude/longitude grid.
    """
    product, param_code, var_name = VARIABLES[atlite_name]
    is_invariant = product == "e5.oper.invariant"
    chunks = sanitize_chunks(chunks, time="time")

    tspecs = _temporal_file_specs(product, param_code, start, end)
    sspecs = _spatial_specs(north, south, west, east)
    logger.info(
        f"{atlite_name}: {len(tspecs)} file(s), {len(sspecs)} spatial segment(s)"
    )

    n = len(tspecs)
    # we use a custom thread pool to limit the number of simultaneous request to THREDDS
    with dask.config.set(pool=_FETCH_POOL):
        # get list of locations of all downloaded files.
        all_paths = compute(
            *[
                delayed(_fetch_file)(
                    _build_url(ts, ss, var_name),
                    var_name,
                    atlite_name,
                    ss["lat_s"],
                    ss["lat_e"],
                    ss["lon_s"],
                    ss["lon_e"],
                    product,
                    ts["time_coord"],
                    tmpdir,
                )
                for ss in sspecs
                for ts in tspecs
            ]
        )

    # open all relevant files. requests which cross the zero meridian will have two
    # files - one for data east of the meridian, one for data west of the meridian.
    # those need to be concatenated along the longitude.

    # invariant will only have one timestep, so it doesn't require concatenation along time
    if is_invariant:
        if len(sspecs) == 1:
            da = xr.open_dataset(str(all_paths[0]), engine="zarr")[atlite_name]
        else:
            das = [
                xr.open_dataset(str(p), engine="zarr")[atlite_name] for p in all_paths
            ]
            da = xr.concat(das, dim="longitude")
        da = da.sortby("longitude")
        da.encoding = {}
        return da

    # everything else has a time dimension, so files must be concatenated along time and space.
    if len(sspecs) == 1:
        da = xr.open_mfdataset(
            [str(p) for p in all_paths],
            engine="zarr",
            concat_dim="time",
            combine="nested",
            chunks={},
        )[atlite_name]
    else:
        ds_a = xr.open_mfdataset(
            [str(p) for p in all_paths[:n]],
            engine="zarr",
            concat_dim="time",
            combine="nested",
            chunks={},
        )
        ds_b = xr.open_mfdataset(
            [str(p) for p in all_paths[n:]],
            engine="zarr",
            concat_dim="time",
            combine="nested",
            chunks={},
        )
        da = xr.concat([ds_a, ds_b], dim="longitude")[atlite_name]

    da = da.sortby("longitude")
    da = da.sortby("time").sel(time=slice(str(start), str(end)))
    if chunks != _ZARR_DISK_CHUNKS:
        da = da.chunk(chunks)
    da.encoding = {}
    return da


def _fetch_file(
    url: str,
    var_name: str,
    atlite_name: str,
    lat_s: int,
    lat_e: int,
    lon_s: int,
    lon_e: int,
    product: str,
    time_coord: np.ndarray | None,
    tmpdir: Path,
) -> Path:
    """
    Caching function, also enables recovery from a failed run.
    Due to the large number of files required, we benefit a lot
    from reading and writing them in parallel. We use zarr for
    this, as the zarr engine is thread-safe by default, whereas
    the HDF5 engine used by NetCDF is not, and requires serialising
    reads and writes.

    Parameters
    ----------
    url : str
        DAP2 URL including constraint expression.
    var_name : str
        NCAR variable name.
    atlite_name : str
        Variable name used by atlite.
    lat_s, lat_e, lon_s, lon_e : int
        Inclusive ERA5 grid index bounds.
    product : str
        NCAR product directory.
    time_coord : numpy.ndarray or None
        Hourly time coordinate for the downloaded values.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.

    Returns
    -------
    pathlib.Path
        Path to the Zarr store.
    """
    # the url features the constraint expression, so it uniquely identifies a file
    # incl. spatial and temporal subset.
    key = hashlib.md5(url.encode()).hexdigest()[:16]
    cache = tmpdir / f"era5ncar_{key}.zarr"
    if cache.exists():
        logger.debug(f"cache hit: {cache.name}")
        return cache

    da = _download_to_array(
        url, var_name, atlite_name, lat_s, lat_e, lon_s, lon_e, product, time_coord
    )
    # atomic write
    tmp = cache.with_name(cache.name + ".tmp")
    encoding = None
    if da.name is not None and "time" in da.dims:
        zarr_chunks = tuple(
            min(_ZARR_DISK_CHUNKS.get(dim, size), size)
            for dim, size in zip(da.dims, da.shape, strict=True)
        )
        encoding = {da.name: {"chunks": zarr_chunks}}
    da.to_zarr(tmp, mode="w", encoding=encoding)
    tmp.rename(cache)
    logger.debug(f"cached: {cache.name}")
    return cache


# We use tenacity to retry downloads when they fail due to network errors or
# rate limits. Exponential backoff ensures we don't hit the server with
# a lot of retries at the same time. Retries only happen at the below
# network-related errors
_TRANSIENT_EXCEPTIONS = (urllib.error.URLError, OSError, TimeoutError)


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
    wait=tenacity.wait_exponential_jitter(initial=4, max=120, jitter=4),
    stop=tenacity.stop_after_attempt(8),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _download_to_array(
    url: str,
    var_name: str,
    atlite_name: str,
    lat_s: int,
    lat_e: int,
    lon_s: int,
    lon_e: int,
    product: str,
    time_coord: np.ndarray | None,
) -> xr.DataArray:
    """
    Download one ERA5 file.

    Parameters
    ----------
    url : str
        DAP2 URL including constraint expression.
    var_name : str
        NCAR variable name.
    atlite_name : str
        Variable name used by atlite.
    lat_s, lat_e, lon_s, lon_e : int
        Inclusive ERA5 grid index bounds.
    product : str
        NCAR product directory.
    time_coord : numpy.ndarray or None
        Hourly time coordinate for the downloaded values.

    Returns
    -------
    xarray.DataArray
        Downloaded variable with ``time`` where applicable and native ERA5
        latitude/longitude coordinates.
    """
    ds_pydap = open_url(f"dap2://{url[8:]}")
    arr = np.asarray(ds_pydap[var_name][var_name][:])

    if product == "e5.oper.fc.sfc.accumu":
        # forecast variables have a 2D time index [forecast_init][forecast_time]
        # we flatten them here for consistent processing upstream
        # note - the values are NOT actually accumulated within a forecast,
        # the timeseries is already deaccumnulated
        data = arr.reshape(-1, arr.shape[2], arr.shape[3])
    elif product == "e5.oper.an.sfc":
        data = arr
    elif product == "e5.oper.invariant":
        # invariant variables are stored as a single timestep in 1970
        # we need to extract the data
        data = arr[0]
    else:
        raise ValueError(f"Unknown product: {product!r}")

    # assign lat/lon values
    lat_vals = 90.0 - np.arange(lat_s, lat_e + 1) * _ERA5_RES
    raw_lons = np.arange(lon_s, lon_e + 1) * _ERA5_RES
    lon_vals = np.where(raw_lons >= 180.0, raw_lons - 360.0, raw_lons)

    if time_coord is None:
        return xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={"latitude": lat_vals, "longitude": lon_vals},
            name=atlite_name,
        )
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": time_coord, "latitude": lat_vals, "longitude": lon_vals},
        name=atlite_name,
    )


def _is_native_grid(cutout: Cutout) -> bool:
    """
    Return whether the cutout uses the native ERA5 resolution.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid.

    Returns
    -------
    bool
        True when both spatial resolutions are 0.25 degrees.
    """
    return abs(cutout.dx - _ERA5_RES) < 1e-9 and abs(cutout.dy - _ERA5_RES) < 1e-9


def _with_periodic_longitude(da: xr.DataArray, target_lon: np.ndarray) -> xr.DataArray:
    """
    Add a temporary +180 longitude column for interpolation across the seam.

    Atlite-facing coordinates remain [-180, 180). The duplicate is only used as
    an interpolation source when a target longitude lies beyond the largest
    available positive source longitude and the -180 column is present.

    Parameters
    ----------
    da : xarray.DataArray
        Native-grid variable with a ``longitude`` coordinate.
    target_lon : numpy.ndarray
        Target longitudes requested by the cutout.

    Returns
    -------
    xarray.DataArray
        ``da`` unchanged, or ``da`` with a temporary duplicate seam column at
        ``+180`` for interpolation.
    """
    lon = da.longitude.values
    if lon.size == 0 or np.nanmax(target_lon) <= np.nanmax(lon) + 1e-9:
        return da
    if np.any(np.isclose(lon, 180.0, atol=1e-9)):
        return da
    if not np.any(np.isclose(lon, -180.0, atol=1e-9)):
        return da

    seam = da.sel(longitude=[-180.0], method="nearest", tolerance=1e-6)
    seam = seam.assign_coords(longitude=[180.0])
    return xr.concat([da, seam], dim="longitude").sortby("longitude")


def _regrid_to_target(arrays: RawArrays, cutout: Cutout) -> RawArrays:
    """
    Regrid raw variables to the cutout grid. NCAR stores ERA5 at a 0.25/0.25 deg.
    grid, so the best we can do is interpolate within it.

    The CDS source in era5.py is a better choice if you need data at a grid that is
    not a multiple of 0.25/0.25 deg., as they do sophisticated interpolation from the
    raw spectral and Gaussian grids.

    This function is here for feature parity with era5.py.

    Parameters
    ----------
    arrays : dict
        Native-grid data arrays keyed by variable name.
    cutout : atlite.Cutout
        Cutout defining the target grid.

    Returns
    -------
    dict
        Data arrays on the cutout grid.
    """
    target_lat = cutout.coords["y"].values
    target_lon = cutout.coords["x"].values
    if _is_native_grid(cutout):
        return {
            name: da.sel(
                latitude=target_lat,
                longitude=target_lon,
                method="nearest",
                tolerance=1e-6,
            )
            for name, da in arrays.items()
        }
    else:
        logger.warning(
            f"NCAR ERA5 regrids from native 0.25 deg. to dx={cutout.dx}, dy={cutout.dy} "
            f"(target shape y={len(target_lat)}, x={len(target_lon)}); "
            "results may differ from CDS/MIR."
        )
        return {
            # if data is close to zero longitude, we mirror it for interpolation purposes
            name: _with_periodic_longitude(da, target_lon).interp(
                latitude=target_lat, longitude=target_lon, method="linear"
            )
            for name, da in arrays.items()
        }


def _rename_and_clean_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize spatial coordinates to atlite conventions.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``latitude`` and ``longitude`` coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with ``y``, ``x``, ``lat`` and ``lon`` coordinates.
    """
    ds = ds.rename({"latitude": "y", "longitude": "x"})
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5),
        y=np.round(ds.y.astype(float), 5),
    )
    ds = maybe_swap_spatial_dims(ds)
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds


# ---------------------------------------------------------------------------
# Per-feature handlers (mirror atlite.datasets.era5.get_data_<feature>)
# ---------------------------------------------------------------------------


def get_data_wind(cutout: Cutout, tmpdir: Path) -> xr.Dataset:
    """
    Retrieve and prepare wind variables.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``wnd100m``, ``wnd_shear_exp``, ``wnd_azimuth`` and
        ``roughness``.
    """
    arrays = _fetch_vars(["u10", "v10", "u100", "v100", "fsr"], cutout, tmpdir)
    arrays = _regrid_to_target(arrays, cutout)
    ds = _rename_and_clean_coords(xr.Dataset(arrays))

    for h in (10, 100):
        ds[f"wnd{h}m"] = np.sqrt(ds[f"u{h}"] ** 2 + ds[f"v{h}"] ** 2).assign_attrs(
            units="m s**-1", long_name=f"{h} metre wind speed"
        )
    ds["wnd_shear_exp"] = (
        np.log(ds["wnd10m"] / ds["wnd100m"]) / np.log(10 / 100)
    ).assign_attrs(units="", long_name="wind shear exponent")

    azimuth = xr.apply_ufunc(np.arctan2, ds["u100"], ds["v100"], dask="allowed")
    ds["wnd_azimuth"] = xr.where(
        azimuth >= 0, azimuth, azimuth + 2 * np.pi
    ).assign_attrs(units="m s**-1", long_name="100 metre U wind component")

    ds = ds.drop_vars(["u100", "v100", "u10", "v10", "wnd10m"])
    ds = ds.rename({"fsr": "roughness"})
    ds["roughness"] = ds["roughness"].assign_attrs(
        units="m", long_name="Forecast surface roughness"
    )
    return ds


def get_data_influx(cutout: Cutout, tmpdir: Path) -> xr.Dataset:
    """
    Retrieve and prepare solar influx variables.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.

    Returns
    -------
    xarray.Dataset
        Dataset containing influx, albedo and solar position variables.
    """
    arrays = _fetch_vars(["ssrd", "ssr", "fdir", "tisr"], cutout, tmpdir)

    # Avoid negative albedo artifacts at off-grid day/night boundaries.
    if not _is_native_grid(cutout):
        arrays["ssr"] = arrays["ssr"].where(
            arrays["ssr"] <= arrays["ssrd"], other=arrays["ssrd"]
        )

    arrays = _regrid_to_target(arrays, cutout)
    ds = _rename_and_clean_coords(xr.Dataset(arrays))

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
        ds[a] = ds[a] / 3600.0
        ds[a].attrs["units"] = "W m**-2"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift=pd.to_timedelta("-30 minutes"))
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp])
    return ds


def get_data_temperature(cutout: Cutout, tmpdir: Path) -> xr.Dataset:
    """
    Retrieve and prepare temperature variables.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.

    Returns
    -------
    xarray.Dataset
        Dataset containing air, soil and dewpoint temperatures.
    """
    arrays = _fetch_vars(["t2m", "d2m", "stl4"], cutout, tmpdir)
    arrays = _regrid_to_target(arrays, cutout)
    ds = _rename_and_clean_coords(xr.Dataset(arrays))
    ds = ds.rename(
        {
            "t2m": "temperature",
            "stl4": "soil temperature",
            "d2m": "dewpoint temperature",
        }
    )
    for name in ("temperature", "soil temperature", "dewpoint temperature"):
        ds[name].attrs["units"] = "K"
    return ds


def get_data_runoff(cutout: Cutout, tmpdir: Path) -> xr.Dataset:
    """
    Retrieve and prepare runoff.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``runoff``.
    """
    arrays = _fetch_vars(["ro"], cutout, tmpdir)
    arrays = _regrid_to_target(arrays, cutout)
    ds = _rename_and_clean_coords(xr.Dataset(arrays))
    ds = ds.rename({"ro": "runoff"})
    ds["runoff"].attrs["units"] = "m"
    return ds


def get_data_height(cutout: Cutout, tmpdir: Path) -> xr.Dataset:
    """
    Retrieve and prepare surface height.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid.
    tmpdir : pathlib.Path
        Directory for cached Zarr stores.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``height``.
    """
    arrays = _fetch_vars(["z"], cutout, tmpdir)
    arrays = _regrid_to_target(arrays, cutout)
    ds = _rename_and_clean_coords(xr.Dataset(arrays))
    ds = _add_height(ds)
    ds["height"].attrs["units"] = "m**2 s**-2"
    return ds


_HANDLERS: dict[str, Handler] = {
    "wind": get_data_wind,
    "influx": get_data_influx,
    "temperature": get_data_temperature,
    "runoff": get_data_runoff,
    "height": get_data_height,
}

_SANITIZERS: dict[str, Sanitizer] = {
    "wind": sanitize_wind,
    "influx": sanitize_influx,
    "runoff": sanitize_runoff,
}


# ---------------------------------------------------------------------------
# atlite Cutout entrypoint
# ---------------------------------------------------------------------------


def get_data(
    cutout: Cutout,
    feature: str,
    tmpdir: str | Path | None = None,
    lock: Any = None,
    **creation_parameters: Any,
) -> xr.Dataset:
    """
    Retrieve ERA5 feature data from NCAR THREDDS.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    feature : str
        Feature name to retrieve. Must be one of ``features``.
    tmpdir : str or pathlib.Path, optional
        Directory for cached Zarr stores.
    lock : object, optional
        Accepted for compatibility with :func:`atlite.datasets.era5.get_data`.
    **creation_parameters
        Additional creation parameters. ``sanitize`` controls whether standard
        atlite sanitizers are applied.

    Returns
    -------
    xarray.Dataset
        Prepared dataset for the requested feature.
    """
    if feature not in _HANDLERS:
        raise NotImplementedError(f"Feature {feature!r} not supported by era5_ncar")

    logger.info(f"Requesting data for feature {feature}...")

    cache_dir = Path(tmpdir) if tmpdir is not None else Path(mkdtemp())

    ds = _HANDLERS[feature](cutout, cache_dir)

    sanitize = creation_parameters.get("sanitize", True)
    if sanitize and feature in _SANITIZERS:
        ds = _SANITIZERS[feature](ds)

    if feature not in static_features:
        ds = ds.reindex(time=cutout.coords["time"])

    return ds
