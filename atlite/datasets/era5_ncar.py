# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Download ERA5 data from NCAR THREDDS (ds633.0) via OPeNDAP (DAP2) with CEs.

Design
------
- file_urls() assembles dodsC URLs directly from the NCAR file naming convention
  — no catalog fetch needed.
- _bbox_isel() maps the bounding box to ERA5 0.25° integer grid indices.
- _download_to_array() opens one file via pydap.open_url and returns a DataArray.
- _fetch_file() wraps _download_to_array with a disk cache in tmpdir: a file
  keyed by MD5(url) is written atomically; hits skip the network entirely.
- _load_var() fetches all files for one variable in parallel, then opens them
  lazily via xr.open_mfdataset so downstream computation reads from disk in
  chunks rather than holding all arrays in RAM simultaneously.
- tenacity retries each network fetch on any transient error.
"""

import hashlib
import logging
import math
import threading
from calendar import monthrange
from datetime import date
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import tenacity
import xarray as xr
from dask import compute, delayed
from pydap.client import open_url

from atlite.gis import maybe_swap_spatial_dims

logger = logging.getLogger(__name__)

_FETCH_SEMAPHORE = threading.Semaphore(8)

crs = 4326

features = {
    "influx": ["ssrd", "ssr", "fdir", "tisr"],
}

_DODC_BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000"
_ERA5_RES = 0.25  # degree
_ERA5_EPOCH = np.datetime64("1900-01-01T00:00", "h")

SOLAR_VARIABLES = {
    "ssrd": ("e5.oper.fc.sfc.accumu", "128_169_ssrd", "SSRD"),
    "ssr":  ("e5.oper.fc.sfc.accumu", "128_176_ssr",  "SSR"),
    "fdir": ("e5.oper.fc.sfc.accumu", "228_021_fdir", "FDIR"),
    "tisr": ("e5.oper.fc.sfc.accumu", "128_212_tisr", "TISR"),
}


# ---------------------------------------------------------------------------
# URL assembly
# ---------------------------------------------------------------------------

def _month_bounds_in_range(start: date, end: date) -> list[tuple[date, date]]:
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
    m += delta
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return y, m


def _halfmonth_bounds_in_range(start: date, end: date) -> list[tuple[date, date]]:
    """
    Return half-month (period_start, next_period_start) pairs covering [start, end].

    When start falls exactly on the 1st or 16th the preceding half-month is also
    included, because accumulated ERA5 forecast files for that day begin at 06:00 UTC
    so hours 00–06 live in the previous file.
    """
    if start > end:
        return []

    def cur_start(d: date) -> date:
        return date(d.year, d.month, 1) if d.day <= 15 else date(d.year, d.month, 16)

    def next_start(s: date) -> date:
        return date(s.year, s.month, 16) if s.day == 1 else date(*_add_month(s.year, s.month, 1), 1)

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


def file_urls(
    product: str,
    param_code: str,
    var_name: str,
    start: date,
    end: date,
    lat_s: int,
    lat_e: int,
    lon_s: int,
    lon_e: int,
) -> list[str]:
    """
    Assemble dodsC URLs with embedded DAP2 constraint expressions for [start, end].

    Each URL includes a CE that fully constrains spatial dimensions and, for
    forecast products, the time dimensions, so _fetch_file needs no size queries.

    n_init for forecast files is derived from (last_day - first_day).days * 2;
    n_hour is always 12 (ERA5 forecast steps 1–12 h).
    """
    _N_HOUR = 11  # 12 forecast steps; DAP2 stop index is inclusive
    spatial_ce = f"[{lat_s}:{lat_e}][{lon_s}:{lon_e}]"
    urls = []

    if product == "e5.oper.invariant":
        urls.append(
            f"{_DODC_BASE}/e5.oper.invariant/197901/"
            "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
        )
    elif product == "e5.oper.an.sfc":
        for first_day, last_day in _month_bounds_in_range(start, end):
            ym = first_day.strftime("%Y%m")
            n_time = (last_day - first_day).days * 24 + 23  # inclusive stop index
            ce = f"?{var_name}[0:{n_time}]{spatial_ce},time[0:{n_time}]"
            urls.append(
                f"{_DODC_BASE}/{product}/{ym}/"
                f"{product}.{param_code}.ll025sc."
                f"{first_day.strftime('%Y%m%d')}00_{last_day.strftime('%Y%m%d')}23.nc"
                f"{ce}"
            )
    elif product == "e5.oper.fc.sfc.accumu":
        for first_day, last_day in _halfmonth_bounds_in_range(start, end):
            ym = first_day.strftime("%Y%m")
            n_init = (last_day - first_day).days * 2 - 1  # inclusive stop index
            ce = (
                f"?{var_name}[0:{n_init}][0:{_N_HOUR}]{spatial_ce}"
                f",forecast_initial_time[0:{n_init}]"
                f",forecast_hour[0:{_N_HOUR}]"
            )
            urls.append(
                f"{_DODC_BASE}/{product}/{ym}/"
                f"{product}.{param_code}.ll025sc."
                f"{first_day.strftime('%Y%m%d')}06_{last_day.strftime('%Y%m%d')}06.nc"
                f"{ce}"
            )
    else:
        raise ValueError(f"Unknown product: {product!r}")
    return urls


# ---------------------------------------------------------------------------
# Spatial index helpers
# ---------------------------------------------------------------------------

def _bbox_isel(north: float, south: float, west: float, east: float) -> dict:
    """
    Convert a WGS84 bounding box to ERA5 0.25° grid integer index slices.

    Indices computed from lat[i] = 90 - i*0.25, lon[j] = j*0.25 — no
    coordinate array fetch needed.
    """
    lat_start = math.ceil((90 - north) / _ERA5_RES)
    lat_stop  = math.floor((90 - south) / _ERA5_RES) + 1
    lon_start = math.ceil((west + 180) / _ERA5_RES)
    lon_stop  = math.floor((east + 180) / _ERA5_RES) + 1
    return {
        "latitude":  slice(lat_start, lat_stop),
        "longitude": slice(lon_start, lon_stop),
    }


# ---------------------------------------------------------------------------
# Per-file fetch
# ---------------------------------------------------------------------------

def _cache_path(tmpdir: Path, url: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()[:16]
    return tmpdir / f"era5ncar_{key}.zarr"


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(Exception),
    wait=tenacity.wait_exponential(multiplier=2, min=4, max=120),
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
) -> xr.DataArray:
    """Download one ERA5 accumu file via pydap and return a DataArray."""
    # dap2:// avoids pydap's protocol-detection warning for https:// URLs.
    with _FETCH_SEMAPHORE:
        ds_pydap = open_url(f"dap2://{url[8:]}")
        arr     = np.asarray(ds_pydap[var_name][var_name][:])
        fit_raw = np.asarray(ds_pydap["forecast_initial_time"][:])  # h since 1900-01-01
        fhr     = np.asarray(ds_pydap["forecast_hour"][:])          # 1..12

    fit_dt = _ERA5_EPOCH + fit_raw.astype("timedelta64[h]")
    times  = (fit_dt[:, None] + fhr[None, :].astype("timedelta64[h]")).ravel()

    lat_vals = 90.0 - np.arange(lat_s, lat_e + 1) * _ERA5_RES
    lon_vals = -180.0 + np.arange(lon_s, lon_e + 1) * _ERA5_RES

    return xr.DataArray(
        arr.reshape(-1, arr.shape[2], arr.shape[3]),
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": lat_vals, "longitude": lon_vals},
        name=atlite_name,
    )


def _fetch_file(
    url: str,
    var_name: str,
    atlite_name: str,
    lat_s: int,
    lat_e: int,
    lon_s: int,
    lon_e: int,
    tmpdir: Path,
) -> Path:
    """
    Return path to a cached Zarr store for one ERA5 file, downloading if absent.

    The cache key is MD5(url), which encodes variable, period, and spatial CE.
    The write is atomic (write to .tmp dir, rename on success) so a crashed
    download leaves no corrupt store.  Zarr is thread-safe so no lock is needed.
    """
    cache = _cache_path(tmpdir, url)
    if cache.exists():
        logger.debug("cache hit: %s", cache.name)
        return cache

    da = _download_to_array(url, var_name, atlite_name, lat_s, lat_e, lon_s, lon_e)
    tmp = cache.with_name(cache.name + ".tmp")
    da.to_zarr(tmp, mode="w")
    tmp.rename(cache)
    logger.debug("cached: %s", cache.name)
    return cache


# ---------------------------------------------------------------------------
# Per-variable loader
# ---------------------------------------------------------------------------

def _load_var(
    atlite_name: str,
    north: float,
    south: float,
    west: float,
    east: float,
    start: date,
    end: date,
    tmpdir: Path,
) -> xr.DataArray:
    """Fetch all files for one variable in parallel, return a DataArray."""
    product, param_code, var_name = SOLAR_VARIABLES[atlite_name]

    isel = _bbox_isel(north, south, west, east)
    lat_s = isel["latitude"].start
    lat_e = isel["latitude"].stop - 1
    lon_s = isel["longitude"].start
    lon_e = isel["longitude"].stop - 1

    urls = file_urls(product, param_code, var_name, start, end, lat_s, lat_e, lon_s, lon_e)
    logger.info("%s: %d files", atlite_name, len(urls))

    paths = compute(*[
        delayed(_fetch_file)(url, var_name, atlite_name, lat_s, lat_e, lon_s, lon_e, tmpdir)
        for url in urls
    ])

    ds = xr.open_mfdataset(
        [str(p) for p in paths],
        engine="zarr",
        concat_dim="time",
        combine="nested",
    )
    da = ds[atlite_name].sortby("time").sel(time=slice(str(start), str(end)))
    da.encoding = {}  # don't let cache-file encoding bleed into the cutout write
    return da


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_solar_timeseries(
    north: float,
    south: float,
    west: float,
    east: float,
    start: date,
    end: date,
    tmpdir: Path,
    lock=None,
) -> xr.Dataset:
    """
    Download all four atlite solar variables from NCAR THREDDS and return a
    merged xarray Dataset with a 1D hourly time axis.

    Variables returned (raw ERA5 accumulated values):
      ssrd  Surface solar radiation downwards  [J m-2]
      ssr   Surface net solar radiation        [J m-2]
      fdir  Direct solar radiation at surface  [J m-2]
      tisr  TOA incident solar radiation       [J m-2 s]

    Parameters
    ----------
    north, south, west, east : float
        Bounding box in degrees (lat/lon WGS84).
    start, end : date
        Inclusive date range.
    tmpdir : Path
        Directory for per-file Zarr cache. Pass a persistent path to enable
        crash recovery; entries that already exist are skipped.
    lock : ignored
        Accepted for API compatibility with atlite's get_features; Zarr is
        thread-safe so no lock is needed.
    """
    arrays = compute(*[
        delayed(_load_var)(name, north, south, west, east, start, end, tmpdir)
        for name in SOLAR_VARIABLES
    ])
    return xr.merge(list(arrays))


# ---------------------------------------------------------------------------
# atlite Cutout entrypoint
# ---------------------------------------------------------------------------

def get_data(cutout, feature, tmpdir=None, lock=None, **kwargs):
    """
    Retrieve data from NCAR THREDDS for the given cutout and feature.

    This is the atlite dataset entrypoint called by cutout.prepare().
    Currently only 'influx' is supported, returning raw ERA5 accumulated
    solar radiation variables (ssrd, ssr, fdir, tisr).

    tmpdir is provided by atlite's cutout.prepare(); when callers pass a
    persistent path, previously downloaded files are reused on restart.
    """
    coords = cutout.coords
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()

    time_index = coords["time"].to_index()
    start = time_index[0].date()
    end = time_index[-1].date()

    cache_dir = Path(tmpdir) if tmpdir is not None else Path(mkdtemp())

    if feature == "influx":
        ds = get_solar_timeseries(
            north=y1, south=y0, west=x0, east=x1,
            start=start, end=end,
            tmpdir=cache_dir,
            lock=lock,
        )
    else:
        raise NotImplementedError(f"Feature {feature!r} not supported by era5_ncar")

    ds = ds.rename({"latitude": "y", "longitude": "x"})
    ds = maybe_swap_spatial_dims(ds)
    ds = ds.reindex(time=coords["time"])
    return ds
