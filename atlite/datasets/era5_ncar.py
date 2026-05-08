# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Download ERA5 data from NCAR THREDDS (ds633.0) via OPeNDAP (DAP2) with CEs.

Design
------
- _temporal_file_specs() assembles dodsC base URLs and time CE fragments from
  the NCAR file naming convention — no catalog fetch needed.
- _spatial_specs() maps the bounding box to one or two ERA5 0.25° grid spatial
  segments. ERA5 on NCAR uses 0–360 longitude (lon[j] = j*0.25, j=0 at 0°E),
  so bboxes that straddle the prime meridian produce two non-contiguous index
  ranges that must be fetched separately and concatenated.
- _build_url() combines a temporal spec and a spatial spec into a full DAP2 CE URL.
- _download_to_array() opens one file via pydap.open_url and returns a DataArray.
- _fetch_file() wraps _download_to_array with a disk cache in tmpdir: a file
  keyed by MD5(url) is written atomically; hits skip the network entirely.
- _load_var() fetches all (temporal-file × spatial-segment) combinations in
  one parallel batch, then opens them lazily via xr.open_mfdataset so downstream
  computation reads from disk in chunks rather than holding all arrays in RAM.
- tenacity retries each network fetch on any transient error.
"""

import hashlib
import logging
import math
import threading
import warnings
from calendar import monthrange
from datetime import date
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import tenacity
import xarray as xr
from dask import compute, delayed
from pydap.client import open_url

from atlite.datasets.era5 import sanitize_influx
from atlite.gis import maybe_swap_spatial_dims
from atlite.pv.solar_position import SolarPosition

logger = logging.getLogger(__name__)

_FETCH_SEMAPHORE = threading.Semaphore(8)

crs = 4326

features = {
    "influx": [
        "influx_toa",
        "influx_direct",
        "influx_diffuse",
        "albedo",
        "solar_altitude",
        "solar_azimuth",
    ],
}

_DODC_BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000"
_ERA5_RES = 0.25  # degree
_BBOX_PAD = 0.5   # degrees added to each side for edge interpolation support
_ERA5_EPOCH = np.datetime64("1900-01-01T00:00", "h")
_N_HOUR = 11  # 12 ERA5 forecast steps; DAP2 stop index is inclusive

SOLAR_VARIABLES = {
    "ssrd": ("e5.oper.fc.sfc.accumu", "128_169_ssrd", "SSRD"),
    "ssr":  ("e5.oper.fc.sfc.accumu", "128_176_ssr",  "SSR"),
    "fdir": ("e5.oper.fc.sfc.accumu", "228_021_fdir", "FDIR"),
    "tisr": ("e5.oper.fc.sfc.accumu", "128_212_tisr", "TISR"),
}


# ---------------------------------------------------------------------------
# Temporal helpers
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


def _temporal_file_specs(
    product: str, param_code: str, start: date, end: date
) -> list[dict]:
    """
    Return one record per ERA5 file covering [start, end].

    Each record contains:
      base_url   — dodsC URL of the NetCDF file (no CE appended)
      time_ce    — precomputed DAP2 index expression for the time dimension(s),
                   e.g. "[0:743]" for analysis or "[0:29][0:11]" for forecast
      extra_vars — comma-prefixed CE fragments for the auxiliary time variables
                   that must be requested alongside the data variable

    _build_url() inserts the spatial CE between time_ce and extra_vars.
    """
    specs = []

    if product == "e5.oper.invariant":
        specs.append({
            "base_url": (
                f"{_DODC_BASE}/e5.oper.invariant/197901/"
                "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
            ),
            "time_ce": "",
            "extra_vars": "",
        })

    elif product == "e5.oper.an.sfc":
        for first_day, last_day in _month_bounds_in_range(start, end):
            ym = first_day.strftime("%Y%m")
            n_time = (last_day - first_day).days * 24 + 23  # inclusive stop index
            fname = (
                f"{product}.{param_code}.ll025sc."
                f"{first_day.strftime('%Y%m%d')}00_{last_day.strftime('%Y%m%d')}23.nc"
            )
            specs.append({
                "base_url": f"{_DODC_BASE}/{product}/{ym}/{fname}",
                "time_ce": f"[0:{n_time}]",
                "extra_vars": f",time[0:{n_time}]",
            })

    elif product == "e5.oper.fc.sfc.accumu":
        for first_day, last_day in _halfmonth_bounds_in_range(start, end):
            ym = first_day.strftime("%Y%m")
            n_init = (last_day - first_day).days * 2 - 1  # inclusive stop index
            fname = (
                f"{product}.{param_code}.ll025sc."
                f"{first_day.strftime('%Y%m%d')}06_{last_day.strftime('%Y%m%d')}06.nc"
            )
            specs.append({
                "base_url": f"{_DODC_BASE}/{product}/{ym}/{fname}",
                "time_ce": f"[0:{n_init}][0:{_N_HOUR}]",
                "extra_vars": (
                    f",forecast_initial_time[0:{n_init}]"
                    f",forecast_hour[0:{_N_HOUR}]"
                ),
            })

    else:
        raise ValueError(f"Unknown product: {product!r}")

    return specs


def _build_url(tspec: dict, sspec: dict, var_name: str) -> str:
    """Assemble a full DAP2 CE URL from a temporal file spec and a spatial segment spec."""
    spatial_ce = f"[{sspec['lat_s']}:{sspec['lat_e']}][{sspec['lon_s']}:{sspec['lon_e']}]"
    return (
        f"{tspec['base_url']}"
        f"?{var_name}{tspec['time_ce']}{spatial_ce}{tspec['extra_vars']}"
    )


# ---------------------------------------------------------------------------
# Spatial index helpers
# ---------------------------------------------------------------------------

def _spatial_specs(
    north: float, south: float, west: float, east: float
) -> list[dict]:
    """
    Map a WGS84 bounding box to one or two ERA5 0.25° grid spatial segments.

    ERA5 on NCAR uses 0–360 longitude (lon[j] = j*0.25, j=0 at 0°E prime
    meridian, j=1439 at 359.75°E). Bboxes that straddle the prime meridian
    produce two non-contiguous index ranges:
      segment 0: [j_west : 1439]  western (negative-longitude) portion
      segment 1: [0 : j_east]     eastern (positive-longitude) portion
    Non-straddling bboxes produce a single segment.

    Latitude runs 90°N → −90°S: lat[i] = 90 − i*0.25 (i=0 at 90°N, i=720 at −90°S).
    """
    lat_s = math.ceil((90 - north) / _ERA5_RES)
    lat_e = math.floor((90 - south) / _ERA5_RES)

    # Python modulo maps negative longitudes to 0–360: -4 % 360 = 356
    west_360 = west % 360
    east_360 = east % 360
    j_west = math.ceil(west_360 / _ERA5_RES)
    j_east = math.floor(east_360 / _ERA5_RES)

    base = {"lat_s": lat_s, "lat_e": lat_e}
    if j_west <= j_east:
        return [{**base, "lon_s": j_west, "lon_e": j_east}]
    else:
        return [
            {**base, "lon_s": j_west, "lon_e": 1439},
            {**base, "lon_s": 0,      "lon_e": j_east},
        ]


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
    raw_lons = np.arange(lon_s, lon_e + 1) * _ERA5_RES                    # 0–360
    lon_vals = np.where(raw_lons > 180.0, raw_lons - 360.0, raw_lons)     # → −180..180

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

    tspecs = _temporal_file_specs(product, param_code, start, end)
    sspecs = _spatial_specs(north, south, west, east)
    logger.info("%s: %d file(s), %d spatial segment(s)", atlite_name, len(tspecs), len(sspecs))

    # Download all (temporal × spatial) combinations in one parallel batch.
    # Ordering: outer loop over sspecs, inner over tspecs, so all_paths[:n]
    # are the temporal files for sspecs[0] and all_paths[n:] for sspecs[1].
    n = len(tspecs)
    all_paths = compute(*[
        delayed(_fetch_file)(
            _build_url(ts, ss, var_name),
            var_name, atlite_name,
            ss["lat_s"], ss["lat_e"], ss["lon_s"], ss["lon_e"],
            tmpdir,
        )
        for ss in sspecs
        for ts in tspecs
    ])

    if len(sspecs) == 1:
        da = xr.open_mfdataset(
            [str(p) for p in all_paths],
            engine="zarr", concat_dim="time", combine="nested",
        )[atlite_name]
    else:
        ds_a = xr.open_mfdataset(
            [str(p) for p in all_paths[:n]],
            engine="zarr", concat_dim="time", combine="nested",
        )
        ds_b = xr.open_mfdataset(
            [str(p) for p in all_paths[n:]],
            engine="zarr", concat_dim="time", combine="nested",
        )
        da = xr.concat([ds_a, ds_b], dim="longitude").sortby("longitude")[atlite_name]

    da = da.sortby("time").sel(time=slice(str(start), str(end)))
    da.encoding = {}
    return da

# ---------------------------------------------------------------------------
# Solar timeseries download and postprocessing
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
    """Download all four solar accumulation variables and return a raw Dataset."""


def _postprocess_influx(ds: xr.Dataset) -> xr.Dataset:
    """Convert raw solar variables (ssrd, ssr, fdir, tisr) to atlite influx variables.

    Mirrors atlite.datasets.era5.get_data_influx / sanitize_influx.
    Accepts ds with native latitude/longitude dimension names.
    """
    ds = ds.rename({"latitude": "y", "longitude": "x"})
    ds = maybe_swap_spatial_dims(ds)
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])

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

    ds = sanitize_influx(ds)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift=pd.to_timedelta("-30 minutes"))
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    return xr.merge([ds, sp])


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
        
    TODO:
    - verify congruence between interpolated grids for larger cutouts in CDS and NCAR
    """
    coords = cutout.coords
    x0, x1 = coords["x"].min().item(), coords["x"].max().item()
    y0, y1 = coords["y"].min().item(), coords["y"].max().item()

    time_index = coords["time"].to_index()
    start = time_index[0].date()
    end = time_index[-1].date()

    cache_dir = Path(tmpdir) if tmpdir is not None else Path(mkdtemp())

    # Pad bounding box so bilinear interpolation has support at target grid edges.
    north = min(90.0, y1 + _BBOX_PAD)
    south = max(-90.0, y0 - _BBOX_PAD)
    west = x0 - _BBOX_PAD
    east = x1 + _BBOX_PAD

    # Target latitude/longitude values for regridding raw downloads.
    # DataArrays from _load_var use "latitude"/"longitude" dims (before
    # _postprocess_influx renames them to "y"/"x").
    target_lat = coords["y"].values
    target_lon = coords["x"].values

    if feature == "influx":
        arrays = {}
        for atlite_name in ("ssrd", "ssr", "fdir", "tisr"):
            arrays[atlite_name] = _load_var(
                atlite_name, north, south, west, east, start, end, cache_dir
            )

        # ERA5 can have ssr > ssrd at the day/night boundary (accumulation artifact).
        # Clip before interpolation so the unphysical values aren't amplified by bilinear
        # interpolation (target points near the boundary get small ssrd but large ssr
        # from the daytime side, yielding extreme negative albedo).
        arrays["ssr"] = arrays["ssr"].where(
            arrays["ssr"] <= arrays["ssrd"], other=arrays["ssrd"]
        )

        # Regrid raw variables to target resolution before computing derived quantities.
        # This matches the CDS ordering (interpolate → compute albedo/diffuse/clip).
        # At native 0.25° resolution target points fall exactly on source grid points
        # and linear interp returns exact values — never use method="nearest".

        for atlite_name in list(arrays.keys()):
            arrays[atlite_name] = arrays[atlite_name].interp(
                latitude=target_lat, longitude=target_lon, method="linear"
            )
        ds = _postprocess_influx(xr.Dataset(arrays))
    else:
        raise NotImplementedError(f"Feature {feature!r} not supported by era5_ncar")

    ds = ds.reindex(time=coords["time"])
    return ds
