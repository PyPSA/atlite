# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Module for creating cutouts from the SARAH2 dataset."""

from __future__ import annotations

import gc
import logging
import shutil
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.warp import Resampling

from atlite.gis import regrid
from atlite.pv.solar_position import SolarPosition

if TYPE_CHECKING:
    from atlite._types import PathLike

logger = logging.getLogger(__name__)


# Model, CRS and Resolution Settings
crs = 4326
dx = 0.05
dy = 0.05
dt = "30min"
features = {
    "influx": [
        "influx_direct",
        "influx_diffuse",
        "solar_altitude",
        "solar_azimuth",
    ],
}
static_features: dict[str, list[str]] = {}

# EUMETSAT Data Store collection holding the SARAH-3 products
SARAH_COLLECTION_ID = "EO:EUM:DAT:0863"
SARAH_API_SUBDIR = "sarah_raw"
SARAH_API_PATTERNS = {"SID": "SIDin*", "SIS": "SISin*"}
SARAH_API_RETRIES = 3
SARAH_API_DEFAULT_WORKERS = 15


def _api_time_range(coords: dict[str, Any]) -> tuple[str, str]:
    """Return the ISO start/end strings covering the days of the cutout."""
    index = coords["time"].to_index()
    start = index[0].floor("D")
    end = index[-1].floor("D") + pd.Timedelta(hours=23, minutes=30)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return start.strftime(fmt), end.strftime(fmt)


def _download_nc(product: Any, target: Path, retries: int = SARAH_API_RETRIES) -> str:
    """Download a product ZIP, extract its NetCDF file and remove the ZIP."""
    zip_path = target.with_suffix(".zip")
    # Extract to a temporary name so an interrupted run never leaves a
    # partial ``.nc`` file that looks valid.
    tmp = target.with_suffix(".part")

    for attempt in range(1, retries + 1):
        try:
            with product.open() as src, zip_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            with zipfile.ZipFile(zip_path) as zf:
                member = next(
                    name for name in zf.namelist() if name.lower().endswith(".nc")
                )
                with zf.open(member) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

            tmp.replace(target)
        except Exception as exc:
            logger.warning(
                "%s attempt %d/%d failed: %s", product, attempt, retries, exc
            )
            if attempt < retries:
                time.sleep(5 * attempt)
        else:
            return "file extracted"
        finally:
            zip_path.unlink(missing_ok=True)

    tmp.unlink(missing_ok=True)
    return "failed"


def download_from_api(
    start: str,
    end: str,
    target_dir: PathLike,
    max_workers: int = SARAH_API_DEFAULT_WORKERS,
) -> Path:
    """
    Download SARAH SID and SIS NetCDF files from the EUMETSAT Data Store.

    Credentials are read by the EUMDAC library from the file written by
    ``eumdac set-credentials <consumer_key> <consumer_secret>``.

    Parameters
    ----------
    start, end : str
        ISO timestamps (``YYYY-MM-DDTHH:MM:SSZ``) delimiting the search.
    target_dir : PathLike
        Directory in which the ``SID`` and ``SIS`` sub-directories are created.
    max_workers : int, default 15
        Number of parallel downloads.

    Returns
    -------
    pathlib.Path
        The directory containing the downloaded files.

    Raises
    ------
    ImportError
        If the optional ``eumdac`` package is not installed.
    RuntimeError
        If no credentials are found, nothing matches the time range or any
        product fails to download.
    """
    try:
        import eumdac
        from eumdac.cli_helpers import (
            CredentialsFileNotFoundError,
            load_credentials,
        )
    except ImportError as exc:
        raise ImportError(
            "Downloading SARAH data requires the 'eumdac' package. "
            "Install it with `pip install eumdac`."
        ) from exc

    try:
        credentials = load_credentials()
    except CredentialsFileNotFoundError as exc:
        raise RuntimeError(
            "No EUMETSAT credentials found. Set them once with "
            "`eumdac set-credentials <consumer_key> <consumer_secret>`."
        ) from exc

    datastore = eumdac.DataStore(eumdac.AccessToken(credentials))
    collection = datastore.get_collection(SARAH_COLLECTION_ID)

    target_dir = Path(target_dir)
    pending = [
        (var, product, product.size)
        for var, pattern in SARAH_API_PATTERNS.items()
        for product in collection.search(dtstart=start, dtend=end, title=pattern)
    ]
    if not pending:
        raise RuntimeError(f"No SARAH products found between {start} and {end}.")

    # product.size is in kilobytes
    size_gib = sum(size for _, _, size in pending) / 1024**2
    logger.warning(
        "SARAH download: %d products, estimated size %.2f GiB.",
        len(pending),
        size_gib,
    )

    for var in SARAH_API_PATTERNS:
        (target_dir / var).mkdir(parents=True, exist_ok=True)

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_nc, product, target_dir / var / f"{product}.nc"
            ): (
                var,
                product,
            )
            for var, product, _ in pending
        }
        for i, future in enumerate(as_completed(futures), 1):
            var, product = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                logger.warning("%s failed: %s", product, exc)
                status = "failed"
            if status == "failed":
                failed.append(str(product))
            logger.info("[%d/%d] %s %s -> %s", i, len(pending), var, product, status)

    if failed:
        raise RuntimeError(f"{len(failed)} SARAH products failed to download: {failed}")
    return target_dir


def cleanup(tmpdir: PathLike) -> None:
    """
    Remove the raw SARAH files downloaded from the API into ``tmpdir``.

    Called by :func:`atlite.data.cutout_prepare` once the cutout is written.
    """
    # Drop lingering references so file handles of the lazily opened NetCDF
    # files are released (required on Windows before deleting).
    gc.collect()
    shutil.rmtree(Path(tmpdir) / SARAH_API_SUBDIR, ignore_errors=True)


def get_filenames(sarah_dir: str | PathLike, coords: dict[str, Any]) -> pd.DataFrame:
    """
    Build a DataFrame of SIS and SID file paths for the requested time range.

    Parameters
    ----------
    sarah_dir : str or PathLike
        Root directory containing SARAH NetCDF files.
    coords : dict
        Coordinate mapping with a ``"time"`` key whose index defines the
        requested temporal range.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``"sis"`` and ``"sid"`` indexed by date.
    """

    def _filenames_starting_with(name: str) -> pd.Series[str]:
        pattern = str(Path(sarah_dir) / "**" / f"{name}*.nc")
        files = pd.Series([str(f) for f in Path(sarah_dir).rglob(f"{name}*.nc")])
        assert not files.empty, (
            f"No files found at {pattern}. Make sure "
            f"sarah_dir points to the correct directory!"
        )

        files.index = pd.to_datetime(files.str.extract(r"SI.in(\d{8})", expand=False))
        return files.sort_index()

    files = pd.concat(
        {
            "sis": _filenames_starting_with("SIS"),
            "sid": _filenames_starting_with("SID"),
        },
        join="inner",
        axis=1,
    )

    start = coords["time"].to_index()[0].floor("D")
    end = coords["time"].to_index()[-1].floor("D")

    if (start < files.index[0]) or (end > files.index[-1]):
        logger.error(
            "Files in %s do not cover the whole time span:\t%s until %s",
            sarah_dir,
            start,
            end,
        )

    return files.loc[(files.index >= start) & (files.index <= end)].sort_index()


def interpolate(
    ds: xr.Dataset | xr.DataArray, dim: str = "time"
) -> xr.Dataset | xr.DataArray:
    """
    Linearly interpolate NaN values along a dimension.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input data with potential NaN gaps.
    dim : str, default "time"
        Dimension along which to interpolate.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Data with NaN values filled by linear interpolation.
    """

    def _interpolate1d(
        y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        nan = np.isnan(y)
        if nan.all() or not nan.any():
            return y

        def x(z: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[np.intp]]:
            return z.nonzero()[0]

        y = np.array(y)
        y[nan] = np.interp(x(nan), x(~nan), y[~nan])
        return y

    def _interpolate(a: Any) -> Any:
        return a.map_blocks(
            partial(np.apply_along_axis, _interpolate1d, -1), dtype=a.dtype
        )

    data_vars = ds.data_vars.values() if isinstance(ds, xr.Dataset) else (ds,)
    dtypes = {da.dtype for da in data_vars}
    assert len(dtypes) == 1, "interpolate only supports datasets with homogeneous dtype"

    result: xr.Dataset | xr.DataArray = xr.apply_ufunc(
        _interpolate,
        ds,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        output_dtypes=[dtypes.pop()],
        output_sizes={dim: len(ds.indexes[dim])},
        dask="allowed",
        keep_attrs=True,
    )
    return result


def as_slice(bounds: slice | tuple[float, float], pad: bool = True) -> slice:
    """
    Convert coordinate bounds to a slice, optionally with small padding.

    Parameters
    ----------
    bounds : slice or tuple of float
        Existing slice or ``(start, stop)`` tuple.
    pad : bool, default True
        If *bounds* is a tuple, add ±0.01 padding.

    Returns
    -------
    slice
        Slice suitable for ``.sel()`` indexing.
    """
    if not isinstance(bounds, slice):
        bounds = bounds + (-0.01, 0.01)  # type: ignore[assignment]
        bounds = slice(*bounds)
    return bounds


def hourly_mean(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute hourly means from 30-minute data by averaging consecutive pairs.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 30-minute temporal resolution.

    Returns
    -------
    xr.Dataset
        Dataset resampled to hourly resolution.
    """
    ds1 = ds.isel(time=slice(None, None, 2))
    ds2 = ds.isel(time=slice(1, None, 2))
    ds2 = ds2.assign_coords(time=ds2.indexes["time"] - pd.Timedelta(30, "m"))
    ds = (ds1 + ds2) / 2
    ds.attrs = ds1.attrs
    for v in ds.variables:
        ds[v].attrs = ds1[v].attrs
    return ds


def get_data(
    cutout: Any,
    feature: str,
    tmpdir: PathLike,
    lock: Any = None,
    monthly_requests: bool = False,
    **creation_parameters: Any,
) -> xr.Dataset:
    """
    Retrieve and process SARAH solar irradiance data for a cutout.

    Parameters
    ----------
    cutout : Cutout
        Target cutout defining the spatial and temporal extent.
    feature : str
        Feature name (e.g. ``"influx"``).
    tmpdir : PathLike
        Temporary directory for intermediate files.
    lock : lock-like, optional
        Lock for thread-safe file access.
    monthly_requests : bool, default False
        Whether to split requests by month.
    **creation_parameters
        Either ``sarah_dir`` or ``sarah_from_api=True`` must be given.
        Optional keys: ``parallel`` (bool), ``sarah_interpolate`` (bool),
        ``sarah_from_api`` (bool, default False; downloads the files from the
        EUMETSAT API into ``tmpdir`` and ignores ``sarah_dir``),
        ``sarah_api_workers`` (int, default 15).

    Returns
    -------
    xr.Dataset
        Dataset with ``influx_direct``, ``influx_diffuse``,
        ``solar_altitude``, and ``solar_azimuth`` variables.
    """
    assert cutout.dt in ("30min", "30T", "h", "1h")

    coords = cutout.coords
    chunks = cutout.chunks

    creation_parameters.setdefault("parallel", False)
    creation_parameters.setdefault("sarah_interpolate", True)

    if creation_parameters.get("sarah_from_api", False):
        start, end = _api_time_range(coords)
        sarah_dir = download_from_api(
            start,
            end,
            Path(tmpdir) / SARAH_API_SUBDIR,
            max_workers=int(
                creation_parameters.get("sarah_api_workers", SARAH_API_DEFAULT_WORKERS)
            ),
        )
    else:
        sarah_dir = creation_parameters["sarah_dir"]

    files = get_filenames(sarah_dir, coords)
    open_kwargs = {"chunks": chunks, "parallel": creation_parameters["parallel"]}
    ds_sis = xr.open_mfdataset(files.sis, combine="by_coords", **open_kwargs)[["SIS"]]
    ds_sid = xr.open_mfdataset(files.sid, combine="by_coords", **open_kwargs)[["SID"]]
    ds = xr.merge([ds_sis, ds_sid])
    ds = ds.sel(lon=as_slice(cutout.extent[:2]), lat=as_slice(cutout.extent[2:]))
    ds = ds.assign_coords(
        lon=ds.lon.astype(float).round(4), lat=ds.lat.astype(float).round(4)
    )

    ds = cast(
        "xr.Dataset",
        interpolate(ds) if creation_parameters["sarah_interpolate"] else ds.fillna(0),
    )

    if cutout.dt not in ["30min", "30T"]:
        ds = hourly_mean(ds)

    if (cutout.dx != dx) or (cutout.dy != dy):
        ds = cast(
            "xr.Dataset",
            regrid(ds, coords["lon"], coords["lat"], resampling=Resampling.average),
        )

    dif_attrs = {"long_name": "Surface Diffuse Shortwave Flux", "units": "W m-2"}
    ds["influx_diffuse"] = (ds["SIS"] - ds["SID"]).assign_attrs(**dif_attrs)
    ds = ds.rename({"SID": "influx_direct"}).drop_vars("SIS")
    ds = ds.assign_coords(x=ds.coords["lon"], y=ds.coords["lat"])

    ds = ds.swap_dims({"lon": "x", "lat": "y"})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift="0H")
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    return xr.merge([ds, sp])
