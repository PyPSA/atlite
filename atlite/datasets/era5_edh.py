# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Download ERA5 data from the Earth Data Hub (EDH).

EDH stores a mirror of ERA5 in a Zarr format. Dataset parameters:

- URL: https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr
- Auth: ``EARTHDATAHUB_API_KEY`` env var, or a netrc entry in ``./.netrc``
  or ``~/.netrc`` (must be ``chmod 600``).
- Time coordinate: ``valid_time``, hourly, from 1940-01-01 to the present.
- Latitude: descending from 90.0 to -90.0 in 0.25 degree steps, 721 points.
- Longitude: ascending from 0.0 to 359.75 in 0.25 degree steps, 1440 points.
- Native chunks: ``valid_time=4320, latitude=64, longitude=64``.
"""

from __future__ import annotations

import base64
import logging
import netrc
import os
import warnings
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from dask.utils import SerializableLock
from obstore.store import HTTPStore
from zarr.storage import ObjectStore

from atlite.datasets.era5 import (
    _add_height,
    sanitize_influx,
    sanitize_runoff,
    sanitize_wind,
)
from atlite.pv.solar_position import SolarPosition

if TYPE_CHECKING:
    from atlite.cutout import Cutout


Handler = Callable[["Cutout"], xr.Dataset]
Sanitizer = Callable[[xr.Dataset], xr.Dataset]


logger = logging.getLogger(__name__)

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


# Per-feature raw variables to pull from the EDH zarr.
_FEATURE_VARS: dict[str, list[str]] = {
    "wind": ["u10", "v10", "u100", "v100", "fsr"],
    "influx": ["ssrd", "ssr", "fdir", "tisr"],
    "temperature": ["t2m", "d2m", "stl4"],
    "runoff": ["ro"],
    "height": ["z"],
}


_EDH_URL = (
    "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
)


def _get_edh_auth_header() -> str:
    """
    Resolve EDH credentials and return the HTTP Basic ``Authorization`` header.

    Credential lookup order:

    1. ``EARTHDATAHUB_API_KEY`` environment variable (login defaults to ``edh``).
    2. ``./.netrc`` in the current working directory.
    3. The user's home-directory netrc, resolved by :mod:`netrc` itself

    Netrc files are parsed with :mod:`netrc`, which requires ``chmod 600`` on
    entries that carry a password. Shared by the obstore store opened in
    :func:`_open_edh` and the connectivity probe in the test suite.

    Raises
    ------
    RuntimeError
        If no credentials are found.
    """
    host = _EDH_URL.split("/")[2]

    key = os.environ.get("EARTHDATAHUB_API_KEY")
    if key:
        login, password = "edh", key
    else:
        login = password = None
        # None lets netrc.netrc() find the home-folder file itself, which is
        # platform-aware (.netrc on Unix, _netrc on Windows).
        candidates: list[str | None] = [str(Path.cwd() / ".netrc"), None]
        for arg in candidates:
            try:
                auth = netrc.netrc(arg).authenticators(host)
            except FileNotFoundError:
                continue
            except netrc.NetrcParseError as err:
                label = arg or "home netrc"
                logger.error(
                    "Could not parse %s (%s). Earth Data Hub credentials in "
                    "this file will be ignored. If it holds your DestinE API "
                    "key, ensure it is chmod 600 and retry.",
                    label,
                    err,
                )
                continue
            if auth is not None and auth[2] is not None:
                login, _account, password = auth
                login = login or ""
                break

    if password is None:
        raise RuntimeError(
            f"Earth Data Hub access needs a DestinE API key. Provide it by either:\n"
            f"  1) setting the EARTHDATAHUB_API_KEY environment variable, or\n"
            f"  2) adding an entry to your netrc (./.netrc, ~/.netrc, or ~/_netrc\n"
            f"     on Windows; must be chmod 600):\n"
            f"        machine {host}\n"
            f"        login edh\n"
            f"        password <your-api-key>\n"
            f"Get or refresh your key at https://earthdatahub.destine.eu/account-settings"
        )

    auth = base64.b64encode(f"{login}:{password}".encode()).decode()
    return f"Basic {auth}"


def _open_edh() -> xr.Dataset:
    """
    Open the ERA5 dataset hosted on Earth Data Hub.

    The dataset is a remote Zarr store read over HTTPS through ``obstore``.
    ``obstore`` is faster than the ``fsspec`` store natively used by xarray
    and implements retries, which is important for large cutouts where we
    make hundreds of requests and one of them is likely to fail.

    Returns
    -------
    xarray.Dataset
        Dataset object, dask-backed with the store's native
        ``(4320, 64, 64)`` chunks.
    """
    # uses obstore.HTTPStore for performance and because it implements retries unlike
    # native fsspec.HTTPStore
    store = HTTPStore.from_url(
        _EDH_URL,
        client_options={"default_headers": {"Authorization": _get_edh_auth_header()}},
        retry_config={
            "max_retries": 8,
            "retry_timeout": timedelta(minutes=10),
            "backoff": {
                "init_backoff": timedelta(seconds=5),
                "max_backoff": timedelta(minutes=2),
                "base": 2,
            },
        },  # we observed O(1) ClientPayloadError during multi h downloads
    )
    ds = xr.open_dataset(ObjectStore(store, read_only=True), chunks={}, engine="zarr")
    # get rid of unnecessary coordinates
    for coord in ("number", "surface"):
        if coord in ds.coords:
            ds = ds.reset_coords(coord, drop=True)
    return ds


def _subset_spatial(ds: xr.Dataset, cutout: Cutout) -> xr.Dataset:
    """
    Select the cutout's bounding box and convert from atlite coordinates to
    EDH coordinates.

    atlite coordinate system:
        - x: -180:180
        - y: -90:90

    EDH coordinate system:
        - x: 0:360
        - y: 90:-90

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset on the EDH grid with descending ``latitude`` coordinates and
        ascending ``longitude`` coordinates in the half-open interval
        ``[0, 360)``.
    cutout : atlite.Cutout
        Cutout defining the target bbox via its ``x``/``y`` coords.

    Returns
    -------
    xarray.Dataset
        Subset with ``latitude`` (descending) and ``longitude``
        rewrapped to -180..180 ascending.
    """
    x = cutout.coords["x"].values
    y = cutout.coords["y"].values

    x_lo = float(x.min()) % 360
    x_hi = float(x.max()) % 360
    if x_lo <= x_hi:
        sub = ds.sel(longitude=slice(x_lo, x_hi))
    else:
        # bbox straddles the 0/360 seam — fetch the two halves and concat.
        east = ds.sel(longitude=slice(x_lo, 360.0))
        west = ds.sel(longitude=slice(0.0, x_hi))
        sub = xr.concat([east, west], dim="longitude")

    sub = sub.sel(latitude=slice(float(y.max()), float(y.min())))
    new_lon = ((sub.longitude + 180) % 360) - 180
    sub = sub.assign_coords(longitude=new_lon)
    # Rewrapping can leave longitude non-monotonic (wide seam-straddling
    # bboxes); sort only when needed to avoid an unnecessary Dask graph layer.
    if new_lon.size > 1 and np.any(np.diff(new_lon.values) < 0):
        sub = sub.sortby("longitude")
    return sub


def _subset_temporal(ds: xr.Dataset, cutout: Cutout) -> xr.Dataset:
    """
    Select the cutout's time slice.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a ``valid_time`` coordinate.
    cutout : atlite.Cutout
        Cutout defining the target time range.

    Returns
    -------
    xarray.Dataset
        Subset with the time axis renamed to ``time``.
    """
    t = cutout.coords["time"].values
    sub = ds.sel(valid_time=slice(t[0], t[-1]))
    return sub.rename({"valid_time": "time"})


def _rename_and_clean_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename ``latitude``/``longitude`` to atlite's ``y``/``x`` and apply the
    working chunk shape used by the rest of the pipeline.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``latitude`` and ``longitude`` coords, already subset to
        the cutout's bbox and time range.

    Returns
    -------
    xarray.Dataset
        Dataset with ``y``, ``x`` (both ascending) and
        ``lat``/``lon`` aliases, rechunked.
    """
    ds = ds.rename({"latitude": "y", "longitude": "x"})
    # EDH stores latitude descending; atlite expects ascending.
    ds = ds.isel(y=slice(None, None, -1))
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5),
        y=np.round(ds.y.astype(float), 5),
    )
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    # rechunk. EDH-sized chunks are large and result in a large RAM footprint.
    # we rechunk to a size that is 1/6th in the time dimension, keeping
    # the 64x64 spatial dimensions. If changing this, it's best to
    # use a chunk size that cleanly divides the original 4320x64x64 dimension
    chunks = {k: v for k, v in {"time": 360, "y": 64, "x": 64}.items() if k in ds.dims}
    if chunks:
        # unify_chunks reconciles the 1-D index coordinates -- which .chunk()
        # turns into single-chunk dask arrays -- with data variables whose
        # spatial dims may be multi-chunk, e.g. after the 0/360 seam concat
        # in _subset_spatial.
        ds = ds.chunk(chunks).unify_chunks()
    return ds


def _load_feature(cutout: Cutout, feature: str, static: bool = False) -> xr.Dataset:
    """
    Open the EDH store, pull the feature's raw variables and subset them to
    the cutout's bbox (and time range, unless ``static``).

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    feature : str
        Feature name; key into ``_FEATURE_VARS``.
    static : bool, optional
        If True, collapse the time axis to its first step and skip the
        temporal subset (for time-invariant features such as ``height``).

    Returns
    -------
    xarray.Dataset
        Raw feature variables on atlite's ``y``/``x`` grid, rechunked to
        ``cutout.chunks``.
    """
    ds = _open_edh()[_FEATURE_VARS[feature]]
    if static:
        ds = ds.isel(valid_time=0, drop=True)
        ds = _subset_spatial(ds, cutout)
    else:
        ds = _subset_temporal(ds, cutout)
        ds = _subset_spatial(ds, cutout)
    return _rename_and_clean_coords(ds)


def get_data_wind(cutout: Cutout) -> xr.Dataset:
    """
    Retrieve and prepare wind variables.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``wnd100m``, ``wnd_shear_exp``, ``wnd_azimuth`` and
        ``roughness``.
    """
    ds = _load_feature(cutout, "wind")

    for h in (10, 100):
        ds[f"wnd{h}m"] = np.sqrt(ds[f"u{h}"] ** 2 + ds[f"v{h}"] ** 2).assign_attrs(
            units="m s**-1", long_name=f"{h} metre wind speed"
        )
    # Dividing by the float64 scalar np.log(0.1) promotes the result to
    # float64; cast back to float32 -- the shear exponent needs no more.
    ds["wnd_shear_exp"] = (
        (np.log(ds["wnd10m"] / ds["wnd100m"]) / np.log(10 / 100))
        .astype(np.float32)
        .assign_attrs(units="", long_name="wind shear exponent")
    )

    # span the whole circle: 0 is north, π/2 east, π south, 3π/2 west. The
    # `+ 2π` float64 scalar promotes the result; cast back to float32.
    azimuth = np.arctan2(ds["u100"], ds["v100"])
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi).astype(
        np.float32
    )

    ds = ds.drop_vars(["u100", "v100", "u10", "v10", "wnd10m"])
    ds = ds.rename({"fsr": "roughness"})
    ds["roughness"] = ds["roughness"].assign_attrs(
        units="m", long_name="Forecast surface roughness"
    )
    return ds


def get_data_influx(cutout: Cutout) -> xr.Dataset:
    """
    Retrieve and prepare solar influx variables.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``influx_toa``, ``influx_direct``, ``influx_diffuse``,
        ``albedo``, ``solar_altitude`` and ``solar_azimuth``.
    """
    ds = _load_feature(cutout, "influx")

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

    # ERA5 radiation is the mean over the previous hour; centre solar position
    # on the interval midpoint (see PyPSA/atlite#158).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp = SolarPosition(ds, time_shift=pd.to_timedelta("-30 minutes"))
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})
    # SolarPosition computes in float64; float32 is ample for stored solar
    # geometry (~1e-6 rad) and halves these variables on disk.
    sp = sp.astype(np.float32)

    return xr.merge([ds, sp])


def get_data_temperature(cutout: Cutout) -> xr.Dataset:
    """
    Retrieve and prepare temperature variables.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``temperature``, ``soil temperature`` and
        ``dewpoint temperature``.
    """
    ds = _load_feature(cutout, "temperature")
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


def get_data_runoff(cutout: Cutout) -> xr.Dataset:
    """
    Retrieve and prepare runoff.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``runoff``.
    """
    ds = _load_feature(cutout, "runoff")
    ds = ds.rename({"ro": "runoff"})
    ds["runoff"].attrs["units"] = "m"
    return ds


def get_data_height(cutout: Cutout) -> xr.Dataset:
    """
    Retrieve and prepare surface height.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``height``.
    """
    ds = _load_feature(cutout, "height", static=True)
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


def get_data(
    cutout: Cutout,
    feature: str,
    tmpdir: str | Path | None = None,
    lock: SerializableLock | None = None,
    **creation_parameters: object,
) -> xr.Dataset:
    """
    Retrieve ERA5 feature data from Earth Data Hub.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout defining the target grid and time range.
    feature : str
        Feature name to retrieve. Must be one of ``features``.
    tmpdir : str or pathlib.Path, optional
        Unused; accepted for interface compatibility with
        :func:`atlite.datasets.era5.get_data`.
    lock : dask.utils.SerializableLock, optional
        Unused; accepted for interface compatibility with
        :func:`atlite.datasets.era5.get_data`.
    **creation_parameters
        Additional creation parameters. ``sanitize`` controls whether the
        standard atlite sanitizers are applied (default True).

    Returns
    -------
    xarray.Dataset
        Prepared dataset for the requested feature.

    Raises
    ------
    ValueError
        If the cutout grid is not the native 0.25°×0.25°.
    NotImplementedError
        If ``feature`` is not yet implemented.
    """
    native_res = 0.25
    if not (np.isclose(cutout.dx, native_res) and np.isclose(cutout.dy, native_res)):
        raise ValueError(
            "era5-edh only supports the native 0.25°×0.25° grid. "
            "For other resolutions, use module='era5' (CDS)."
        )
    if feature not in _HANDLERS:
        raise NotImplementedError(f"Feature {feature!r} not supported by era5_edh")

    logger.info(f"Requesting data for feature {feature}...")
    ds = _HANDLERS[feature](cutout)

    sanitize = creation_parameters.get("sanitize", True)
    if sanitize and feature in _SANITIZERS:
        ds = _SANITIZERS[feature](ds)

    if feature not in static_features:
        ds = ds.sel(time=cutout.coords["time"])

    return ds
