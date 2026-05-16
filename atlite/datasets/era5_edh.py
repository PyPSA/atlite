# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Download ERA5 data from the Earth Data Hub (EDH). EDH stores a mirror
of ERA5 in a convenient Zarr format. Dataset parameters:
URL: https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr
Auth: .netrc in user $HOME folder
Time coord: valid_time, hourly, 1940-01-01 → today,
Latitude: descending 90 → -90, step -0.25, length 721.
Longitude: 0–360 ascending, step 0.25, length 1440.
Chunks: (4320, 64, 64)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

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


def _open_edh() -> xr.Dataset:
    """
    Open the ERA5 dataset hosted on Earth Data Hub.

    Returns
    -------
    xarray.Dataset
        Dataset object, dask-backed with the store's native
        ``(4320, 64, 64)`` chunks.

    Notes
    -----
    The store is opened with its native chunking and rechunked later, after
    spatial/temporal subsetting (see :func:`_rename_and_clean_coords`). Opening
    with a fine ``chunks`` argument instead would size the dask graph to the
    *whole* 1940..today store (~750k steps -> millions of tasks per variable),
    which costs gigabytes of graph overhead regardless of cutout size.
    """
    _DATASET_URL = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
    # passed to aiohttp to enable authentication via .netrc file in $HOME
    _STORAGE_OPTIONS: dict[str, Any] = {"client_kwargs": {"trust_env": True}}

    ds = xr.open_dataset(
        _DATASET_URL,
        storage_options=_STORAGE_OPTIONS,
        chunks={},
        engine="zarr",
    )
    # get rid of unnecessary coordinates
    for coord in ("number", "surface"):
        if coord in ds.coords:
            ds = ds.reset_coords(coord, drop=True)
    return ds


def _subset_spatial(ds: xr.Dataset, cutout: Cutout) -> xr.Dataset:
    """
    Select the cutout's bounding box from an EDH dataset.

    The cutout uses atlite's coordinate system (longitude -180..180,
    latitude -90..90, ascending). EDH stores ERA5 with longitude 0..360
    ascending and latitude 90..-90 descending. This translates the bbox
    into EDH coordinates, handles wrap-around at the 0/360 seam by
    fetching the two halves and concatenating, then rewraps longitudes
    on the result back to -180..180.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset on the EDH grid with ``latitude`` (descending) and
        ``longitude`` (0..360 ascending) coordinates.
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
    return sub.assign_coords(longitude=new_lon)


def _subset_temporal(ds: xr.Dataset, cutout: Cutout) -> xr.Dataset:
    """
    Select the cutout's time slice from an EDH dataset.

    Renames the time axis from EDH's ``valid_time`` to atlite's ``time``.

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


def _rename_and_clean_coords(ds: xr.Dataset, cutout: Cutout) -> xr.Dataset:
    """
    Rename ``latitude``/``longitude`` to atlite's ``y``/``x`` and apply the
    cutout's dask chunking.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``latitude`` and ``longitude`` coords, already subset to
        the cutout's bbox and time range.
    cutout : atlite.Cutout
        Cutout whose ``chunks`` (set via ``Cutout(..., chunks=...)``) control
        the dask block size of the returned data, mirroring how the ``era5``
        (CDS) module threads ``cutout.chunks`` into its readers.

    Returns
    -------
    xarray.Dataset
        Dataset with ``y``, ``x`` (rounded to 5 decimals, both ascending) and
        ``lat``/``lon`` aliases, rechunked to ``cutout.chunks``.

    Notes
    -----
    The rechunk runs here, on the already-subset array, not on the raw store.
    ``ds`` now spans only the cutout (a few native time chunks), so the
    rechunk graph is small. Rechunking the full 1940..today store instead
    would build a multi-million-task graph and exhaust memory.
    """
    ds = ds.rename({"latitude": "y", "longitude": "x"})
    # EDH stores latitude descending; atlite expects ascending.
    ds = ds.isel(y=slice(None, None, -1))
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5),
        y=np.round(ds.y.astype(float), 5),
    )
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    chunks = {k: v for k, v in (cutout.chunks or {}).items() if k in ds.dims}
    if chunks:
        # unify_chunks reconciles the 1-D index coordinates -- which .chunk()
        # turns into single-chunk dask arrays -- with data variables whose
        # spatial dims may be multi-chunk, e.g. after the 0/360 seam concat
        # in _subset_spatial. Without it, ds.chunksizes raises
        # "inconsistent chunks along dimension x".
        ds = ds.chunk(chunks).unify_chunks()
    return ds


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
    ds = _open_edh()[_FEATURE_VARS["wind"]]
    ds = _subset_spatial(ds, cutout)
    ds = _subset_temporal(ds, cutout)
    ds = _rename_and_clean_coords(ds, cutout)

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
    ds = _open_edh()[_FEATURE_VARS["influx"]]
    ds = _subset_spatial(ds, cutout)
    ds = _subset_temporal(ds, cutout)
    ds = _rename_and_clean_coords(ds, cutout)

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
    ds = _open_edh()[_FEATURE_VARS["temperature"]]
    ds = _subset_spatial(ds, cutout)
    ds = _subset_temporal(ds, cutout)
    ds = _rename_and_clean_coords(ds, cutout)
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
    ds = _open_edh()[_FEATURE_VARS["runoff"]]
    ds = _subset_spatial(ds, cutout)
    ds = _subset_temporal(ds, cutout)
    ds = _rename_and_clean_coords(ds, cutout)
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
    ds = _open_edh()[_FEATURE_VARS["height"]].isel(valid_time=0, drop=True)
    ds = _subset_spatial(ds, cutout)
    ds = _rename_and_clean_coords(ds, cutout)
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
    lock: Any = None,
    **creation_parameters: Any,
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
        Currently unused. Reserved for an optional intermediate zarr cache
        gated by a future ``cache_dir`` creation parameter.
    lock : object, optional
        Accepted for signature compatibility with
        :func:`atlite.datasets.era5.get_data`. Not used.
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
    _NATIVE_RES = 0.25
    if not (np.isclose(cutout.dx, _NATIVE_RES) and np.isclose(cutout.dy, _NATIVE_RES)):
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
        ds = ds.reindex(time=cutout.coords["time"])

    return ds
