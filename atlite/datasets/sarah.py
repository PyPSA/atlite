# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module containing specific operations for creating cutouts from the SARAH2
dataset.
"""

from __future__ import annotations

import logging
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def get_filenames(sarah_dir: str | PathLike, coords: dict[str, Any]) -> pd.DataFrame:
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

    return xr.apply_ufunc(
        _interpolate,
        ds,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        output_dtypes=[dtypes.pop()],
        output_sizes={dim: len(ds.indexes[dim])},
        dask="allowed",
        keep_attrs=True,
    )


def as_slice(bounds: slice | tuple[float, float], pad: bool = True) -> slice:
    if not isinstance(bounds, slice):
        bounds = bounds + (-0.01, 0.01)  # type: ignore[assignment]
        bounds = slice(*bounds)
    return bounds


def hourly_mean(ds: xr.Dataset) -> xr.Dataset:
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
    assert cutout.dt in ("30min", "30T", "h", "1h")

    coords = cutout.coords
    chunks = cutout.chunks

    sarah_dir = creation_parameters["sarah_dir"]
    creation_parameters.setdefault("parallel", False)
    creation_parameters.setdefault("sarah_interpolate", True)

    files = get_filenames(sarah_dir, coords)
    open_kwargs = {"chunks": chunks, "parallel": creation_parameters["parallel"]}
    ds_sis = xr.open_mfdataset(files.sis, combine="by_coords", **open_kwargs)[["SIS"]]
    ds_sid = xr.open_mfdataset(files.sid, combine="by_coords", **open_kwargs)[["SID"]]
    ds = xr.merge([ds_sis, ds_sid])
    ds = ds.sel(lon=as_slice(cutout.extent[:2]), lat=as_slice(cutout.extent[2:]))
    ds = ds.assign_coords(
        lon=ds.lon.astype(float).round(4), lat=ds.lat.astype(float).round(4)
    )

    ds = interpolate(ds) if creation_parameters["sarah_interpolate"] else ds.fillna(0)

    if cutout.dt not in ["30min", "30T"]:
        ds = hourly_mean(ds)

    if (cutout.dx != dx) or (cutout.dy != dy):
        ds = regrid(ds, coords["lon"], coords["lat"], resampling=Resampling.average)

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
