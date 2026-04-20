# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for creating cutouts from the NCEP dataset.

DEPRECATED
----------

The ncep dataset module has not been ported to atlite v0.2, yet. Use atlite v0.0.2 to use it,
for the time being!
"""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Generator

    from atlite._types import PathLike

engine: str = "pynio"
crs: int = 4326


def convert_lons_lats_ncep(
    ds: xr.Dataset, xs: slice | np.ndarray[Any, Any], ys: slice | np.ndarray[Any, Any]
) -> xr.Dataset:
    """Select and rename NCEP longitude/latitude coordinates, handling wraparound.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``lon_0`` and ``lat_0`` coordinates.
    xs : slice or np.ndarray
        Longitude selection range or array.
    ys : slice or np.ndarray
        Latitude selection range or array.

    Returns
    -------
    xr.Dataset
        Dataset with coordinates renamed to ``x``/``y`` and ``lon``/``lat``.
    """
    if not isinstance(xs, slice):
        first, second, last = np.asarray(xs)[[0, 1, -1]]
        xs = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))
    if not isinstance(ys, slice):
        first, second, last = np.asarray(ys)[[0, 1, -1]]
        ys = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))

    ds = ds.sel(lat_0=ys)

    if len(ds.coords["lon_0"].sel(lon_0=slice(xs.start + 360.0, xs.stop + 360.0))):
        ds = xr.concat(
            [ds.sel(lon_0=slice(xs.start + 360.0, xs.stop + 360.0)), ds.sel(lon_0=xs)],
            dim="lon_0",
        )
        ds = ds.assign_coords(
            lon_0=np.where(
                ds.coords["lon_0"].values <= 180,
                ds.coords["lon_0"].values,
                ds.coords["lon_0"].values - 360.0,
            )
        )
    else:
        ds = ds.sel(lon_0=xs)

    ds = ds.rename({"lon_0": "x", "lat_0": "y"})
    return ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])


def convert_time_hourly_ncep(ds: xr.Dataset, drop_time_vars: bool = True) -> xr.Dataset:
    """Stack initial and forecast times into a single hourly time dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``initial_time0_hours`` and ``forecast_time0`` dimensions.
    drop_time_vars : bool, optional
        Drop auxiliary time variables (default ``True``).

    Returns
    -------
    xr.Dataset
        Dataset with a unified ``time`` coordinate.
    """
    ds = ds.stack(time=("initial_time0_hours", "forecast_time0")).assign_coords(
        time=np.ravel(
            ds.coords["initial_time0_hours"]
            + ds.coords["forecast_time0"]
            - pd.Timedelta("1h")
        )
    )
    if drop_time_vars:
        ds = ds.drop(["initial_time0", "initial_time0_encoded"])
    return ds


def convert_unaverage_ncep(ds: xr.Dataset) -> xr.Dataset:
    """Convert running-average variables (``*_avg``) to instantaneous values.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with variables ending in ``_avg``.

    Returns
    -------
    xr.Dataset
        Dataset with un-averaged variables replacing the originals.
    """

    def unaverage(da: xr.DataArray, dim: str = "forecast_time0") -> xr.DataArray:
        coords = da.coords[dim]
        y = da * xr.DataArray(
            np.arange(1, len(coords) + 1), dims=[dim], coords={dim: coords}
        )
        return y - y.shift(**{dim: 1}).fillna(0.0)

    for k, da in ds.items():
        assert isinstance(k, str)
        if k.endswith("_avg"):
            ds[k[: -len("_avg")]] = unaverage(da)
            ds = ds.drop(k)

    return ds


def convert_unaccumulate_ncep(ds: xr.Dataset) -> xr.Dataset:
    """Convert accumulated variables (``*_acc``) to per-timestep values.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with variables ending in ``_acc``.

    Returns
    -------
    xr.Dataset
        Dataset with de-accumulated variables replacing the originals.
    """

    def unaccumulate(da: xr.DataArray, dim: str = "forecast_time0") -> xr.DataArray:
        return da - da.shift(**{dim: 1}).fillna(0.0)

    for k, da in ds.items():
        assert isinstance(k, str)
        if k.endswith("_acc"):
            ds[k[: -len("_acc")]] = unaccumulate(da)
            ds = ds.drop(k)

    return ds


def convert_clip_lower(
    ds: xr.Dataset, variable: str, a_min: float, value: float
) -> xr.Dataset:
    """Replace values at or below a threshold with a fill value.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    variable : str
        Name of the variable to clip.
    a_min : float
        Threshold; values <= ``a_min`` are replaced.
    value : float
        Replacement value.

    Returns
    -------
    xr.Dataset
        Dataset with clipped variable.
    """
    ds[variable] = ds[variable].where(ds[variable] > a_min).fillna(value)
    return ds


def prepare_wnd10m_ncep(
    fn: PathLike,
    yearmonth: tuple[int, int],
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare 10-m wind speed from NCEP U/V components.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    yearmonth : tuple of (int, int)
        Year and month identifier.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``wnd10m`` variable.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds)
        ds["wnd10m"] = np.sqrt(
            ds["VGRD_P0_L103_GGA0"] ** 2 + ds["UGRD_P0_L103_GGA0"] ** 2
        )
        ds = ds.drop(["VGRD_P0_L103_GGA0", "UGRD_P0_L103_GGA0"])
        yield yearmonth, ds


def prepare_influx_ncep(
    fn: PathLike,
    yearmonth: tuple[int, int],
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare downward shortwave radiation flux from NCEP data.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    yearmonth : tuple of (int, int)
        Year and month identifier.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``influx`` variable.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_unaverage_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"DSWRF_P8_L1_GGA0": "influx"})
        ds = convert_clip_lower(ds, "influx", a_min=0.1, value=0.0)
        yield yearmonth, ds


def prepare_outflux_ncep(
    fn: PathLike,
    yearmonth: tuple[int, int],
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare upward shortwave radiation flux from NCEP data.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    yearmonth : tuple of (int, int)
        Year and month identifier.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``outflux`` variable.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_unaverage_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"USWRF_P8_L1_GGA0": "outflux"})
        ds = convert_clip_lower(ds, "outflux", a_min=3.0, value=0.0)
        yield yearmonth, ds


def prepare_temperature_ncep(
    fn: PathLike,
    yearmonth: tuple[int, int],
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare 2-m air temperature from NCEP data.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    yearmonth : tuple of (int, int)
        Year and month identifier.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``temperature`` variable.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"TMP_P0_L103_GGA0": "temperature"})
        yield yearmonth, ds


def prepare_soil_temperature_ncep(
    fn: PathLike,
    yearmonth: tuple[int, int],
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare soil temperature from NCEP data.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    yearmonth : tuple of (int, int)
        Year and month identifier.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``soil temperature`` variable.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"TMP_P0_2L106_GGA0": "soil temperature"})
        yield yearmonth, ds


def prepare_runoff_ncep(
    fn: PathLike,
    yearmonth: tuple[int, int],
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare surface runoff from NCEP data.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    yearmonth : tuple of (int, int)
        Year and month identifier.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``runoff`` variable.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = ds.fillna(0.0)
        ds = convert_unaccumulate_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"WATR_P8_L1_GGA0": "runoff"})
        yield yearmonth, ds


def prepare_height_ncep(
    fn: PathLike,
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    yearmonths: list[tuple[int, int]],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare geopotential height from NCEP data.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    yearmonths : list of tuple of (int, int)
        Year-month pairs to yield the same height data for.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``height`` variable for each yearmonth.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = ds.rename({"HGT_P0_L105_GGA0": "height"})
        for ym in yearmonths:
            yield ym, ds


def prepare_roughness_ncep(
    fn: PathLike,
    yearmonth: tuple[int, int],
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    engine: str = engine,
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Prepare surface roughness from NCEP data.

    Parameters
    ----------
    fn : PathLike
        Path to the GRIB2 file.
    yearmonth : tuple of (int, int)
        Year and month identifier.
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    engine : str, optional
        xarray backend engine.

    Yields
    ------
    tuple of (tuple of (int, int), xr.Dataset)
        ``(yearmonth, dataset)`` with ``roughness`` variable.
    """
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = ds.rename({"SFCR_P8_L1_GGA0": "roughness"})
        ds = ds.assign_coords(year=yearmonth[0]).assign_coords(month=yearmonth[1])
        yield yearmonth, ds


def prepare_meta_ncep(
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    year: int,
    month: int,
    template: str,
    height_config: dict[str, Any],
    module: Any,
    engine: str = engine,
) -> xr.Dataset:
    """Prepare cutout metadata including coordinates and height.

    Parameters
    ----------
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    year : int
        Reference year.
    month : int
        Reference month.
    template : str
        Glob-able file path template with ``{year}`` and ``{month}`` placeholders.
    height_config : dict
        Configuration dict for height preparation (must include ``tasks_func``).
    module : Any
        Dataset module reference.
    engine : str, optional
        xarray backend engine.

    Returns
    -------
    xr.Dataset
        Metadata dataset with coordinates, time, and ``height``.
    """
    fn = next(glob.iglob(template.format(year=year, month=month)))  # noqa: PTH207
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = ds.coords.to_dataset()
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds, drop_time_vars=False)
        meta = ds.load()

    xs = ds["x"].values
    ys = ds["y"].values

    height_config = height_config.copy()
    height_tasks_func = height_config.pop("tasks_func")
    (height_task,) = height_tasks_func(
        xs, ys, [(year, month)], meta_attrs={}, **height_config
    )
    height_prepare_func = height_task.pop("prepare_func")
    _, ds = next(height_prepare_func(**height_task))

    meta["height"] = ds["height"]

    return meta


def tasks_monthly_ncep(
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    yearmonths: list[tuple[int, int]],
    prepare_func: Any,
    template: str,
    meta_attrs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build per-month task dicts for NCEP data preparation.

    Parameters
    ----------
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    yearmonths : list of tuple of (int, int)
        Year-month pairs to create tasks for.
    prepare_func : callable
        Preparation function to invoke per task.
    template : str
        Glob-able file path template with ``{year}`` and ``{month}`` placeholders.
    meta_attrs : dict
        Metadata attributes (unused, kept for interface consistency).

    Returns
    -------
    list of dict
        Task dictionaries keyed for the preparation function.
    """
    return [
        {
            "prepare_func": prepare_func,
            "xs": xs,
            "ys": ys,
            "fn": next(glob.iglob(template.format(year=ym[0], month=ym[1]))),  # noqa: PTH207
            "engine": engine,
            "yearmonth": ym,
        }
        for ym in yearmonths
    ]


def tasks_height_ncep(
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    yearmonths: list[tuple[int, int]],
    prepare_func: Any,
    template: str,
    meta_attrs: dict[str, Any],
    **extra_args: Any,
) -> list[dict[str, Any]]:
    """Build a single task dict for NCEP height data preparation.

    Parameters
    ----------
    xs : slice or np.ndarray
        Longitude selection.
    ys : slice or np.ndarray
        Latitude selection.
    yearmonths : list of tuple of (int, int)
        Year-month pairs passed through to the preparation function.
    prepare_func : callable
        Preparation function to invoke.
    template : str
        Glob-able file path to the height data.
    meta_attrs : dict
        Metadata attributes (unused, kept for interface consistency).
    **extra_args
        Additional keyword arguments forwarded to the task dict.

    Returns
    -------
    list of dict
        Single-element list with the height task dictionary.
    """
    return [
        dict(
            prepare_func=prepare_func,
            xs=xs,
            ys=ys,
            yearmonths=yearmonths,
            fn=next(glob.iglob(template)),  # noqa: PTH207
            **extra_args,
        )
    ]


weather_data_config: dict[str, dict[str, Any]] = {}
try:
    from atlite import config  # type: ignore[attr-defined]

    weather_data_config = {
        "influx": {
            "tasks_func": tasks_monthly_ncep,
            "prepare_func": prepare_influx_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "{year}{month:0>2}/dswsfc.*.grb2",
            ),
        },
        "outflux": {
            "tasks_func": tasks_monthly_ncep,
            "prepare_func": prepare_outflux_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "{year}{month:0>2}/uswsfc.*.grb2",
            ),
        },
        "temperature": {
            "tasks_func": tasks_monthly_ncep,
            "prepare_func": prepare_temperature_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "{year}{month:0>2}/tmp2m.*.grb2",
            ),
        },
        "soil temperature": {
            "tasks_func": tasks_monthly_ncep,
            "prepare_func": prepare_soil_temperature_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "{year}{month:0>2}/soilt1.*.grb2",
            ),
        },
        "wnd10m": {
            "tasks_func": tasks_monthly_ncep,
            "prepare_func": prepare_wnd10m_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "{year}{month:0>2}/wnd10m.*.grb2",
            ),
        },
        "runoff": {
            "tasks_func": tasks_monthly_ncep,
            "prepare_func": prepare_runoff_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "{year}{month:0>2}/runoff.*.grb2",
            ),
        },
        "roughness": {
            "tasks_func": tasks_monthly_ncep,
            "prepare_func": prepare_roughness_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "{year}{month:0>2}/flxf.gdas.*.grb2",
            ),
        },
        "height": {
            "tasks_func": tasks_height_ncep,
            "prepare_func": prepare_height_ncep,
            "template": os.path.join(
                config.ncep_dir,
                "height/cdas1.20130101.splgrbanl.grb2",
            ),
        },
    }
except ImportError:
    pass

meta_data_config: dict[str, Any] = {}
try:
    from atlite import config  # type: ignore[attr-defined]

    meta_data_config = {
        "prepare_func": prepare_meta_ncep,
        "template": os.path.join(
            config.ncep_dir,
            "{year}{month:0>2}/tmp2m.*.grb2",
        ),
        "height_config": weather_data_config["height"],
    }
except (ImportError, KeyError):
    pass
