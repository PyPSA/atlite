# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for creating cutouts from the CORDEX dataset.

DEPRECATED
----------

The cordex dataset module has not been ported to atlite v0.2, yet. Use atlite v0.0.2 to use it,
for the time being!
"""

from __future__ import annotations

import glob
import os
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np

    from atlite._types import PathLike

# Model and CRS Settings
model = "MPI-M-MPI-ESM-LR"

crs = 4326  # TODO
# something like the following is correct
# crs = pyproj.crs.DerivedGeographicCRS(4326, pcrs.coordinate_operation.RotatedLatitudeLongitudeConversion(??))
# RotProj(dict(proj='ob_tran', o_proj='latlong', lon_0=180, o_lon_p=-162, o_lat_p=39.25))


def rename_and_clean_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename rotated coordinates and drop auxiliary variables.

    Parameters
    ----------
    ds : xr.Dataset
        CORDEX dataset with rotated lon/lat coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with ``rlon``/``rlat`` renamed to ``x``/``y`` and
        ``bnds``, ``height``, ``rotated_pole`` removed if present.
    """
    ds = ds.rename({"rlon": "x", "rlat": "y"})
    return ds.drop(
        (set(ds.coords) | set(ds.data_vars)) & {"bnds", "height", "rotated_pole"}
    )


def prepare_data_cordex(
    fn: PathLike,
    year: int,
    months: list[int],
    oldname: str,
    newname: str,
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Load and prepare time-varying CORDEX data, yielding per-month slices.

    Parameters
    ----------
    fn : PathLike
        Path to the NetCDF file.
    year : int
        Year to extract.
    months : list of int
        Months to extract.
    oldname : str
        Original variable name in the dataset.
    newname : str
        Target variable name after renaming.
    xs : slice or np.ndarray
        Spatial selection along x.
    ys : slice or np.ndarray
        Spatial selection along y.

    Yields
    ------
    tuple of ((int, int), xr.Dataset)
        ``(year, month)`` key and the corresponding monthly dataset slice.
    """
    with xr.open_dataset(fn) as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.rename({oldname: newname})
        ds = ds.sel(x=xs, y=ys)

        if newname in {"influx", "outflux"}:
            ds = ds.assign_coords(
                time=(
                    pd.to_datetime(ds.coords["time"].values) - pd.Timedelta(hours=1.5)
                )
            )
        elif newname in {"runoff"}:
            t = pd.to_datetime(ds.coords["time"].values)
            ds = ds.reindex(method="bfill", time=(t - pd.Timedelta(hours=3.0)).union(t))

        for m in months:
            yield (year, m), ds.sel(time=f"{year}-{m}")


def prepare_static_data_cordex(
    fn: PathLike,
    year: int,
    months: list[int],
    oldname: str,
    newname: str,
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Load and prepare static (time-invariant) CORDEX data.

    Parameters
    ----------
    fn : PathLike
        Path to the NetCDF file.
    year : int
        Year key for the yielded tuples.
    months : list of int
        Months to yield entries for.
    oldname : str
        Original variable name in the dataset.
    newname : str
        Target variable name after renaming.
    xs : slice or np.ndarray
        Spatial selection along x.
    ys : slice or np.ndarray
        Spatial selection along y.

    Yields
    ------
    tuple of ((int, int), xr.Dataset)
        ``(year, month)`` key and the static dataset (same for each month).
    """
    with xr.open_dataset(fn) as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.rename({oldname: newname})
        ds = ds.sel(x=xs, y=ys)

        for m in months:
            yield (year, m), ds


def prepare_weather_types_cordex(
    fn: PathLike,
    year: int,
    months: list[int],
    oldname: str,
    newname: str,
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
) -> Generator[tuple[tuple[int, int], xr.Dataset], None, None]:
    """Load and prepare CORDEX weather type classification data.

    Parameters
    ----------
    fn : PathLike
        Path to the NetCDF file.
    year : int
        Year to extract.
    months : list of int
        Months to extract.
    oldname : str
        Original variable name in the dataset.
    newname : str
        Target variable name after renaming.
    xs : slice or np.ndarray
        Unused, kept for interface consistency.
    ys : slice or np.ndarray
        Unused, kept for interface consistency.

    Yields
    ------
    tuple of ((int, int), xr.Dataset)
        ``(year, month)`` key and the corresponding monthly dataset slice.
    """
    with xr.open_dataset(fn) as ds:
        ds = ds.rename({oldname: newname})
        for m in months:
            yield (year, m), ds.sel(time=f"{year}-{m}")


def prepare_meta_cordex(
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    year: int,
    month: int,
    template: str,
    height_config: dict[str, Any],
    module: Any,
    model: str = "MPI-M-MPI-ESM-LR",
) -> xr.Dataset:
    """Build metadata dataset for a CORDEX cutout including height.

    Parameters
    ----------
    xs : slice or np.ndarray
        Spatial selection along x.
    ys : slice or np.ndarray
        Spatial selection along y.
    year : int
        Reference year.
    month : int
        Reference month.
    template : str
        Glob template for locating NetCDF files.
    height_config : dict
        Configuration for height data retrieval.
    module : Any
        Dataset module reference.
    model : str, optional
        Climate model identifier.

    Returns
    -------
    xr.Dataset
        Coordinate metadata dataset with height variable.
    """
    fn = next(glob.iglob(template.format(year=year, model=model)))  # noqa: PTH207
    with xr.open_dataset(fn) as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.coords.to_dataset()
        meta = ds.sel(time=f"{year}-{month}", x=xs, y=ys).load()

    xs = ds["x"].values
    ys = ds["y"].values

    height_config = height_config.copy()
    height_tasks_func = height_config.pop("tasks_func")
    (height_task,) = height_tasks_func(
        xs, ys, [(year, month)], meta_attrs={}, **height_config
    )
    height_prepare_func = height_task.pop("prepare_func")
    _, ds = height_prepare_func(**height_task)[0]

    meta["height"] = ds["height"]

    return meta


def tasks_yearly_cordex(
    xs: slice | np.ndarray[Any, Any],
    ys: slice | np.ndarray[Any, Any],
    yearmonths: list[tuple[int, int]],
    prepare_func: Any,
    template: str,
    oldname: str,
    newname: str,
    meta_attrs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create yearly preparation task dicts for CORDEX data retrieval.

    Parameters
    ----------
    xs : slice or np.ndarray
        Spatial selection along x.
    ys : slice or np.ndarray
        Spatial selection along y.
    yearmonths : list of (int, int)
        ``(year, month)`` pairs to process.
    prepare_func : callable
        Function to call for data preparation.
    template : str
        Glob template for locating NetCDF files.
    oldname : str
        Original variable name in the dataset.
    newname : str
        Target variable name after renaming.
    meta_attrs : dict
        Cutout metadata attributes; must contain ``"model"`` key.

    Returns
    -------
    list of dict
        One task dict per year with keys needed by ``prepare_func``.
    """
    model = meta_attrs["model"]

    if not isinstance(xs, slice):
        first, second, last = xs.values[[0, 1, -1]]
        xs = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))
    if not isinstance(ys, slice):
        first, second, last = ys.values[[0, 1, -1]]
        ys = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))

    return [
        {
            "prepare_func": prepare_func,
            "xs": xs,
            "ys": ys,
            "oldname": oldname,
            "newname": newname,
            "fn": next(glob.iglob(template.format(year=year, model=model))),  # noqa: PTH207
            "year": year,
            "months": list(map(itemgetter(1), yearmonths)),
        }
        for year, yearmonths in groupby(yearmonths, itemgetter(0))
    ]


cordex_dir = os.environ["ATLITE_CORDEX_DIR"]

weather_data_config: dict[str, dict[str, Any]] = {
    "influx": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_data_cordex,
        "oldname": "rsds",
        "newname": "influx",
        "template": os.path.join(cordex_dir, "{model}", "influx", "rsds_*_{year}*.nc"),
    },
    "outflux": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_data_cordex,
        "oldname": "rsus",
        "newname": "outflux",
        "template": os.path.join(cordex_dir, "{model}", "outflux", "rsus_*_{year}*.nc"),
    },
    "temperature": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_data_cordex,
        "oldname": "tas",
        "newname": "temperature",
        "template": os.path.join(
            cordex_dir, "{model}", "temperature", "tas_*_{year}*.nc"
        ),
    },
    "humidity": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_data_cordex,
        "oldname": "hurs",
        "newname": "humidity",
        "template": os.path.join(
            cordex_dir, "{model}", "humidity", "hurs_*_{year}*.nc"
        ),
    },
    "wnd10m": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_data_cordex,
        "oldname": "sfcWind",
        "newname": "wnd10m",
        "template": os.path.join(cordex_dir, "{model}", "wind", "sfcWind_*_{year}*.nc"),
    },
    "roughness": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_static_data_cordex,
        "oldname": "rlst",
        "newname": "roughness",
        "template": os.path.join(cordex_dir, "{model}", "roughness", "rlst_*.nc"),
    },
    "runoff": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_data_cordex,
        "oldname": "mrro",
        "newname": "runoff",
        "template": os.path.join(cordex_dir, "{model}", "runoff", "mrro_*_{year}*.nc"),
    },
    "height": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_static_data_cordex,
        "oldname": "orog",
        "newname": "height",
        "template": os.path.join(cordex_dir, "{model}", "altitude", "orog_*.nc"),
    },
    "CWT": {
        "tasks_func": tasks_yearly_cordex,
        "prepare_func": prepare_weather_types_cordex,
        "oldname": "CWT",
        "newname": "CWT",
        "template": os.path.join(
            cordex_dir, "{model}", "weather_types", "CWT_*_{year}*.nc"
        ),
    },
}

meta_data_config: dict[str, Any] = {
    "prepare_func": prepare_meta_cordex,
    "template": os.path.join(cordex_dir, "{model}", "temperature", "tas_*_{year}*.nc"),
    "height_config": weather_data_config["height"],
}
