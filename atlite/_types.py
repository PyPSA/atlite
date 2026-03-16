# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
import xarray as xr
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

NDArray: TypeAlias = np.ndarray[Any, np.dtype[np.floating[Any]]]
NDArrayInt: TypeAlias = np.ndarray[Any, np.dtype[np.signedinteger[Any]]]
NDArrayBool: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
DataArray: TypeAlias = xr.DataArray
Dataset: TypeAlias = xr.Dataset
PathLike: TypeAlias = str | Path
NumericArray: TypeAlias = NDArray | DataArray
Number: TypeAlias = int | float | np.number[Any]
GeoDataFrame: TypeAlias = gpd.GeoDataFrame
GeoSeries: TypeAlias = gpd.GeoSeries
Geometry: TypeAlias = BaseGeometry
CrsLike: TypeAlias = str | int | CRS | dict[str, Any] | None
SparseMatrix: TypeAlias = sp.lil_matrix | sp.csr_matrix


class CutoutPrepareConfig(TypedDict, total=False):
    datasets: list[str]
    months: list[int]
    start_year: int
    end_year: int


class DatasetConfig(TypedDict, total=False):
    module: str
    version: str
    years: list[int]


class ConversionConfig(TypedDict, total=False):
    data_source: str
    temperature: bool
    wind_speed: bool
    solar_irradiance: bool


class PVConfig(TypedDict, total=False):
    tracking: Literal["fixed", "horizontal", "vertical", "two_axis"]
    orientation: Literal["south", "fixed"]
    tilt: float | None
    azimuth: float | None
    racking: Literal[
        "open_rack_cell_glued_back",
        "close_mount_cell_glued_back",
        "open_rack_polymer_thinfilm_copper_covered_edge",
    ]


class ERA5RetrievalParams(TypedDict, total=False):
    product: str
    area: list[float]
    grid: str
    chunks: dict[str, int] | None
    tmpdir: str | Path | None
    lock: Any | None
    data_format: Literal["grib", "netcdf"]
    year: list[str]
    month: list[str] | str
    day: list[str] | str
    time: str | list[str]
    variable: str | list[str]


class SarahCreationParams(TypedDict, total=False):
    sarah_dir: str | Path
    parallel: bool
    sarah_interpolate: bool


class GebcoCreationParams(TypedDict, total=False):
    gebco_path: str | Path


class TaskDict(TypedDict, total=False):
    prepare_func: Callable[..., Any]
    xs: Any
    ys: Any
    yearmonths: list[tuple[int, int]]
    fn: str | Path
    year: int
    month: int | list[int]
    yearmonth: tuple[int, int]
    engine: str
    oldname: str
    newname: str
    template: str
    drop_time_vars: bool


class CSPConfig(TypedDict, total=False):
    turbine: str
    capacity: float


class WindConfig(TypedDict, total=False):
    turbine: str
    capacity: float
    hub_height: float | None


class LayoutConfig(TypedDict, total=False):
    layout: DataArray | None
    capacity: float | None


class ShapeConfig(TypedDict, total=False):
    shapes: Sequence[Geometry] | None
    shapes_crs: CrsLike


class AggregationConfig(TypedDict, total=False):
    matrix: SparseMatrix | DataArray | None
    index: Any
    per_unit: bool
    return_capacity: bool
    aggregate_time: Literal["sum", "mean"] | bool | None
    capacity_factor: bool
    capacity_factor_timeseries: bool
    show_progress: bool
    dask_kwargs: dict[str, Any]
