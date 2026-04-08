# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
import xarray as xr
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

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

TrackingType: TypeAlias = (
    Literal["horizontal", "tilted_horizontal", "vertical", "dual"] | None
)
ClearskyModel: TypeAlias = Literal["simple", "enhanced"]
TrigonModel: TypeAlias = Literal["simple", "perez"]
IrradiationType: TypeAlias = Literal["total", "direct", "diffuse", "ground"]
HeatPumpSource: TypeAlias = Literal["air", "soil"]
OrientationName: TypeAlias = Literal["latitude_optimal", "constant", "latitude"]
DataFormat: TypeAlias = Literal["grib", "netcdf"]


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
