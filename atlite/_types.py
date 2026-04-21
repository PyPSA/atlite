# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import xarray as xr
from pyproj import CRS

NDArray: TypeAlias = np.ndarray[Any, np.dtype[np.floating[Any]]]
NDArrayInt: TypeAlias = np.ndarray[Any, np.dtype[np.signedinteger[Any]]]
NDArrayBool: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
PathLike: TypeAlias = str | Path
NumericArray: TypeAlias = NDArray | xr.DataArray
Number: TypeAlias = int | float | np.number[Any]
CrsLike: TypeAlias = str | int | CRS | dict[str, Any]
ConvertResult: TypeAlias = (
    xr.DataArray | xr.Dataset | tuple[xr.DataArray | xr.Dataset, xr.DataArray]
)

TrackingType: TypeAlias = Literal["horizontal", "tilted_horizontal", "vertical", "dual"]
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
