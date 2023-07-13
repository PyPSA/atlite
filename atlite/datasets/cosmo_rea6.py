# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2023 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and preparing data from COSMO REA6 reanalysis.
"""
from __future__ import annotations

import bz2
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import requests
import xarray as xr
from rasterio.warp import Resampling
from scipy.interpolate import griddata
from tqdm import tqdm

from atlite.gis import get_coords, regrid

# avoid circular imports for type hints
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from atlite import Cutout


logger = logging.getLogger(__name__)
logging.captureWarnings(True)

# Model, CRS and Resolution Settings
crs = 4326
dx = 0.0875
dy = 0.0545
dt = "1h"
features = {
    "wind": [
        # "wnd10m",
        # "wnd40m",
        # "wnd60m",
        # "wnd80m",
        # "wnd100m",
        "wnd125m",
        # "wnd150m",
        "wnd175m",
        # "wnd200m",
        "roughness",
    ],
}
static_features = {}


def build_filename(
    year: int,
    month: int,
    dt: Literal[
        "daily", "1D", "D", "hourly", "1H", "H", "monthly", "1M", "M", "1MS", "MS"
    ],
    height: Literal[10, 40, 60, 80, 100, 125, 150, 175, 200],
) -> str:
    """
    Reconstruct a filename used on the opendata.dwd.de server.

    Parameters
    ----------
    year : int
        _description_
    month : int
        _description_
    dt : Literal[&quot;daily&quot;, &quot;1D&quot;, &quot;D&quot;, &quot;hourly&quot;, &quot;1H&quot;, &quot;H&quot;, &quot;monthly&quot;, &quot;1M&quot;, &quot;M&quot;, &quot;1MS&quot;, &quot;MS&quot;]
        _description_
    height : Literal[10, 40, 60, 80, 100, 125, 150, 175, 200]
        _description_

    Returns
    -------
    str
    """
    assert dt in [
        "daily",
        "1D",
        "D",
        "hourly",
        "1H",
        "H",
        "monthly",
        "1M",
        "M",
        "1MS",
        "MS",
    ]
    assert year in range(1995, 2020)
    if year == 2019:  # data only until August available
        assert month in range(1, 9), "COSMO REA6: 2019 only months Jan-Aug available"
    else:
        assert month in range(1, 13)
    available_heights = (10, 40, 60, 80, 100, 125, 150, 175, 200)
    assert height in available_heights, (
        f"wind height level {height} not available in COSMO REA6\n"
        f"available height levels are {available_heights}"
    )

    if dt in ["hourly", "1H", "H"]:
        file_suffix = ".nc4"
    elif dt in ["daily", "1D", "D"]:
        file_suffix = ".DayMean.nc4"
    elif dt in ["monthly", "1M", "M", "1MS", "MS"]:
        file_suffix = ".MonMean.nc"

    filename = f"WS_{height:03d}m.2D.{year}{month:02d}{file_suffix}"
    return filename


def download_file(
    download_url: str, filepath: str | Path, show_progress: bool = True
) -> None:
    """
    Savely download a large file from a URL.

    If the download is interrupted, the incompletely downloaded  file will be deleted.

    Parameters
    ----------
    download_url : str
        URL where the file is located online.
    filepath : str | Path
        Local path where the file should be saved
    show_progress : bool, optional
        Whether to show a progressbar, by default True
    """
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            if show_progress:
                with tqdm.wrapattr(
                    open(filepath, "wb"),
                    "write",
                    miniters=1,
                    desc=download_url.split("/")[-1],
                    total=int(r.headers.get("content-length", 0)),
                ) as f:
                    for chunk in r.iter_content(chunk_size=4096):
                        f.write(chunk)
            else:
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        # if chunk:
                        f.write(chunk)

        logger.info(f"download complete, file saved to\n{filepath}")
    except KeyboardInterrupt:
        # open and close again to free the incompletely downloaded file:
        with open(filepath, "rb") as f:
            pass
        # delete the incompletely downloaded file
        filepath.unlink()
        raise KeyboardInterrupt


def maybe_download_file(
    download_url: str, filepath: str | Path, show_progress: bool = True
) -> None:
    """
    Check if local file exists, otherwise download the file.

    If the download is interrupted, the incompletely downloaded file will be deleted.

    Parameters
    ----------
    download_url : str
        URL where the file is located online.
    filepath : str | Path
        Local path where the file should be saved
    show_progress : bool, optional
        Whether to show a progressbar, by default True
    """
    if Path(filepath).exists():
        logger.info(f"skipping download: {filepath} exists already")
    else:
        download_file(download_url, filepath, show_progress)


def maybe_download_COSMO_constant(out_dir: str | Path) -> Path:
    """
    Downloads the constant data from COMSO REA6 if it does not already exist in
    `out_dir.`

    Parameters
    ----------
    out_dir : str or Path
        directory where the file should be searched and saved for.

    Returns
    -------
    filepath : Path
        filepath of the downloaded file
    """
    url = "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/constant/COSMO_REA6_CONST_withOUTsponge.nc.bz2"
    zipfile_path = Path(out_dir) / "COSMO_REA6_CONST_withOUTsponge.nc.bz2"
    filepath = Path(out_dir) / "COSMO_REA6_CONST_withOUTsponge.nc"

    if filepath.exists():
        logger.info("static COSMO REA6 fields already downloaded")

    else:
        download_file(url, zipfile_path)
        # decompress bz2 file:
        with open(filepath, "wb") as new_file, bz2.BZ2File(zipfile_path, "rb") as file:
            for data in iter(lambda: file.read(100 * 1024), b""):
                new_file.write(data)
        # delete zip file
        zipfile_path.unlink()

    return filepath


def download_wind_from_opendata_dwd(
    year: int,
    month: int,
    dt: Literal[
        "daily", "1D", "D", "hourly", "1H", "H", "monthly", "1M", "M", "1MS", "MS"
    ],
    height: Literal[10, 40, 60, 80, 100, 125, 150, 175, 200],
    out_dir: str | Path,
    show_progress: bool = True,
) -> Path:
    """
    Download wind data from opendata.dwd.de.

    Parameters
    ----------
    year : int
        from 1995 to 2019
    month : int
        Month of the year
    dt : str in {"daily", "hourly", "monthly"}
        time resolution
    height : int
        Height above ground, possible values are
        (10, 40, 60, 80, 100, 125, 150, 175, 200)
    out_dir : str or Path
        directory where to store the file
    show_progress : bool
        Whether download progress is shown or not

    Returns
    -------
    filepath : Path
        path and name where the file has been downloaded to.
    """
    filename = build_filename(int(year), int(month), dt, height)
    filepath = Path(out_dir) / filename
    # common url path for all converted REA6 fields
    base_url = "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted"
    # url for wind speed at a given height on opendata.dwd.de
    if dt in ["hourly", "1H", "H"]:
        dt_dwd = "hourly"
    elif dt in ["daily", "1D", "D"]:
        dt_dwd = "daily"
    elif dt in ["monthly", "1M", "M", "1MS", "MS"]:
        dt_dwd = "monthly"
    file_dir_url = f"{base_url}/{dt_dwd}/2D/WS_{height:03d}/"
    # url pointing to the netcdf file on opendata.dwd.de
    download_url = f"{file_dir_url}{filename}"
    maybe_download_file(download_url, filepath, show_progress)
    return filepath


def _subfeature_to_heigt(subfeature_string: str) -> int:
    return int("".join(x for x in subfeature_string if x.isdigit()))


def _height_to_subfeature(height: int) -> str:
    return f"wnd{int(height)}m"


def get_filenames(
    cosmo_rea6_dir: str | Path,
    cutout: Cutout,
    subfeature: Literal[
        "wnd10m",
        "wnd40m",
        "wnd60m",
        "wnd80m",
        "wnd100m",
        "wnd125m",
        "wnd150m",
        "wnd175m",
        "wnd200m",
    ],
    try_download: bool,
) -> list[Path]:
    """
    Get all files in the COSMO REA6 data directory that match the required
    height level, temporal resolution and time range.

    Parameters
    ----------
    cosmo_rea6_dir : str
    cutout : atlite.Cutout
    subfeature : str
        Subfeature sring decoding the height of the wind speed Needs to have the form
        "wnd{height}m".
    try_download : bool
        Wheter the required files should be downloaded from atlite or manually from the
        user.

    Returns
    -------
    files : list
        list of all files required for the cutout temporal extent and the subfeature.
        This list can be passed to `:func:xarray.open_mfdataset`
    """
    coords = cutout.coords
    included_year_months = (
        coords["time"].dt.year.to_series().astype(str)
        + "_"
        + coords["time"].dt.month.to_series().astype(str).str.zfill(2)
    ).unique()

    files = []
    for ym in included_year_months:
        if try_download:
            files.append(
                download_wind_from_opendata_dwd(
                    year=ym.split("_")[0],
                    month=ym.split("_")[1],
                    dt=cutout.dt,
                    height=_subfeature_to_heigt(subfeature),
                    out_dir=cosmo_rea6_dir,
                    show_progress=True,
                )
            )
        else:
            files.append(
                Path(cosmo_rea6_dir)
                / build_filename(
                    year=ym.split("_")[0],
                    month=ym.split("_")[1],
                    dt=cutout.dt,
                    height=_subfeature_to_heigt(subfeature),
                )
            )

    for f in files:
        if not f.exists():
            warnings.warn(
                (
                    f"{f} is not downloaded yet and will not be added to cutout.\n\nPlease "
                    "download it manually from"
                    "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted"
                    "or set 'try_download=True'"
                )
            )
    return files


def regrid_to_rectilinear(
    ds: xr.Dataset,
    cutout: Cutout,
    data_variable: str,
) -> xr.Dataset:
    """
    Regrid the curvilinear COMSO grid to rectilinear grid used in atlite.

    Parameters
    ----------
    ds : _type_
        _description_
    cutout : _type_
        _description_
    data_variable : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    x1, x2, y1, y2 = cutout.extent
    # get dataset with coords with more or less equal spacing as in COSMO REA6
    ds_regridded = get_coords(
        x=slice(x1, x2),
        y=slice(y1, y2),
        time=slice(
            cutout.coords["time"].min().values, cutout.coords["time"].max().values
        ),
        dx=dx,  # use mean dx from COSMO curvilinear grid
        dy=dy,  # use mean dy from COSMO curvilinear grid
        dt=cutout.dt,
    )

    xx, yy = np.meshgrid(ds_regridded.x.values, ds_regridded.y.values)

    if len(ds[data_variable].dims) == 3:
        assert len(ds_regridded["time"]) == len(ds["time"])
        arr = (
            np.zeros((len(ds_regridded.y), len(ds_regridded.x), len(ds_regridded.time)))
            * np.nan
        )

        # slow loop over time dimension
        for i, t in enumerate(
            tqdm(
                ds_regridded["time"],
                desc=f"transforming {data_variable} to rectilinear grid",
            )
        ):
            try:
                arr[:, :, i] = griddata(
                    points=(ds["RLON"].values.flatten(), ds["RLAT"].values.flatten()),
                    values=ds[data_variable].sel(time=t).values.flatten(),
                    xi=(xx, yy),
                    method="nearest",
                )
            except KeyError:
                # cosmo starts at 01:00:00 of a day, get_coords() at 00:00:00
                pass

        ds_regridded[data_variable] = xr.DataArray(
            data=arr,
            coords={
                "y": ds_regridded.y,
                "x": ds_regridded.x,
                "time": ds_regridded.time,
            },
            attrs=ds[data_variable].attrs,
        )
    elif len(ds[data_variable].dims) == 2:
        arr = np.zeros((len(ds_regridded.y), len(ds_regridded.x))) * np.nan

        arr[:, :] = griddata(
            points=(ds["RLON"].values.flatten(), ds["RLAT"].values.flatten()),
            values=ds[data_variable].values.flatten(),
            xi=(xx, yy),
            method="nearest",
        )

        ds_regridded[data_variable] = xr.DataArray(
            data=arr,
            coords={"y": ds_regridded.y, "x": ds_regridded.x},
            attrs=ds[data_variable].attrs,
        )
    else:
        raise ValueError("DataArray to be regridded needs to be 2D or 3D.")

    return ds_regridded


def get_roughness_subfeature(cutout: Cutout, cosmo_rea6_dir: str | Path) -> xr.Dataset:
    file = maybe_download_COSMO_constant(cosmo_rea6_dir)
    ds = xr.open_dataset(file)
    ds = regrid_to_rectilinear(ds, cutout, "Z0")
    ds = ds.rename({"Z0": "roughness"})
    return ds


def get_wind_heightlevel_subfeature(
    subfeature: str,
    cutout: Cutout,
    cosmo_rea6_dir: str | Path,
    try_download: bool,
    parallel: bool,
) -> xr.Dataset:
    files = get_filenames(
        cosmo_rea6_dir,
        cutout,
        subfeature,
        try_download=try_download,
    )

    open_kwargs = dict(chunks=cutout.chunks, parallel=parallel)
    ds = xr.open_mfdataset(files, combine="by_coords", **open_kwargs)
    ds = regrid_to_rectilinear(ds, cutout, "wind_speed")
    ds = ds.rename({"wind_speed": subfeature})
    return ds


def get_data_wind(
    cutout: Cutout,
    height_levels: list[str],
    cosmo_rea6_dir: str | Path,
    try_download: bool,
    parallel: bool,
) -> xr.Dataset:
    subfeature_ds_list = []
    for subfeature in height_levels:
        logger.info(f"preparing wind subfeature '{subfeature}'")
        if subfeature.startswith("wnd"):
            subfeature_ds_list.append(
                get_wind_heightlevel_subfeature(
                    subfeature,
                    cutout,
                    cosmo_rea6_dir,
                    try_download,
                    parallel,
                )
            )
        else:  # roughness
            subfeature_ds_list.append(
                get_roughness_subfeature(
                    cutout,
                    cosmo_rea6_dir,
                )
            )

    ds = xr.merge(subfeature_ds_list)

    coords = cutout.coords
    if (cutout.dx != dx) or (cutout.dy != dy):
        ds = regrid(ds, coords["lon"], coords["lat"], resampling=Resampling.average)
    return ds


def get_data(cutout, feature, tmpdir, lock=None, **creation_params):
    """
    This function (down)loads COSMO REA6 data and regrids it to match a given
    `:class:atlite.Cutout`.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.cosmo_rea6.features`
    **creation_parameters :
        Mandatory arguments are:
            * 'cosmo_rea6_dir': str of Path
            Directory of the stored COMSO REA6 data.
        Possible arguments are:
            * 'parallel', bool.
                Whether to load stored files in parallel mode. Default is False.
            * 'try_download' : bool
                Whether a data file should be downloaded from opendata.dwd.de if it is
                not available in `cosmo_rea6_dir`.

    Returns
    -------
    xarray.Dataset
        Dataset with the retrieved variables.
    """
    assert cutout.dt in ("1D", "D", "1H", "H", "1M", "M", "1MS", "MS")
    assert "cosmo_rea6_dir" in creation_params.keys()

    creation_params.setdefault("parallel", False)
    creation_params.setdefault("try_download", True)

    levels = features["wind"]

    ds = get_data_wind(
        cutout,
        levels,
        creation_params["cosmo_rea6_dir"],
        creation_params["try_download"],
        creation_params["parallel"],
    )
    return ds
