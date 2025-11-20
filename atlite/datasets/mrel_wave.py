# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for curating the already downloaded wave data of MREL (ECHOWAVE).

For further reference see:
[1] Matías A., George L., The ECHOWAVE Hindcast: A 30-years high resolution database
for wave energy applications in North Atlantic European waters, Renewable Energy,
Volume 236, 2024, 121391,ISSN 0960-1481, https://doi.org/10.1016/j.renene.2024.121391
"""

import logging
import numpy as np
import xarray as xr
from rasterio.warp import Resampling

from atlite.gis import regrid

logger = logging.getLogger(__name__)

crs = 4326
dx = 0.03
dy = 0.03

features = {"hs": "wave_height", "fp": "wave_period"}

def _rename_and_clean_coords(ds, cutout):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y', fix roundings and grid dimensions.
    """
    coords = cutout.coords
    
    if "longitude" in ds and "latitude" in ds:
        ds = ds.rename({"longitude": "x", "latitude": "y"})
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    if (cutout.dx != dx) or (cutout.dy != dy):
        ds = regrid(ds, coords["x"], coords["y"], resampling=Resampling.average)

    return ds


def sanitize_wave_height(ds):
    """
    Sanitize retrieved wave height data.
    """
    ds["wave_height"] = ds["wave_height"].clip(min=0.0)
    return ds


def sanitize_wave_period(ds):
    """
    Sanitize retrieved wave height data.
    """
    ds["wave_period"] = ds["wave_period"].clip(min=0.0)
    return ds


def _bounds(coords, pad: float = 0) -> dict[str, slice]:
    """
    Convert coordinate bounds to slice and pad if requested.
    """
    x0, x1 = coords["x"].min().item() - pad, coords["x"].max().item() + pad
    y0, y1 = coords["y"].min().item() - pad, coords["y"].max().item() + pad

    return {"x": slice(x0, x1), "y": slice(y0, y1)}

def get_data(cutout, feature, tmpdir, **creation_parameters):
    """
    Load stored MREL (ECHOWAVE) data and reformat to matching the given cutout.

    This function loads and resamples the stored MREL data for a given
    `atlite.Cutout`.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.mrel_wave.features`
    **creation_parameters :
        Mandatory arguments are:
            * 'data_path', str. Directory of the stored MREL data.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.
    """

    if "data_path" not in creation_parameters:
        logger.error('Argument "data_path" not defined')
        raise ValueError('Argument "data_path" not defined')
    path = creation_parameters["data_path"]

    ds = xr.open_dataset(path)
    ds = _rename_and_clean_coords(ds, cutout)
    bounds = _bounds(cutout.coords, pad=creation_parameters.get("pad", 0))
    ds = ds.sel(**bounds)

    # invert the wave peak frequency to obrain wave peak period
    ds['tp'] = (1 / ds['fp'])
    
    ds = ds[list(features.keys())].rename(features)
    for feature in features.values():
        sanitize_func = globals().get(f"sanitize_{feature}")
        if sanitize_func is not None:
            ds = sanitize_func(ds)

    return ds
