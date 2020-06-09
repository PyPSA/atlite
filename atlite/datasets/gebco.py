#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for loading gebco data

"""

import numpy as np
from tempfile import mkstemp
import os
import rasterio as rio
import xarray as xr

import logging
logger = logging.getLogger(__name__)

projection = 'latlong'
features = {'height': ['height']}


def get_data_gebco_height(xs, ys, gebco_path):
    x, X = xs[[0, -1]]
    y, Y = ys[[0, -1]]

    dx = (X - x) / (len(xs) - 1)
    dy = (Y - y) / (len(ys) - 1)

    with rio.open(gebco_path) as dataset:
        window = dataset.window(x - dx/2, y - dy/2, X + dx/2, Y + dy/2)
        gebco = dataset.read(indexes=1,
                             window=window,
                             out_shape=(len(ys), len(xs)),
                             resampling=Resampling.average)
        tags = dataset.tags(bidx=1)

    return xr.DataArray(gebco, coords=[("y", ys), ("x", xs)], name='height', attrs=tags)


def get_data(cutout, feature, tmpdir, **creation_parameters):
    """
    Get the gebco height data.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Takes no effect, only here for consistency with other dataset modules.
    tmpdir : str
        Takes no effect, only here for consistency with other dataset modules.
    **creation_parameters :
        Must include `gebco_path`.

    Returns
    -------
    xr.Dataset
    """
    if 'gebco_path' not in creation_parameters:
        logger.error('Argument "gebco_path" not defined')
    path = creation_parameters['gebco_path']

    coords = cutout.coords
    #assign time dimesion even if not used
    return get_data_gebco_height(coords['x'], coords['y'], path)\
            .to_dataset()\
            .assign_coords(cutout.coords)


