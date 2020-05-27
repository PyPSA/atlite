#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for loading gebco data

"""

import numpy as np
from tempfile import mkstemp
import os
import subprocess
import xarray as xr
from dask import delayed

import logging
logger = logging.getLogger(__name__)


features = {'height'}


def get_data_gebco_height(xs, ys, gebco_path=None):
    # gebco bathymetry heights for underwater
    cornersc = np.array(((xs[0], ys[0]), (xs[-1], ys[-1])))
    minc = np.minimum(*cornersc)
    maxc = np.maximum(*cornersc)
    span = (maxc - minc) / (np.asarray((len(xs), len(ys))) - 1)
    minx, miny = minc - span / 2.
    maxx, maxy = maxc + span / 2.

    fd, target = mkstemp(suffix='.nc')
    os.close(fd)

    delete_target = True

    try:
        cp = subprocess.run(['gdalwarp', '-of', 'NETCDF',
                             '-ts', str(len(xs)), str(len(ys)),
                             '-te', str(minx), str(miny), str(maxx), str(maxy),
                             '-r', 'average',
                             gebco_path, target],
                            capture_output=True,
                            check=True
                            )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"gdalwarp encountered an error, gebco height was not resampled:\n"
            f"{e.stderr}")

    except OSError:
        logger.warning("gdalwarp was not found for resampling gebco. "
                       "Next-neighbour interpolation will be used instead!")
        target = gebco_path
        delete_target = False

    with xr.open_dataset(target) as ds_gebco:
        height = (ds_gebco.rename({'lon': 'x', 'lat': 'y', 'Band1': 'height'})
                          .reindex(x=xs, y=ys, method='nearest')
                          .load()['height'])

    if delete_target:
        os.unlink(target)

    return height


def get_data(cutout, feature, tmpdir, **creation_parameters):
    if 'gebco_path' not in creation_parameters:
        logger.error('Argument "gebco_path" not defined')
    path = creation_parameters['gebco_path']
    coords = cutout.coords

    return delayed(get_data_gebco_height)(coords['x'], coords['y'], path)

