# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for operations shared among all dataset sources.
"""

import cdsapi
import os
import subprocess
import numpy as np
import xarray as xr
import weakref
from tempfile import mkstemp

import logging
logger = logging.getLogger(__name__)


def noisy_unlink(path):
    logger.info(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(
            f"Unable to delete file {path}, since it is still in use. You will have to clean up yourself!!")


def retrieve_data(product, chunks=None, tmpdir=None, **updates):
    """Download data like ERA5 from the Climate Data Store (CDS)"""

    # Default request
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'day': list(range(1, 31 + 1)),
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        'month': list(range(1, 12 + 1)),
        # 'area': [50, -1, 49, 1], # North, West, South, East. Default: global
        # 'grid': [0.25, 0.25], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
    }
    request.update(updates)

    assert {'year', 'month', 'variable'}.issubset(
        request), "Need to specify at least 'variable', 'year' and 'month'"

    result = cdsapi.Client(
        progress=False
    ).retrieve(
        product,
        request
    )

    fd, target = mkstemp(suffix='.nc', dir=tmpdir)
    os.close(fd)

    logger.info("Downloading request for {} variables to {}".format(
        len(request['variable']), target))

    result.download(target)

    ds = xr.open_dataset(target, chunks=chunks or {})
    if tmpdir is None:
        logger.debug(f"Adding finalizer for {target}")
        weakref.finalize(ds._file_obj._manager, noisy_unlink, target)

    return ds


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
