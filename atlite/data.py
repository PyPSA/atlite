# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pandas as pd
import xarray as xr
import os
from numpy import atleast_1d
from tempfile import mkstemp, mkdtemp
from shutil import rmtree
from dask import delayed, compute
from dask.utils import SerializableLock
from dask.diagnostics import ProgressBar
import logging
logger = logging.getLogger(__name__)

from .datasets import modules as datamodules


def get_features(cutout, module, features, tmpdir=None):
    """
    Load the feature data for a given module.

    This get the data for a set of features from a module. All modules in
    `atlite.datasets` are allowed.
    """
    parameters = cutout.data.attrs
    lock = SerializableLock()
    datasets = []
    get_data = datamodules[module].get_data

    for feature in features:
        feature_data = delayed(get_data)(cutout, feature, tmpdir=tmpdir,
                                         lock=lock, **parameters)
        datasets.append(feature_data)

    datasets = compute(*datasets)

    ds = xr.merge(datasets, compat='equals')
    for v in ds:
        ds[v].attrs['module'] = module
    return ds


def available_features(module=None):
    """
    Inspect the available features of all or a selection of modules.

    Parameters
    ----------
    module : str/list, optional
        Module name(s) which to inspect. The default None will result in all
        modules

    Returns
    -------
    pd.Series
        A Series of all variables. The MultiIndex indicated which module
        provides the variable and with which feature name the variable can be
        obtained.

    """
    features = {name: m.features for name, m in datamodules.items()}
    features =  pd.DataFrame(features).unstack().dropna() \
                  .rename_axis(index=['module', 'feature']).rename('variables')
    if module is not None:
        features = features.reindex(atleast_1d(module), level='module')
    return features.explode()


def non_bool_dict(d):
    """Convert bool to int for netCDF4 storing"""
    return {k: v if not isinstance(v, bool) else int(v) for k,v in d.items()}


def cutout_prepare(cutout, features=slice(None), tmpdir=None, overwrite=False):
    """
    Prepare all or a selection of features in a cutout.

    This function loads the feature data of a cutout, e.g. influx or runoff.
    When not specifying the `feature` argument, all available features will be
    loaded. The function compares the variables which are already included in
    the cutout with the available variables of the modules specified by the
    cutout. It detects missing variables and stores them into the netcdf file
    of the cutout.


    Parameters
    ----------
    cutout : atlite.Cutout
    features : str/list, optional
        Feature(s) to be prepared. The default slice(None) results in all
        available features.
    tmpdir : str/Path, optional
        Directory in which temporary files (for example retrieved ERA5 netcdf
        files) are stored. If set, the directory will not be deleted and the
        intermediate files can be examined.
    overwrite : bool, optional
        Whether to overwrite variables which are already included in the
        cutout. The default is False.

    Returns
    -------
    cutout : atlite.Cutout
        Cutout with prepared data. The variables are stored in `cutout.data`.

    """
    if tmpdir is None:
        tmpdir = mkdtemp()
        keep_tmpdir = False
    else:
        keep_tmpdir = True

    ds = None

    try:
        logger.info(f'Storing temporary files in {tmpdir}')

        modules = atleast_1d(cutout.module)
        features = atleast_1d(features)
        prepared = set(atleast_1d(cutout.data.attrs['prepared_features']))

        # target is series of all available variables for given module and features
        target = available_features(modules).loc[:, features].drop_duplicates()

        for module in target.index.unique('module'):
            missing_vars = target[module]
            if not overwrite:
                missing_vars = missing_vars[lambda v: ~v.isin(cutout.data)]
            if missing_vars.empty:
                continue
            logger.info(f'Calculating and writing with module {module}:')
            missing_features = missing_vars.index.unique('feature')
            ds = get_features(cutout, module, missing_features, tmpdir=tmpdir)
            prepared |= set(missing_features)

            ds = (cutout.data
                  .merge(ds[missing_vars.values])
                  .assign_attrs(
                      prepared_features=list(prepared),
                      **non_bool_dict(cutout.data.attrs),
                      **ds.attrs))

            # write data to tmp file, copy it to original data, this is much safer
            # than appending variables
            directory, filename = os.path.split(str(cutout.path))
            fd, tmp = mkstemp(suffix=filename, dir=directory)
            os.close(fd)

            with ProgressBar():
                ds.to_netcdf(tmp)
            ds.close()

            # make sure we are only closing data, if it points to the file we want to update
            if (
                    cutout.data._file_obj is not None and
                    cutout.data._file_obj._filename == str(cutout.path.resolve())
            ):
                cutout.data.close()

            if cutout.path.exists():
                cutout.path.unlink()
            os.rename(tmp, cutout.path)

            cutout.data = xr.open_dataset(cutout.path, chunks=cutout.chunks)

    finally:
        if ds is not None:
            ds.close()

        if not keep_tmpdir:
            rmtree(tmpdir)

    return cutout
