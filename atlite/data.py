# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pandas as pd
import xarray as xr
from numpy import atleast_1d
from tempfile import mkdtemp
from shutil import rmtree
from dask.diagnostics import ProgressBar
import logging
logger = logging.getLogger(__name__)

from .datasets import modules as datamodules


def get_features(cutout, module, features, tmpdir=None):
    """
    Load the feature data for a given module.

    This get the data for a set of features from a module. All modules in
    `atlite.datasets` all allowed.
    """
    parameters = cutout.data.attrs
    datasets = []
    get_data = datamodules[module].get_data

    for feature in features:
        feature_data = get_data(cutout, feature, tmpdir=tmpdir, **parameters)
        datasets.append(feature_data)

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

    modules = atleast_1d(cutout.module)
    features = atleast_1d(features)
    prepared = set(cutout.data.attrs['prepared_features'])

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
        # make sure we don't loose any unused dimension by selecting
        ds = ds[missing_vars.values].assign_coords(ds.coords)

        ds = ds.assign_attrs(**cutout.data.attrs)
        prepared |= set(missing_features)
        ds = ds.assign_attrs(prepared_features = list(prepared))
        # convert bool to int for netCDF4 storing
        ds.attrs.update({k: v if not isinstance(v, bool) else int(v)
                         for k,v in ds.attrs.items()})

        with ProgressBar():
            if cutout.path.exists():
                mode = 'a'
            else:
                mode = 'w'

            ds.to_netcdf(cutout.path, mode=mode)

    if not keep_tmpdir:
        rmtree(tmpdir)

    cutout.data = xr.open_dataset(cutout.path, chunks=cutout.chunks)

    return cutout

