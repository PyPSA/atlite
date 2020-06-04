# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps
import numpy as np
import pandas as pd
import xarray as xr
from numpy import atleast_1d
from tempfile import mkdtemp
from shutil import rmtree
from dask.diagnostics import ProgressBar
from xarray.core.groupby import DatasetGroupBy
import logging
logger = logging.getLogger(__name__)

from .datasets import modules as datamodules


class requires_windowed(object):
    def __init__(self, features, windows=None, allow_dask=False):
        self.features = features
        self.windows = windows
        self.allow_dask = allow_dask

    def __call__(self, f):
        @wraps(f)
        def wrapper(cutout, *args, **kwargs):
            features = kwargs.pop('features', self.features)
            window_type = kwargs.pop('windows', self.windows)
            windows = create_windows(
                cutout, features, window_type, self.allow_dask)

            return f(cutout, *args, windows=windows, **kwargs)

        return wrapper


def create_windows(cutout, features, window_type, allow_dask):
    features = set(
        features if features is not None else cutout.available_features)
    missing_features = features - \
        set(cutout.data.attrs.get('prepared_features', []))

    if missing_features:
        logger.error(f"The following features need to be prepared first: "
                     f"{', '.join(missing_features)}. Will try anyway!")

    if window_type is False:
        return [cutout.data]
    else:
        return Windows(cutout, features, window_type, allow_dask)


class Windows(object):
    def __init__(self, cutout, features, window_type=None, allow_dask=False):
        group_kws = {}
        if window_type is None:
            group_kws['grouper'] = pd.Grouper(freq="M")
        elif isinstance(window_type, str):
            group_kws['grouper'] = pd.Grouper(freq=window_type)
        elif isinstance(window_type, (int, pd.Index, np.array)):
            group_kws['bins'] = window_type
        elif isinstance(window_type, dict):
            group_kws.update(window_type)
        else:
            raise RuntimeError(
                f"Type of `window_type` (`{type(window_type)}`) "
                "is unsupported")

        # vars = cutout.data.data_vars.keys()
        # if cutout.dataset_module:
        #     mfeatures = cutout.dataset_module.features
        #     dataset_vars = sum((mfeatures[f] for f in features), [])
        #     vars = vars & dataset_vars
        self.data = cutout.data
        self.group_kws = group_kws

        if self.data.chunks is None or allow_dask:
            self.maybe_load = lambda it: it
        else:
            self.maybe_load = lambda it: (ds.load() for ds in it)

        self.groupby = DatasetGroupBy(self.data, self.data.coords['time'],
                                      **self.group_kws)

    def __iter__(self):
        return self.maybe_load(self.groupby._iter_grouped())

    def __len__(self):
        return len(self.groupby)


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

    # target is series of all available variables for given module and features
    target = available_features(modules).loc[:, features].drop_duplicates()
    cutout.data.attrs['prepared_features'] = list(target.index.unique('feature'))

    for module in target.index.unique('module'):
        missing_vars = target[module]
        if not overwrite:
            missing_vars = missing_vars[lambda v: ~v.isin(cutout.data)]
        if missing_vars.empty:
            continue
        logger.info(f'Calculating and writing with module {module}:')
        missing_features = missing_vars.index.unique('feature')
        ds = get_features(cutout, module, missing_features, tmpdir=tmpdir)
        ds = ds[missing_vars.values]
        ds = ds.assign_attrs(cutout.data.attrs)

        # netCDF4 does not permit boolean values. Convert to str to preserve information
        ds.attrs.update({k:v if not isinstance(v, bool) else str(v) for k,v in ds.attrs.items()})

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

