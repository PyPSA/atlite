# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Management of data retrieval and structure.
"""

import logging
import os
from functools import wraps
from shutil import rmtree
from tempfile import mkdtemp, mkstemp

import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
from numpy import atleast_1d

from atlite.datasets import modules as datamodules

logger = logging.getLogger(__name__)


def get_features(
    cutout,
    module,
    features,
    tmpdir=None,
    monthly_requests=False,
    concurrent_requests=False,
):
    """
    Load the feature data for a given module.

    This get the data for a set of features from a module. All modules
    in `atlite.datasets` are allowed.
    """
    parameters = cutout.data.attrs
    lock = SerializableLock()
    datasets = []
    get_data = datamodules[module].get_data

    for feature in features:
        feature_data = delayed(get_data)(
            cutout,
            feature,
            tmpdir=tmpdir,
            lock=lock,
            monthly_requests=monthly_requests,
            concurrent_requests=concurrent_requests,
            **parameters,
        )
        datasets.append(feature_data)

    datasets = compute(*datasets)

    ds = xr.merge(datasets, compat="equals")
    for v in ds:
        ds[v].attrs["module"] = module
        fd = datamodules[module].features.items()
        ds[v].attrs["feature"] = [k for k, l in fd if v in l].pop()
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
    features = (
        pd.DataFrame(features)
        .unstack()
        .dropna()
        .rename_axis(index=["module", "feature"])
        .rename("variables")
    )
    if module is not None:
        features = features.reindex(atleast_1d(module), level="module")
    return features.explode()


def non_bool_dict(d):
    """
    Convert bool to int for netCDF4 storing.
    """
    return {k: v if not isinstance(v, bool) else int(v) for k, v in d.items()}


def maybe_remove_tmpdir(func):
    """Use this wrapper to make tempfile deletion compatible with windows machines."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("tmpdir", None):
            res = func(*args, **kwargs)
        else:
            kwargs["tmpdir"] = mkdtemp()
            try:
                res = func(*args, **kwargs)
            finally:
                rmtree(kwargs["tmpdir"])
        return res

    return wrapper


@maybe_remove_tmpdir
def cutout_prepare(
    cutout,
    features=None,
    tmpdir=None,
    overwrite=False,
    compression={"zlib": True, "complevel": 9, "shuffle": True},
    show_progress=False,
    dask_kwargs=None,
    monthly_requests=False,
    concurrent_requests=False,
):
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
    compression : None/dict, optional
        Compression level to use for all features which are being prepared.
        The compression is handled via xarray.Dataset.to_netcdf(...), for details see:
        https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html .
        To efficiently reduce cutout sizes, specify the number of 'least_significant_digits': n here.
        To disable compression, set "complevel" to None.
        Default is {'zlib': True, 'complevel': 9, 'shuffle': True}.
    show_progress : bool, optional
        If True, a progress bar is shown. The default is False.
    dask_kwargs : dict, default {}
        Dict with keyword arguments passed to `dask.compute`.
    monthly_requests : bool, optional
        If True, the data is requested on a monthly basis in ERA5. This is useful for
        large cutouts, where the data is requested in smaller chunks. The
        default is False
    concurrent_requests : bool, optional
        If True, the monthly data requests are posted concurrently.
        Only has an effect if `monthly_requests` is True. The default is False.

    Returns
    -------
    cutout : atlite.Cutout
        Cutout with prepared data. The variables are stored in `cutout.data`.

    """
    if dask_kwargs is None:
        dask_kwargs = {}

    if cutout.prepared and not overwrite:
        logger.info("Cutout already prepared.")
        return cutout

    logger.info(f"Storing temporary files in {tmpdir}")

    modules = atleast_1d(cutout.module)
    features = atleast_1d(features) if features else slice(None)
    prepared = set(atleast_1d(cutout.data.attrs["prepared_features"]))

    # target is series of all available variables for given module and features
    target = available_features(modules).loc[:, features].drop_duplicates()

    for module in target.index.unique("module"):
        missing_vars = target[module]
        if not overwrite:
            missing_vars = missing_vars[lambda v: ~v.isin(cutout.data)]
        if missing_vars.empty:
            continue
        logger.info(f"Calculating and writing with module {module}:")
        missing_features = missing_vars.index.unique("feature")
        ds = get_features(
            cutout,
            module,
            missing_features,
            tmpdir=tmpdir,
            monthly_requests=monthly_requests,
            concurrent_requests=concurrent_requests,
        )
        prepared |= set(missing_features)

        cutout.data.attrs.update(dict(prepared_features=list(prepared)))
        attrs = non_bool_dict(cutout.data.attrs)
        attrs.update(ds.attrs)

        # Add optional compression to the newly prepared features
        if compression:
            for v in missing_vars:
                ds[v].encoding.update(compression)

        ds = cutout.data.merge(ds[missing_vars.values]).assign_attrs(**attrs)

        # write data to tmp file, copy it to original data, this is much safer
        # than appending variables
        directory, filename = os.path.split(str(cutout.path))
        fd, tmp = mkstemp(suffix=filename, dir=directory)
        os.close(fd)

        logger.debug("Writing cutout to file...")
        # Delayed writing for large cutout
        # cf. https://stackoverflow.com/questions/69810367/python-how-to-write-large-netcdf-with-xarray
        write_job = ds.to_netcdf(tmp, compute=False)
        if show_progress:
            with ProgressBar(minimum=2):
                write_job.compute(**dask_kwargs)
        else:
            write_job.compute(**dask_kwargs)
        if cutout.path.exists():
            cutout.data.close()
            cutout.path.unlink()
        os.rename(tmp, cutout.path)

        cutout.data = xr.open_dataset(cutout.path, chunks=cutout.chunks)

    return cutout
