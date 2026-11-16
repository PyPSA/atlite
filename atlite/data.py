# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Management of data retrieval and structure."""

from __future__ import annotations

import gc
import logging
import os
import time
from functools import wraps
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp, mkstemp
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from dask import compute as dask_compute
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
from numpy import atleast_1d

from atlite.datasets import modules as datamodules
from atlite.schema import (
    SCHEMA_VERSION_ATTR,
    require_current_schema_version,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from atlite._types import DataFormat, PathLike
    from atlite.cutout import Cutout

logger = logging.getLogger(__name__)


def _cleanup_sarah_files(cleanup_path: Path, max_retries: int = 3) -> None:
    """
    Safely cleanup SARAH raw files with retry logic for Windows file locks.

    On Windows, files may be locked by Dask or xarray even after computation
    completes. This function retries deletion with increasing wait times to
    allow OS to release locks.

    Parameters
    ----------
    cleanup_path : Path
        Directory to cleanup
    max_retries : int, default 3
        Number of retry attempts

    Notes
    -----
    If cleanup fails after all retries, a warning is logged but the function
    does not raise an error. This ensures prepare() doesn't crash even if
    cleanup fails (the cutout file has already been successfully created).
    """
    if not cleanup_path.exists():
        return

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"Cleaning up SARAH raw files from {cleanup_path} (attempt {attempt + 1}/{max_retries})"
            )
            rmtree(cleanup_path)
            logger.info(f"Successfully cleaned up SARAH files from {cleanup_path}")
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (attempt + 1)  # Exponential backoff: 0.5, 1.0, 1.5 seconds
                logger.debug(
                    f"PermissionError on attempt {attempt + 1}/{max_retries}, "
                    f"waiting {wait_time}s before retry: {e}"
                )
                gc.collect()  # Force garbage collection to release file handles
                time.sleep(wait_time)
            else:
                logger.warning(
                    f"Could not cleanup SARAH raw files from {cleanup_path} after {max_retries} attempts. "
                    f"Please delete manually if needed. Error: {e}"
                )
                return
        except Exception as e:
            logger.warning(
                f"Unexpected error cleaning up SARAH raw files from {cleanup_path}: {e}"
            )
            return


def get_features(
    cutout: Cutout,
    module: str,
    features: Iterable[str],
    data_format: DataFormat,
    tmpdir: PathLike | None = None,
    monthly_requests: bool = False,
    concurrent_requests: bool = False,
) -> xr.Dataset:
    """
    Load feature datasets for a cutout module.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout for which data is retrieved.
    module : str
        Name of the dataset module.
    features : iterable of str
        Feature names to retrieve from the module.
    data_format : str
        Data format requested from the dataset backend.
    tmpdir : str or pathlib.Path, optional
        Directory for intermediate files.
    monthly_requests : bool, optional
        Whether to split requests into monthly chunks.
    concurrent_requests : bool, optional
        Whether to issue monthly requests concurrently.

    Returns
    -------
    xarray.Dataset
        Merged dataset containing the requested features.
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
            data_format=data_format,
            lock=lock,
            monthly_requests=monthly_requests,
            concurrent_requests=concurrent_requests,
            **parameters,
        )
        datasets.append(feature_data)

    datasets = dask_compute(*datasets)

    ds: xr.Dataset = xr.merge(datasets, compat="equals")
    for v in ds:
        da = ds[v]
        da.attrs["module"] = module
        fd = datamodules[module].features.items()
        da.attrs["feature"] = [k for k, l in fd if v in l].pop()

        if da.chunks is not None:
            da.encoding["chunksizes"] = [c[0] for c in da.chunks]

    return ds


def available_features(module: str | Sequence[str] | None = None) -> pd.Series[str]:
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
    features_frame = (
        pd
        .DataFrame(features)
        .unstack()
        .dropna()
        .rename_axis(index=["module", "feature"])
        .rename("variables")
    )
    if module is not None:
        features_frame = features_frame.reindex(atleast_1d(module), level="module")
    return features_frame.explode()


def non_bool_dict(d: dict[str, Any]) -> dict[str, Any]:
    """
    Convert boolean dictionary values to integers.

    Parameters
    ----------
    d : dict
        Dictionary to convert.

    Returns
    -------
    dict
        Dictionary with boolean values replaced by ``0`` or ``1``.
    """
    return {k: v if not isinstance(v, bool) else int(v) for k, v in d.items()}


def maybe_remove_tmpdir(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Wrap a function to manage a temporary directory.

    Parameters
    ----------
    func : callable
        Function accepting a ``tmpdir`` keyword argument.

    Returns
    -------
    callable
        Wrapped function that creates and removes a temporary directory when
        ``tmpdir`` is not provided.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("tmpdir"):
            res: Any = func(*args, **kwargs)
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
    cutout: Cutout,
    features: str | Sequence[str] | None = None,
    tmpdir: PathLike | None = None,
    data_format: DataFormat = "grib",
    overwrite: bool = False,
    compression: dict[str, Any] | None = None,
    show_progress: bool = False,
    dask_kwargs: dict[str, Any] | None = None,
    monthly_requests: bool = False,
    concurrent_requests: bool = False,
) -> Cutout:
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
        The cutout to process.
    features : str/list, optional
        Feature(s) to be prepared. The default slice(None) results in all
        available features.
    tmpdir : str/Path, optional
        Directory in which temporary files (for example retrieved ERA5 netcdf
        files) are stored. If set, the directory will not be deleted and the
        intermediate files can be examined. The path must be a valid path.
    data_format : str, optional
        The data format used to retrieve the data. Only relevant for ERA5 data. The default is 'grib'.
    overwrite : bool, optional
        Whether to overwrite variables which are already included in the
        cutout. The default is False.
    compression : None/dict, optional
        Variable encoding for compression, applied to all variables of the
        cutout when the file is written. The compression is handled via
        xarray.Dataset.to_netcdf(...), for details see:
        https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html .
        The default uses zlib compression after BitRound quantization to 14
        mantissa bits (relative error below 2**-15 = 3.1e-5), see Klöwer et al.
        (2021), https://doi.org/10.1038/s43588-021-00156-2 . For lossless
        compression, remove 'quantize_mode' and 'significant_digits'. zstd
        ({'compression': 'zstd'}) is faster but cannot be read with pip
        installations of netCDF4 on Windows.
        Default is {'zlib': True, 'complevel': 4, 'shuffle': True,
        'quantize_mode': 'BitRound', 'significant_digits': 14}.
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


    Raises
    ------
    ValueError
        If ``tmpdir`` is None.
    FileNotFoundError
        If ``tmpdir`` does not exist.
    atlite.schema.IncompatibleCutoutError
        If features is to be added to an existing cutout that is not the recommended
        schema version of the installed version of ``atlite``.
    RuntimeError
        If the data of a module sets the cutout's schema version attribute.

    """  # noqa: DOC502 (IncompatibleCutoutError is raised by require_current_schema_version)
    if dask_kwargs is None:
        dask_kwargs = {}

    if compression is None:
        compression = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "quantize_mode": "BitRound",
            "significant_digits": 14,
        }

    if cutout.prepared and not overwrite:
        logger.info("Cutout already prepared.")
        return cutout

    require_current_schema_version(cutout.schema_version, cutout.path)

    if tmpdir is None:
        raise ValueError("tmpdir cannot be None")
    temp_dir_path = Path(tmpdir)
    if not temp_dir_path.is_dir():
        raise FileNotFoundError(f"The tmpdir: {temp_dir_path} does not exist.")
    logger.info("Storing temporary files in %s", tmpdir)

    modules_list = atleast_1d(cutout.module).tolist()
    features_normalized = atleast_1d(features) if features else slice(None)
    prepared = set(atleast_1d(cutout.data.attrs["prepared_features"]))

    target = (
        available_features(modules_list).loc[:, features_normalized].drop_duplicates()
    )

    for module in target.index.unique("module"):
        missing_vars = target[module]
        if not overwrite:
            missing_vars = missing_vars[lambda v: ~v.isin(cutout.data)]
        if missing_vars.empty:
            continue
        logger.info("Calculating and writing with module %s:", module)
        missing_features = missing_vars.index.unique("feature")
        ds = get_features(
            cutout,
            module,
            missing_features,
            data_format=data_format,
            tmpdir=tmpdir,
            monthly_requests=monthly_requests,
            concurrent_requests=concurrent_requests,
        )
        if SCHEMA_VERSION_ATTR in ds.attrs:
            raise RuntimeError(
                f"Data of module '{module}' must not contain attribute '{SCHEMA_VERSION_ATTR}'."
            )
        prepared |= set(missing_features)

        cutout.data.attrs.update({"prepared_features": list(prepared)})
        attrs = non_bool_dict(cutout.data.attrs)
        attrs.update(ds.attrs)

        ds = cutout.data.merge(ds[missing_vars.values]).assign_attrs(**attrs)

        if compression:
            for v in ds.data_vars:
                ds[v].encoding.update(compression)

        directory, filename = os.path.split(str(cutout.path))
        fd, tmp = mkstemp(suffix=filename, dir=directory)
        os.close(fd)

        logger.debug("Writing cutout to file...")
        write_job = ds.to_netcdf(tmp, compute=False)
        if show_progress:
            with ProgressBar(minimum=2):
                write_job.compute(**dask_kwargs)
        else:
            write_job.compute(**dask_kwargs)
        if cutout.path.exists():
            cutout.data.close()
            cutout.path.unlink()
        Path(tmp).rename(cutout.path)

        # Clean up SARAH raw files if tracked
        # Must close all file handles before attempting deletion (especially on Windows)
        if "_cleanup_sarah_dir" in ds.attrs:
            # Close dataset to release Dask file handles
            ds.close()
            
            # Try to close existing cutout data if it exists
            if hasattr(cutout, "data") and cutout.data is not None:
                try:
                    cutout.data.close()
                except Exception:
                    pass
            
            # Force garbage collection to release file handles
            gc.collect()
            gc.collect()
            
            # Cleanup with retry logic (handles Windows file locking)
            cleanup_path = Path(ds.attrs["_cleanup_sarah_dir"])
            _cleanup_sarah_files(cleanup_path)

        cutout.data = xr.open_dataset(cutout.path, chunks="auto")
        del ds, write_job

    return cutout
