# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
General utility functions for internal use.
"""

import logging
import re
import textwrap
from pathlib import Path

import pandas as pd
import xarray as xr

from atlite.datasets import modules as datamodules
from atlite.gis import maybe_swap_spatial_dims

logger = logging.getLogger(__name__)


def migrate_from_cutout_directory(old_cutout_dir, path):
    """
    Convert an old style cutout directory to new style netcdf file.
    """
    old_cutout_dir = Path(old_cutout_dir)
    with xr.open_dataset(old_cutout_dir / "meta.nc") as meta:
        newname = f"{old_cutout_dir.name}.nc"
        module = meta.attrs["module"]
        minX, maxX = meta.indexes["x"][[0, -1]]
        minY, maxY = sorted(meta.indexes["y"][[0, -1]])
        minT, maxT = meta.indexes["time"][[0, -1]].strftime("%Y-%m")

        logger.warning(
            textwrap.dedent(
                f"""
            Found an old-style directory-like cutout. It can manually be
            recreated using

            cutout = atlite.Cutout("{newname}",
                                   module="{module}",
                                   time=slice("{minT}", "{maxT}"),
                                   x=slice({minX}, {maxX}),
                                   y=slice({minY}, {maxY})
            cutout.prepare()

            but we are trying to offer an automated migration as well ...
        """
            )
        )

        try:
            data = xr.open_mfdataset(
                str(old_cutout_dir / "[12]*.nc"), combine="by_coords"
            )
            data.attrs.update(meta.attrs)
        except xr.MergeError:
            logger.exception(
                "Automatic migration failed. Re-create the cutout "
                "with the command above!"
            )
            raise

    data = maybe_swap_spatial_dims(data)
    module = data.attrs["module"]
    data.attrs["prepared_features"] = list(datamodules[module].features)
    for v in data:
        data[v].attrs["module"] = module
        fd = datamodules[module].features.items()
        features = [k for k, l in fd if v in l]
        data[v].attrs["feature"] = features.pop() if features else "undefined"

    path = Path(path).with_suffix(".nc")
    logger.info(
        f"Writing cutout data to {path}. When done, load it again using"
        f"\n\n\tatlite.Cutout('{path}')"
    )
    data.to_netcdf(path)
    return data


def timeindex_from_slice(timeslice):
    end = pd.Timestamp(timeslice.end) + pd.offsets.DateOffset(months=1)
    return pd.date_range(timeslice.start, end, freq="1h", closed="left")


class arrowdict(dict):
    """
    A subclass of dict, which allows you to get items in the dict using the
    attribute syntax!
    """

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(e.args[0])

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self):
        dict_keys = []
        for k in self.keys():
            if isinstance(k, str):
                m = self._re_pattern.match(k)
                if m:
                    dict_keys.append(m.string)
        return dict_keys


class CachedAttribute:
    """
    Computes attribute value and caches it in the instance.

    From the Python Cookbook (Denis Otkidach) This decorator allows you
    to create a property which can be computed once and accessed many
    times. Sort of like memoization.
    """

    # For python 3.8 >= use functoolts.cached_property instead.

    def __init__(self, method, name=None, doc=None):
        # record the unbound-method and the name
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = doc or method.__doc__

    def __get__(self, inst, cls):
        if inst is None:
            # instance attribute accessed on class, return self
            # You get here if you write `Foo.bar`
            return self
        # compute, cache and return the instance's attribute value
        result = self.method(inst)
        # setattr redefines the instance's attribute so this doesn't get called
        # again
        setattr(inst, self.name, result)
        return result
