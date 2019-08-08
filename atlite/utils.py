# -*- coding: utf-8 -*-

## Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Renewable Energy Atlas Lite (Atlite)

Light-weight version of Aarhus RE Atlas for converting weather data to power systems data
"""

import progressbar as pgb
from pathlib import Path
from contextlib import contextmanager
import pandas as pd
import xarray as xr
import sys
import os
import pkg_resources
import re

from . import config

import logging
logger = logging.getLogger(__name__)

def make_optional_progressbar(show, prefix, max_value=None):
    if show:
        widgets = [
            pgb.widgets.Percentage(),
            ' ', pgb.widgets.SimpleProgress(format='(%s)' % pgb.widgets.SimpleProgress.DEFAULT_FORMAT),
            ' ', pgb.widgets.Bar(),
            ' ', pgb.widgets.Timer(),
            ' ', pgb.widgets.ETA()
        ]
        if not prefix.endswith(": "):
            prefix = prefix.strip() + ": "
        maybe_progressbar = pgb.ProgressBar(prefix=prefix, widgets=widgets, max_value=max_value)
    else:
        maybe_progressbar = lambda x: x

    return maybe_progressbar

def migrate_from_cutout_directory(old_cutout_dir, name, cutout_fn, cutoutparams):
    logger.warning("Found an old-style directory-like cutout. Use `prepare` to transfer that data.")

    old_cutout_dir = Path(old_cutout_dir)
    with xr.open_dataset(old_cutout_dir / "meta.nc") as meta:
        data = xr.open_mfdataset(str(old_cutout_dir / "[12]*.nc"))
        data.attrs.update(meta.attrs)
    data.attrs['prepared_features'] = list(sys.modules['atlite.datasets.' + data.attrs["module"]].features)

    return data

def timeindex_from_slice(timeslice):
    end = pd.Timestamp(timeslice.end) + pd.offsets.DateOffset(months=1)
    return pd.date_range(timeslice.start, end, freq="1h", closed="left")

def construct_filepath(path):
    """Construct the absolute file path from the provided 'path' as per the packages convention.

    Paths which are already absolute are returned unchanged.
    Relative paths are converted into absolute paths.
    The convention for relative paths is:
    They are considered relative to the current 'config.config_path'.
    If the 'config_path' is not defined, just return the relative path.
    """

    if os.path.isabs(path):
        return path
    elif path.startswith('<ATLITE>'):
        return pkg_resources.resource_filename(__name__, path[8:])
    elif config.config_path is None:
        # If config_path is not defined assume the user know what per does
        return path
    else:
        return os.path.join(os.path.dirname(config.config_path), path)

class arrowdict(dict):
    """
    A subclass of dict, which allows you to get
    items in the dict using the attribute syntax!
    """

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(e.args[0])

    _re_pattern = re.compile('[a-zA-Z_][a-zA-Z0-9_]*')

    def __dir__(self):
        dict_keys = []
        for k in self.keys():
            if isinstance(k, str):
                m = self._re_pattern.match(k)
                if m:
                    dict_keys.append(m.string)
        return dict_keys

