# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for providing configuration and configuration management to Atlite.
"""

import os
import pkg_resources
import yaml
import logging
logger = logging.getLogger(__name__)


_FILE_NAME = ".atlite.config.yaml"
_FILE_SEARCH_PATH = os.path.join(os.path.expanduser("~"), _FILE_NAME)
_DEFAULT_FILE_NAME = "config.default.yaml"
_DEFAULT_SEARCH_PATH = pkg_resources.resource_filename(__name__, _DEFAULT_FILE_NAME)

# List of all supported attributes for the config
ATTRS = []

# Implemented attributes
cutout_dir = None
windturbine_dir = None
solarpanel_dir = None
ncep_dir = None
cordex_dir = None
sarah_dir = None

# Path of the configuration file.
# Automatically updated when using provided API.
config_path = ""

def read(path):
    """Read and set the configuration based on the file in 'path'."""

    if not os.path.isfile(path):
        raise TypeError("Invalid configuration file path: "
                        "{p}".format(p=path))
    
    with open(path, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    config_dict['config_path'] = path
    update(config_dict)

    logger.info("Configuration from {p} successfully read.".format(p=path))

def save(path, overwrite=False):
    """Write the current configuration into a config file in the specified path.

    Parameters
    ----------
    path : string or os.path
        Including name of the new config file.
    overwrite : boolean
        (Default: False) Allow overwriting of existing files.
    
    """

    if os.path.exists(path) and overwrite is False:
        raise FileExistsError("Overwriting disallowed for {p}".format(p=path))

    # New path now points to the current config
    global config_path
    config_path = path

    # Construct attribute dict
    global ATTRS
    _update_variables()
    
    config = {key:globals()[key] for key in ATTRS}
       
    with open(path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

def update(config_dict):
    """Update the existing config based on the `config_dict` dictionary; resets `config_path`."""

    globals().update(config_dict)
    _update_variables()

def reset():
    """Reset the configuration to its initial values."""

    # Test for file existence in order to not try to read
    # non-existing configuration files at this point (do not confuse the user)
    for path in [_DEFAULT_SEARCH_PATH, _FILE_SEARCH_PATH]:
        if os.path.isfile(path):
            read(path)

    # Notify user of empty config
    if not config_path:
        logger.warn("No valid configuration file found in default and home directories. "
                    "No configuration is loaded, manual configuration required.")

def _update_variables():
    """Update list of provided attributes by the module."""

    global ATTRS

    ATTRS = {k for k,v in globals().items() if not k.startswith("_") and not callable(v)}

    # Manually remove imported modules and the attribute itself from the list
    ATTRS = ATTRS - {"ATTRS", "logging",
                     "logger", "os", "pkg_resources", "yaml"}


# Load the configuration at first module import
reset()