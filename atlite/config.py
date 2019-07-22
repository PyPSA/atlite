## Copyright 2019 Johannes Hampp (Justus-Liebig University Giessen)

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


import os
import pkg_resources
import yaml
import logging
logger = logging.getLogger(__name__)


_FILE_NAME = "config.yaml"
_DEFAULT_FILE_NAME = "default.config.yaml"

# Search paths for the yaml-configuration file
_SEARCH_PATHS = (
    # User home directory - Custom
    os.path.join(os.path.expanduser("~"), ".atlite", _FILE_NAME),
    # Package install directory - Custom
    pkg_resources.resource_filename(__name__, _FILE_NAME),
    # Package install directory - Default
    pkg_resources.resource_filename(__name__, _DEFAULT_FILE_NAME)
)

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
config_path = None

def read(path):
    """Read and set the configuration based on the file in 'path'."""

    if not os.path.isfile(path):
        raise TypeError("Invalid configuration file path: "
                        "{p}".format(p=path))
    
    with open(path, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    config_dict.update({"config_path": path})
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
    
def _update_variables():
    """Update list of provided attributes by the module."""

    global ATTRS

    ATTRS = {k for k,v in globals().items() if not k.startswith("_") and not callable(v)}

    # Manually remove imported modules and the attribute itself from the list
    ATTRS = ATTRS - {"ATTRS", "logging",
                     "logger", "os", "pkg_resources", "yaml"}


# Try to load configuration from standard paths
for path in _SEARCH_PATHS:
    if os.path.isfile(path):
        # Don't check if the file is actually what it claims to be
        # also: consider a read without error a success.
        read(path)
        break

# Notify user of empty config
if not config_path:
    logger.warn("No valid configuration file found in default paths. "
                "No configuration is loaded, manual configuration required.")