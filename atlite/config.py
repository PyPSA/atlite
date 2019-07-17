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
import logging
logger = logging.getLogger(__name__)

class Config(object):

    _FILE_NAME = "config.yaml"
    _DEFAULT_FILE_NAME = ".atlite.default.config.yaml"

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
    # This set of supported attributes can be expanded by
    # adding the attribute name directly to this tuple
    _ATTRS = (
        "cutout_dir",
        "ncep_dir",
        "cordex_dir",
        "sarah_dir",
        "windturbine_dir",
        "solarpanel_dir",
        "gebco_path",
        # For information purposes; updated during config creation
        "config_path" 
    )

    def __init__(self, config_dict=None, config_path=None):
        """Create an empty config object, filling it from a dictionary or a config file.

        If neither `config_dict` nor `config_path` are given, then only an empty object is created.


        Parameters
        ----------
        config_dict : dict
            Dictionary containing the values for a new config, takes priority
            over reading configurations from file.
            (Default: None)
        config_path : string or os.path
            Full path of a valid config file (including arbitrary name).
            (Default: None)
        
        """

        # Add attributes to the instance
        # Manual adding, since __set_attr__ was overwritten
        for a in Config._ATTRS:
            self.__dict__[a] = None

        
        # Try to find a working config
        if config_dict and config_path:
            raise TypeError("Only one of config_dict or config_path "
                            "may be specified at a time")
        elif isinstance(config_dict, dict):
            # Prefer config explicitly provided as a dict
            self.update(config_dict)
        elif config_path is not None:
            # Provide a fully specified path for reading the config from
            self.read(config_path)
        elif config_path:
            self.read(path)
    
    def read(self, path):
        """Read and set the configuration based on the file in 'path'."""
        import os
        import yaml

        if not os.path.isfile(path):
            raise TypeError("Invalid configuration file path: "
                            "{p}".format(p=path))
        
        with open(path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            self.update(config_dict)

        logger.info("Configuration from {p} successfully read.".format(p=path))
        self.__setattr__("config_path", path)
    
    def save(self, path, overwrite=False):
        """Write the current configuration into a config file in the specified path.

        Parameters
        ----------
        path : string or os.path
            Including name of the new config file.
        overwrite : boolean
            (Default: False) Allow overwriting of existing files.
        
        """
        import yaml

        if os.path.exists(path) and overwrite is False:
            raise FileExistsError("Overwriting disallowed for {p}".format(p=path))
 
        # New path now points to the current config
        self.__setattr__("config_path", path)
 
        # Construct attribute dict
        config = {key:self.__getattribute__(key) for key in Config._ATTRS}
       
        with open(path, "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

    def update(self, config_dict):
        """Update the existing config based on the `config_dict` dictionary and reset `config_path`."""

        for key, val in config_dict.items():
            self.__setattr__(key, val)
        
        self.__setattr__("config_path", None)

    def __setattr__(self, key, value):
        """Only allow existing parameters to be set.

        This way, to include new configs, they will first have to be specified
        in this file, providing a single point of reference for all supported
        configs.
        This also prevents the user from accidentally setting configs which do
        not exist (e.g. typos) and then resulting in unexpected behaviour."""

        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise TypeError("Unknown configuration key {k}".format(k=key))

# Create config object to share amongst package modules
# When loading the package, try to load the config from
# a few predefined locations.
config = Config()

# Try to load configuration from standard paths
for path in config._SEARCH_PATHS:
    if os.path.isfile(path):
        # Don't check if the file is actually what it claims to be
        # also: consider a read without error a success.
        config.read(path)
        break

# Notify user of empty config
if not config.config_path:
    logger.warn("No valid configuration file found in default paths. "
                "No configuration is loaded, manual configuration required.")