from pathlib import Path
import yaml
import atlite
import pkg_resources
import logging
logger = logging.getLogger(__name__)

class Config(object):
    _FILE_NAME = Path(".atlite.config.yaml")
    _FILE_SEARCH_PATH = Path.home().joinpath(_FILE_NAME)
    _DEFAULT_FILE_NAME = Path("config.default.yaml")
    _DEFAULT_SEARCH_PATH = Path(pkg_resources.resource_filename("atlite", str(_DEFAULT_FILE_NAME)))

    # Functions called on config change
    _UPDATE_HOOKS = []

    def resolve_filepath(self, path):
        """Construct the absolute file path from the provided 'path' as per the packages convention.

        Paths which are already absolute are returned unchanged.
        Relative paths are converted into absolute paths.
        The convention for relative paths is:
        They are considered relative to the current 'config.config_path'.
        If the 'config_path' is not defined, just return the relative path.

        Returns
        -------
        path : pathlib.Path
            A Path object pointing to the resolved location.
        """

        if path is None:
            return path
        
        path = Path(path)

        if path.is_absolute():
            return path
        elif path.parts[0] == '<ATLITE>':
            return Path(pkg_resources.resource_filename("atlite", str(Path(*path.parts[1:]))))
        elif self._config_path is None:
            # If config_path is not defined assume user know what per does
            return path
        elif Path(self._config_path) == path:
            # Avoid recursion resolving the same path over and over again
            return path
        else:
            return Path(self._config_path).parent.joinpath(path)

    @property
    def cutout_dir(self):
        return self.resolve_filepath(self._cutout_dir)
    
    @property
    def ncep_dir(self):
        return self.resolve_filepath(self._ncep_dir)

    @property
    def cordex_dir(self):
        return self.resolve_filepath(self._cordex_dir)

    @property
    def sarah_dir(self):
        return self.resolve_filepath(self._sarah_dir)

    @property
    def windturbine_dir(self):
        return self.resolve_filepath(self._windturbine_dir)

    @property
    def solarpanel_dir(self):
        return self.resolve_filepath(self._solarpanel_dir)

    @property
    def gebco_path(self):
        return self.resolve_filepath(self._gebco_path)

    @property
    def config_path(self):
        return self.resolve_filepath(self._config_path)

    def __init__(self, config_dict=None, config_path=None):
        """Create a config object using a dictionary or by specifying a config file path.

        If neither `config_dict` nor `config_path` are given, then try to read a config
        file from one of the default locations.

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

        self._cutout_dir = None
        self._ncep_dir = None
        self._cordex_dir = None
        self._sarah_dir = None
        self._windturbine_dir = None
        self._solarpanel_dir = None
        self._gebco_path = None
        self._config_path = None

        # Try to find a working config
        
        # Prefer config explicitly provided as a dict
        if isinstance(config_dict, dict):
            logger.info(f"Loading config from dict.")
            self.update(config_dict)
        elif config_path is not None:
            # Provide a fully specified path for reading the config from
            logger.info(f"Loading config from {str(config_path)}.")
            self.read(config_path)
        else:
            # Try to load configuration from standard paths
            for path in [Config._FILE_SEARCH_PATH, Config._DEFAULT_SEARCH_PATH]:
                if path.is_file():
                    logger.info(f"Loading config from {str(path)}.")
                    self.read(path)
                    # Either successfully read or invalid file
                    break

    def read(self, path):
        """Read and set the configuration based on the file in 'path'."""

        path = Path(path)
        if not path.is_file():
            raise TypeError(f"Invalid configuration file path {str(path)}.")
        
        with open(path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            self.update(config_dict)

        logger.info(f"Configuration from {str(path)} successfully read.")
        self.update(config_path=path)
    
    def save(self, path, overwrite=False):
        """Write the current configuration into a config file in the specified path.

        Parameters
        ----------
        path : string or os.path
            Including name of the new config file.
        overwrite : boolean
            (Default: False) Allow overwriting of existing files.
        
        """

        path = Path(path)

        if path.exists() and overwrite is False:
            raise FileExistsError(f"Overwriting disallowed for {str(path)}")
 
        # New path now points to the current config
        self.update(config_path=str(path))
 
        # Construct attribute dict, make sure to only store string representations
        config = {k:str(v) for k,v in self.__dict__.items()}
       
        with open(path, "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

    def update(self, config_dict=None, **kwargs):
        """Update the existing config.
        
        Use a dictionary `config_dict` for updating using a single object
        or provide assignment expressions to the config variables you
        want to change.
        Using this method ensures that all internal dependencies relying
        on path information are also also correctly updated.
        
        Parameters
        ----------
        config_dict : dict
            (Default: dict()). Dictionary with pairs of key value pairs:
            <config var. name>:<new value>.
        **kwargs
            Any existing or new config variable and its new value to store
            in the atlite.config.
        """

        if config_dict is None:
            config_dict = dict()
        updates = kwargs
        updates.update(config_dict)

        for key, val in updates.items():
            self.__setattr__("_"+str(key), val)

        for func in self._UPDATE_HOOKS:
            func()

def reset():
    """Reset the configuration to its initial values at atlite import."""

    global config
    config = Config()

# Module's config object to be accessed by other modules
# This can be exchanged for another Config object to interchange
# the configuration quickly
# TODO : If "config" is assigned a different Config object, then
# TODO : functions in _UPDATE_HOOKS need to be called (are currently not)
config = None

# Init the configuration at module import
reset()