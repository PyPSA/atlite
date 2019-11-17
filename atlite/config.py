import yaml
import atlite
import pkg_resources
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def _make_property(var, ispath=True):
    if ispath:
        def getvar(self):
            return self.resolve_filepath(getattr(self, f"_{var}"))
    else:
        def getvar(self):
            return getattr(self, f"_{var}")
    def setvar(self):
        setattr(self, f"_{var}")
        self.update_hooks
    # Maybe add link to online documentation to doc-string
    return property(getvar, setvar, doc=f"Get or set config variable {var}")

def add_properties(cls):
    for var in cls._VARIABLES:
        setattr(cls, var, _make_property(var))
    return cls

@add_properties
class Config:
    _VARIABLES = ["config_path", "cutout_dir", "sarah_dir", "windturbine_dir",
                  "solarpanel_dir", "gebco_path"]

    _FILE_NAME = Path(".atlite.config.yaml")
    _FILE_SEARCH_PATH = Path.home().joinpath(_FILE_NAME)
    _DEFAULT_FILE_NAME = Path("config.default.yaml")
    _DEFAULT_SEARCH_PATH = Path(pkg_resources.resource_filename("atlite", str(_DEFAULT_FILE_NAME)))

    # Functions called on config change
    _UPDATE_HOOKS = []

    @classmethod
    def from_dict(cls, config_dict):
        return cls(config_dict=config_dict)

    @classmethod
    def from_file(cls, config_path):
        return cls(config_path=config_path)

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

        for var in self._VARIABLES:
            setattr(self, f"_{var}", None)

        # Prefer config explicitly provided as a dict
        if isinstance(config_dict, dict):
            logger.info(f"Loading config from dict.")
            config_dict.setdefault("config_path", self._FILE_NAME)
            self.update(config_dict)
        elif config_path is not None:
            # Provide a fully specified path for reading the config from
            logger.info(f"Loading config from {config_path}.")
            self.read(config_path)
        else:
            # Try to load configuration from standard paths
            for path in [self._FILE_SEARCH_PATH, self._DEFAULT_SEARCH_PATH]:
                if path.is_file():
                    logger.info(f"Loading config from {path}.")
                    self.read(path)
                    # Either successfully read or invalid file
                    break

        self._UPDATE_HOOKS = []

    def read(self, path):
        """Read and set the configuration based on the file in 'path'."""

        path = Path(path)
        if not path.is_file():
            raise TypeError(f"Invalid configuration file path {path}.")

        with open(path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            config_dict.setdefault("config_path", path)
            self.update(config_dict)

        logger.info(f"Configuration from {path} successfully read.")

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

        if path.exists() and not overwrite:
            raise FileExistsError(f"File {path} already exists. "
                                  "Overwrite with save(path, overwrite=True).")

        # New path now points to the current config
        self.config_path = str(path)

        # Construct attribute dict, make sure to only store string representations
        config = {var: str(getattr(self, f"_{var}")) for var in self._VARIABLES}

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
            Dictionary with pairs of key value pairs:
            <config var. name>: <new value>
        **kwargs
            Any existing or new config variable and its new value
        """

        if config_dict is None:
            config_dict = dict()
        updates = kwargs
        updates.update(config_dict)

        for key, val in updates.items():
            setattr(self, f"_{key}", val)

        for func in self._UPDATE_HOOKS:
            func(self)

    def register_update_hook(self, func):
        self._UPDATE_HOOKS.append(func)

    def resolve_filepath(self, path):
        """
        Construct the absolute file path from the provided 'path' as per the
        packages convention.

        Paths which are already absolute are returned unchanged. Relative paths
        are converted into absolute paths. The convention for relative paths
        is: They are considered relative to the current 'config.config_path'.
        If the 'config_path' is not defined, just return the relative path.

        Returns
        -------
        path : pathlib.Path
            A Path object pointing to the resolved location.
        """

        if path is None:
            return None

        path = Path(path)

        if path.is_absolute():
            return path
        elif path.parts[0] == '<ATLITE>':
            return Path(pkg_resources.resource_filename("atlite", str(Path(*path.parts[1:]))))
        else:
            return Path(self._config_path).parent.joinpath(path)

def ensure_config(config):
    if isinstance(config, Config):
        return config
    elif isinstance(config, (str, Path)):
        return Config.from_file(config)
    elif isinstance(config, dict):
        return Config.from_dict(config)
    else:
        return Config()

fallback_config = Config()
