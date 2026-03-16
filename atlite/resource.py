# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module for providing access to external ressources, like windturbine or pv
panel configurations.
"""

from __future__ import annotations

import json
import logging
import re
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import requests
import yaml
from dask.array import radians
from scipy.signal import fftconvolve

from atlite.utils import arrowdict

logger = logging.getLogger(name=__name__)

RESOURCE_DIRECTORY = Path(__file__).parent / "resources"
WINDTURBINE_DIRECTORY = RESOURCE_DIRECTORY / "windturbine"
SOLARPANEL_DIRECTORY = RESOURCE_DIRECTORY / "solarpanel"
CSPINSTALLATION_DIRECTORY = RESOURCE_DIRECTORY / "cspinstallation"

if TYPE_CHECKING:
    from typing import TypedDict

    from typing_extensions import NotRequired

    from atlite._types import DataArray, NDArray, PathLike

    class TurbineConfig(TypedDict):
        V: NDArray
        POW: NDArray
        P: float
        hub_height: float | int
        name: NotRequired[str]
        manufacturer: NotRequired[str]
        source: NotRequired[str]

    class PanelConfig(TypedDict):
        model: NotRequired[Literal["huld", "bofinger"]]
        efficiency: NotRequired[float]
        A: NotRequired[float]
        B: NotRequired[float]
        C: NotRequired[float]
        name: NotRequired[str]
        source: NotRequired[str]

    class CSPConfig(TypedDict):
        efficiency: DataArray
        path: PathLike
        name: NotRequired[str]
        source: NotRequired[str]


def get_windturbineconfig(
    turbine: str | PathLike | dict[str, Any], add_cutout_windspeed: bool = True
) -> TurbineConfig:
    """
    Load the wind 'turbine' configuration.

    Parameters
    ----------
    turbine : str or pathlib.Path or dict
        if str:
            The name of a preshipped turbine from alite.resources.windturbine .
            Alternatively, if a str starting with 'oedb:<name>' is passed the Open
            Energy Database is searched for a turbine with the matching '<name>'
            and if found that turbine configuration is used. See
            `atlite.resource.get_oedb_windturbineconfig(...)`
        if `pathlib.Path` is provided the configuration is read from this local
            path instead
        if dict:
            a user provided config dict. Needs to have the keys "POW", "V", "P", and
            "hub_height". Values for "POW" and "V" need to be list or np.ndarray with
            equal length.
    add_cutout_windspeed : bool = True
        If True and in case the power curve does not end with a zero, will add zero power
        output at the highest wind speed in the power curve. If False, a warning will be
        raised if the power curve does not have a cut-out wind speed.

    Returns
    -------
    config : dict
        Config with details on the turbine

    """
    if not isinstance(turbine, (str, Path, dict)):
        raise KeyError(
            f"`turbine` must be a str, pathlib.Path or dict, but is {type(turbine)}."
        )

    if isinstance(turbine, str) and turbine.startswith("oedb:"):
        conf = cast(
            "dict[str, Any]", get_oedb_windturbineconfig(turbine[len("oedb:") :])
        )

    elif isinstance(turbine, (str, Path)):
        if isinstance(turbine, str):
            turbine_path = windturbines[turbine.replace(".yaml", "")]

        elif isinstance(turbine, Path):
            turbine_path = turbine

        with Path(turbine_path).open() as f:
            conf = yaml.safe_load(f)
            conf = {
                "V": np.array(conf["V"]),
                "POW": np.array(conf["POW"]),
                "hub_height": conf["HUB_HEIGHT"],
                "P": np.max(conf["POW"]),
            }

    elif isinstance(turbine, dict):
        conf = turbine

    return _validate_turbine_config_dict(
        cast("dict[str, Any]", conf), add_cutout_windspeed
    )


def get_solarpanelconfig(panel: str | PathLike) -> PanelConfig:
    """
    Load the 'panel'.yaml file from local disk and provide a solar panel dict.

    Parameters
    ----------
    panel : str or pathlib.Path
        if str is provided the name of a preshipped panel
            from alite.resources.solarpanel is expected.
        if `pathlib.Path` is provided the configuration
            is read from this local path instead

    Returns
    -------
    config : dict
        Config with details on the solarpanel

    """
    assert isinstance(panel, (str, Path))

    if isinstance(panel, str):
        panel_path = solarpanels[panel.replace(".yaml", "")]

    elif isinstance(panel, Path):
        panel_path = panel

    with Path(panel_path).open() as f:
        return cast("PanelConfig", yaml.safe_load(f))


def get_cspinstallationconfig(installation: str | PathLike) -> CSPConfig:
    """
    Load the 'installation'.yaml file from local disk to provide the system
    efficiencies.

    Parameters
    ----------
    installation : str or pathlib.Path
        if str is provided the name of a preshipped CSP installation
            from alite.resources.cspinstallation is expected.
        if `pathlib.Path` is provided the configuration
            is read from this local path instead

    Returns
    -------
    config : dict
        Config with details on the CSP installation.

    """
    assert isinstance(installation, (str, Path))

    if isinstance(installation, str):
        installation_path = cspinstallations[installation.replace(".yaml", "")]

    elif isinstance(installation, Path):
        installation_path = installation

    with Path(installation_path).open() as f:
        config = cast("dict[str, Any]", yaml.safe_load(f))
    config["path"] = installation_path

    da = pd.DataFrame(config["efficiency"]).set_index(["altitude", "azimuth"])

    da = da.to_xarray()["value"]

    da = da.rename({"azimuth": "azimuth [deg]", "altitude": "altitude [deg]"})
    da = da.assign_coords(
        {
            "altitude": radians(da["altitude [deg]"]),
            "azimuth": radians(da["azimuth [deg]"]),
        }
    )
    da = da.swap_dims({"altitude [deg]": "altitude", "azimuth [deg]": "azimuth"})

    da = da.chunk("auto")

    da /= 1.0e2

    config["efficiency"] = da

    return cast("CSPConfig", config)


def solarpanel_rated_capacity_per_unit(panel: str | PathLike | PanelConfig) -> float:
    """
    Return the rated capacity per unit of a solar panel configuration.

    Parameters
    ----------
    panel : str or pathlib.Path or dict
        Solar panel configuration or reference to one.

    Returns
    -------
    float
        Rated capacity per unit area or per panel, depending on the model.
    """
    if isinstance(panel, (str, Path)):
        panel = get_solarpanelconfig(panel)

    model = panel.get("model", "huld")
    if model == "huld":
        return cast("float", panel["efficiency"])
    if model == "bofinger":
        A, B, C = itemgetter("A", "B", "C")(panel)
        return cast("float", (A + B * 1000.0 + C * np.log(1000.0)) * 1e3)
    raise ValueError(f"Unknown panel model: {model}")


def windturbine_rated_capacity_per_unit(
    turbine: str | PathLike | TurbineConfig,
) -> float:
    """
    Return the rated capacity of a wind turbine configuration.

    Parameters
    ----------
    turbine : str or pathlib.Path or dict
        Wind turbine configuration or reference to one.

    Returns
    -------
    float
        Rated turbine capacity.
    """
    if isinstance(turbine, (str, Path)):
        turbine = get_windturbineconfig(turbine)

    return turbine["P"]


def windturbine_smooth(
    turbine: TurbineConfig, params: dict[str, float] | None | bool = None
) -> TurbineConfig:
    """
    Smooth the powercurve in `turbine` with a gaussian kernel.

    Parameters
    ----------
    turbine : dict
        Turbine config with at least V and POW
    params : dict
        Allows adjusting fleet availability eta, mean Delta_v and
        stdev sigma. Defaults to values from Andresen's paper: 0.95,
        1.27 and 2.29, respectively.

    Returns
    -------
    turbine : dict
        Turbine config with a smoothed power curve

    References
    ----------
    G. B. Andresen, A. A. Søndergaard, M. Greiner, Validation of
    Danish wind time series from a new global renewable energy atlas
    for energy system analysis, Energy 93, Part 1 (2015) 1074–1088.

    """
    if params is None or params is True:
        params = {}

    params = cast("dict[str, float]", params)
    eta: float = params.get("eta", 0.95)
    Delta_v: float = params.get("Delta_v", 1.27)
    sigma: float = params.get("sigma", 2.29)

    def kernel(v_0: NDArray) -> NDArray:
        return (  # type: ignore[no-any-return]
            1.0
            / np.sqrt(2 * np.pi * sigma * sigma)
            * np.exp(-(v_0 - Delta_v) * (v_0 - Delta_v) / (2 * sigma * sigma))
        )

    def smooth(velocities: NDArray, power: NDArray) -> tuple[NDArray, NDArray]:
        velocities_reg = np.linspace(-50.0, 50.0, 1001)
        power_reg = np.interp(velocities_reg, velocities, power)
        kernel_reg = kernel(velocities_reg)

        convolution = 0.1 * fftconvolve(power_reg, kernel_reg, mode="same")

        velocities_new = np.linspace(0.0, 35.0, 72)
        power_new = eta * np.interp(velocities_new, velocities_reg, convolution)

        return velocities_new, power_new

    turbine = turbine.copy()
    turbine["V"], turbine["POW"] = smooth(turbine["V"], turbine["POW"])
    turbine["P"] = cast("float", float(np.max(turbine["POW"])))

    if any(turbine["POW"][np.where(turbine["V"] == 0.0)] > 1e-2):
        logger.warning(
            "Oversmoothing detected with parameters eta=%f, Delta_v=%f, sigma=%f. "
            "Turbine generates energy at 0 m/s wind speeds.",
            eta,
            Delta_v,
            sigma,
        )

    return turbine


def _max_v_is_zero_pow(turbine: TurbineConfig) -> bool:
    return cast(
        "bool", bool(np.any(turbine["POW"][turbine["V"] == turbine["V"].max()] == 0))
    )


def _validate_turbine_config_dict(
    turbine: dict[str, Any], add_cutout_windspeed: bool
) -> TurbineConfig:
    """
    Checks the turbine config dict format and power curve.

    Parameters
    ----------
    turbine : dict
        turbine configuration dict. Needs the keys "POW", "V", "P", and "hub_height".
        Values for "V" and "POW" need to be list or np.ndarray.
    add_cutout_windspeed : bool
        If True and in case the power curve does not end with a zero, will add zero power
        output at the highest wind speed in the power curve. If False, a warning will be
        raised if the power curve does not have a cut-out wind speed.

    Returns
    -------
    dict
        validated and potentially modified turbine config dict

    """
    if not all(key in turbine for key in ("POW", "V", "P", "hub_height")):
        err_msg = (
            "turbine config dict needs at least the following keys: ['POW', 'V', 'P', "
            f"'hub_height']\nbut are currently: {list(turbine.keys())}"
        )
        raise ValueError(err_msg)

    if not all(isinstance(turbine[p], (np.ndarray, list)) for p in ("POW", "V")):
        err_msg = "turbine entries 'POW' and 'V' must be np.ndarray or list"
        raise ValueError(err_msg)

    if any(isinstance(turbine[p], list) for p in ("POW", "V")):
        turbine["V"] = np.array(turbine["V"])
        turbine["POW"] = np.array(turbine["POW"])

    if len(turbine["POW"]) != len(turbine["V"]):
        err_msg = "turbine wind speed and power arrays do not have equal length."
        raise ValueError(err_msg)

    if not np.all(np.diff(turbine["V"]) >= 0):
        err_msg = (
            "wind speed 'V' in the turbine config dict is expected to be increasing, "
            f"but is currently not in ascending order:\n{turbine['V']}"
        )
        raise ValueError(err_msg)

    if add_cutout_windspeed is True and not _max_v_is_zero_pow(
        cast("TurbineConfig", turbine)
    ):
        turbine["V"] = np.pad(turbine["V"], (0, 1), "maximum")
        turbine["POW"] = np.pad(turbine["POW"], (0, 1), "constant", constant_values=0)
        logger.info(
            "adding a cut-out wind speed to the turbine power curve at V=%s m/s.",
            turbine["V"][-1],
        )

    if not _max_v_is_zero_pow(cast("TurbineConfig", turbine)):
        logger.warning(
            "The power curve does not have a cut-out wind speed, i.e. the power"
            " output corresponding to the\nhighest wind speed is not zero. You can"
            " either change the power curve manually or set\n"
            "'add_cutout_windspeed=True' in the Cutout.wind conversion method."
        )
    return cast("TurbineConfig", turbine)


def get_oedb_windturbineconfig(
    search: int | str | None = None, **search_params: Any
) -> TurbineConfig:
    """
    Download a windturbine configuration from the OEDB database.

    Download the configuration of a windturbine model from the OEDB database
    into the local 'windturbine_dir'.
    The OEDB database can be viewed here:
    https://openenergy-platform.org/dataedit/view/supply/wind_turbine_library
    (2019-07-22)
    Only one turbine configuration is downloaded at a time, if the
    search parameters yield an ambigious result, no data is downloaded.

    Parameters
    ----------
    search : int|str
        Smart search parameter, if int use as model id, if str look in name or turbine_type
    **search_params : dict
        Recognized arguments are 'id', 'name', 'turbine_type' and 'manufacturer'

    Returns
    -------
    turbineconfig : dict
        The turbine configuration in the format from 'atlite.ressource.get_turbineconf(name)'.

    Example
    -------
    >>> get_oedb_windturbineconfig(10)
    {'V': ..., 'POW': ..., ...}

    >>> get_oedb_windturbineconfig(name="E-53/800", manufacturer="Enercon")
    {'V': ..., 'POW': ..., ...}

    """
    # Parse information of different allowed 'turbine' values
    if isinstance(search, int):
        search_params.setdefault("id", search)
        search = None

    # Retrieve and cache OEDB turbine data
    OEDB_URL = "https://openenergy-platform.org/api/v0/schema/supply/tables/wind_turbine_library/rows"

    # Cache turbine request locally
    global _oedb_turbines

    if _oedb_turbines is None:
        # Get the turbine list
        result = requests.get(OEDB_URL)

        # Convert JSON to dataframe for easier filtering
        # Only consider turbines with power curves available
        df = pd.DataFrame.from_dict(result.json())
        _oedb_turbines = df[df.has_power_curve]

    logger.info(
        "Searching turbine power curve in OEDB database using %s.",
        ", ".join(f"{k}='{v}'" for (k, v) in search_params.items()),
    )

    # Working copy
    df = _oedb_turbines
    selector = True
    if search is not None:
        selector &= df.name.str.contains(
            search, case=False
        ) | df.turbine_type.str.contains(search, case=False)
    if "id" in search_params:
        selector &= df.id == int(search_params["id"])
    if "name" in search_params:
        selector &= df.name.str.contains(search_params["name"], case=False)
    if "turbine_type" in search_params:
        selector &= df.turbine_type.str.contains(search_params["name"], case=False)
    if "manufacturer" in search_params:
        selector &= df.manufacturer.str.contains(
            search_params["manufacturer"], case=False
        )

    df = df.loc[selector]

    if len(df) < 1:
        raise RuntimeError("No turbine found.")
    if len(df) > 1:
        raise RuntimeError(
            f"Provided information corresponds to {len(df)} turbines,"
            " use `id` for an unambiguous search.\n"
            + str(df[["id", "manufacturer", "turbine_type"]])
        )

    # Convert to series for simpliticty
    ds = df.iloc[0]

    # convert power from kW to MW
    power = np.array(json.loads(ds.power_curve_values)) / 1e3

    hub_height = ds.hub_height

    if not hub_height:
        hub_height = 100
        logger.warning(
            "No hub_height defined in dataset. Manual clean-up required."
            "Assuming a hub_height of 100m for now."
        )
    elif isinstance(hub_height, str):
        hub_heights = [float(t) for t in re.split(r"\s*;\s*", hub_height.strip()) if t]

        if len(hub_heights) > 1:
            hub_height = np.mean(hub_heights, dtype=int)
            logger.warning(
                "Multiple values for hub_height in dataset (%s). "
                "Manual clean-up required. Using the averge %dm for now.",
                hub_heights,
                hub_height,
            )
        else:
            hub_height = hub_heights[0]

    turbineconf = {
        "name": ds.turbine_type.strip(),
        "manufacturer": ds.manufacturer.strip(),
        "source": f"Original: {ds.source}. Via OEDB {OEDB_URL}",
        "hub_height": hub_height,
        "V": np.array(json.loads(ds.power_curve_wind_speeds)),
        "POW": power,
        "P": power.max(),
    }

    # Cache in windturbines
    global windturbines
    charmap = str.maketrans("/- ", "___")
    name = "{manufacturer}_{name}".format(**turbineconf).translate(charmap)
    windturbines[name] = turbineconf

    return turbineconf  # type: ignore[return-value]


# Global caches
_oedb_turbines = None
windturbines = arrowdict({p.stem: p for p in WINDTURBINE_DIRECTORY.glob("*.yaml")})
solarpanels = arrowdict({p.stem: p for p in SOLARPANEL_DIRECTORY.glob("*.yaml")})
cspinstallations = arrowdict(
    {p.stem: p for p in CSPINSTALLATION_DIRECTORY.glob("*.yaml")}
)
