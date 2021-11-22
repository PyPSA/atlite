# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for providing access to external ressources, like windturbine or pv panel configurations.
"""

from .utils import arrowdict
import yaml
from operator import itemgetter
import numpy as np
from scipy.signal import fftconvolve
from pathlib import Path
import requests
import pandas as pd
import json
import re
import pkg_resources

import logging

logger = logging.getLogger(name=__name__)


RESOURCE_DIRECTORY = Path(pkg_resources.resource_filename(__name__, "resources"))
WINDTURBINE_DIRECTORY = RESOURCE_DIRECTORY / "windturbine"
SOLARPANEL_DIRECTORY = RESOURCE_DIRECTORY / "solarpanel"
CSPINSTALLATION_DIRECTORY = RESOURCE_DIRECTORY / "cspinstallation"


def get_windturbineconfig(turbine):
    """Load the wind 'turbine' configuration.

    The configuration can either be one from local storage, then 'turbine' is
    considered part of the file base name '<turbine>.yaml' in config.windturbine_dir.
    Alternatively the configuration can be downloaded from the Open Energy Database (OEDB),
    in which case 'turbine' is a dictionary used for selecting a turbine from the database.

    Parameters
    ----------
    turbine : str
        Name of the local turbine file.
        Alternatively a dict for selecting a turbine from the Open Energy
        Database, in this case the key 'source' should be contained. For all
        other key arguments to retrieve the matching turbine, see
        atlite.resource.download_windturbineconfig() for details.
    """

    if isinstance(turbine, str) and turbine.startswith("oedb:"):
        return get_oedb_windturbineconfig(turbine[len("oedb:") :])

    if isinstance(turbine, str):
        if not turbine.endswith(".yaml"):
            turbine += ".yaml"

        turbine = WINDTURBINE_DIRECTORY / turbine

    with open(turbine, "r") as f:
        conf = yaml.safe_load(f)

    return dict(
        V=np.array(conf["V"]),
        POW=np.array(conf["POW"]),
        hub_height=conf["HUB_HEIGHT"],
        P=np.max(conf["POW"]),
    )


def get_solarpanelconfig(panel):
    """Load the 'panel'.yaml file from local disk and provide a solar panel dict."""

    if isinstance(panel, str):
        if not panel.endswith(".yaml"):
            panel += ".yaml"

        panel = SOLARPANEL_DIRECTORY / panel

    with open(panel, "r") as f:
        conf = yaml.safe_load(f)

    return conf


def get_cspinstallationconfig(installation):
    """Load the 'installation'.yaml file from local disk to provide the system efficiencies.

    Parameters
    ----------
    installation : str
        Name of CSP installation kind. Must correspond to name of one of the files
        in resources/cspinstallation.

    Returns
    -------
    config : dict
        Config with details on the CSP installation.
    """

    if isinstance(installation, str):
        if not installation.endswith(".yaml"):
            installation += ".yaml"

    installation = CSPINSTALLATION_DIRECTORY / installation

    # Load and set expected index columns
    with open(installation, "r") as f:
        config = yaml.safe_load(f)

    config["path"] = installation

    ## Convert efficiency dict to xr.DataArray and convert units to deg -> rad, % -> p.u.
    da = pd.DataFrame(config["efficiency"]).set_index(["altitude", "azimuth"])

    # Handle as xarray DataArray early - da will be 'return'-ed
    da = da.to_xarray()["value"]

    # Solar altitude + azimuth expected in deg for better readibility
    # calculations use solar position in rad
    # Convert da to new coordinates and drop old
    da = da.rename({"azimuth": "azimuth [deg]", "altitude": "altitude [deg]"})
    da = da.assign_coords(
        {
            "altitude": np.deg2rad(da["altitude [deg]"]),
            "azimuth": np.deg2rad(da["azimuth [deg]"]),
        }
    )
    da = da.swap_dims({"altitude [deg]": "altitude", "azimuth [deg]": "azimuth"})

    da = da.chunk("auto")

    # Efficiency unit from % to p.u.
    da /= 1.0e2

    config["efficiency"] = da

    return config


def solarpanel_rated_capacity_per_unit(panel):
    # unit is m^2 here

    if isinstance(panel, (str, Path)):
        panel = get_solarpanelconfig(panel)

    model = panel.get("model", "huld")
    if model == "huld":
        return panel["efficiency"]
    elif model == "bofinger":
        # one unit in the capacity layout is interpreted as one panel of a
        # capacity (A + 1000 * B + log(1000) * C) * 1000W/m^2 * (k / 1000)
        A, B, C = itemgetter("A", "B", "C")(panel)
        return (A + B * 1000.0 + C * np.log(1000.0)) * 1e3


def windturbine_rated_capacity_per_unit(turbine):
    if isinstance(turbine, (str, Path)):
        turbine = get_windturbineconfig(turbine)

    return turbine["P"]


def windturbine_smooth(turbine, params=None):
    """
    Smooth the powercurve in `turbine` with a gaussian kernel

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

    if params is None or params == True:
        params = {}

    eta = params.get("eta", 0.95)
    Delta_v = params.get("Delta_v", 1.27)
    sigma = params.get("sigma", 2.29)

    def kernel(v_0):
        # all velocities in m/s
        return (
            1.0
            / np.sqrt(2 * np.pi * sigma * sigma)
            * np.exp(-(v_0 - Delta_v) * (v_0 - Delta_v) / (2 * sigma * sigma))
        )

    def smooth(velocities, power):
        # interpolate kernel and power curve to the same, regular velocity grid
        velocities_reg = np.linspace(-50.0, 50.0, 1001)
        power_reg = np.interp(velocities_reg, velocities, power)
        kernel_reg = kernel(velocities_reg)

        # convolve power and kernel
        # the downscaling is necessary because scipy expects the velocity
        # increments to be 1., but here, they are 0.1
        convolution = 0.1 * fftconvolve(power_reg, kernel_reg, mode="same")

        # sample down so power curve doesn't get too long
        velocities_new = np.linspace(0.0, 35.0, 72)
        power_new = eta * np.interp(velocities_new, velocities_reg, convolution)

        return velocities_new, power_new

    turbine = turbine.copy()
    turbine["V"], turbine["POW"] = smooth(turbine["V"], turbine["POW"])
    turbine["P"] = np.max(turbine["POW"])

    if any(turbine["POW"][np.where(turbine["V"] == 0.0)] > 1e-2):
        logger.warning(
            "Oversmoothing detected with parameters eta=%f, Delta_v=%f, sigma=%f. "
            "Turbine generates energy at 0 m/s wind speeds.",
            eta,
            Delta_v,
            sigma,
        )

    return turbine


def get_oedb_windturbineconfig(search=None, **search_params):
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
        "Searching turbine power curve in OEDB database using "
        + ", ".join(f"{k}='{v}'" for (k, v) in search_params.items())
        + "."
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
    elif len(df) > 1:
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

    return turbineconf


# Global caches
_oedb_turbines = None
windturbines = arrowdict({p.stem: p for p in WINDTURBINE_DIRECTORY.glob("*.yaml")})
solarpanels = arrowdict({p.stem: p for p in SOLARPANEL_DIRECTORY.glob("*.yaml")})
cspinstallations = arrowdict(
    {p.stem: p for p in CSPINSTALLATION_DIRECTORY.glob("*.yaml")}
)
