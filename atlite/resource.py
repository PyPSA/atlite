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

import os
import yaml
from operator import itemgetter
import numpy as np
from scipy.signal import fftconvolve
from pathlib import Path
import requests
import pandas as pd
import json
import re


import logging
logger = logging.getLogger(name=__name__)

from . import config
from .utils import construct_filepath, arrowdict


def get_windturbineconfig(turbine):
    """Load the wind 'turbine' configuration.

    The configuration can either be one from local storage, then 'turbine' is
    considered part of the file base name '<turbine>.yaml' in config.windturbine_dir.
    Alternatively the configuration can be downloaded from the Open Energy Database (OEDB),
    in which case 'turbine' is a dictionary used for selecting a turbine from the database.

    Parameter
    ---------
    turbine : str|dict
        Name of the local turbine file.
        Alternatively a dict for selecting a turbine from the Open Energy
        Database, in this case the key 'source' should be contained. For all
        other key arguments to retrieve the matching turbine, see
        atlite.resource.download_windturbineconfig() for details.
    """

    turbineconf = None

    if isinstance(turbine, str):
        if turbine.startswith('oedb:'):
            turbineconf = download_windturbineconfig(turbine, store_locally=False)
        else:
            turbine = {'filename':turbine, 'source':'local'}
    if isinstance(turbine, dict):
        if turbine.get('source') is None:
            logger.warning("No key 'source':'oedb' provided with the turbine dictionary."
                           "I am assuming and adding it for now, but still nag you about it.")
            turbine['source'] = 'oedb'

        if turbine['source'] == 'oedb':
            turbineconf = download_windturbineconfig(turbine, store_locally=False)
        elif turbine['source'] == "local":
            res_name = os.path.join(config.windturbine_dir, turbine['filename']+".yaml")
            res_name = construct_filepath(res_name)

            with open(res_name, "r") as turbine_file:
                turbineconf = yaml.safe_load(turbine_file)
        else:
            raise ValueError("Not a valid 'source'.")

    if turbineconf is None:
        raise ValueError("No matching turbine configuration found.")

    V, POW, hub_height = itemgetter('V', 'POW', 'HUB_HEIGHT')(turbineconf)
    return dict(V=np.array(V), POW=np.array(POW), hub_height=hub_height, P=np.max(POW))

def get_solarpanelconfig(panel):
    """Load the 'panel'.yaml file from local disk and provide a solar panel dict."""

    res_name = os.path.join(config.solarpanel_dir, panel+".yaml")

    res_name = construct_filepath(res_name)

    with open(res_name, "r") as panel_file:
        panelconf = yaml.safe_load(panel_file)

    return panelconf

def solarpanel_rated_capacity_per_unit(panel):
    # unit is m^2 here

    if isinstance(panel, str):
        panel = get_solarpanelconfig(panel)

    model = panel.get('model', 'huld')
    if model == 'huld':
        return panel['efficiency']
    elif model == 'bofinger':
        # one unit in the capacity layout is interpreted as one panel of a
        # capacity (A + 1000 * B + log(1000) * C) * 1000W/m^2 * (k / 1000)
        A, B, C = itemgetter('A', 'B', 'C')(panel)
        return (A + B * 1000. + C * np.log(1000.))*1e3

def windturbine_rated_capacity_per_unit(turbine):
    if isinstance(turbine, str):
        turbine = get_windturbineconfig(turbine)

    return turbine['P']

def windturbine_smooth(turbine, params={}):
    '''
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
    '''

    if not isinstance(params, dict):
        params = {}

    eta = params.setdefault('eta', 0.95)
    Delta_v = params.setdefault('Delta_v', 1.27)
    sigma = params.setdefault('sigma', 2.29)

    def kernel(v_0):
        # all velocities in m/s
        return (1./np.sqrt(2*np.pi*sigma*sigma) *
                np.exp(-(v_0 - Delta_v)*(v_0 - Delta_v)/(2*sigma*sigma) ))

    def smooth(velocities, power):
        # interpolate kernel and power curve to the same, regular velocity grid
        velocities_reg = np.linspace(-50., 50., 1001)
        power_reg = np.interp(velocities_reg, velocities, power)
        kernel_reg = kernel(velocities_reg)

        # convolve power and kernel
        # the downscaling is necessary because scipy expects the velocity
        # increments to be 1., but here, they are 0.1
        convolution = 0.1*fftconvolve(power_reg, kernel_reg, mode='same')

        # sample down so power curve doesn't get too long
        velocities_new = np.linspace(0., 35., 72)
        power_new = eta * np.interp(velocities_new, velocities_reg, convolution)

        return velocities_new, power_new

    turbine = turbine.copy()
    turbine['V'], turbine['POW'] = smooth(turbine['V'], turbine['POW'])
    turbine['P'] = np.max(turbine['POW'])

    if any(turbine['POW'][np.where(turbine['V'] == 0.0)] > 1e-2):
        logger.warning(f"Oversmoothing detected with parameters {params}. "
                    "Turbine generates energy at 0 m/s wind speeds"
        )

    return turbine

def download_windturbineconfig(turbine, store_locally=True):
    """Download a windturbine configuration from the OEDB database.

    Download the configuration of a windturbine model from the OEDB database
    into the local 'windturbine_dir'.
    The OEDB database can be viewed here:
    https://openenergy-platform.org/dataedit/view/supply/turbine_library
    (2019-07-22)
    Only one turbine configuration is downloaded at a time, if the
    search parameters yield an ambigious result, no data is downloaded.

    Parameters
    ----------
    turbine : dict
        Search parameters, either provide the turbine model id (takes priority)
        or a manufacturer and turbine name, e.g. the following are identical:
        {'id':10}, {'id':10, name:'E-53/800', 'manufacturer':'Unknown'},
        {name:'E-53/800', 'manufacturer':'Enercon'}
    store_locally : bool
        (Default: True) Whether the downloaded config should be stored locally
        in config.windturbine_dir.

    Returns
    -------
    turbineconf : dict
        The turbine configuration downloaded and stored is also returned as a dict.
        Has the same format as returned by 'atlite.ressource.get_turbineconf(name)'.
    """

    ## Parse information of different allowed 'turbine' values
    parsed = False
    # Assume id
    if isinstance(turbine, int):
        turbine = {'id': turbine}
        parsed = True
    elif isinstance(turbine,str):
        s = turbine
        turbine = {}
        # Clean the string
        s = s.strip()
        s = s.replace('oedb:','')

        # 'turbine' is just a str(-inged) id
        if s.isdigit() and parsed is False:
            turbine.setdefault('id', int(s))
            parsed = True

        # 'turbine' is a name or combi of manufacturer + name
        # Matches e.g. "TurbineName", "Manu1/Manu2_Turbine/number", "Man. Turb."
        # Split on white-spaces, underscore and pipes.
        m = re.split("[\s|_]+",s, maxsplit=1)
        if m and parsed is False:
            if len(m) == 1:
                turbine.setdefault('name', m[0])
            elif len(m) == 2:
                turbine.setdefault('manufacturer', m[0])
                turbine.setdefault('name', m[1])
            parsed = True

    # Fail because we were unable to parse until here
    if parsed is False:
        logger.info(f"Unable to parse the turbine '{turbine}'.")

    ## Retrieve and cache OEDB turbine data
    OEDB_URL = 'https://openenergy-platform.org/api/v0/schema/supply/tables/turbine_library/rows'

    # Cache turbine request locally
    global _oedb_turbines

    if _oedb_turbines is None:
        try:
            # Get the turbine list
            result = requests.get(OEDB_URL)
        except:
            logger.info(f"Connection to OEDB failed.")
            raise

        # Convert JSON to dataframe for easier filtering
        # Only consider turbines with power curves available
        df = pd.DataFrame.from_dict(result.json())
        _oedb_turbines = df[df.has_power_curve]


    logger.info("Searching turbine power curve in OEDB database using " +
                ", ".join(f"{k}='{v}'" for (k,v) in turbine.items()) + ".")
    # Working copy
    df = _oedb_turbines
    if turbine.get('id'):
        df = df[df.id == int(turbine['id'])]
    if turbine.get('name'):
        df = df[df.turbine_type.str.contains(turbine['name'], case=False)]
    if turbine.get('manufacturer'):
        df = df[df.manufacturer.str.contains(turbine['manufacturer'], case=False)]


    if len(df) < 1 :
        logger.info("No turbine found.")
        return None
    elif len(df) > 1 :
        logger.info(f"Provided information corresponds to {len(df)} turbines: \n"
                    f"{df[['id','manufacturer','turbine_type']].head(3)}. \n"
                    f"Use an 'id' for an unambiguous search.")
        return None
    elif len(df) == 1:
        # Convert to series for simpliticty
        ds = df.iloc[0]

    # convert power from kW to MW
    power = np.array(json.loads(ds.power_curve_values)) / 1e3

    turbineconf = {
        "name": ds.turbine_type.strip(),
        "manufacturer": ds.manufacturer.strip(),
        "source": f"Original: {ds.source}. Via OEDB {OEDB_URL}",
        "HUB_HEIGHT": ds.hub_height,
        "V": json.loads(ds.power_curve_wind_speeds),
        "POW": power.tolist(),
    }

    # Other convinience assumptions
    if not turbineconf['HUB_HEIGHT']:
        turbineconf['HUB_HEIGHT'] = 100
        logger.warning(f"No HUB_HEIGHT defined in dataset. Manual clean-up required."
                       f"Assuming a HUB_HEIGHT of {turbineconf['HUB_HEIGHT']}m for now.")

    if ";" in str(turbineconf['HUB_HEIGHT']):
        hh = [float(t.strip()) for t in turbineconf['HUB_HEIGHT'].strip().split(";") if t]

        if len(hh) > 1:
            turbineconf['HUB_HEIGHTS'] = hh
            turbineconf['HUB_HEIGHT'] = np.mean(hh, dtype=int)
            logger.warning(f"Multiple HUB_HEIGHTS in dataset ({turbineconf['HUB_HEIGHTS']}). "
                           f"Manual clean-up is required. "
                           f"Using the average {turbineconf['HUB_HEIGHT']}m for now.")
        else:
            turbineconf['HUB_HEIGHT'] = hh[0]

    if store_locally is True:
        filename = (f"{turbineconf['manufacturer']}_{turbineconf['name']}.yaml"
                     .replace('/','_').replace(' ','_').replace('-','_'))
        filepath = construct_filepath(os.path.join(config.windturbine_dir, filename))

        with open(filepath, 'w') as turbine_file:
            yaml.dump(turbineconf, turbine_file)

        _update_resource_dictionaries()
        logger.info(f"Turbine configuration downloaded to '{filepath}'.")


    return turbineconf

# Global caches
_oedb_turbines = None
windturbines = arrowdict()
solarpanels = arrowdict()

def _update_resource_dictionaries():
    global turbines, panels

    windturbines.clear()
    windturbines.update({p.stem: p.stem
                         for p in
                         Path(construct_filepath(config.windturbine_dir)).glob("*.yaml")})

    solarpanels.clear()
    solarpanels.update({p.stem: p.stem
                        for p in Path(construct_filepath(config.solarpanel_dir)).glob("*.yaml")})

_update_resource_dictionaries()
config._update_hooks.append(_update_resource_dictionaries)
