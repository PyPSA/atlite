# -*- coding: utf-8 -*-

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
import xarray as xr
import numpy as np
import requests
import pandas as pd
import json
import yaml
import os

from . import config
from . import utils

import logging
logger = logging.getLogger(__name__)

def extrapolate_wind_speed(ds, to_height, from_height=None):
    """Extrapolate the wind speed from a given height above ground to another.

    If ds already contains a key refering to wind speeds at the desired to_height,
    no conversion is done and the wind speeds are directly returned.

    Extrapolation of the wind speed follows the logarithmic law as desribed in [1].

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the wind speed time-series at 'from_height' with key
        'wnd{height:d}m' and the surface orography with key 'roughness' at the
        geographic locations of the wind speeds.
    from_height : int
        (Optional)
        Height (m) from which the wind speeds are interpolated to 'to_height'.
        If not provided, the closest height to 'to_height' is selected.
    to_height : int|float
        Height (m) to which the wind speeds are extrapolated to.

    Returns
    -------
    da : xarray.DataArray
        DataArray containing the extrapolated wind speeds. Name of the DataArray
        is 'wnd{to_height:d}'.

    References
    ----------
    [1] Equation (2) in Andresen, G. et al (2015):
        'Validation of Danish wind time series from a new global renewable
        energy atlas for energy system analysis'.
    [2] https://en.wikipedia.org/w/index.php?title=Roughness_length&oldid=862127433,
        Retrieved 2019-02-15.
    """


    # Fast lane
    to_name   = "wnd{h:0d}m".format(h=int(to_height))
    if to_name in ds:
        return ds[to_name]

    if from_height is None:
        # Determine closest height to to_name
        heights = np.asarray([int(s[3:-1]) for s in ds if s.startswith("wnd")])

        if len(heights) == 0:
            raise AssertionError("Wind speed is not in dataset")

        from_height = heights[np.argmin(np.abs(heights-to_height))]

    from_name = "wnd{h:0d}m".format(h=int(from_height))

    # Sanitise roughness for logarithm
    # 0.0002 corresponds to open water [2]
    ds['roughness'].values[ds['roughness'].values <= 0.0] = 0.0002

    # Wind speed extrapolation
    wnd_spd = ds[from_name] * ( np.log(to_height /ds['roughness'])
                              / np.log(from_height/ds['roughness']))

    wnd_spd.attrs.update({"long name":
                            "extrapolated {ht} m wind speed using logarithmic "
                            "method with roughness and {hf} m wind speed"
                            "".format(ht=to_height, hf=from_height),
                          "units" : "m s**-1"})

    return wnd_spd.rename(to_name)


# Cache
_oedb_turbines = None

def download_turbineconf(turbine, store_locally=True):
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
    
    OEDB_URL = 'https://openenergy-platform.org/api/v0/schema/supply/tables/turbine_library/rows'

    # Cache turbine request locally
    global _oedb_turbines

    if _oedb_turbines is None:
        try: 
            # Get the turbine list
            result = requests.get(OEDB_URL)
        except requests.exceptions.RequestException as e:
            logger.info(f"Connection to OEDB failed with:\n\n{str(e)}")
            return None

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
        df = df[df.name == turbine['name']]
    if turbine.get('manufacturer'):
        df = df[df.manufacturer == turbine['manufacturer']]


    if len(df) < 1 :
        logger.info("No turbine found.")
        return None
    elif len(df) > 1 :
        logger.info(f"Provided information corresponds to {len(df)} turbines: \n"
                    f"{df[['id','name','manufacturer']].head(3)}. \n"
                    f"Use an 'id' for an unambiguous search.")
        return None
    elif len(df) == 1:
        # Convert to series for simpliticty
        ds = df.iloc[0]

    # convert power from kW to MW
    power = np.array(json.loads(ds.power_curve_values)) / 1e3

    turbineconf = {
        "name": ds['name'].strip(),
        "manufacturer": ds.manufacturer.strip(),
        "source": f"Original: {ds.source}. Via OEDB {OEDB_URL}",
        "HUB_HEIGHT": ds.hub_height.strip(),
        "V": json.loads(ds.power_curve_wind_speeds),
        "POW": power.tolist(),
    }

    if ";" in turbineconf['HUB_HEIGHT']:
        h = np.mean([float(t.strip()) for t in turbineconf['HUB_HEIGHT'].split(";")], dtype=int)

        turbineconf['HUB_HEIGHTS'] = turbineconf['HUB_HEIGHT']
        turbineconf['HUB_HEIGHT'] = h

        logger.warning(f"Multiple HUB_HEIGHTS in dataset ({turbineconf['HUB_HEIGHTS']}). "
                       f"Manual clean-up is required. "
                       f"Using the average {turbineconf['HUB_HEIGHT']}m for now.")


    if store_locally is True:
        filename = (f"{turbineconf['manufacturer']}_{turbineconf['name']}.yaml"
                     .replace('/','_').replace(' ','_'))
        filepath = utils.construct_filepath(os.path.join(config.windturbine_dir, filename))

        with open(filepath, 'w') as turbine_file:
            yaml.dump(turbineconf, turbine_file)
            
        logger.info(f"Turbine configuration downloaded to '{filepath}'.")


    return turbineconf