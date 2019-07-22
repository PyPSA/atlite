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

def download_turbineconf(requested_turbine=None):
    """Download a windturbine configuration from the OEDB database.
    
    Download the configuration of a windturbine model from the OEDB database
    into the local 'windturbine_dir'.
    The OEDB database can be viewed here:
    https://openenergy-platform.org/dataedit/view/supply/turbine_library
    (2019-07-22)
    Only one turbine configuration is downloaded at a time, if the 
    search parameters yield an ambigious result, not data is downloaded.
    
    Parameters
    ----------
    requested_turbine : dict
        Search parameters, either provide the turbine model id (takes priority)
        or a manufacturer and turbine name, e.g. the following are identical:
        {'id':10}, {'id':10, 'manufacturer':'Unknown', name:'E-53/800'},
        {'manufacturer':'Enercon', name:'E-53/800'}
        
    Returns
    -------
    turbineconf : dict
        The turbine configuration downloaded and stored is also returned as a dict.
        Has the same format as returned by 'atlite.ressource.get_turbineconf(name)'.
    """
    
    # Get the turbine list
    oedb_url = 'https://openenergy-platform.org/api/v0/schema/supply/tables/turbine_library/rows'
    result = requests.get(oedb_url)

    # Convert JSON to dataframe for easier filtering
    # Only consider turbines with power curves available
    df = pd.DataFrame.from_dict(result.json())
    df = df[df.has_power_curve]

    if requested_turbine.get('id'):
        logger.info("Searching turbine power curve in OEDB database using id "
                    "'{id}'".format(id=requested_turbine['id'])
                   )
        ds = df[df.id == requested_turbine['id']]
        
    else:
        if not isinstance(requested_turbine.get('name'),str) or
           not isinstance(requested_turbine.get('manufacturer'),str):
            
            logger.error("'name' and 'manufacturer' must be provided. "
                         "Alternatively provide a turbine id from OEDB.")
            return None
        logger.info("Searching turbine power curve in OEDB database using "
                    "manufacturer '{m}' and name '{n}'."
                    "".format(m=requested_turbine['manufacturer'],
                              n=requested_turbine['name']))
            
        ds = df[df.name == requested_turbine['name']]
        ds = ds[ds.manufacturer == requested_turbine['manufacturer']]

    if len(ds) < 1 :
        logger.info("No turbine found for {rt}.".format(rt=requested_turbine))
    elif len(ds) > 2 :
        logger.info("Provided information corresponds to more than one turbine. "
                    "Provide id to unambigious lookup. {rt}."
                    "".format(rt=requested_turbine))
    elif len(ds) == 1:
        # Convert to series for simpliticty
        ds = ds.iloc[0]

    # convert power from kW to MW
    power = np.array(json.loads(ds.power_curve_values)) / 1e3

    turbineconf = {
        "name": ds['name'].strip(),
        "manufacturer": ds.manufacturer.strip(),
        "source": "Original: {origin}. "
                  "Via OEDB {secondary}".format(origin=ds.source, secondary=oedb_url),
        "HUB_HEIGHT": ds.hub_height.strip(),
        "V": json.loads(ds.power_curve_wind_speeds),
        "POW": power.tolist(),
    }

    filename = "{m}_{n}.yaml".format(m=turbineconf['manufacturer'],
                                     n=turbineconf['name'])
    filename = filename.replace("/","_")
    filename = filename.replace(" ","_")
    filepath = utils.construct_filepath(os.path.join(config.windturbine_dir, filename))

    with open(filepath, 'w') as turbine_file:
        yaml.dump(turbineconf, turbine_file)
        
    logger.info("Turbine configuration downloaded to '{fp}'.".format(fp=filepath))
    if ";" in turbineconf['HUB_HEIGHT']:
        logger.info("Multiple HUB_HEIGHTS in dataset. Manual clean-up required.")

    return turbineconf