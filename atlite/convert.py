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

from __future__ import absolute_import

import xarray as xr
import numpy as np
import os

## heat demand

def convert_heat_demand(ds, threshold = 15., a = 1., constant = 0.):
    """
    Convert outside temperature into heat demand using the degree-day
    approximation.

    Parameters
    ----------
    threshold : float
        Outside temperature in degrees Celsius above which there is no heat demand.
    a : float
        Linear factor relating heat demand to outside temperature.
    constant : float
        Constant part of heat demand that does not depend on outside temperature (e.g. due to water heating).
    """

    #Temperature is in Kelvin
    T = ds['temperature']
    threshold += 273.15
    heat_demand = a*(threshold - T)

    heat_demand.values[heat_demand.values < 0.] = 0.

    return constant + heat_demand


def heat_demand(cutout, matrix=None, index=None, **params):
    return cutout.convert_and_aggregate(convert_func=convert_heat_demand,
                                        matrix=matrix, index=index, **params)

## wind

try:
    from REatlas_client import reatlas_client
    def get_turbineconfig_from_reatlas(turbine):
        fn = os.path.join(os.path.dirname(reatlas_client.__file__), 'TurbineConfig', turbine + '.cfg')
        return reatlas_client.turbineconf_to_powercurve_object(fn)

    have_reatlas = True
except ImportError:
    have_reatlas = False

def convert_wind(ds, V, POW, hub_height):
    ds['roughness'].values[ds['roughness'].values <= 0.0] = 0.0002
    wnd_hub = ds['wnd10m'] * np.log(hub_height/ds['roughness']) / np.log((10.0)/ds['roughness'])
    wind_energy = xr.DataArray(np.interp(wnd_hub,V,POW), coords=wnd_hub.coords)
    return wind_energy

def wind(cutout, matrix=None, index=None, **params):
    if 'turbine' in params:
        assert have_reatlas, "REatlas client is necessary for loading turbine configs"

        turbine = params.pop('turbine')
        turbineconfig = get_turbineconfig_from_reatlas(turbine)
        params['V'] = turbineconfig['V']
        params['POW'] = turbineconfig['POW']
        params['hub_height'] = turbineconfig['HUB_HEIGHT']
    return cutout.convert_and_aggregate(convert_func=convert_wind,
                                        matrix=matrix, index=index, **params)
