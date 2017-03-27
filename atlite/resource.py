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

import os
from six import string_types
from operator import itemgetter

try:
    from REatlas_client import reatlas_client

    have_reatlas = True
except ImportError:
    have_reatlas = False

def get_windturbineconfig(turbine):
    assert have_reatlas, "REatlas client is necessary for loading turbine configs"

    fn = os.path.join(os.path.dirname(reatlas_client.__file__), 'TurbineConfig', turbine + '.cfg')
    turbineconf = reatlas_client.turbineconf_to_powercurve_object(fn)
    V, POW, hub_height = itemgetter('V', 'POW', 'HUB_HEIGHT')(turbineconf)
    return dict(V=V, POW=POW, hub_height=hub_height)

def get_solarpanelconfig(panel):
    assert have_reatlas, "REatlas client is necessary for loading solar panel configs"

    fn = os.path.join(os.path.dirname(reatlas_client.__file__), 'SolarPanelData', panel + '.cfg')
    return reatlas_client.solarpanelconf_to_solar_panel_config_object(fn)

def solarpanel_rated_capacity_per_unit(panel):
    # unit is m^2 here

    # one unit in the capacity layout is interpreted as one panel of a
    # capacity (A + 1000 * B + log(1000) * C) * 1000W/m^2 * (k / 1000)

    if isinstance(panel, string_types):
        panel = get_solarpanelconfig(panel)

    A, B, C = itemgetter('A', 'B', 'C')(panelconf)
    return (A + B * 1000. + C * np.log(1000.))*1e3

def windturbine_rated_capacity_per_unit(turbine):
    if isinstance(turbine, string_types):
        turbine = get_turbineconfig(panel)

    return max(turbine['POW'])
