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

from __future__ import absolute_import

import os
from six import string_types
from operator import itemgetter
import numpy as np
from scipy.signal import fftconvolve

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

    A, B, C = itemgetter('A', 'B', 'C')(panel)
    return (A + B * 1000. + C * np.log(1000.))*1e3

def windturbine_rated_capacity_per_unit(turbine):
    if isinstance(turbine, string_types):
        turbine = get_windturbineconfig(turbine)

    return max(turbine['POW'])

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

    eta = params.get('eta', 0.95)
    Delta_v = params.get('Delta_v', 1.27)
    sigma = params.get('sigma', 2.29)

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

    return turbine
