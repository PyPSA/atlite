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
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import scipy as sp, scipy.sparse

from .aggregate import aggregate_sum, aggregate_matrix
from .shapes import spdiag, compute_indicatormatrix

from .pv.solar_position import SolarPosition
from .pv.irradiation import TiltedTotalIrradiation
from .pv.solar_panel_model import SolarPanelModel

def convert_and_aggregate(cutout, convert_func, matrix=None,
                          index=None, layout=None,
                          shapes=None, shapes_proj='latlong',
                          **convert_kwds):
    assert cutout.prepared, "The cutout has to be prepared first."

    if shapes is not None:
        if isinstance(shapes, pd.Series) and index is None:
            index = shapes.index

        matrix = cutout.indicatormatrix(shapes, shapes_proj)

    if matrix is not None:
        matrix = sp.sparse.csr_matrix(matrix)

        if layout is not None:
            matrix = matrix.dot(spdiag(layout.ravel()))

        if index is None:
            index = pd.RangeIndex(matrix.shape[0])
        aggregate_func = aggregate_matrix
        aggregate_kwds = dict(matrix=matrix, index=index)
    else:
        aggregate_func = aggregate_sum
        aggregate_kwds = {}

    results = []

    yearmonths = cutout.coords['year-month'].to_index()

    for ym in yearmonths:
        with xr.open_dataset(cutout.datasetfn(ym)) as ds:
            da = convert_func(ds, **convert_kwds).load()
        results.append(aggregate_func(da, **aggregate_kwds))
    if 'time' in results[0]:
        results = xr.concat(results, dim='time')
    else:
        results = sum(results)
    return results


## temperature


def convert_temperature(ds):
    """Return outside temperature (useful for e.g. heat pump T-dependent
    coefficient of performance).
    """

    #Temperature is in Kelvin
    return ds['temperature'] - 273.15


def temperature(cutout, **params):
    return cutout.convert_and_aggregate(convert_func=convert_temperature, **params)



## heat demand

def convert_heat_demand(ds, threshold=15., a=1., constant=0., hour_shift=0.):
    """
    Convert outside temperature into daily heat demand using the
    degree-day approximation.

    Since "daily average temperature" means different things in
    different time zones and since xarray coordinates do not handle
    time zones gracefully like pd.DateTimeIndex, you can provide an
    hour_shift to redefine when the day starts.

    E.g. for Moscow in winter, hour_shift = 4, for New York in winter,
    hour_shift = -5

    This time shift applies across the entire spatial scope of ds for
    all times. More fine-grained control will be built in a some
    point, i.e. space- and time-dependent time zones.

    WARNING: Because the original data is provided every month, at the
    month boundaries there is untidiness if you use a time shift. The
    resulting xarray will have duplicates in the index for the parts
    of the day in each month at the boundary. You will have to
    re-average these based on the number of hours in each month for
    the duplicated day.

    Parameters
    ----------
    threshold : float
        Outside temperature in degrees Celsius above which there is no
        heat demand.
    a : float
        Linear factor relating heat demand to outside temperature.
    constant : float
        Constant part of heat demand that does not depend on outside
        temperature (e.g. due to water heating).
    hour_shift : float
        Time shift relative to UTC for taking daily average
    """

    #Temperature is in Kelvin; take daily average
    T = ds['temperature']
    T.coords['time'].values += np.timedelta64(dt.timedelta(hours=hour_shift))

    T = ds['temperature'].resample("D", dim="time", how="mean")
    threshold += 273.15
    heat_demand = a*(threshold - T)

    heat_demand.values[heat_demand.values < 0.] = 0.

    return constant + heat_demand

def heat_demand(cutout, **params):
    return cutout.convert_and_aggregate(convert_func=convert_heat_demand, **params)


## solar thermal collectors

def convert_solar_thermal(ds, c0=0.8, c1=3., t_store=80., angle=45.):
    """
    Convert downward short-wave radiation flux and outside temperature
    into time series for solar thermal collectors.

    Mathematical model and defaults for c0, c1 based on model in
    Henning and Palzer, Renewable and Sustainable Energy Reviews 30
    (2014) 1003-1018

    WARNING: Angles with Earth's surface are not yet implemented.

    Parameters
    ----------
    c0 : float
        Optical efficiency
    c1 : float
        Heat loss coefficient (units of W/(m^2 K))
    t_store : float
        Storage temperature (units of degrees Celsius)
    angle : float
        Placeholder for angle with horizontal facing south
    """

    # convert storage temperature to Kelvin in line with reanalysis data
    t_store += 273.15

    # Downward shortwave radiation flux is in W/m^2
    # http://rda.ucar.edu/datasets/ds094.0/#metadata/detailed.html?_do=y
    influx = ds['influx']

    # overall efficiency
    eta = c0 - c1*((t_store - ds['temperature'])/influx)

    return (influx*eta).where(influx > 0.).fillna(0.)


def solar_thermal(cutout, **params):
    return cutout.convert_and_aggregate(convert_func=convert_solar_thermal, **params)


## turbine and panel data can be read in from reatlas

try:
    from REatlas_client import reatlas_client
    def get_turbineconfig_from_reatlas(turbine):
        fn = os.path.join(os.path.dirname(reatlas_client.__file__), 'TurbineConfig', turbine + '.cfg')
        return reatlas_client.turbineconf_to_powercurve_object(fn)

    def get_solarpanelconfig_from_reatlas(panel):
        fn = os.path.join(os.path.dirname(reatlas_client.__file__), 'SolarPanelData', panel + '.cfg')
        return reatlas_client.solarpanelconf_to_solar_panel_config_object(fn)

    have_reatlas = True
except ImportError:
    have_reatlas = False

## wind

def convert_wind(ds, V, POW, hub_height):
    ds['roughness'].values[ds['roughness'].values <= 0.0] = 0.0002
    wnd_hub = ds['wnd10m'] * (np.log(hub_height/ds['roughness']) / np.log((10.0)/ds['roughness']))
    wind_energy = xr.DataArray(np.interp(wnd_hub,V,POW), coords=wnd_hub.coords)
    return wind_energy

def wind(cutout, **params):
    if 'turbine' in params:
        assert have_reatlas, "REatlas client is necessary for loading turbine configs"

        turbine = params.pop('turbine')
        turbineconfig = get_turbineconfig_from_reatlas(turbine)
        params['V'] = turbineconfig['V']
        params['POW'] = turbineconfig['POW']
        params['hub_height'] = turbineconfig['HUB_HEIGHT']

    return cutout.convert_and_aggregate(convert_func=convert_wind, **params)


## solar PV

def convert_pv(ds, settings, panelconfig):
    solar_position = SolarPosition(ds, settings)
    irradiation = TiltedTotalIrradiation(ds, solar_position, settings)
    solar_panel = SolarPanelModel(ds, irradiation, panelconfig)
    ac_power = solar_panel['AC power']
    return ac_power

def pv(cutout, **params):
    '''
    Example for the <settings> keyword argument:

    settings = {'panel' : 'Scheuten215IG',
                'surface slope' : 0.0,
                'surface azimuth' : 0.0,
                'simulate REatlas' : True,
                'clearsky model' : 'Simple'}

    Simulate REatlas option:
        True : Simulates the REatlas routines, clearsky model is ignored
        False : Executes debugged code, make sure to specify a clearsky model

    Reindl clearsky model options for diffuse irradiation component:
        'Simple'   - clearness of sky index required
        'Enhanced' - clearness of sky index, ambient air temperature, relative humidity required
    '''

    if 'settings' in params:
        assert have_reatlas, "REatlas client is necessary for loading solar panel configs"

        settings = params.pop('settings')
        panelconfig = get_solarpanelconfig_from_reatlas(settings['panel'])
        params['settings'] = settings
        params['panelconfig'] = panelconfig
    return cutout.convert_and_aggregate(convert_func=convert_pv, **params)

## hydro

def convert_runoff(ds):
    runoff = ds['runoff'] * ds['height']
    return runoff

def runoff(cutout, **params):
    return cutout.convert_and_aggregate(convert_func=convert_runoff, **params)
