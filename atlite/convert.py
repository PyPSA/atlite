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
import scipy as sp, scipy.sparse

from .aggregate import aggregate_sum, aggregate_matrix
from .shapes import spdiag, compute_indicatormatrix

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

def heat_demand(cutout, **params):
    return cutout.convert_and_aggregate(convert_func=convert_heat_demand, **params)

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

def wind(cutout, **params):
    if 'turbine' in params:
        assert have_reatlas, "REatlas client is necessary for loading turbine configs"

        turbine = params.pop('turbine')
        turbineconfig = get_turbineconfig_from_reatlas(turbine)
        params['V'] = turbineconfig['V']
        params['POW'] = turbineconfig['POW']
        params['hub_height'] = turbineconfig['HUB_HEIGHT']

    return cutout.convert_and_aggregate(convert_func=convert_wind, **params)

## hydro

def convert_runoff(ds):
    runoff = ds['runoff'] * ds['height']
    return runoff

def runoff(cutout, **params):
    return cutout.convert_and_aggregate(convert_func=convert_runoff, **params)
