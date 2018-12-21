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

import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import scipy as sp, scipy.sparse
from six import string_types
from operator import itemgetter
import progressbar as pgb

from .aggregate import aggregate_sum, aggregate_matrix
from .gis import spdiag, compute_indicatormatrix

from .pv.solar_position import SolarPosition
from .pv.irradiation import TiltedIrradiation
from .pv.solar_panel_model import SolarPanelModel
from .pv.orientation import get_orientation, SurfaceOrientation

from .resource import (get_windturbineconfig, get_solarpanelconfig,
                       windturbine_rated_capacity_per_unit,
                       solarpanel_rated_capacity_per_unit,
                       windturbine_smooth)

def convert_and_aggregate(cutout, convert_func, matrix=None,
                          index=None, layout=None, shapes=None,
                          shapes_proj='latlong', per_unit=False,
                          return_capacity=False, capacity_factor=False,
                          show_progress=True, **convert_kwds):
    """
    Convert and aggregate a weather-based renewable generation time-series.

    NOTE: Not meant to be used by the user him or herself. Rather it is a
    gateway function that is called by all the individual time-series
    generation functions like pv and wind. Thus, all its parameters are also
    available from these.

    Parameters (passed through as **params)
    ---------------------------------------
    matrix : sp.sparse.csr_matrix or None
        If given, it is used to aggregate the `grid_cells` to buses.
    index : pd.Index
        Buses
    layout : X x Y - np.array or xr.DataArray
        The capacity to be build in each of the `grid_cells`.
    shapes : list or pd.Series of shapely.geometry.Polygon
        If given, matrix is constructed as indicatormatrix of the polygons, its
        index determines the bus index on the time-series.
    shapes_proj : str or pyproj.Proj
        Defaults to 'latlong'. If different to the map projection of the
        cutout, the shapes are reprojected using pyproj.transform to match
        cutout.projection (defaults to 'latlong').
    per_unit : boolean
        Returns the time-series in per-unit units, instead of in MW (defaults
        to False).
    return_capacity : boolean
        Additionally returns the installed capacity at each bus corresponding
        to `layout` (defaults to False).
    capacity_factor : boolean
        If True, the capacity factor of the chosen resource for each grid cell
        is computed.
    show_progress : boolean|string
        Whether to show a progress bar if boolean and its label if given as a
        string (defaults to True).

    Returns
    -------
    resource : xr.DataArray
        Time-series of renewable generation aggregated to buses, if
        `matrix` or equivalents are provided else the total sum of
        generated energy.
    units : xr.DataArray (optional)
        The installed units per bus in MW corresponding to `layout`
        (only if `return_capacity` is True).

    Internal Parameters (provided by f.ex. wind and pv)
    ---------------------------------------------------
    convert_func : Function
        Callback like convert_wind, convert_pv
    """
    assert cutout.prepared, "The cutout has to be prepared first."

    if shapes is not None:
        if isinstance(shapes, pd.Series) and index is None:
            index = shapes.index

        matrix = cutout.indicatormatrix(shapes, shapes_proj)

    if layout is not None:
        if isinstance(layout, xr.DataArray):
            layout = layout.reindex_like(cutout.meta).stack(spatial=('y', 'x')).values
        else:
            assert layout.shape == cutout.shape
        matrix = (layout.reshape((1,-1))
                  if matrix is None
                  else sp.sparse.csr_matrix(matrix).dot(spdiag(layout.ravel())))

    if matrix is not None:
        matrix = sp.sparse.csr_matrix(matrix)

        if index is None:
            index = pd.RangeIndex(matrix.shape[0])
        aggregate_func = aggregate_matrix
        aggregate_kwds = dict(matrix=matrix, index=index)
    else:
        aggregate_func = aggregate_sum
        aggregate_kwds = {}

    results = []

    yearmonths = cutout.coords['year-month'].to_index()

    if show_progress is not False:
        if isinstance(show_progress, string_types):
            prefix = show_progress
        else:
            func_name = (convert_func.__name__[len('convert_'):]
                         if convert_func.__name__.startswith('convert_')
                         else convert_func.__name__)
            prefix = 'Convert and aggregate `{}`: '.format(func_name)
        widgets = [
            pgb.widgets.Percentage(),
            ' ', pgb.widgets.SimpleProgress(format='(%s)' % pgb.widgets.SimpleProgress.DEFAULT_FORMAT),
            ' ', pgb.widgets.Bar(),
            ' ', pgb.widgets.Timer(),
            ' ', pgb.widgets.ETA()
        ]
        maybe_progressbar = pgb.ProgressBar(prefix=prefix, widgets=widgets, max_value=len(yearmonths))
    else:
        maybe_progressbar = lambda x: x

    for ym in maybe_progressbar(yearmonths):
        with xr.open_dataset(cutout.datasetfn(ym)) as ds:
            if 'view' in cutout.meta.attrs:
                ds = ds.sel(**cutout.meta.attrs['view'])
            da = convert_func(ds, **convert_kwds)
            results.append(aggregate_func(da, **aggregate_kwds).load())
    if 'time' in results[0].coords:
        results = xr.concat(results, dim='time')
    else:
        results = sum(results)

    if capacity_factor:
        assert aggregate_func is aggregate_sum, \
            "The arguments `matrix`, `shapes` and `layout` are incompatible with capacity_factor"
        results /= len(cutout.meta['time'])

    if per_unit or return_capacity:
        assert aggregate_func is aggregate_matrix, \
            "One of `matrix`, `shapes` and `layout` must be given for `per_unit`"
        capacity = xr.DataArray(np.asarray(matrix.sum(axis=1)).squeeze(), [index])

    if per_unit:
        results = (results / capacity).fillna(0.)

    if return_capacity:
        return results, capacity
    else:
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



## soil temperature


def convert_soil_temperature(ds):
    """Return soil temperature (useful for e.g. heat pump T-dependent
    coefficient of performance).
    """

    #Temperature is in Kelvin

    #There are nans where there is sea; by setting them
    #to zero we guarantee they do not contribute when multiplied
    #by matrix in atlite/aggregate.py
    return (ds['soil temperature'] - 273.15).fillna(0.)

def soil_temperature(cutout, **params):
    return cutout.convert_and_aggregate(convert_func=convert_soil_temperature, **params)



## heat demand

def convert_heat_demand(ds, threshold, a, constant, hour_shift):
    #Temperature is in Kelvin; take daily average
    T = ds['temperature']
    T.coords['time'].values += np.timedelta64(dt.timedelta(hours=hour_shift))

    T = ds['temperature'].resample("D", dim="time", how="mean")
    threshold += 273.15
    heat_demand = a*(threshold - T)

    heat_demand.values[heat_demand.values < 0.] = 0.

    return constant + heat_demand

def heat_demand(cutout, threshold=15., a=1., constant=0., hour_shift=0., **params):
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

    Note
    ----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.
    """

    return cutout.convert_and_aggregate(convert_func=convert_heat_demand,
                                        threshold=threshold, a=a,
                                        constant=constant,
                                        hour_shift=hour_shift,
                                        **params)


## solar thermal collectors

def convert_solar_thermal(ds, orientation, trigon_model, clearsky_model, c0, c1, t_store):
    # convert storage temperature to Kelvin in line with reanalysis data
    t_store += 273.15

    # Downward shortwave radiation flux is in W/m^2
    # http://rda.ucar.edu/datasets/ds094.0/#metadata/detailed.html?_do=y
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation)
    irradiation = TiltedIrradiation(ds, solar_position, surface_orientation, trigon_model, clearsky_model)

    # overall efficiency; can be negative, so need to remove negative values below
    eta = c0 - c1*((t_store - ds['temperature'])/irradiation)

    output = irradiation*eta

    return (output).where(output > 0.).fillna(0.)


def solar_thermal(cutout, orientation={'slope': 45., 'azimuth': 180.},
                  trigon_model="simple",
                  clearsky_model="simple",
                  c0=0.8, c1=3., t_store=80.,
                  **params):
    """
    Convert downward short-wave radiation flux and outside temperature
    into time series for solar thermal collectors.

    Mathematical model and defaults for c0, c1 based on model in [1].

    Parameters
    ----------
    cutout : cutout
    orientation : dict or str or function
        Panel orientation with slope and azimuth (units of degrees), or
        'latitude_optimal'.
    trigon_model : str
        Type of trigonometry model
    clearsky_model : str or None
        Type of clearsky model for diffuse irradiation. Either
        `simple' or `enhanced'.
    c0, c1 : float
        Parameters for model in [1] (defaults to 0.8 and 3., respectively)
    t_store : float
        Store temperature in degree Celsius

    Note
    ----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    References
    ----------
    [1] Henning and Palzer, Renewable and Sustainable Energy Reviews 30
        (2014) 1003-1018
    """

    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(convert_func=convert_solar_thermal,
                                        orientation=orientation,
                                        trigon_model=trigon_model,
                                        clearsky_model=clearsky_model,
                                        c0=c0, c1=c1, t_store=t_store,
                                        **params)


## wind

def convert_wind(ds, turbine):
    V, POW, hub_height, P = itemgetter('V', 'POW', 'hub_height', 'P')(turbine)

    ds['roughness'].values[ds['roughness'].values <= 0.0] = 0.0002

    for data_height in (100, 10):
        data_name = 'wnd%dm' % data_height
        if data_name in ds.data_vars: break
    else:
        raise AssertionError("Wind speed is not in dataset")

    wnd_hub = ds[data_name] * (np.log(hub_height/ds['roughness']) /
                               np.log(data_height/ds['roughness']))
    wind_energy = xr.DataArray(np.interp(wnd_hub, V, np.asarray(POW)/P),
                               coords=wnd_hub.coords)
    return wind_energy

def wind(cutout, turbine, smooth=False, **params):
    """
    Generate wind generation time-series

    Extrapolates 10m wind speed with monthly surface roughness to hub
    height and evaluates the power curve.

    Parameters
    ----------
    turbine : str or dict
        Name of a turbine known by the reatlas client or a
        turbineconfig dictionary with the keys 'hub_height' for the
        hub height and 'V', 'POW' defining the power curve.
    smooth : bool or dict
        If True smooth power curve with a gaussian kernel as
        determined for the Danish wind fleet to Delta_v = 1.27 and
        sigma = 2.29. A dict allows to tune these values.

    Note
    ----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    References
    ----------
    [1] Andresen G B, Søndergaard A A and Greiner M 2015 Energy 93, Part 1
        1074 – 1088. doi:10.1016/j.energy.2015.09.071
    """

    if isinstance(turbine, string_types):
        turbine = get_windturbineconfig(turbine)

    if smooth:
        turbine = windturbine_smooth(turbine, params=smooth)

    return cutout.convert_and_aggregate(convert_func=convert_wind, turbine=turbine,
                                        **params)

## solar PV

def convert_pv(ds, panel, orientation, trigon_model='simple', clearsky_model='simple'):
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation)
    irradiation = TiltedIrradiation(ds, solar_position, surface_orientation,
                                    trigon_model=trigon_model,
                                    clearsky_model=clearsky_model)
    solar_panel = SolarPanelModel(ds, irradiation, panel)
    return solar_panel

def pv(cutout, panel, orientation, clearsky_model=None, **params):
    '''
    Convert downward-shortwave, upward-shortwave radiation flux and
    ambient temperature into a pv generation time-series.

    Parameters
    ----------
    panel : str or dict
        Panel name known to the reatlas client or a panel config
        dictionary with the parameters for the electrical model in [3].
    orientation : str, dict or callback
        Panel orientation can be chosen from either
        'latitude_optimal', a constant orientation {'slope': 0.0,
        'azimuth': 0.0} or a callback function with the same signature
        as the callbacks generated by the
        `atlite.pv.orientation.make_*' functions.
    clearsky_model : str or None
        Either the 'simple' or the 'enhanced' Reindl clearsky
        model. The default choice of None will choose dependending on
        data availability, since the 'enhanced' model also
        incorporates ambient air temperature and relative humidity.

    Returns
    -------
    pv : xr.DataArray
        Time-series or capacity factors based on additional general
        conversion arguments.

    Note
    ----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    References
    ----------
    [1] Soteris A. Kalogirou. Solar Energy Engineering: Processes and Systems,
        pages 49–117,469–516. Academic Press, 2009. ISBN 0123745012.
    [2] D.T. Reindl, W.A. Beckman, and J.A. Duffie. Diffuse fraction correla-
        tions. Solar Energy, 45(1):1 – 7, 1990.
    [3] Hans Georg Beyer, Gerd Heilscher and Stefan Bofinger. A Robust Model
        for the MPP Performance of Different Types of PV-Modules Applied for
        the Performance Check of Grid Connected Systems, Freiburg, June 2004.
        Eurosun (ISES Europe Solar Congress).
    '''

    if isinstance(panel, string_types):
        panel = get_solarpanelconfig(panel)
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(convert_func=convert_pv,
                                        panel=panel, orientation=orientation,
                                        clearsky_model=clearsky_model,
                                        **params)

## hydro

def convert_runoff(ds):
    runoff = ds['runoff'] * ds['height']
    return runoff

def runoff(cutout, smooth=None, lower_threshold_quantile=None,
           normalize_using_yearly=None, **params):
    result = cutout.convert_and_aggregate(convert_func=convert_runoff, **params)

    if smooth is not None:
        if smooth is True: smooth = 24*7
        if "return_capacity" in params.keys():
            result = result[0].rolling(time=smooth, min_periods=1).mean(), result[1]
        else:
            result = result.rolling(time=smooth, min_periods=1).mean()

    if lower_threshold_quantile is not None:
        if lower_threshold_quantile is True: lower_threshold_quantile = 5e-3
        lower_threshold = pd.Series(result.values.ravel()).quantile(lower_threshold_quantile)
        result.values[result.values < lower_threshold] = 0.

    if normalize_using_yearly is not None:
        normalize_using_yearly_i = normalize_using_yearly.index
        if isinstance(normalize_using_yearly_i, pd.DatetimeIndex):
            normalize_using_yearly_i = normalize_using_yearly_i.year
        else:
            normalize_using_yearly_i = normalize_using_yearly_i.astype(int)

        years = (pd.Series(pd.to_datetime(result.coords['time'].values).year)
                 .value_counts().loc[lambda x: x>8700].index
                 .intersection(normalize_using_yearly_i))
        assert len(years), "Need at least a full year of data (more is better)"
        years_overlap = slice(str(min(years)), str(max(years)))

        dim = result.dims[1 - result.get_axis_num('time')]
        result *= ((xr.DataArray(normalize_using_yearly.loc[years_overlap].sum(), dims=[dim]) /
                   result.sel(time=years_overlap).sum('time'))
                   .reindex(countries=result.coords['countries']))

    return result
