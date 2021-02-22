# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
All functions for converting weather data into energy system model data.
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
from operator import itemgetter
from pathlib import Path
from dask.diagnostics import ProgressBar
from scipy.sparse import csr_matrix

import logging
logger = logging.getLogger(__name__)

from .aggregate import aggregate_matrix
from .gis import spdiag

from .pv.solar_position import SolarPosition
from .pv.irradiation import TiltedIrradiation
from .pv.solar_panel_model import SolarPanelModel
from .pv.orientation import get_orientation, SurfaceOrientation

from . import hydro as hydrom
from . import wind as windm

from .resource import (get_windturbineconfig, get_solarpanelconfig,
                       windturbine_smooth)


def convert_and_aggregate(cutout, convert_func, matrix=None,
                          index=None, layout=None, shapes=None,
                          shapes_crs=4326, per_unit=False,
                          return_capacity=False, capacity_factor=False,
                          show_progress=True, dask_kwargs={}, **convert_kwds):
    """
    Convert and aggregate a weather-based renewable generation time-series.

    NOTE: Not meant to be used by the user him or herself. Rather it is a
    gateway function that is called by all the individual time-series
    generation functions like pv and wind. Thus, all its parameters are also
    available from these.

    Parameters
    -----------
    matrix : N x S - xr.DataArray or sp.sparse.csr_matrix or None
        If given, it is used to aggregate the grid cells to buses.
        N is the number of buses, S the number of spatial coordinates, in the
        order of `cutout.grid`.
    index : pd.Index
        Index of Buses.
    layout : X x Y - xr.DataArray
        The capacity to be build in each of the `grid_cells`.
    shapes : list or pd.Series of shapely.geometry.Polygon
        If given, matrix is constructed as indicatormatrix of the polygons, its
        index determines the bus index on the time-series.
    shapes_crs : pyproj.CRS or compatible
        If different to the map crs of the cutout, the shapes are
        transformed to match cutout.crs (defaults to EPSG:4326).
    per_unit : boolean
        Returns the time-series in per-unit units, instead of in MW (defaults
        to False).
    return_capacity : boolean
        Additionally returns the installed capacity at each bus corresponding
        to `layout` (defaults to False).
    capacity_factor : boolean
        If True, the static capacity factor of the chosen resource for each
        grid cell is computed.
    show_progress : boolean, default True
        Whether to show a progress bar.
    dask_kwargs : dict, default {}
        Dict with keyword arguments passed to `dask.compute`.

    Other Parameters
    -----------------
    convert_func : Function
        Callback like convert_wind, convert_pv


    Returns
    -------
    resource : xr.DataArray
        Time-series of renewable generation aggregated to buses, if
        `matrix` or equivalents are provided else the total sum of
        generated energy.
    units : xr.DataArray (optional)
        The installed units per bus in MW corresponding to `layout`
        (only if `return_capacity` is True).

    """

    func_name = convert_func.__name__.replace('convert_', '')
    logger.info(f"Convert and aggregate '{func_name}'.")
    da = convert_func(cutout.data, **convert_kwds)

    no_args = all(v is None for v in [layout, shapes, matrix])

    if no_args:
        if per_unit or return_capacity:
            raise ValueError("One of `matrix`, `shapes` and `layout` must be "
                             "given for `per_unit` or `return_capacity`")
        if capacity_factor:
            return maybe_progressbar(da.mean('time'), show_progress)
        else:
            return maybe_progressbar(da.sum('time'), show_progress)


    if shapes is not None:
        geoseries_like = (pd.Series, gpd.GeoDataFrame, gpd.GeoSeries)
        if isinstance(shapes, geoseries_like) and index is None:
            index = shapes.index
        matrix = cutout.indicatormatrix(shapes, shapes_crs)


    if layout is not None:
        assert isinstance(layout, xr.DataArray)
        layout = layout.reindex_like(cutout.data).stack(spatial=['y', 'x'])
        if matrix is None:
            matrix = layout.expand_dims('new')
        else:
            matrix = csr_matrix(matrix) * spdiag(layout)

    index = pd.RangeIndex(matrix.shape[0]) if index is None else index
    matrix = csr_matrix(matrix)
    results = aggregate_matrix(da, matrix=matrix, index=index)


    if per_unit or return_capacity:
        caps = matrix.sum(-1)
        capacity = xr.DataArray(np.asarray(caps).flatten(), [index])
        capacity.attrs['units'] ='MW'

    if per_unit:
        results = (results / capacity.where(capacity!=0)).fillna(0.)
        results.attrs['units'] = 'p.u.'
    else:
        results.attrs['units'] = 'MW'


    if return_capacity:
        return maybe_progressbar(results, show_progress, **dask_kwargs), capacity
    else:
        return maybe_progressbar(results, show_progress, **dask_kwargs)


def maybe_progressbar(ds, show_progress, **kwargs):
    """Load a xr.dataset with dask arrays either with or without progressbar."""
    if show_progress:
        with ProgressBar(minimum=2):
            ds.load(**kwargs)
    else:
        ds.load(**kwargs)
    return ds


# temperature
def convert_temperature(ds):
    """Return outside temperature (useful for e.g. heat pump T-dependent
    coefficient of performance).
    """

    # Temperature is in Kelvin
    return ds['temperature'] - 273.15


def temperature(cutout, **params):
    return cutout.convert_and_aggregate(
        convert_func=convert_temperature, **params)


# soil temperature
def convert_soil_temperature(ds):
    """Return soil temperature (useful for e.g. heat pump T-dependent
    coefficient of performance).
    """

    # Temperature is in Kelvin

    # There are nans where there is sea; by setting them
    # to zero we guarantee they do not contribute when multiplied
    # by matrix in atlite/aggregate.py
    return (ds['soil temperature'] - 273.15).fillna(0.)


def soil_temperature(cutout, **params):
    return cutout.convert_and_aggregate(
        convert_func=convert_soil_temperature, **params)


# heat demand
def convert_heat_demand(ds, threshold, a, constant, hour_shift):
    # Temperature is in Kelvin; take daily average
    T = ds['temperature']
    T = T.assign_coords(time=(T.coords['time'] +
                              np.timedelta64(dt.timedelta(hours=hour_shift))))

    T = T.resample(time="1D").mean(dim='time')
    threshold += 273.15
    heat_demand = a * (threshold - T)

    heat_demand = heat_demand.clip(min=0.)

    return (constant + heat_demand).rename('heat_demand')


def heat_demand(
        cutout,
        threshold=15.,
        a=1.,
        constant=0.,
        hour_shift=0.,
        **params):
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


# solar thermal collectors
def convert_solar_thermal(ds, orientation, trigon_model, clearsky_model,
                          c0, c1, t_store):
    # convert storage temperature to Kelvin in line with reanalysis data
    t_store += 273.15

    # Downward shortwave radiation flux is in W/m^2
    # http://rda.ucar.edu/datasets/ds094.0/#metadata/detailed.html?_do=y
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation)
    irradiation = TiltedIrradiation(ds, solar_position, surface_orientation,
                                    trigon_model, clearsky_model)

    # overall efficiency; can be negative, so need to remove negative values
    # below
    eta = c0 - c1 * ((t_store - ds['temperature']) /
                     irradiation.where(irradiation!=0)).fillna(0)

    output = irradiation * eta

    return output.where(output > 0., 0.)


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
        'simple' or 'enhanced'.
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


# wind
def convert_wind(ds, turbine):
    """Convert wind speeds for turbine to wind energy generation."""

    V, POW, hub_height, P = itemgetter('V', 'POW', 'hub_height', 'P')(turbine)

    wnd_hub = windm.extrapolate_wind_speed(ds, to_height=hub_height)

    def _interpolate(da):
        return np.interp(da, V, POW / P)

    return xr.apply_ufunc(_interpolate, wnd_hub,
                          input_core_dims=[[]],
                          output_core_dims=[[]],
                          output_dtypes=[wnd_hub.dtype],
                          dask="parallelized")


def wind(cutout, turbine, smooth=False, **params):
    """
    Generate wind generation time-series

    Extrapolates 10m wind speed with monthly surface roughness to hub
    height and evaluates the power curve.

    Parameters
    ----------
    turbine : str or dict
        A turbineconfig dictionary with the keys 'hub_height' for the
        hub height and 'V', 'POW' defining the power curve.
        Alternatively a str refering to a local or remote turbine configuration
        as accepted by atlite.resource.get_windturbineconfig().
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

    if isinstance(turbine, (str, Path)):
        turbine = get_windturbineconfig(turbine)

    if smooth:
        turbine = windturbine_smooth(turbine, params=smooth)

    return cutout.convert_and_aggregate(
        convert_func=convert_wind, turbine=turbine, **params)


# solar PV
def convert_pv(
        ds,
        panel,
        orientation,
        trigon_model='simple',
        clearsky_model='simple'):
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation)
    irradiation = TiltedIrradiation(ds, solar_position, surface_orientation,
                                    trigon_model=trigon_model,
                                    clearsky_model=clearsky_model)
    solar_panel = SolarPanelModel(ds, irradiation, panel)
    return solar_panel


def pv(cutout, panel, orientation, clearsky_model=None, **params):
    """
    Convert downward-shortwave, upward-shortwave radiation flux and
    ambient temperature into a pv generation time-series.

    Parameters
    ----------
    panel : str or dict
        Panel config dictionary with the parameters for the electrical
        model in [3]. Alternatively, name of yaml file stored in
        atlite.config.solarpanel_dir.
    orientation : str, dict or callback
        Panel orientation can be chosen from either
        'latitude_optimal', a constant orientation {'slope': 0.0,
        'azimuth': 0.0} or a callback function with the same signature
        as the callbacks generated by the
        'atlite.pv.orientation.make_*' functions.
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
    """

    if isinstance(panel, (str, Path)):
        panel = get_solarpanelconfig(panel)
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(convert_func=convert_pv,
                                        panel=panel, orientation=orientation,
                                        clearsky_model=clearsky_model,
                                        **params)

# hydro
def convert_runoff(ds, weight_with_height=True):
    runoff = ds['runoff']

    if weight_with_height:
        runoff = runoff * ds['height']

    return runoff


def runoff(cutout, smooth=None, lower_threshold_quantile=None,
           normalize_using_yearly=None, **params):
    result = cutout.convert_and_aggregate(convert_func=convert_runoff, **params)

    if smooth is not None:
        if smooth is True:
            smooth = 24 * 7
        if "return_capacity" in params.keys():
            result = result[0].rolling(
                time=smooth, min_periods=1).mean(), result[1]
        else:
            result = result.rolling(time=smooth, min_periods=1).mean()

    if lower_threshold_quantile is not None:
        if lower_threshold_quantile is True:
            lower_threshold_quantile = 5e-3
        lower_threshold = pd.Series(result.values.ravel())\
                            .quantile(lower_threshold_quantile)
        result = result.where(result >= lower_threshold, 0.)

    if normalize_using_yearly is not None:
        normalize_using_yearly_i = normalize_using_yearly.index
        if isinstance(normalize_using_yearly_i, pd.DatetimeIndex):
            normalize_using_yearly_i = normalize_using_yearly_i.year
        else:
            normalize_using_yearly_i = normalize_using_yearly_i.astype(int)

        years = (pd.Series(pd.to_datetime(result.coords['time'].values).year)
                 .value_counts().loc[lambda x: x > 8700].index
                 .intersection(normalize_using_yearly_i))
        assert len(years), "Need at least a full year of data (more is better)"
        years_overlap = slice(str(min(years)), str(max(years)))

        dim = result.dims[1 - result.get_axis_num('time')]
        result *= (
            (xr.DataArray(
                normalize_using_yearly.loc[years_overlap].sum(),
                dims=[dim]) /
                result.sel(
                time=years_overlap).sum('time')) .reindex(
                countries=result.coords['countries']))

    return result


def hydro(cutout, plants, hydrobasins, flowspeed=1, weight_with_height=False,
          show_progress=True, **kwargs):
    """
    Compute inflow time-series for `plants` by aggregating over catchment
    basins from `hydrobasins`

    Parameters
    ----------
    plants : pd.DataFrame
        Run-of-river plants or dams with lon, lat columns.
    hydrobasins : str|gpd.GeoDataFrame
        Filename or GeoDataFrame of one level of the HydroBASINS dataset.
    flowspeed : float
        Average speed of water flows to estimate the water travel time from
        basin to plant (default: 1 m/s).
    weight_with_height : bool
        Whether surface runoff should be weighted by potential height (probably
        better for coarser resolution).
    show_progress : bool
        Whether to display progressbars.

    References
    ----------
    [1] Liu, Hailiang, et al. "A validated high-resolution hydro power
    time-series model for energy systems analysis." arXiv preprint
    arXiv:1901.08476 (2019).

    [2] Lehner, B., Grill G. (2013): Global river hydrography and network
    routing: baseline data and new approaches to study the world’s large river
    systems. Hydrological Processes, 27(15): 2171–2186. Data is available at
    www.hydrosheds.org.
    """
    basins = hydrom.determine_basins(
        plants, hydrobasins, show_progress=show_progress)

    matrix = cutout.indicatormatrix(basins.shapes)
    # compute the average surface runoff in each basin
    matrix_normalized = matrix / matrix.sum(axis=1)
    runoff = cutout.runoff(matrix=matrix_normalized, index=basins.shapes.index,
                           weight_with_height=weight_with_height,
                           show_progress=show_progress, **kwargs)
    # The hydrological parameters are in units of "m of water per day" and so
    # they should be multiplied by 1000 and the basin area to convert to m3
    # d-1 = m3 h-1 / 24
    runoff *= (1000. / 24.) * \
        xr.DataArray(basins.shapes.to_crs(dict(proj="cea")).area)

    return hydrom.shift_and_aggregate_runoff_for_plants(
        basins, runoff, flowspeed, show_progress)
