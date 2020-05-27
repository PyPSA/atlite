# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for downloading and curating data from ECMWFs ERA5 dataset (via CDS).
"""

from .common import retrieve_data, get_data_gebco_height
import os
import pandas as pd
import numpy as np
import xarray as xr
from dask import delayed
from ..gis import maybe_swap_spatial_dims


import logging
logger = logging.getLogger(__name__)


# Model and Projection Settings
projection = 'latlong'

features = {
    'height': ['height'],
    'wind': [
        'wnd100m',
        'roughness'],
    'influx': [
        'influx_toa',
        'influx_direct',
        'influx_diffuse',
        'influx',
        'albedo'],
    'temperature': [
        'temperature',
        'soil_temperature'],
    'runoff': ['runoff']}

static_features = {'height'}


def _add_height(ds):
    """Convert geopotential 'z' to geopotential height following [1]

    References
    ----------
    [1] ERA5: surface elevation and orography, retrieved: 10.02.2019
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+surface+elevation+and+orography

    """
    g0 = 9.80665
    z = ds['z']
    if 'time' in z.coords:
        z = z.isel(time=0, drop=True)
    ds['height'] = z / g0
    ds = ds.drop_vars('z')
    return ds


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords['x'].min().item(), coords['x'].max().item()
    y0, y1 = coords['y'].min().item(), coords['y'].max().item()
    return [y1, x0, y0, x1]


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y'

    Optionally (add_lon_lat, default:True) preserves latitude and longitude columns as 'lat' and 'lon'.
    """

    ds = ds.rename({'longitude': 'x', 'latitude': 'y'})
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds


def get_data_wind(retrieval_params):
    ds = retrieve_data(
        variable=[
            '100m_u_component_of_wind',
            '100m_v_component_of_wind',
            'forecast_surface_roughness'],
        **retrieval_params)
    ds = _rename_and_clean_coords(ds)

    ds['wnd100m'] = (np.sqrt(ds['u100']**2 + ds['v100']**2)
                     .assign_attrs(units=ds['u100'].attrs['units'],
                                   long_name="100 metre wind speed"))
    ds = ds.drop_vars(['u100', 'v100'])
    ds = ds.rename({'fsr': 'roughness'})

    return ds


def sanitize_wind(ds):
    ds['roughness'] = ds['roughness'].where(ds['roughness'] >= 0.0, 2e-4)
    return ds


def get_data_influx(retrieval_params):
    ds = retrieve_data(
        variable=[
            'surface_net_solar_radiation',
            'surface_solar_radiation_downwards',
            'toa_incident_solar_radiation',
            'total_sky_direct_solar_radiation_at_surface'],
        **retrieval_params)

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename({'fdir': 'influx_direct', 'tisr': 'influx_toa'})
    with np.errstate(divide='ignore', invalid='ignore'):
        ds['albedo'] = (((ds['ssrd'] - ds['ssr']) / ds['ssrd']).fillna(0.)
                        .assign_attrs(units='(0 - 1)', long_name='Albedo'))
    ds['influx_diffuse'] = (
        (ds['ssrd'] - ds['influx_direct'])
        .assign_attrs(units='J m**-2',
                      long_name='Surface diffuse solar radiation downwards'))
    ds = ds.drop_vars(['ssrd', 'ssr'])

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    for a in ('influx_direct', 'influx_diffuse', 'influx_toa'):
        ds[a] = ds[a] / (60. * 60.)
        ds[a].attrs['units'] = 'W m**-2'

    return ds


def sanitize_inflow(ds):
    for a in ('influx_direct', 'influx_diffuse', 'influx_toa'):
        ds[a] = ds[a].clip(min=0.)
    return ds


def get_data_temperature(retrieval_params):
    ds = retrieve_data(variable=['2m_temperature', 'soil_temperature_level_4'],
                       **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({'t2m': 'temperature', 'stl4': 'soil temperature'})

    return ds


def get_data_runoff(retrieval_params):
    ds = retrieve_data(variable=['runoff'], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({'ro': 'runoff'})

    return ds


def sanitize_runoff(ds):
    ds['runoff'] = ds['runoff'].clip(min=0.)
    return ds


def get_data_height(retrieval_params):
    ds = retrieve_data(variable='orography', **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = _add_height(ds)

    return ds


def get_data(
        coords,
        period,
        feature,
        sanitize=True,
        tmpdir=None,
        **creation_parameters):

    assert tmpdir is not None

    retrieval_params = {'product': 'reanalysis-era5-single-levels',
                        'area': _area(coords),
                        'tmpdir': tmpdir,
                        'chunks': creation_parameters.pop('chunks', None)}

    if {'dx', 'dy'}.issubset(creation_parameters):
        retrieval_params['grid'] = [
            creation_parameters.pop('dx'),
            creation_parameters.pop('dy')]

    if isinstance(period, pd.Period):
        retrieval_params['year'] = period.year
        if isinstance(period.freq, pd.tseries.offsets.MonthOffset):
            retrieval_params['month'] = period.month
        elif isinstance(period.freq, pd.tseries.frequencies.Day):
            retrieval_params['month'] = period.month
            retrieval_params['day'] = period.day
    elif isinstance(period, pd.Timestamp):
        retrieval_params.update(year=period.year, month=period.month,
                                day=period.day, time=period.strftime("%H:00"))
    else:
        raise TypeError(f"{period} should be one of pd.Timestamp or pd.Period")

    gebco_path = creation_parameters.pop('gebco_path', None)
    if gebco_path and feature == 'height':
        return delayed(get_data_gebco_height)(
            coords.indexes['x'], coords.indexes['y'], gebco_path)

    if creation_parameters:
        logger.debug(
            f"Unused creation_parameters: {', '.join(creation_parameters)}")

    # Reference of the quantities
    # https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
    # (shortName) | (name)                                      | (paramId)
    # tisr        | TOA incident solar radiation                | 212
    # ssrd        | Surface Solar Rad Downwards                 | 169
    # ssr         | Surface net Solar Radiation                 | 176
    # fdir        | Total sky direct solar radiation at surface | 228021
    # ro          | Runoff                                      | 205
    # 2t          | 2 metre temperature                         | 167
    # sp          | Surface pressure                            | 134
    # stl4        | Soil temperature level 4                    | 236
    # fsr         | Forecast surface roughnes                   | 244

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")
    if func is None:
        raise NotImplementedError(
            f"Feature '{feature}' has not been implemented for dataset era5")

    ds = delayed(func)(retrieval_params)

    if sanitize and sanitize_func is not None:
        ds = delayed(sanitize_func)(ds)

    return ds
