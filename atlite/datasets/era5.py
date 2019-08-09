# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Renewable Energy Atlas Lite (Atlite)

Light-weight version of Aarhus RE Atlas for converting weather data to power systems data
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from dask import delayed

import logging
logger = logging.getLogger(__name__)

from ..utils import timeindex_from_slice
from .common import retrieve_data, get_data_gebco_height

# Model and Projection Settings
projection = 'latlong'

features = {
    'height': ['height'],
    'wind': ['wnd100m', 'roughness'],
    'influx': ['influx_toa', 'influx_direct', 'influx_diffuse', 'influx', 'albedo'],
    'temperature': ['temperature', 'soil_temperature'],
    'runoff': ['runoff']
}

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
    ds['height'] = z/g0
    ds = ds.drop('z')
    return ds

def _area(xs, ys):
    # North, West, South, East. Default: global
    return [ys.start, xs.start, ys.stop, xs.stop]

def _rename_and_clean_coords(ds, add_lon_lat=True):
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y'

    Optionally (add_lon_lat, default:True) preserves latitude and longitude columns as 'lat' and 'lon'.
    """

    ds = ds.rename({'longitude': 'x', 'latitude': 'y'})
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def get_coords(time, x, y, **creation_parameters):
    # Reference of the quantities
    # https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
    # Geopotential is aka Orography in the CDS:
    # https://confluence.ecmwf.int/pages/viewpage.action?pageId=78296105
    #
    # (shortName) | (name)                        | (paramId)
    # z           | Geopotential (CDS: Orography) | 129

    # time = timeindex_from_slice(time)

    ds = xr.Dataset({'longitude': np.r_[-180:180:0.25], 'latitude': np.r_[90:-90:-0.25],
                     'time': pd.date_range(start="1979", end="now", freq="h")})

    ds = _rename_and_clean_coords(ds)
    ds = ds.sel(x=x, y=y, time=time)

    return ds


def get_data_wind(kwds):
    ds = retrieve_data(variable=['100m_u_component_of_wind',
                                 '100m_v_component_of_wind',
                                 'forecast_surface_roughness'], **kwds)
    ds = _rename_and_clean_coords(ds)

    ds['wnd100m'] = (np.sqrt(ds['u100']**2 + ds['v100']**2)
                     .assign_attrs(units=ds['u100'].attrs['units'],
                                   long_name="100 metre wind speed"))
    ds = ds.drop(['u100', 'v100'])
    ds = ds.rename({'fsr': 'roughness'})

    return ds

def get_data_influx(kwds):
    ds = retrieve_data(variable=['surface_net_solar_radiation',
                                 'surface_solar_radiation_downwards',
                                 'toa_incident_solar_radiation',
                                 'total_sky_direct_solar_radiation_at_surface'],
                       **kwds)

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename({'fdir': 'influx_direct', 'tisr': 'influx_toa'})
    with np.errstate(divide='ignore', invalid='ignore'):
        ds['albedo'] = (((ds['ssrd'] - ds['ssr'])/ds['ssrd']).fillna(0.)
                        .assign_attrs(units='(0 - 1)', long_name='Albedo'))
    ds['influx_diffuse'] = ((ds['ssrd'] - ds['influx_direct'])
                            .assign_attrs(units='J m**-2',
                                          long_name='Surface diffuse solar radiation downwards'))
    ds = ds.drop(['ssrd', 'ssr'])

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    for a in ('influx_direct', 'influx_diffuse', 'influx_toa'):
        ds[a] = ds[a].clip(min=0.) / (60.*60.)
        ds[a].attrs['units'] = 'W m**-2'

    return ds

def get_data_temperature(kwds):
    ds = retrieve_data(variable=['2m_temperature', 'soil_temperature_level_4'], **kwds)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({'t2m': 'temperature', 'stl4': 'soil temperature'})

    return ds

def get_data_runoff(kwds):
    ds = retrieve_data(variable=['runoff'], **kwds)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({'ro': 'runoff'})
    ds['runoff'] = ds['runoff'].clip(min=0.)


    return ds

def get_data_height(kwds):
    ds = retrieve_data(variable='orography', **kwds)

    ds = _rename_and_clean_coords(ds)
    ds = _add_height(ds)

    return ds

def get_data(coords, time, feature, x, y, chunks=None, **creation_parameters):
    kwds = {'product': 'reanalysis-era5-single-levels',
            'chunks': chunks, 'area': _area(x, y)}

    if {'dx', 'dy'}.issubset(creation_parameters):
        kwds['grid'] = [creation_parameters.pop('dx'), creation_parameters.pop('dy')]

    if isinstance(time, pd.Period):
        kwds['year'] = time.year
        if isinstance(time.freq, pd.tseries.offsets.MonthOffset):
            kwds['month'] = time.month
        elif isinstance(time.freq, pd.tseries.frequencies.Day):
            kwds['month'] = time.month
            kwds['day'] = time.day
    elif isinstance(time, pd.Timestamp):
        kwds.update(year=time.year, month=time.month, day=time.day, time=time.strftime("%H:00"))
    else:
        raise TypeError(f"{time} should be one of pd.Timestamp or pd.Period")

    gebco_fn = creation_parameters.pop('gebco_fn', None)
    if gebco_fn is not None and feature == 'height':
        coords = get_coords(x=x, y=y, **creation_parameters)
        return delayed(get_data_gebco_height)(coords.indexes['x'], coords.indexes['y'], gebco_fn)

    if creation_parameters:
        logger.debug(f"Unused creation_parameters: {', '.join(creation_parameters)}")

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
    if func is None:
        raise NotImplementedError(f"Feature '{feature}' has not been implemented for dataset era5")

    return delayed(func)(kwds)
