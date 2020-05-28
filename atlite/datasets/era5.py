# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for downloading and curating data from ECMWFs ERA5 dataset (via CDS).

For further reference see
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from dask import delayed
from tempfile import mkstemp
import weakref
import cdsapi

from ..gis import maybe_swap_spatial_dims


import logging
logger = logging.getLogger(__name__)


# Model and Projection Settings
projection = 'latlon'

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


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords['x'].min().item(), coords['x'].max().item()
    y0, y1 = coords['y'].min().item(), coords['y'].max().item()
    return [y1, x0, y0, x1]


def retrieval_times(coords):
    time = pd.Series(coords['time'])
    time_span = time[0] - time[len(time)-1]
    if len(time) == 1:
        return [{'year': d.year, 'month': d.month, 'day': d.day,
                 'time': d.strftime("%H:00")} for d in time]
    if time_span.days <= 1:
        return [{'year': d.year, 'month': d.month, 'day': d.day}
                for d in time.dt.date.unique()]
    elif time_span.days < 90:
        return [{'year': year, 'month': month}
                for month in time.dt.month.unique()
                for year in time.dt.year.unique()]
    else:
        return [{'year': year} for year in time.dt.year.unique()]


def noisy_unlink(path):
    logger.info(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def retrieve_data(product, chunks=None, tmpdir=None, **updates):
    """Download data like ERA5 from the Climate Data Store (CDS)"""

    # Default request
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'day': list(range(1, 31 + 1)),
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        'month': list(range(1, 12 + 1)),
        # 'area': [50, -1, 49, 1], # North, West, South, East. Default: global
        # 'grid': [0.25, 0.25], # Latitude/longitude grid: east-west (longitude)
        # and north-south resolution (latitude). Default: 0.25 x 0.25
    }
    request.update(updates)

    assert {'year', 'month', 'variable'}.issubset(
        request), "Need to specify at least 'variable', 'year' and 'month'"

    result = cdsapi.Client(
        progress=False
    ).retrieve(
        product,
        request
    )

    fd, target = mkstemp(suffix='.nc', dir=tmpdir)
    os.close(fd)

    logger.info("Downloading request for {} variables to {}".format(
        len(request['variable']), target))

    result.download(target)

    ds = xr.open_dataset(target, chunks=chunks or {})
    if tmpdir is None:
        logger.debug(f"Adding finalizer for {target}")
        weakref.finalize(ds._file_obj._manager, noisy_unlink, target)

    return ds



def get_data(cutout, feature, tmpdir, **creation_parameters):

    assert tmpdir is not None

    coords = cutout.coords
    sanitize = creation_parameters.get('sanitize', True)

    retrieval_params = {'product': 'reanalysis-era5-single-levels',
                        'area': _area(coords),
                        'tmpdir': tmpdir,
                        'chunks': creation_parameters.pop('chunks', None),
                        'grid': [cutout.dx, cutout.dy]}

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")


    datasets = []
    for d in retrieval_times(coords):
        retrieval_params.update(d)
        ds = delayed(func)(retrieval_params)
        if sanitize and sanitize_func is not None:
            ds = delayed(sanitize_func)(ds)
        if feature in static_features:
            return ds
        datasets.append(ds)
    return delayed(xr.concat)(datasets, dim='time')
