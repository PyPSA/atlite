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
import numpy as np
import xarray as xr
from tempfile import mkstemp
import weakref
import cdsapi
import logging
from numpy import atleast_1d
from ..gis import maybe_swap_spatial_dims

# Null context for running a with statements wihout any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib
    @contextlib.contextmanager
    def nullcontext():
        yield

logger = logging.getLogger(__name__)

# Model and CRS Settings
crs = 4326

features = {
    'height': ['height'],
    'wind': [
        'wnd100m',
        'roughness'],
    'influx': [
        'influx_toa',
        'influx_direct',
        'influx_diffuse',
        'albedo'],
    'temperature': [
        'temperature',
        'soil temperature'],
    'runoff': ['runoff']}

static_features = {'height'}


def _add_height(ds):
    """Convert geopotential 'z' to geopotential height following [1].

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
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and longitude
    columns as 'lat' and 'lon'.
    """
    ds = ds.rename({'longitude': 'x', 'latitude': 'y'})
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(x=np.round(ds.x.astype(float), 5),
                          y=np.round(ds.y.astype(float), 5))
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds


def get_data_wind(retrieval_params):
    """Get wind data for given retrieval parameters."""
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
    """Sanitize retrieved wind data."""
    ds['roughness'] = ds['roughness'].where(ds['roughness'] >= 0.0, 2e-4)
    return ds


def get_data_influx(retrieval_params):
    """Get influx data for given retrieval parameters."""
    ds = retrieve_data(
        variable=[
            'surface_net_solar_radiation',
            'surface_solar_radiation_downwards',
            'toa_incident_solar_radiation',
            'total_sky_direct_solar_radiation_at_surface'],
        **retrieval_params)

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename({'fdir': 'influx_direct', 'tisr': 'influx_toa'})
    ds['albedo'] = (((ds['ssrd'] - ds['ssr']) /
                     ds['ssrd'].where(ds['ssrd'] != 0))
                    .fillna(0.)
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
    """Sanitize retrieved inflow data."""
    for a in ('influx_direct', 'influx_diffuse', 'influx_toa'):
        ds[a] = ds[a].clip(min=0.)
    return ds


def get_data_temperature(retrieval_params):
    """Get wind temperature for given retrieval parameters."""
    ds = retrieve_data(variable=['2m_temperature', 'soil_temperature_level_4'],
                       **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({'t2m': 'temperature', 'stl4': 'soil temperature'})

    return ds


def get_data_runoff(retrieval_params):
    """Get runoff data for given retrieval parameters."""
    ds = retrieve_data(variable=['runoff'], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({'ro': 'runoff'})

    return ds


def sanitize_runoff(ds):
    """Sanitize retrieved runoff data."""
    ds['runoff'] = ds['runoff'].clip(min=0.)
    return ds


def get_data_height(retrieval_params):
    """Get height data for given retrieval parameters."""
    ds = retrieve_data(variable='orography', **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = _add_height(ds)

    return ds


def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords['x'].min().item(), coords['x'].max().item()
    y0, y1 = coords['y'].min().item(), coords['y'].max().item()
    return [y1, x0, y0, x1]


def retrieval_times(coords, static=False):
    """
    Get list of retrieval cdsapi arguments for time dimension in coordinates.

    If static is False, this function creates a query for each year in the
    time axis in coords. This ensures not running into query limits of the
    cdsapi. If static is True, the function return only one set of parameters
    for the very first time point.

    Parameters
    ----------
    coords : atlite.Cutout.coords

    Returns
    -------
    list of dicts witht retrieval arguments

    """
    time = coords['time'].to_index()
    if static:
        return {'year': str(time[0].year), 'month': str(time[0].month),
                'day': str(time[0].day), 'time': time[0].strftime("%H:00")}

    times = []
    for year in time.year.unique():
        t = time[time.year==year]
        query = {'year': str(year),
                 'month': list(t.month.unique()),
                 'day': list(t.day.unique()),
                 'time': ["%02d:00" %h for h in t.hour.unique()]}
        times.append(query)
    return times


def noisy_unlink(path):
    """Delete file at given path."""
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def retrieve_data(product, chunks=None, tmpdir=None, lock=None, **updates):
    """
    Download data like ERA5 from the Climate Data Store (CDS).

    If you want to track the state of your request go to
    https://cds.climate.copernicus.eu/cdsapp#!/yourrequests
    """
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf'}
    request.update(updates)

    assert {'year', 'month', 'variable'}.issubset(request), (
            "Need to specify at least 'variable', 'year' and 'month'")

    client = cdsapi.Client(info_callback=logger.debug,
                            debug=logging.DEBUG >= logging.root.level)
    result = client.retrieve(product, request)

    if lock is None:
        lock = nullcontext()

    with lock:
        fd, target = mkstemp(suffix='.nc', dir=tmpdir); os.close(fd)

        yearstr = ', '.join(atleast_1d(request['year']))
        variables = atleast_1d(request['variable'])
        varstr = ''.join(['\t * ' + v + f' ({yearstr})\n' for v in variables])
        logger.info(f"CDS: Downloading variables\n{varstr}")
        result.download(target)

    ds = xr.open_dataset(target, chunks=chunks or {})
    if tmpdir is None:
        logger.debug(f"Adding finalizer for {target}")
        weakref.finalize(ds._file_obj._manager, noisy_unlink, target)

    return ds



def get_data(cutout, feature, tmpdir, lock=None, **creation_parameters):
    """
    Retrieve data from ECMWFs ERA5 dataset (via CDS).

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.era5.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    **creation_parameters :
        Additional keyword arguments. The only effective argument is 'sanitize'
        (default True) which sets sanitization of the data on or off.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

    """
    coords = cutout.coords

    sanitize = creation_parameters.get('sanitize', True)

    retrieval_params = {'product': 'reanalysis-era5-single-levels',
                        'area': _area(coords),
                        'chunks': cutout.chunks,
                        'grid': [cutout.dx, cutout.dy],
                        'tmpdir': tmpdir,
                        'lock': lock}

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    logger.info(f'Requesting data for feature {feature}...')

    def retrieve_once(time):
        ds = func({**retrieval_params, **time})
        if sanitize and sanitize_func is not None:
            ds = sanitize_func(ds)
        return ds

    if feature in static_features:
        return retrieve_once(retrieval_times(coords, static=True)).squeeze()

    datasets = map(retrieve_once, retrieval_times(coords))

    return xr.concat(datasets, dim='time').sel(time=coords['time'])
