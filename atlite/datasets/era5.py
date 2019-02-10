## Copyright 2016-2017 Jonas Hoersch (FIAS), Tom Brown (FIAS), Markus Schlott
## (FIAS)

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

import os
import pandas as pd
import numpy as np
import xarray as xr
import shutil
from six.moves import range
from contextlib import contextmanager
from tempfile import mkstemp
import logging
logger = logging.getLogger(__name__)

try:
    import cdsapi
    has_cdsapi = True
except ImportError:
    has_cdsapi = False

# Model and Projection Settings
projection = 'latlong'

@contextmanager
def _get_data(target=None, product='reanalysis-era5-single-levels', chunks=None, **updates):

    if not has_cdsapi:
        raise RuntimeError(
            "Need installed cdsapi python package available from "
            "https://cds.climate.copernicus.eu/api-how-to"
        )

    # 'orography',
    request = {
        'product_type':'reanalysis',
        'format':'netcdf',
        'variable':[
            '100m_u_component_of_wind','100m_v_component_of_wind','2m_temperature',
            'forecast_surface_roughness','runoff','soil_temperature_level_4',
            'surface_net_solar_radiation','surface_pressure','surface_solar_radiation_downwards',
            'toa_incident_solar_radiation','total_sky_direct_solar_radiation_at_surface'
        ],
        'year':'2000',
        'month':[
            '01','02','03', '04','05','06', '07','08','09', '10','11','12'
        ],
        'day':[
            '01','02','03', '04','05','06', '07','08','09', '10','11','12',
            '13','14','15', '16','17','18', '19','20','21', '22','23','24',
            '25','26','27', '28','29','30', '31'
        ],
        'time':[
            '00:00','01:00','02:00', '03:00','04:00','05:00',
            '06:00','07:00','08:00', '09:00','10:00','11:00',
            '12:00','13:00','14:00', '15:00','16:00','17:00',
            '18:00','19:00','20:00', '21:00','22:00','23:00'
        ],
        'area': [50, -1, 49, 1], # North, West, South, East. Default: global
        'grid': [0.25, 0.25], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
    }
    request.update(updates)


    result = cdsapi.Client().retrieve(
        product,
        request
    )

    if target is None:
        fd, target = mkstemp(suffix='.nc')
        os.close(fd)

    logger.info("Downloading request for {} variables to {}".format(len(request['variable']), target))

    result.download(target)

    with xr.open_dataset(target, chunks=chunks) as ds:
        yield ds

    os.unlink(target)

def _add_height(ds):
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+surface+elevation+and+orography
    g = 9.80665
    z = ds.pop('z')
    if 'time' in z.coords:
        z = z.isel(time=0, drop=True)
    ds['height'] = z/g
    return ds

def _area(xs, ys):
    # North, West, South, East. Default: global
    return [ys.start, xs.start, ys.stop, xs.stop]

def _rename_and_clean_coords(ds, add_lon_lat=True):
    ds = ds.rename({'longitude': 'x', 'latitude': 'y'})
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def prepare_meta_era5(xs, ys, year, month, module):
    with _get_data(variable='orography',
                   year=year, month=month, day=1,
                   area=_area(xs, ys)) as ds:
        ds = _rename_and_clean_coords(ds)
        ds = _add_height(ds)

        t = pd.Timestamp(year=year, month=month, day=1)
        ds['time'] = pd.date_range(t, t + pd.DateOffset(months=1),
                                   freq='1h', closed='left')

        return ds.load()

def prepare_for_sarah(year, month, xs, ys, dx, dy, chunks=None):
    area = _area(xs, ys)
    grid = [dx, dy]

    with _get_data(area=area, grid=grid, year=year, month=month,
                   variable=['2m_temperature',
                             'toa_incident_solar_radiation',
                             'surface_solar_radiation_downwards',
                             'surface_net_solar_radiation'],
                   chunks=chunks) as ds:
        ds = _rename_and_clean_coords(ds, add_lon_lat=False)

        ds = ds.rename({'t2m': 'temperature'})
        ds = ds.rename({'tisr': 'influx_toa'})

        logger.debug("Calculate albedo")
        ds['albedo'] = (((ds['ssrd'] - ds['ssr'])/ds['ssrd']).fillna(0.)
                        .assign_attrs(units='(0 - 1)', long_name='Albedo'))
        ds = ds.drop(['ssrd', 'ssr'])

        logger.debug("Fixing units of influx_toa")
        # Convert from energy to power J m**-2 -> W m**-2
        ds['influx_toa'] /= 60.*60.
        ds['influx_toa'].attrs['units'] = 'W m**-2'

        logger.debug("Yielding ERA5 results")
        yield ds.chunk(chunks)
        logger.debug("Cleaning up ERA5")

def prepare_month_era5(year, month, xs, ys):
    area = _area(xs, ys)

    # https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation
    with _get_data(area=area, year=year, month=month,
                   variable=[
                       '100m_u_component_of_wind','100m_v_component_of_wind','2m_temperature',
                       'runoff','soil_temperature_level_4',
                       'surface_net_solar_radiation','surface_pressure','surface_solar_radiation_downwards',
                       'toa_incident_solar_radiation','total_sky_direct_solar_radiation_at_surface'
                   ]) as ds, \
         _get_data(area=area, year=year, month=month, day=1,
                   variable=['forecast_surface_roughness', 'orography']) as ds_m:

        ds_m = ds_m.isel(time=0, drop=True)
        ds = xr.merge([ds, ds_m], join='left')

        ds = _rename_and_clean_coords(ds)
        ds = _add_height(ds)

        # Help on the radiation quantities is in
        # https://www.ecmwf.int/sites/default/files/radiation_in_mars.pdf
        # TISR Top Incident Solar Radiation 212
        # SSRD Surface Solar Rad Downwards 169
        # SSR Surface net Solar Radiation 176
        # FDIR Direct solar radiation at surface 21.228
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

        ds['wnd100m'] = (np.sqrt(ds['u100']**2 + ds['v100']**2)
                        .assign_attrs(units=ds['u100'].attrs['units'],
                                    long_name="100 metre wind speed"))
        ds = ds.drop(['u100', 'v100'])

        # FSR Forecast surface roughness 244
        ds = ds.rename({'fsr': 'roughness'})

        # RO Runoff 205
        # T2m 2 metre temperature 167
        # SP Surface pressure 134
        # STL4 Soil temperature level 4 236
        ds = ds.rename({'ro': 'runoff',
                        't2m': 'temperature',
                        'sp': 'pressure',
                        'stl4': 'soil temperature'})

        yield (year, month), ds


def tasks_monthly_era5(xs, ys, yearmonths, prepare_func, meta_attrs):
    if not isinstance(xs, slice):
        xs = slice(*xs.values[[0, -1]])
    if not isinstance(ys, slice):
        ys = slice(*ys.values[[0, -1]])

    return [dict(prepare_func=prepare_func, xs=xs, ys=ys, year=year, month=month)
            for year, month in yearmonths]

weather_data_config = {
    '_': dict(tasks_func=tasks_monthly_era5,
              prepare_func=prepare_month_era5)
}

meta_data_config = dict(prepare_func=prepare_meta_era5)
