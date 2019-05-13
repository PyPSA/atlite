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

from ..utils import timeindex_from_slice

# Model and Projection Settings
projection = 'latlong'

@contextmanager
def _get_data(target=None, product='reanalysis-era5-single-levels', chunks=None, **updates):
    """Download ERA5 data from the Climate Data Store (CDS)"""

    if not has_cdsapi:
        raise RuntimeError(
            "Need installed cdsapi python package available from "
            "https://cds.climate.copernicus.eu/api-how-to"
        )

    # Default request
    request = {
        'product_type':'reanalysis',
        'format':'netcdf',
        'day':[
            '01','02','03','04','05','06','07','08','09','10','11','12',
            '13','14','15','16','17','18','19','20','21','22','23','24',
            '25','26','27','28','29','30','31'
        ],
        'time':[
            '00:00','01:00','02:00','03:00','04:00','05:00',
            '06:00','07:00','08:00','09:00','10:00','11:00',
            '12:00','13:00','14:00','15:00','16:00','17:00',
            '18:00','19:00','20:00','21:00','22:00','23:00'
        ],
        # 'area': [50, -1, 49, 1], # North, West, South, East. Default: global
        # 'grid': [0.25, 0.25], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
    }
    request.update(updates)

    assert {'year', 'month', 'variable'}.issubset(request), "Need to specify at least 'variable', 'year' and 'month'"

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

    time = timeindex_from_slice(time)

    ds = xr.Dataset({'longitude': np.r_[-180:180:0.3], 'latitude': np.r_[90:-90:-0.3]})
    ds = _rename_and_clean_coords(ds)
    ds = ds.sel(x=x, y=y)

    ds['time'] = time

def get_data(coords, date, feature, x, y, chunks=None, **creation_parameters):
    kwds = {'chunks': chunks, 'area': _area(x, y), 'year': date.year, 'month': date.month}

    if {'dx', 'dy'}.issubset(creation_parameters):
        kwds['grid'] = [creation_parameters['dx'], creation_parameters['dy']]

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

    if feature == "wind":
        with _get_data(variable=['100m_u_component_of_wind',
                                 '100m_v_component_of_wind',
                                 'forecast_surface_roughness'],
                       **kwds) as ds:

            ds = _rename_and_clean_coords(ds)

            ds['wnd100m'] = (np.sqrt(ds['u100']**2 + ds['v100']**2)
                            .assign_attrs(units=ds['u100'].attrs['units'],
                                        long_name="100 metre wind speed"))
            ds = ds.drop(['u100', 'v100'])

            ds = ds.rename({'fsr': 'roughness'})

            yield ds

    elif feature == "influx":

        with _get_data(variable=['surface_net_solar_radiation',
                                 'surface_solar_radiation_downwards',
                                 'toa_incident_solar_radiation',
                                 'total_sky_direct_solar_radiation_at_surface'],
                       **kwds) as ds:

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

            yield ds

    elif feature == "temperature":

        with _get_data(variable=['2m_temperature', 'soil_temperature_level_4'], **kwds) as ds:

            ds = _rename_and_clean_coords(ds)
            ds = ds.rename({'t2m': 'temperature', 'stl4': 'soil temperature'})

            yield ds

    elif feature == "runoff":

        with _get_data(variable=['runoff'], **kwds) as ds:

            ds = _rename_and_clean_coords(ds)
            ds = ds.rename({'ro': 'runoff'})

            yield ds

    elif feature == "height":
        with _get_data(variable='orography', day=1, time="00:00",
                       **kwds) as ds:

            ds = _rename_and_clean_coords(ds)
            ds = _add_height(ds)

            yield ds

    else:
        raise NotImplementedError(f"Feature '{feature}' has not been implemented for dataset era5")
