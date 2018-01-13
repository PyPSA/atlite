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
import tempfile
import shutil
from six.moves import range
from contextlib import contextmanager
import logging
logger = logging.getLogger(__name__)

try:
    from ecmwfapi import ECMWFDataServer
    has_ecmwfapi = True
except ImportError:
    has_ecmwfapi = False

# Model and Projection Settings
projection = 'latlong'

@contextmanager
def _get_data(target, chunks=None, **updates):
    if not has_ecmwfapi:
        raise RuntimeError(
            "Need installed ecmwfapi python package available from "
            "https://software.ecmwf.int/wiki/display/WEBAPI/Access+ECMWF+Public+Datasets"
        )

    server = ECMWFDataServer()
    request = {'target': target,
               'class': 'ea',
               'dataset': 'era5',
               'expver': '1',
               'stream': 'oper',
               'levtype': 'sfc',
               'grid': '0.3/0.3',
               'format': 'netcdf'}
    request.update(updates)
    server.retrieve(request)

    with xr.open_dataset(target, chunks=chunks) as ds:
        yield ds

    os.unlink(target)

def _add_height(ds):
    Gma = 6.673e-11*5.975e24/(6.378e6)**2
    z = ds['z']
    if 'time' in z.coords:
        z = z.isel(time=0, drop=True)
    ds['height'] = z/Gma
    ds = ds.drop('z')
    return ds

def _rename_and_clean_coords(ds, add_lon_lat=True):
    ds = ds.rename({'longitude': 'x', 'latitude': 'y'})
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def prepare_meta_era5(xs, ys, year, month, module):
    # Z Geopotential 129

    with _get_data('_meta.nc', type='an',
                   stream='moda',
                   date="{}-{:02}-01".format(year, month),
                   area='{}/{}/{}/{}'.format(ys.start, xs.start, ys.stop, xs.stop),
                   param='129') as ds:
        ds = _rename_and_clean_coords(ds)
        ds = _add_height(ds)

        t = pd.Timestamp(year=year, month=month, day=1)
        ds['time'] = pd.date_range(t, t + pd.offsets.MonthOffset(),
                                   freq='1h', closed='left')

        return ds.load()

def prepare_for_sarah(year, month, xs, ys, dx, dy, chunks=None):
    tmpdir = tempfile.mkdtemp()
    fns = [os.path.join(tmpdir, '_{}{:02}_{}.nc'.format(year, month, i)) for i in range(2)]

    tbeg = pd.Timestamp(year=year, month=month, day=1)
    tend = tbeg + pd.offsets.MonthEnd()
    def s(d): return d.strftime('%Y-%m-%d')
    date1 = s(tbeg)+"/to/"+s(tend)
    date2 = s(tbeg - pd.Timedelta(days=1))+"/to/"+s(tend) # Forecast data starts at 06:00 and 18:00
    area='{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(ys.start, xs.start, ys.stop, xs.stop)
    grid='{:.2f}/{:.2f}'.format(dx, dy)

    with _get_data(fns[0], type='an', date=date1, area=area, grid=grid,
                   time='/'.join('{:02}'.format(s) for s in range(0, 24)),
                   param='167', chunks=chunks) as ds, \
         _get_data(fns[1], type='fc', date=date2, area=area, grid=grid,
                   time='06:00:00/18:00:00',
                   step='1/2/3/4/5/6/7/8/9/10/11/12',
                   param='212/169/176', chunks={}) as ds_fc:
        ds = xr.merge([ds, ds_fc], join='left')
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

    shutil.rmtree(tmpdir)

def prepare_month_era5(year, month, xs, ys):
    tmpdir = tempfile.mkdtemp()

    fns = [os.path.join(tmpdir, '_{}{:02}_{}.nc'.format(year, month, i)) for i in range(3)]
    tbeg = pd.Timestamp(year=year, month=month, day=1)
    tend = tbeg + pd.offsets.MonthEnd()
    def s(d): return d.strftime('%Y-%m-%d')
    date1 = s(tbeg)+"/to/"+s(tend)
    date2 = s(tbeg - pd.Timedelta(days=1))+"/to/"+s(tend)
    date3 = s(tbeg)
    area='{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(ys.start, xs.start, ys.stop, xs.stop)

    # https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation
    with _get_data(fns[0], type='an', date=date1, area=area,
                   time='/'.join('{:02}'.format(s) for s in range(0, 24)),
                   param='134/167/246.228/247.228/236') as ds, \
         _get_data(fns[1], type='fc', date=date2, area=area,
                   time='06:00:00/18:00:00',
                   step='1/2/3/4/5/6/7/8/9/10/11/12',
                   param='212/169/176/21.228/205') as ds_fc, \
         _get_data(fns[2], type='an', date=date3, area=area,
                   stream='moda', # Monthly means of daily means
                   param='129/244') as ds_m:

        ds_m = ds_m.isel(time=0, drop=True)
        ds = xr.merge([ds, ds_fc, ds_m], join='left')

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

    shutil.rmtree(tmpdir)


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
