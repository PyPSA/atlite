#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:15:41 2020

@author: fabian
"""

import os
import xarray as xr
from xarray import Dataset
import pandas as pd
import urllib3
from atlite.datasets.sarah import get_data_era5
urllib3.disable_warnings()

import atlite
from atlite import Cutout
from xarray.testing import assert_allclose, assert_equal


time='2013-01-01'
x0 = -4
y0 = 56
x1 = 1.5
y1 = 61


ref = Cutout(path="era5_test", module="era5", bounds=(x0, y0, x1, y1), time=time)

# assert_allclose(era5.data.wnd100m, era5_odd_bounds.data.wnd100m,
#                 atol=1e-5, rtol=1e-5)


# period = pd.Period(time)
# influx = atlite.datasets.era5.get_data(era5.coords, period, 'influx', tmpdir='era5_temps')
# influx = influx.compute()


# period = pd.Period(time)
# influx = atlite.datasets.era5.get_data(era5.coords, period, 'influx', tmpdir='era5_temps')
# influx = influx.compute()

# temperature = atlite.datasets.era5.get_data(era5.coords, period, 'temperature', tmpdir='era5_temps')
# temperature.compute()
# %%

# if os.path.exists('sarah_test.nc'):
#     os.remove('sarah_test.nc')
# creation_params = dict(time="2013-12-03",
#                        x = [-14, 1.5], y = [50, 61],
#                        sarah_dir='/home/vres/climate-data/sarah_v2')
# sarah = atlite.Cutout(path="sarah_test", module="sarah", **creation_params)

# sarah.prepare()

# data = atlite.datasets.sarah.get_data_sarah(sarah.coords, pd.Period('2013'),
#                                             **creation_params).compute()
# data.influx_direct.isnull().sum().compute()



# sarah.data.influx_direct.sel(time=sarah.data.time[12]).plot()


# import os
# import atlite

# cutout = atlite.Cutout(path="../cutouts/test-cutout-sarah2_europe.nc",
#         bounds=(-12, 41.8, 33. 64.8),
#         module="sarah",
#         time=slice("2013-01-01", "2013-12-31"),
#         sarah_dir='/home/vres/climate-data/sarah_v2')

# cutout.prepare()

