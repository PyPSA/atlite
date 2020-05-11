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
path="era5_test"

if os.path.exists(path + '.nc'):
    os.remove(path + '.nc')

ref = Cutout(path=path, module="era5", bounds=(x0, y0, x1, y1), time=time)
ref.prepare(overwrite=True)

# Backwards compatibility with name and cutout_dir
def test_old_style_loading_args():
    cutout = Cutout(name=path, cutout_dir='.')
    assert_equal(cutout.data.coords.to_dataset(), ref.data.coords.to_dataset())

    cutout = Cutout(name=path)
    assert_equal(cutout.data.coords.to_dataset(), ref.data.coords.to_dataset())


def test_odd_bounds():
    cutout = Cutout(path = path + '_odd_bounds', module="era5", time=time,
                    bounds=(x0-0.1, y0-0.02, x1+0.03, y1+0.13))
    cutout.prepare(overwrite=True)
    assert_allclose(cutout.data.wnd100m, ref.data.wnd100m,
                    atol=1e-5, rtol=1e-5)


def test_compare_with_get_data_era5():
    period = pd.Period(time)
    influx = atlite.datasets.era5.get_data(ref.coords, period, 'influx', tmpdir='.')
    influx = influx.compute()
    assert_allclose(influx.influx_toa, ref.data.influx_toa, atol=1e-5, rtol=1e-5)


def test_get_era5_data_in_sarah_module():
    period = pd.Period(time)
    influx = atlite.datasets.sarah.get_data_era5(ref.coords, period, 'influx',
                                                 tmpdir='.', x=None, y=None, sarah_dir=None)
    influx = influx.compute()
    assert_allclose(influx.influx_toa, ref.data.influx_toa, atol=1e-5, rtol=1e-5)


