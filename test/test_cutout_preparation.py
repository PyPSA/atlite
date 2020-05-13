#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:15:41 2020

@author: fabian
"""

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import urllib3
urllib3.disable_warnings()

import atlite
from atlite import Cutout
from xarray.testing import assert_allclose, assert_equal
import numpy as np

time='2013-01-01'
x0 = -4
y0 = 56
x1 = 1.5
y1 = 61
path="era5_test"

tmp_dir = Path('tmp_files_test')
if tmp_dir.exists():
    for nc in tmp_dir.iterdir():
        os.remove(nc)
    tmp_dir.rmdir()
tmp_dir.mkdir()

ref = Cutout(path=tmp_dir / path, module="era5", bounds=(x0, y0, x1, y1), time=time)
ref.prepare()



def test_old_style_loading_args():
    'Test backwards compatibility with name and cutout_dir'
    cutout = Cutout(name=path, cutout_dir=tmp_dir)
    assert_equal(cutout.data.coords.to_dataset(), ref.data.coords.to_dataset())



def test_odd_bounds():
    '''
    Test cutout creation with odd bounds. The slightly shifted coordinates, e.g.
    55.94 instead of 56 for a lower bound, should not effect on the built
    coordinate system.
    '''
    cutout = Cutout(path = tmp_dir / (path + '_odd_bounds'), module="era5", time=time,
                    bounds=(x0-0.1, y0-0.02, x1+0.03, y1+0.13))
    cutout.prepare()
    assert_allclose(cutout.data.wnd100m, ref.data.wnd100m,
                    atol=1e-5, rtol=1e-5)


def test_compare_with_get_data_era5():
    period = pd.Period(time)
    influx = atlite.datasets.era5.get_data(ref.coords, period, 'influx', tmpdir=tmp_dir)
    influx = influx.compute()
    assert_allclose(influx.influx_toa, ref.data.influx_toa, atol=1e-5, rtol=1e-5)


def test_get_era5_data_in_sarah_module():
    period = pd.Period(time)
    influx = atlite.datasets.sarah.get_data_era5(ref.coords, period, 'influx',
                                tmpdir=tmp_dir, x=None, y=None, sarah_dir=None)
    influx = influx.compute()
    assert_allclose(influx.influx_toa, ref.data.influx_toa, atol=1e-5, rtol=1e-5)


def test_pv():
    '''
    Test the atlite.Cutout.pv function with different settings. Compare
    optimal orientation with flat orientation.
    '''

    orientation = {'slope': 0.0, 'azimuth': 0.0}
    cap_factor = ref.pv(atlite.resource.solarpanels.CdTe, orientation)

    assert cap_factor.sum() > 0

    production = ref.pv(atlite.resource.solarpanels.CdTe, orientation,
                        layout=cap_factor)

    assert (production.sel(time=time+ ' 00:00') == 0)

    # Now compare with optimal orienation
    cap_factor_opt = ref.pv(atlite.resource.solarpanels.CdTe, 'latitude_optimal')

    assert cap_factor_opt.sum() > cap_factor.sum()

    production_opt = ref.pv(atlite.resource.solarpanels.CdTe, 'latitude_optimal',
                            layout=cap_factor_opt)

    assert (production_opt.sel(time=time+ ' 00:00') == 0)

    assert production_opt.sum() > production.sum()



def test_wind():
    '''
    Test the atlite.Cutout.wind function with two different layouts.
    The standard layout proportional to the capacity factors must have a lower
    production than a layout proportionally to the capacity layout squared.
    '''
    cap_factor = ref.wind(atlite.windturbines.Enercon_E101_3000kW)

    assert cap_factor.sum() > 0

    production = ref.wind(atlite.windturbines.Enercon_E101_3000kW,
                          layout=cap_factor)

    assert production.sum() > 0

    # Now create a better layout with same amount of installed power
    better_layout = (cap_factor**2)/(cap_factor**2).sum() * cap_factor.sum()
    better_production = ref.wind(atlite.windturbines.Enercon_E101_3000kW,
                                 layout=better_layout)

    assert better_production.sum() > production.sum()


def test_runoff():
    '''
    Test the atlite.Cutout.runoff function.

    First check if the total of all capacity factors is not null.
    Then compare the runoff at sites which belong to the lower (altitude) half
    of the map, with the runoff at higher sites. The runoff at the lower sites
    (mostly at sea level) should have a smaller capacity factor total and
    production.
    '''
    cap_factor = ref.runoff()
    assert cap_factor.sum() > 0

    q = np.quantile(ref.data.height, 0.5)
    lower_area = ref.data.height <= q
    higher_area = ref.data.height > q
    assert cap_factor.where(lower_area).sum() < cap_factor.where(higher_area).sum()

    low_level_prod = ref.runoff(layout=cap_factor.where(lower_area, 0))
    high_level_prod = ref.runoff(layout=cap_factor.where(higher_area, 0))
    assert low_level_prod.sum() < high_level_prod.sum()


# I don't understand the problems with the crs and projection here leaving this
# out:

# def test_hydro():
#     plants = pd.DataFrame({'lon' : [x0, x1],
#                            'lat': [y0, y1]})
#     basins = gpd.GeoDataFrame(dict(geometry=[ref.grid_cells[0], ref.grid_cells[-1]],
#                                    HYBAS_ID = [0,1],
#                                    DIST_MAIN = 10,
#                                    NEXT_DOWN = None), index=[0,1], crs=dict(proj="aea"))
#     ref.hydro(plants, basins)


def test_dummy_delete_tmp_dir():
    '''
    Ignore this test its only purpose is to delete temporary cutout files
    created for the test run
    '''
    for nc in tmp_dir.iterdir():
        os.remove(nc)
    tmp_dir.rmdir()
