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
import logging
from tempfile import mkdtemp
from shutil import rmtree


# %% Predefine tests for cutout

def all_notnull_test(cutout):
    """Test if no nan's in the prepared data occur"""
    assert cutout.data.notnull().all()


def prepared_features_test(cutout):
    """
    The prepared features series should contain all variables in cuttout.data
    """
    assert set(cutout.prepared_features) == set(cutout.data)



def pv_test(cutout):
    '''
    Test the atlite.Cutout.pv function with different settings. Compare
    optimal orientation with flat orientation.
    '''

    orientation = {'slope': 0.0, 'azimuth': 0.0}
    cap_factor = cutout.pv(atlite.resource.solarpanels.CdTe, orientation)

    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0

    production = cutout.pv(atlite.resource.solarpanels.CdTe, orientation,
                        layout=cap_factor)

    assert production.notnull().all()
    assert (production.sel(time=time+ ' 00:00') == 0)

    # Now compare with optimal orienation
    cap_factor_opt = cutout.pv(atlite.resource.solarpanels.CdTe, 'latitude_optimal')

    assert cap_factor_opt.sum() > cap_factor.sum()

    production_opt = cutout.pv(atlite.resource.solarpanels.CdTe, 'latitude_optimal',
                            layout=cap_factor_opt)

    assert (production_opt.sel(time=time+ ' 00:00') == 0)

    assert production_opt.sum() > production.sum()



def wind_test(cutout):
    '''
    Test the atlite.Cutout.wind function with two different layouts.
    The standard layout proportional to the capacity factors must have a lower
    production than a layout proportionally to the capacity layout squared.
    '''
    cap_factor = cutout.wind(atlite.windturbines.Enercon_E101_3000kW)

    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0

    production = cutout.wind(atlite.windturbines.Enercon_E101_3000kW,
                          layout=cap_factor)

    assert production.notnull().all()
    assert production.sum() > 0

    # Now create a better layout with same amount of installed power
    better_layout = (cap_factor**2)/(cap_factor**2).sum() * cap_factor.sum()
    better_production = cutout.wind(atlite.windturbines.Enercon_E101_3000kW,
                                 layout=better_layout)

    assert better_production.sum() > production.sum()


def runoff_test(cutout):
    '''
    Test the atlite.Cutout.runoff function.

    First check if the total of all capacity factors is not null.
    Then compare the runoff at sites which belong to the lower (altitude) half
    of the map, with the runoff at higher sites. The runoff at the lower sites
    (mostly at sea level) should have a smaller capacity factor total and
    production.
    '''
    cap_factor = cutout.runoff()
    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0

    q = np.quantile(cutout.data.height, 0.5)
    lower_area = cutout.data.height <= q
    higher_area = cutout.data.height > q
    assert cap_factor.where(lower_area).sum() < cap_factor.where(higher_area).sum()

    low_level_prod = cutout.runoff(layout=cap_factor.where(lower_area, 0))
    high_level_prod = cutout.runoff(layout=cap_factor.where(higher_area, 0))
    assert low_level_prod.sum() < high_level_prod.sum()


# I don't understand the problems with the crs and projection here leaving this
# out:

# def test_hydro():
#     plants = pd.DataFrame({'lon' : [x0, x1],
#                            'lat': [y0, y1]})
#     basins = gpd.GeoDataFrame(dict(geometry=[ref_era5.grid_cells[0], ref_era5.grid_cells[-1]],
#                                    HYBAS_ID = [0,1],
#                                    DIST_MAIN = 10,
#                                    NEXT_DOWN = None), index=[0,1], crs=dict(proj="aea"))
#     ref_era5.hydro(plants, basins)


# %% Prepare cutouts to test


time='2013-01-01'
x0 = -4
y0 = 56
x1 = 1.5
y1 = 61
sarah_dir = '/home/vres/climate-data/sarah_v2'


era5path="era5_test"
sarahpath='mixed_test'

tmp_dir = Path(mkdtemp())

ref_era5 = Cutout(path=tmp_dir / era5path, module="era5", bounds=(x0, y0, x1, y1),
                  time=time)
ref_era5.prepare()


if os.path.exists(sarah_dir):
    ref_sarah = Cutout(path=tmp_dir / sarahpath, module=["sarah", "era5"],
                       bounds=(x0, y0, x1, y1), time=time,
                       sarah_dir=sarah_dir)
    ref_sarah.prepare()
    test_sarah = True
else:
    logging.warn("'sarah_dir' is not a valid path, skipping tests for sarah module.")
    test_sarah = False


# %% Pure era5 tests

def test_data_module_arguments_era5():
    """
    All data variables should have an attribute to which module thay belong
    """
    for v in ref_era5.data:
        assert ref_era5.data.attrs['module'] == 'era5'


def test_compare_with_get_data_era5():
    """
    The prepared data should be exactly the same as from the low level function
    """
    influx = atlite.datasets.era5.get_data(ref_era5, 'influx', tmpdir=tmp_dir)
    assert_allclose(influx.influx_toa, ref_era5.data.influx_toa, atol=1e-5, rtol=1e-5)



# %% Apply predefined test functions to cutouts

def test_prepared_features_era5():
    return prepared_features_test(ref_era5)

def test_pv_era5():
    return pv_test(ref_era5)


def test_wind_era5():
    return wind_test(ref_era5)

def test_runoff_era5():
    return runoff_test(ref_era5)


if test_sarah:

    def test_prepared_features_sarah():
        return prepared_features_test(ref_sarah)

    def test_pv_sarah():
        return pv_test(ref_sarah)


    def test_wind_sarah():
        return wind_test(ref_sarah)

    def test_runoff_sarah():
        return runoff_test(ref_sarah)



# %% Finally delete the temporary directory

def test_dummy_delete_tmp_dir():
    '''
    Ignore this test its only purpose is to delete temporary cutout files
    created for the test run
    '''
    rmtree(tmp_dir)
