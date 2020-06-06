#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:15:41 2020

@author: fabian
"""

import os
import pytest
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
    assert (production.sel(time=TIME+ ' 00:00') == 0)

    # Now compare with optimal orienation
    cap_factor_opt = cutout.pv(atlite.resource.solarpanels.CdTe, 'latitude_optimal')

    assert cap_factor_opt.sum() > cap_factor.sum()

    production_opt = cutout.pv(atlite.resource.solarpanels.CdTe, 'latitude_optimal',
                            layout=cap_factor_opt)

    assert (production_opt.sel(time=TIME+ ' 00:00') == 0)

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

    height = cutout.data.height.load()
    q = np.quantile(height, 0.5)
    lower_area = height <= q
    higher_area = height > q
    assert cap_factor.where(lower_area).sum() < cap_factor.where(higher_area).sum()

    low_level_prod = cutout.runoff(layout=cap_factor.where(lower_area, 0))
    high_level_prod = cutout.runoff(layout=cap_factor.where(higher_area, 0))
    assert low_level_prod.sum() < high_level_prod.sum()


# I don't understand the problems with the crs and projection here leaving this
# out:

# def test_hydro():
#     plants = pd.DataFrame({'lon' : [x0, x1],
#                            'lat': [y0, y1]})
#     basins = gpd.GeoDataFrame(dict(geometry=[cutout_era5.grid_cells[0], cutout_era5.grid_cells[-1]],
#                                    HYBAS_ID = [0,1],
#                                    DIST_MAIN = 10,
#                                    NEXT_DOWN = None), index=[0,1], crs=dict(proj="aea"))
#     cutout_era5.hydro(plants, basins)


# %% Prepare cutouts to test


TIME = '2013-01-01'
BOUNDS = (-4, 56, 1.5, 61)
SARAH_DIR = os.getenv('SARAH_DIR', '/home/vres/climate-data/sarah_v2')


@pytest.fixture(scope='session')
def cutout_era5(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("era5")
    cutout = Cutout(path=tmp_path / "era5", module="era5", bounds=BOUNDS, time=TIME)
    cutout.prepare()
    return cutout

@pytest.fixture(scope='session')
def cutout_sarah(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sarah")
    cutout = Cutout(path=tmp_path / "sarah", module=["sarah", "era5"],
                    bounds=BOUNDS, time=TIME, sarah_dir=SARAH_DIR)
    cutout.prepare()
    return cutout





class TestERA5:
    @staticmethod
    def test_data_module_arguments_era5(cutout_era5):
        """
        All data variables should have an attribute to which module thay belong
        """
        for v in cutout_era5.data:
            assert cutout_era5.data.attrs['module'] == 'era5'

    @staticmethod
    def test_compare_with_get_data_era5(cutout_era5, tmp_path):
        """
        The prepared data should be exactly the same as from the low level function
        """
        influx = atlite.datasets.era5.get_data(cutout_era5, 'influx', tmpdir=tmp_path)
        assert_allclose(influx.influx_toa, cutout_era5.data.influx_toa, atol=1e-5, rtol=1e-5)

    @staticmethod
    def test_prepared_features_era5(cutout_era5):
        return prepared_features_test(cutout_era5)

    @staticmethod
    def test_pv_era5(cutout_era5):
        return pv_test(cutout_era5)

    @staticmethod
    def test_wind_era5(cutout_era5):
        return wind_test(cutout_era5)

    @staticmethod
    def test_runoff_era5(cutout_era5):
        return runoff_test(cutout_era5)


@pytest.mark.skipif(not os.path.exists(SARAH_DIR),
                    reason="'sarah_dir' is not a valid path")
class TestSarah:

    @staticmethod
    def test_prepared_features_sarah(cutout_sarah):
        return prepared_features_test(cutout_sarah)

    @staticmethod
    def test_pv_sarah(cutout_sarah):
        return pv_test(cutout_sarah)

    @staticmethod
    def test_wind_sarah(cutout_sarah):
        return wind_test(cutout_sarah)

    @staticmethod
    def test_runoff_sarah(cutout_sarah):
        return runoff_test(cutout_sarah)
