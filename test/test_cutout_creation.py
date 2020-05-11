#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:23:13 2020

@author: fabian
"""

# IDEAS for tests

import pandas as pd
import atlite
from atlite import Cutout
from xarray.testing import assert_allclose, assert_equal
import numpy as np


time='2013-01-01'
x0 = -4.
y0 = 56.
x1 = 1.5
y1 = 61.


ref = Cutout(name="creation_ref", module="era5", bounds=(x0, y0, x1, y1), time=time)

new_feature_tests = False

def test_odd_bounds_coords():
    cutout = Cutout(path="odd_bounds", module="era5", time=time,
                    bounds=(x0-0.1, y0-0.02, x1+0.03, y1+0.13))
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_xy_coords():
    cutout = Cutout(path="xy", module="era5", time=time,
                    x=slice(x0, x1), y = slice(y0, y1))
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_xy_reversed_coords():
    cutout = Cutout(path="xy_r", module="era5", time=time,
                    x=slice(x1, x0), y = slice(y1, y0))
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


if new_feature_tests:
    def test_xy_tuple_coords():
        cutout = Cutout(path="xy_tuple", module="era5", time=time,
                        x=(-14.1, 1.512), y =(50.94, 61.123))
        assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_time_sclice_coords():
    cutout = Cutout(path="time_slice", module="era5",
                    time=slice('2013-01-01', '2013-01-01'),
                    x=slice(x0, x1), y = slice(y0, y1))
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


if new_feature_tests:
    def test_time_tuple_coords():
        cutout = Cutout(path="time_tuple", module="era5",
                        time=('2013-01-01', '2013-01-01'),
                        x=slice(x0, x1), y = slice(y0, y1))
        assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


if new_feature_tests:
    def test_time_period_coords():
        cutout = Cutout(path="time_period", module="era5",
                        time=pd.Period('2013-01-01'),
                        x=slice(x0, x1), y = slice(y0, y1))
        assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_module_assignment():
    assert ref.dataset_module == atlite.datasets.era5


def test_grid_coords():
    gcoords = ref.grid_coordinates()
    spatial = ref.data.stack(spatial=['y', 'x'])['spatial'].data
    spatial = np.array([[s[1], s[0]] for s in spatial])
    np.testing.assert_equal(gcoords, spatial)


# Extent is different from bounds
def test_extent():
    np.testing.assert_array_equal(ref.extent,[x0, x1, y0, y1])


def test_indicator_matrix():
    # This should be the grid cell at the lower left corner
    cell = ref.grid_cells[0]
    indicator = ref.indicatormatrix([cell])
    assert indicator[0, 0] == 1.
    assert indicator.sum() == 1
    # This should be the grid cell at the lower left corner
    cell = ref.grid_cells[-2]
    indicator = ref.indicatormatrix([cell])
    assert indicator[0, -2] == 1.
    assert indicator.sum() == 1

