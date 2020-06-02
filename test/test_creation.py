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


ref = Cutout(path="creation_ref", module="era5", bounds=(x0, y0, x1, y1), time=time)


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



def test_time_sclice_coords():
    cutout = Cutout(path="time_slice", module="era5",
                    time=slice('2013-01-01', '2013-01-01'),
                    x=slice(x0, x1), y = slice(y0, y1))
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())



def test_dx_dy_dt():
    """
    Test the properties dx, dy, dt of atlite.Cutout. The spatial resolution
    can be changed through the creation_params dx and dy, the time resolution
    is hard coded (deep in the modules...) to one hour.
    """
    dx = 0.5
    dy = 1
    cutout = Cutout(path="resolution", module="era5",
                    time=slice('2013-01-01', '2013-01-01'),
                    x=slice(x0, x1), y = slice(y0, y1),
                    dx=dx, dy=dy)
    assert dx == cutout.dx
    assert dy == cutout.dy
    assert 'H' == cutout.dt


def test_available_features():
    modules = ref.available_features.index.unique('module')
    assert len(modules) == 1
    assert modules[0] == 'era5'

    cutout = Cutout(path="sarah_first", module=['sarah', 'era5'],
                    time=slice('2013-01-01', '2013-01-01'),
                    x=slice(x0, x1), y = slice(y0, y1))
    modules = cutout.available_features.index.unique('module')
    assert len(modules) == 2
    assert len(cutout.available_features) > len(ref.available_features)


def test_grid_coords():
    gcoords = ref.grid_coordinates()
    spatial = ref.data.stack(spatial=['y', 'x'])['spatial'].data
    spatial = np.array([[s[1], s[0]] for s in spatial])
    np.testing.assert_equal(gcoords, spatial)


def test_sel():
    cutout = ref.sel(x=slice(x0+2, x1-1), y=slice(y0+1, y1-2))
    assert cutout.coords['x'][0] - ref.coords['x'][0] == 2
    assert cutout.coords['y'][-1] - ref.coords['y'][-1] == -2

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

