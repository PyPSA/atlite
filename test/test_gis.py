#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:23:13 2020

@author: fabian
"""

# IDEAS for tests

import pytest
import pandas as pd
import geopandas as gpd
import xarray as xr
import atlite
import numpy as np
from atlite import Cutout
from atlite.gis import ExclusionContainer
from xarray.testing import assert_allclose, assert_equal
from shapely.geometry import box


TIME = '2013-01-01'
X0 = -4.
Y0 = 56.
X1 = 1.5
Y1 = 61.


@pytest.fixture
def ref():
    return Cutout(path="creation_ref", module="era5", bounds=(X0, Y0, X1, Y1), time=TIME)


def test_transform():
    """Test the affine transform. It always has to point to cell origin."""
    cutout = Cutout(path="resolution", module="era5",
                    time=slice('2013-01-01', '2013-01-01'),
                    x=slice(X0, X1), y = slice(Y0, Y1))
    assert cutout.transform * (0.5, 0.5) == (X0, Y0)
    assert cutout.transform * cutout.shape[::-1] == (X1 + cutout.dx/2,
                                                     Y1 + cutout.dy/2)



def test_grid_coords(ref):
    gcoords = ref.grid[['x', 'y']]
    spatial = ref.data.stack(spatial=['y', 'x'])['spatial'].data
    spatial = np.array([[s[1], s[0]] for s in spatial])
    np.testing.assert_equal(gcoords, spatial)


# Extent is different from bounds
def test_extent(ref):
    np.testing.assert_array_equal(ref.extent,[X0, X1, Y0, Y1])


def test_indicator_matrix(ref):
    # This should be the grid cell at the lower left corner
    cell = ref.grid.geometry[0]
    indicator = ref.indicatormatrix([cell])
    assert indicator[0, 0] == 1.
    assert indicator.sum() == 1
    # This should be the grid cell at the lower left corner
    cell = ref.grid.geometry.iloc[-2]
    indicator = ref.indicatormatrix([cell])
    assert indicator[0, -2] == 1.
    assert indicator.sum() == 1


def test_availability_matrix_flat(ref):
    """
    Indicator matrix and availability matrix for an empty excluder must be
    the same.
    """
    shapes = gpd.GeoSeries([box(X0+1, Y0+1, X1-1, Y1-1)], crs=ref.crs)
    I = ref.indicatormatrix(shapes).sum(0).reshape(ref.shape)
    I = xr.DataArray(I, coords=[ref.coords['y'], ref.coords['x']])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    ds = ref.availabilitymatrix(shapes, excluder)
    assert np.allclose(I, ds.sum('shapes'))


# def test_availability_matrix_rastered(ref):
#     """
#     Indicator matrix and availability matris for an empty excluder must be
#     the same.
#     """
#     shapes = gpd.GeoSeries([box(X0+1, Y0+1, X1-1, Y1-1)], crs=ref.crs)
#     I = ref.indicatormatrix(shapes).sum(0).reshape(ref.shape)
#     I = xr.DataArray(I, coords=[ref.coords['y'], ref.coords['x']])
#     ds = ref.availabilitymatrix(shapes, excluder())
#     assert np.allclose(I, ds.sum('shapes'), rtol=1e-4, atol=0.06)

