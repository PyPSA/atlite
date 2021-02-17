#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:23:13 2020

@author: fabian
"""

# IDEAS for tests

import pytest
import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio as rio
import rasterio.warp
from atlite import Cutout
from atlite.gis import ExclusionContainer, shape_availability
from shapely.geometry import box
from atlite.gis import padded_transform_and_shape
from numpy import isclose
from xarray.testing import assert_allclose, assert_equal


TIME = '2013-01-01'
X0 = -4.
Y0 = 56.
X1 = 1.5
Y1 = 61.
raster_clip = 0.25


@pytest.fixture
def ref():
    return Cutout(path="creation_ref", module="era5", bounds=(X0, Y0, X1, Y1), time=TIME)

@pytest.fixture(scope='session')
def raster(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rasters")
    bounds = (X0, Y0, X1, Y1) # same as in test_gis.py
    res = 0.01
    transform, shape = padded_transform_and_shape(bounds, res)
    mask = np.random.rand(*shape) < raster_clip
    mask = mask.astype(rio.int32)
    path = tmp_path / 'raster.tif'
    with rio.open(path, 'w', driver='GTiff', transform=transform, crs=4326,
                  width=shape[1], height=shape[0], count=1, dtype=mask.dtype) as dst:
        dst.write(mask, indexes=1)
    return path

@pytest.fixture(scope='session')
def raster_reproject(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rasters")
    bounds = rio.warp.transform_bounds(4326, 3035, X0, Y0, X1, Y1)
    res = 1000
    transform, shape = padded_transform_and_shape(bounds, res)
    mask = np.random.rand(*shape) < raster_clip
    mask = mask.astype(rio.int32)
    path = tmp_path / 'raster_reproject.tif'
    with rio.open(path, 'w', driver='GTiff', transform=transform, crs=3035,
                  width=shape[1], height=shape[0], count=1, dtype=mask.dtype) as dst:
        dst.write(mask, indexes=1)
    return path

@pytest.fixture(scope='session')
def raster_codes(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rasters")
    bounds = (X0, Y0, X1, Y1) # same as in test_gis.py
    res = 0.01
    transform, shape = padded_transform_and_shape(bounds, res)
    mask = (np.random.rand(*shape) * 100).astype(int)
    mask = mask.astype(rio.int32)
    path = tmp_path / 'raster_codes.tif'
    with rio.open(path, 'w', driver='GTiff', transform=transform, crs=4326,
                  width=shape[1], height=shape[0], count=1, dtype=mask.dtype) as dst:
        dst.write(mask, indexes=1)
    return path


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
    assert np.allclose(I, ds.sum('shape'))


def test_shape_availability_area(ref):
    """Area of the mask and the shape must be close."""
    shapes = gpd.GeoSeries([box(X0+1, Y0+1, X1-1, Y1-1)], crs=ref.crs)
    res = 100
    excluder = ExclusionContainer(res=res)

    with pytest.raises(AssertionError):
        masked, transform = shape_availability(shapes, excluder)

    shapes = shapes.to_crs(3035)
    masked, transform = shape_availability(shapes, excluder)
    assert np.isclose(shapes.area, masked.sum() * res ** 2)


def test_exclusioncontainer_geometries():
    crs = 3035
    shapes = gpd.GeoSeries([box(X0, Y0, X1, Y1)], crs=crs)
    exclude = gpd.GeoSeries([box(X0/2+X1/2, Y0/2+Y1/2, X1, Y1)], crs=crs)
    res = 0.01

    excluder = ExclusionContainer(crs, res=res)
    excluder.add_geometry(exclude, buffer=1)
    excluder.open_files()
    assert (excluder.geometries[0]['geometry'] != exclude).all()
    excluder.open_files()
    buffered = excluder.geometries[0]['geometry']
    # open again and check that the buffer remains the same
    assert (excluder.geometries[0]['geometry'] == buffered).all()

    # should take GeoDataFrames and the result is the same
    excluder = ExclusionContainer(crs, res=res)
    excluder.add_geometry(exclude.to_frame('geometry'), buffer=1)
    excluder.open_files()
    assert (excluder.geometries[0]['geometry'] == buffered).all()



def test_shape_availability_exclude_geometry(ref):
    """
    When excluding the quarter of the geometry, the eligible area must be a
    forth. Test the inverted case too.
    """
    shapes = gpd.GeoSeries([box(X0, Y0, X1, Y1)], crs=ref.crs)
    exclude = gpd.GeoSeries([box(X0/2+X1/2, Y0/2+Y1/2, X1, Y1)], crs=ref.crs)
    res = 0.01

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_geometry(exclude)
    masked, transform = shape_availability(shapes, excluder)
    area = shapes.geometry[0].area # get area without warning
    assert isclose(3*area/4, masked.sum() * res ** 2)

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_geometry(exclude, invert=True)
    masked, transform = shape_availability(shapes, excluder)
    area = shapes.geometry[0].area # get area without warning
    assert isclose(area/4, masked.sum() * res ** 2)



def test_shape_availability_exclude_raster(ref, raster):
    """When excluding the half of the geometry, the eligible area must be half."""
    shapes = gpd.GeoSeries([box(X0, Y0, X1, Y1)], crs=ref.crs)
    res = 0.01

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster)
    masked, transform = shape_availability(shapes, excluder)
    ratio = masked.sum() / masked.size
    assert round(ratio, 2) == (1 - raster_clip)

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, invert=True)
    masked, transform = shape_availability(shapes, excluder)
    ratio = masked.sum() / masked.size
    assert round(ratio, 2) == raster_clip

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, buffer=res)
    masked, transform = shape_availability(shapes, excluder)
    ratio = masked.sum() / masked.size
    # should be close to zero
    assert round(ratio, 2) < (1-raster_clip)

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, buffer=res, invert=True)
    masked, transform = shape_availability(shapes, excluder)
    ratio2 = masked.sum() / masked.size
    # for the case that we have more excluded area and this is buffered
    if raster_clip < 0.5:
        assert ratio >= ratio2
    else:
        assert ratio <= ratio2



def test_availability_matrix_rastered(ref, raster):
    """
    Availability matrix with a non-zero raster must have less available area
    than the Indicator matrix.
    """
    shapes = gpd.GeoSeries([box(X0+1, Y0+1, X1-1, Y0/2 + Y1/2),
                            box(X0+1, Y0/2 + Y1/2, X1-1, Y1-1)],
                           crs=ref.crs)
    I = np.asarray(ref.indicatormatrix(shapes).todense())
    I = I.reshape(shapes.shape + ref.shape)
    I = xr.DataArray(I, coords=[('shape', shapes.index), ref.coords['y'],
                                ref.coords['x']])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    excluder.add_raster(raster)
    ds = ref.availabilitymatrix(shapes, excluder)
    eligible_share = 1 - raster_clip

    assert isclose(I.sum() * eligible_share, ds.sum(), atol=5)
    assert_allclose(I.sum(['x', 'y']) * eligible_share, ds.sum(['x', 'y']), atol=5)

    # check parallel mode
    excluder = ExclusionContainer(ref.crs, res=0.01)
    excluder.add_raster(raster)
    assert_equal(ds, ref.availabilitymatrix(shapes, excluder, 2))


def test_availability_matrix_rastered_repro(ref, raster_reproject):
    """
    Availability matrix with a non-zero raster must have less available area
    than the Indicator matrix. Test this with a raster of a different crs.
    """
    shapes = gpd.GeoSeries([box(X0+1, Y0+1, X1-1, Y0/2 + Y1/2),
                            box(X0+1, Y0/2 + Y1/2, X1-1, Y1-1)],
                            crs=ref.crs)
    I = np.asarray(ref.indicatormatrix(shapes).todense())
    I = I.reshape(shapes.shape + ref.shape)
    I = xr.DataArray(I, coords=[('shape', shapes.index), ref.coords['y'],
                                ref.coords['x']])
    excluder = ExclusionContainer()
    excluder.add_raster(raster_reproject)
    ds = ref.availabilitymatrix(shapes, excluder)
    eligible_share = 1 - raster_clip

    assert isclose(I.sum() * eligible_share, ds.sum(), atol=5)
    assert_allclose(I.sum(['x', 'y']) * eligible_share, ds.sum(['x', 'y']), atol=5)



def test_shape_availability_exclude_raster_codes(ref, raster_codes):
    """Test exclusion of multiple raster codes."""
    shapes = gpd.GeoSeries([box(X0, Y0, X1, Y1)], crs=ref.crs)
    res = 0.01

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster_codes, codes=range(20))
    masked, transform = shape_availability(shapes, excluder)
    ratio = masked.sum() / masked.size
    assert round(ratio, 1) == 0.8

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster_codes, codes=range(20), invert=True)
    masked, transform = shape_availability(shapes, excluder)
    ratio = masked.sum() / masked.size
    assert round(ratio, 1) == 0.2

    # test with a function
    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster_codes, codes=lambda x: x<20, invert=True)
    masked, transform = shape_availability(shapes, excluder)
    assert ratio == masked.sum() / masked.size

