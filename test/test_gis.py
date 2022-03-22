#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
import functools
from atlite import Cutout
from atlite.gis import ExclusionContainer, shape_availability, pad_extent, regrid
from shapely.geometry import box
from atlite.gis import padded_transform_and_shape
from numpy import isclose, allclose
from xarray.testing import assert_allclose, assert_equal
import pandas as pd


TIME = "2013-01-01"
X0 = -4.0
Y0 = 56.0
X1 = 1.5
Y1 = 61.0
raster_clip = 0.25  # this rastio is excluded (True) in the raster


@pytest.fixture
def ref():
    return Cutout(
        path="creation_ref", module="era5", bounds=(X0, Y0, X1, Y1), time=TIME
    )


@pytest.fixture(scope="session")
def geometry(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("geometries")
    geometry = gpd.GeoSeries(
        [box(X0 / 2 + X1 / 2, Y0 / 2 + Y1 / 2, X1, Y1)],
        crs="EPSG:4326",
        index=[0],
        name="boxes",
    )
    geometry = geometry.to_frame().set_geometry("boxes")

    path = tmp_path / "geometry.gpkg"

    geometry.to_file(path, driver="GPKG")

    return path


@pytest.fixture(scope="session")
def raster(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rasters")
    bounds = (X0, Y0, X1, Y1)  # same as in test_gis.py
    res = 0.01
    transform, shape = padded_transform_and_shape(bounds, res)
    mask = np.random.rand(*shape) < raster_clip
    mask = mask.astype(rio.int32)
    path = tmp_path / "raster.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        transform=transform,
        crs=4326,
        width=shape[1],
        height=shape[0],
        count=1,
        dtype=mask.dtype,
    ) as dst:
        dst.write(mask, indexes=1)
    return path


@pytest.fixture(scope="session")
def raster_reproject(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rasters")
    bounds = rio.warp.transform_bounds(4326, 3035, X0, Y0, X1, Y1)
    res = 1000
    transform, shape = padded_transform_and_shape(bounds, res)
    mask = np.random.rand(*shape) < raster_clip
    mask = mask.astype(rio.int32)
    path = tmp_path / "raster_reproject.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        transform=transform,
        crs=3035,
        width=shape[1],
        height=shape[0],
        count=1,
        dtype=mask.dtype,
    ) as dst:
        dst.write(mask, indexes=1)
    return path


@pytest.fixture(scope="session")
def raster_codes(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rasters")
    bounds = (X0, Y0, X1, Y1)  # same as in test_gis.py
    res = 0.01
    transform, shape = padded_transform_and_shape(bounds, res)
    mask = (np.random.rand(*shape) * 100).astype(int)
    mask = mask.astype(rio.int32)
    path = tmp_path / "raster_codes.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        transform=transform,
        crs=4326,
        width=shape[1],
        height=shape[0],
        count=1,
        dtype=mask.dtype,
    ) as dst:
        dst.write(mask, indexes=1)
    return path


def test_open_closed_checks(ref, geometry, raster):
    """Test atlite.ExclusionContainer(...) file open/closed checks for plausibility. C.f. GH issue #225."""

    res = 0.01
    excluder = ExclusionContainer(ref.crs, res=res)

    # Without raster/shapes, both should evaluate to True
    assert excluder.all_closed and excluder.all_open

    # First add geometries, than raster
    excluder.add_geometry(geometry)
    assert excluder.all_closed and not excluder.all_open

    # Check if still works with 2nd geometry
    excluder.add_geometry(geometry)
    assert excluder.all_closed and not excluder.all_open

    excluder.add_raster(raster)
    assert excluder.all_closed and not excluder.all_open

    # Check if still works with 2nd raster
    excluder.add_raster(raster)
    assert excluder.all_closed and not excluder.all_open

    excluder.open_files()
    assert not excluder.all_closed and excluder.all_open

    # First add raster, then geometries
    excluder = ExclusionContainer(ref.crs, res=res)

    excluder.add_raster(raster)
    assert excluder.all_closed and not excluder.all_open

    # 2nd raster
    excluder.add_raster(raster)
    assert excluder.all_closed and not excluder.all_open

    excluder.add_geometry(geometry)
    assert excluder.all_closed and not excluder.all_open

    # 2nd geometry
    excluder.add_geometry(geometry)
    assert excluder.all_closed and not excluder.all_open

    excluder.open_files()
    assert not excluder.all_closed and excluder.all_open


def test_transform():
    """Test the affine transform. It always has to point to cell origin."""
    cutout = Cutout(
        path="resolution",
        module="era5",
        time=slice("2013-01-01", "2013-01-01"),
        x=slice(X0, X1),
        y=slice(Y0, Y1),
    )
    assert cutout.transform * (0.5, 0.5) == (X0, Y0)
    assert cutout.transform * cutout.shape[::-1] == (
        X1 + cutout.dx / 2,
        Y1 + cutout.dy / 2,
    )


def test_grid_coords(ref):
    gcoords = ref.grid[["x", "y"]]
    spatial = ref.data.stack(spatial=["y", "x"])["spatial"].data
    spatial = np.array([[s[1], s[0]] for s in spatial])
    np.testing.assert_equal(gcoords, spatial)


def test_extent(ref):
    pad = 0.25 / 2
    np.testing.assert_array_equal(ref.extent, [X0 - pad, X1 + pad, Y0 - pad, Y1 + pad])


# Note that bounds is the same as extent but in different order.
def test_bounds(ref):
    np.testing.assert_array_equal(ref.bounds, ref.grid.total_bounds)


def test_regrid():
    """Test the atlite.gis.regrid function with average resampling."""
    # define blocks
    A = 0.25
    B = 0.5
    C = 0.3
    D = 0.1
    ones = np.ones((4, 4))
    fine = np.block([[ones * A, ones * B], [ones * C, ones * D]])
    # add coordinates
    finecoords = np.arange(0.5, 8, 1)
    fine = xr.DataArray(fine, coords=[("y", finecoords), ("x", finecoords)])

    coarsecoords = np.arange(2, 8, 4)
    coarse = xr.DataArray(np.nan, coords=[("y", coarsecoords), ("x", coarsecoords)])

    # apply average resampling
    res = regrid(fine, coarse.x, coarse.y, resampling=5)
    target = np.array([[A, B], [C, D]])
    assert allclose(res, target)
    assert (coarse.x == res.x).all() and (coarse.y == res.y).all()

    # now test multiple layers

    fine = xr.concat([fine] * 10, pd.Index(range(10), name="z"))
    res = regrid(fine, coarse.x, coarse.y, resampling=5)
    target = np.stack([np.array([[A, B], [C, D]])] * 10)
    assert allclose(res, target)
    assert (coarse.x == res.x).all() and (coarse.y == res.y).all()

    # now let the target grid cover a subarea of the original
    fine = fine.sel(z=0, drop=True)
    coarsecoords = np.arange(1, 6, 2)
    coarse = xr.DataArray(np.nan, coords=[("y", coarsecoords), ("x", coarsecoords)])

    # apply average resampling
    res = regrid(fine, coarse.x, coarse.y, resampling=5)
    target = np.array([[A, A, B], [A, A, B], [C, C, D]])
    assert allclose(res, target)
    assert (coarse.x == res.x).all() and (coarse.y == res.y).all()


def test_pad_extent():
    """Test whether padding works with arrays of dimension > 2."""
    src = np.ones((3, 2))
    src_trans = rio.Affine(1, 0, 0, 0, 1, 0)
    dst_trans = rio.Affine(2, 0, 0, 0, 2, 0)
    crs = 4326

    padded, trans = pad_extent(src, src_trans, dst_trans, crs, crs)
    src = np.ones((1, 3, 2))
    padded_ndim, trans_ndim = pad_extent(src, src_trans, dst_trans, crs, crs)
    assert (padded_ndim[0] == padded).all()
    assert trans == trans_ndim

    # second check with large shape
    src = np.ones((1, 2, 3, 2))
    padded_ndim, trans_ndim = pad_extent(src, src_trans, dst_trans, crs, crs)
    assert (padded_ndim[0, 0] == padded).all()
    assert trans == trans_ndim

    # other way round, here it should not pad, since target resolution is lower
    padded_r, trans_r = pad_extent(src, dst_trans, src_trans, crs, crs)
    assert (padded_r == src).all()
    assert trans_r == dst_trans


def test_indicator_matrix(ref):
    # This should be the grid cell at the lower left corner
    cell = ref.grid.geometry[0]
    indicator = ref.indicatormatrix([cell])
    assert indicator[0, 0] == 1.0
    assert indicator.sum() == 1
    # This should be the grid cell at the lower left corner
    cell = ref.grid.geometry.iloc[-2]
    indicator = ref.indicatormatrix([cell])
    assert indicator[0, -2] == 1.0
    assert indicator.sum() == 1


def test_availability_matrix_flat(ref):
    """
    Indicator matrix and availability matrix for an empty excluder must be
    the same.
    """
    shapes = gpd.GeoSeries(
        [box(X0 + 1, Y0 + 1, X1 - 1, Y1 - 1)], crs=ref.crs
    ).rename_axis("shape")
    I = ref.indicatormatrix(shapes).sum(0).reshape(ref.shape)
    I = xr.DataArray(I, coords=[ref.coords["y"], ref.coords["x"]])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    ds = ref.availabilitymatrix(shapes, excluder)
    assert np.allclose(I, ds.sum("shape"))


def test_availability_matrix_flat_parallel(ref):
    """
    Same as `test_availability_matrix_flat` but parallel and without progressbar.
    """
    shapes = gpd.GeoSeries(
        [box(X0 + 1, Y0 + 1, X1 - 1, Y1 - 1)], crs=ref.crs
    ).rename_axis("shape")
    I = ref.indicatormatrix(shapes).sum(0).reshape(ref.shape)
    I = xr.DataArray(I, coords=[ref.coords["y"], ref.coords["x"]])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    ds = ref.availabilitymatrix(shapes, excluder, nprocesses=2)
    assert np.allclose(I, ds.sum("shape"))


def test_availability_matrix_flat_parallel_anonymous_function(ref, raster_codes):
    """
    Test availability matrix in parallel mode with a non-anonymous filter function.
    """
    shapes = gpd.GeoSeries(
        [box(X0 + 1, Y0 + 1, X1 - 1, Y1 - 1)], crs=ref.crs
    ).rename_axis("shape")
    I = ref.indicatormatrix(shapes).sum(0).reshape(ref.shape)
    I = xr.DataArray(I, coords=[ref.coords["y"], ref.coords["x"]])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    func = functools.partial(np.greater_equal, 20)
    excluder.add_raster(raster_codes, codes=func)
    ref.availabilitymatrix(shapes, excluder, nprocesses=2)


def test_availability_matrix_flat_wo_progressbar(ref):
    """
    Same as `test_availability_matrix_flat` but without progressbar.
    """
    shapes = gpd.GeoSeries(
        [box(X0 + 1, Y0 + 1, X1 - 1, Y1 - 1)], crs=ref.crs
    ).rename_axis("shape")
    I = ref.indicatormatrix(shapes).sum(0).reshape(ref.shape)
    I = xr.DataArray(I, coords=[ref.coords["y"], ref.coords["x"]])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    ds = ref.availabilitymatrix(shapes, excluder, disable_progressbar=True)
    assert np.allclose(I, ds.sum("shape"))


def test_availability_matrix_flat_parallel_wo_progressbar(ref):
    """
    Same as `test_availability_matrix_flat` but parallel and without progressbar.
    """
    shapes = gpd.GeoSeries(
        [box(X0 + 1, Y0 + 1, X1 - 1, Y1 - 1)], crs=ref.crs
    ).rename_axis("shape")
    I = ref.indicatormatrix(shapes).sum(0).reshape(ref.shape)
    I = xr.DataArray(I, coords=[ref.coords["y"], ref.coords["x"]])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    ds = ref.availabilitymatrix(
        shapes, excluder, nprocesses=2, disable_progressbar=True
    )
    assert np.allclose(I, ds.sum("shape"))


def test_shape_availability_area(ref):
    """Area of the mask and the shape must be close."""
    shapes = gpd.GeoSeries([box(X0 + 1, Y0 + 1, X1 - 1, Y1 - 1)], crs=ref.crs)
    res = 100
    excluder = ExclusionContainer(res=res)

    with pytest.raises(AssertionError):
        masked, transform = shape_availability(shapes, excluder)

    shapes = shapes.to_crs(3035)
    masked, transform = shape_availability(shapes, excluder)
    assert np.isclose(shapes.area, masked.sum() * res**2)


def test_exclusioncontainer_geometries():
    crs = 3035
    shapes = gpd.GeoSeries([box(X0, Y0, X1, Y1)], crs=crs)
    exclude = gpd.GeoSeries([box(X0 / 2 + X1 / 2, Y0 / 2 + Y1 / 2, X1, Y1)], crs=crs)
    res = 0.01

    excluder = ExclusionContainer(crs, res=res)
    excluder.add_geometry(exclude, buffer=1)
    excluder.open_files()
    assert (excluder.geometries[0]["geometry"] != exclude).all()
    excluder.open_files()
    buffered = excluder.geometries[0]["geometry"]
    # open again and check that the buffer remains the same
    assert (excluder.geometries[0]["geometry"] == buffered).all()

    # should take GeoDataFrames and the result is the same
    excluder = ExclusionContainer(crs, res=res)
    excluder.add_geometry(exclude.to_frame("geometry"), buffer=1)
    excluder.open_files()
    assert (excluder.geometries[0]["geometry"] == buffered).all()


def test_shape_availability_exclude_geometry(ref):
    """
    When excluding the quarter of the geometry, the eligible area must be a
    forth. Test the inverted case too.
    """
    shapes = gpd.GeoSeries([box(X0, Y0, X1, Y1)], crs=ref.crs)
    exclude = gpd.GeoSeries(
        [box(X0 / 2 + X1 / 2, Y0 / 2 + Y1 / 2, X1, Y1)], crs=ref.crs
    )
    res = 0.01

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_geometry(exclude)
    masked, transform = shape_availability(shapes, excluder)
    area = shapes.geometry[0].area  # get area without warning
    assert isclose(3 * area / 4, masked.sum() * res**2)

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_geometry(exclude, invert=True)
    masked, transform = shape_availability(shapes, excluder)
    area = shapes.geometry[0].area  # get area without warning
    assert isclose(area / 4, masked.sum() * res**2)


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
    assert round(ratio, 2) < (1 - raster_clip)

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, buffer=res, invert=True)
    masked, transform = shape_availability(shapes, excluder)
    ratio2 = masked.sum() / masked.size
    # for the case that we have more excluded area and this is buffered
    if raster_clip < 0.5:
        assert ratio >= ratio2
    else:
        assert ratio <= ratio2


def test_shape_availability_excluder_partial_overlap(ref, raster):
    """Test behavior, when a raster only overlaps half of the geometry."""
    bounds = X0 - 2, Y0, X0 + 2, Y1
    area = abs((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
    shapes = gpd.GeoSeries([box(*bounds)], crs=ref.crs)
    res = 0.01

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, codes=[0, 1])
    masked, transform = shape_availability(shapes, excluder)
    assert masked.sum() * (res**2) == area / 2

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, nodata=0)
    masked, transform = shape_availability(shapes, excluder)
    assert masked.sum() * (res**2) > area / 2

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, nodata=1)
    masked, transform = shape_availability(shapes, excluder)
    assert masked.sum() * (res**2) < area / 2


def test_shape_availability_excluder_raster_no_overlap(ref, raster):
    """Check if the allow_no_overlap flag works."""
    bounds = X0 - 10.0, Y0 - 10.0, X0 - 2.0, Y0 - 2.0
    area = abs((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
    shapes = gpd.GeoSeries([box(*bounds)], crs=ref.crs)
    res = 0.01

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster)
    with pytest.raises(ValueError):
        masked, transform = shape_availability(shapes, excluder)

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, allow_no_overlap=True)
    masked, transform = shape_availability(shapes, excluder)
    assert (masked == 0).all()

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, allow_no_overlap=True, codes=[1, 255], invert=True)
    masked, transform = shape_availability(shapes, excluder)
    assert masked.sum() * (res**2) == area

    excluder = ExclusionContainer(ref.crs, res=res)
    excluder.add_raster(raster, allow_no_overlap=True, nodata=0)
    masked, transform = shape_availability(shapes, excluder)
    assert masked.sum() * (res**2) == area


def test_availability_matrix_rastered(ref, raster):
    """
    Availability matrix with a non-zero raster must have less available area
    than the Indicator matrix.
    """
    shapes = gpd.GeoSeries(
        [
            box(X0 + 1, Y0 + 1, X1 - 1, Y0 / 2 + Y1 / 2),
            box(X0 + 1, Y0 / 2 + Y1 / 2, X1 - 1, Y1 - 1),
        ],
        crs=ref.crs,
    ).rename_axis("shape")
    I = np.asarray(ref.indicatormatrix(shapes).todense())
    I = I.reshape(shapes.shape + ref.shape)
    I = xr.DataArray(I, coords=[shapes.index, ref.coords["y"], ref.coords["x"]])
    excluder = ExclusionContainer(ref.crs, res=0.01)
    excluder.add_raster(raster)
    ds = ref.availabilitymatrix(shapes, excluder)
    eligible_share = 1 - raster_clip

    assert isclose(I.sum() * eligible_share, ds.sum(), atol=5)
    assert_allclose(I.sum(["x", "y"]) * eligible_share, ds.sum(["x", "y"]), atol=5)

    excluder = ExclusionContainer(ref.crs, res=0.01)
    excluder.add_raster(raster)
    assert_equal(ds, ref.availabilitymatrix(shapes, excluder, 2))


def test_availability_matrix_rastered_repro(ref, raster_reproject):
    """
    Availability matrix with a non-zero raster must have less available area
    than the Indicator matrix. Test this with a raster of a different crs.
    """
    shapes = gpd.GeoSeries(
        [
            box(X0 + 1, Y0 + 1, X1 - 1, Y0 / 2 + Y1 / 2),
            box(X0 + 1, Y0 / 2 + Y1 / 2, X1 - 1, Y1 - 1),
        ],
        crs=ref.crs,
    ).rename_axis("shape")
    I = np.asarray(ref.indicatormatrix(shapes).todense())
    I = I.reshape(shapes.shape + ref.shape)
    I = xr.DataArray(I, coords=[shapes.index, ref.coords["y"], ref.coords["x"]])
    excluder = ExclusionContainer()
    excluder.add_raster(raster_reproject)
    ds = ref.availabilitymatrix(shapes, excluder)
    eligible_share = 1 - raster_clip

    assert isclose(I.sum() * eligible_share, ds.sum(), atol=5)
    assert_allclose(I.sum(["x", "y"]) * eligible_share, ds.sum(["x", "y"]), atol=5)


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
    excluder.add_raster(raster_codes, codes=lambda x: x < 20, invert=True)
    masked, transform = shape_availability(shapes, excluder)
    assert ratio == masked.sum() / masked.size
