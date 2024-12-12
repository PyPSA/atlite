#!/usr/bin/env python3

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Created on Wed May  6 15:23:13 2020.

@author: fabian
"""

# IDEAS for tests

import numpy as np
import pytest
import rasterio as rio
from xarray.testing import assert_equal

from atlite import Cutout

TIME = "2013-01-01"
X0 = -4.0
Y0 = 56.0
X1 = 1.5
Y1 = 61.0


@pytest.fixture
def ref():
    return Cutout(
        path="creation_ref", module="era5", bounds=(X0, Y0, X1, Y1), time=TIME
    )


def test_name(ref):
    assert ref.name == "creation_ref"


def test_module(ref):
    assert ref.module == "era5"


def test_crs(ref):
    assert ref.crs == "EPSG:4326"


def test_shape(ref):
    assert ref.shape == (21, 23)


def test_extent(ref):
    reference_extent = [-4.125, 1.625, 55.875, 61.125]
    assert all([x == y for x, y in zip(ref.extent, reference_extent)])


def test_bounds(ref):
    reference_extent = [-4.125, 55.875, 1.625, 61.125]
    assert all([x == y for x, y in zip(ref.bounds, reference_extent)])


def test_transform(ref):
    reference_affine = rio.Affine(
        0.25, 0.00, -4.125, 0.00, 0.25, 55.875, 0.00, 0.00, 1.00
    )
    assert reference_affine == ref.transform


def test_equals(ref):
    test_cutout = Cutout(
        path="creation_ref", module="era5", bounds=(X0, Y0, X1, Y1), time=TIME
    )
    assert ref.equals(test_cutout)


def test_transform_r(ref):
    reference_affine = rio.Affine(
        0.25, 0.00, -4.125, 0.00, -0.25, 61.125, 0.00, 0.00, 1.00
    )
    assert reference_affine == ref.transform_r


def test_odd_bounds_coords(ref):
    cutout = Cutout(
        path="odd_bounds",
        module="era5",
        time=TIME,
        bounds=(X0 - 0.1, Y0 - 0.02, X1 + 0.03, Y1 + 0.13),
    )
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_xy_coords(ref):
    cutout = Cutout(
        path="xy", module="era5", time=TIME, x=slice(X0, X1), y=slice(Y0, Y1)
    )
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_xy_reversed_coords(ref):
    cutout = Cutout(
        path="xy_r", module="era5", time=TIME, x=slice(X1, X0), y=slice(Y1, Y0)
    )
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_time_sclice_coords(ref):
    cutout = Cutout(
        path="time_slice",
        module="era5",
        time=slice("2013-01-01", "2013-01-01"),
        x=slice(X0, X1),
        y=slice(Y0, Y1),
    )
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_auto_chunking(ref):
    cutout = Cutout(
        path="auto_chunking",
        module="era5",
        time=slice("2013-01-01", "2013-01-01"),
        x=slice(X0, X1),
        y=slice(Y0, Y1),
        chunks="auto",
    )
    assert_equal(cutout.coords.to_dataset(), ref.coords.to_dataset())


def test_dx_dy_dt():
    """
    Test the properties dx, dy, dt of atlite.Cutout.

    The spatial resolution can be changed through the creation_params dx
    and dy, the time resolution is hard coded (deep in the modules...)
    to one hour.
    """
    dx = 0.5
    dy = 1
    cutout = Cutout(
        path="resolution",
        module="era5",
        time=slice("2013-01-01", "2013-01-01"),
        x=slice(X0, X1),
        y=slice(Y0, Y1),
        dx=dx,
        dy=dy,
    )
    assert dx == cutout.dx
    assert dy == cutout.dy
    assert "h" == cutout.dt


def test_available_features(ref):
    modules = ref.available_features.index.unique("module")
    assert len(modules) == 1
    assert modules[0] == "era5"

    cutout = Cutout(
        path="sarah_first",
        module=["sarah", "era5"],
        time=slice("2013-01-01", "2013-01-01"),
        x=slice(X0, X1),
        y=slice(Y0, Y1),
    )
    modules = cutout.available_features.index.unique("module")
    assert len(modules) == 2
    assert len(cutout.available_features) > len(ref.available_features)


def test_sel(ref):
    cutout = ref.sel(x=slice(X0 + 2, X1 - 1), y=slice(Y0 + 1, Y1 - 2))
    assert cutout.coords["x"][0] - ref.coords["x"][0] == 2
    assert cutout.coords["y"][-1] - ref.coords["y"][-1] == -2


def test_layout_from_area_density(ref):
    density = 0.01  # MW / m^2
    crs = 3035  # in meter
    layout = ref.uniform_density_layout(density, crs=crs)
    assert layout.dims == ("y", "x")
    assert layout.shape == (ref.data.y.size, ref.data.x.size)
    assert np.isclose(ref.area(crs=crs).sum() * density, layout.sum())
