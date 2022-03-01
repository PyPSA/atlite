#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Created on Mon May 11 11:15:41 2020

@author: fabian
"""

import os
import sys
import pytest
import urllib3
import geopandas as gpd

urllib3.disable_warnings()

import atlite
from atlite import Cutout
from shapely.geometry import Point, LineString as Line
from xarray.testing import assert_allclose, assert_equal
import numpy as np
import pandas as pd


# %% Predefine tests for cutout


def all_notnull_test(cutout):
    """Test if no nan's in the prepared data occur"""
    assert cutout.data.notnull().all()


def prepared_features_test(cutout):
    """
    The prepared features series should contain all variables in cuttout.data
    """
    assert set(cutout.prepared_features) == set(cutout.data)


def update_feature_test(cutout, red):
    """atlite should be able to overwrite a feature."""
    red.data = cutout.data.drop_vars("influx_direct")
    red.prepare("influx", overwrite=True)
    assert_equal(red.data.influx_direct, cutout.data.influx_direct)


def merge_test(cutout, other, target_modules):
    merge = cutout.merge(other, compat="override")
    assert set(merge.module) == set(target_modules)


def wrong_recreation(cutout):
    with pytest.warns(UserWarning):
        Cutout(path=cutout.path, module="somethingelse")


def pv_test(cutout):
    """
    Test the atlite.Cutout.pv function with different settings. Compare
    optimal orientation with flat orientation.
    """

    orientation = {"slope": 0.0, "azimuth": 0.0}
    cap_factor = cutout.pv(atlite.resource.solarpanels.CdTe, orientation)

    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0

    production = cutout.pv(
        atlite.resource.solarpanels.CdTe, orientation, layout=cap_factor
    )

    assert production.notnull().all()
    assert production.sel(time=TIME + " 00:00") == 0

    cells = cutout.grid
    cells = cells.assign(regions=["lower"] * 200 + ["upper"] * (len(cells) - 200))
    shapes = cells.dissolve("regions")
    production, capacity = cutout.pv(
        atlite.resource.solarpanels.CdTe,
        orientation,
        layout=cap_factor,
        shapes=shapes,
        return_capacity=True,
    )
    cap_per_region = (
        cells.assign(cap_factor=cap_factor.stack(spatial=["y", "x"]).values)
        .groupby("regions")
        .cap_factor.sum()
    )

    assert all(cap_per_region.round(3) == capacity.round(3))

    # Now compare with optimal orienation
    cap_factor_opt = cutout.pv(atlite.resource.solarpanels.CdTe, "latitude_optimal")

    assert cap_factor_opt.sum() > cap_factor.sum()

    production_opt = cutout.pv(
        atlite.resource.solarpanels.CdTe, "latitude_optimal", layout=cap_factor_opt
    )

    assert production_opt.sel(time=TIME + " 00:00") == 0

    assert production_opt.sum() > production.sum()

    # now use the non simple trigon model
    production_other = cutout.pv(
        atlite.resource.solarpanels.CdTe,
        "latitude_optimal",
        layout=cap_factor_opt,
        trigon_model="other",
    )

    assert production_other.sel(time=TIME + " 00:00") == 0
    # should be roughly the same
    assert (production_other.sum() / production_opt.sum()).round(0) == 1

    # now another solarpanel with bofinger model
    production_other = cutout.pv(
        atlite.resource.solarpanels.KANENA,
        "latitude_optimal",
        layout=cap_factor_opt,
    )

    assert production_other.sel(time=TIME + " 00:00") == 0
    # should be roughly the same
    assert (production_other.sum() / production_opt.sum()).round(0) == 1


def csp_test(cutout):
    """
    Test the atlite.Cutout.csp function with different for different
    settings and technologies.
    """

    ## Test technology = "solar tower"
    st = cutout.csp(atlite.cspinstallations.SAM_solar_tower, capacity_factor=True)

    assert st.notnull().all()
    assert (st >= 0).all()
    assert (st <= 1).all()

    # Efficiencies <= 1 should lead to the conversion to always be less than perfect
    st = cutout.csp(atlite.cspinstallations.SAM_solar_tower)
    ll = cutout.csp(atlite.cspinstallations.lossless_installation)
    assert (st <= ll).all()

    ## Test technology = "parabolic trough"
    pt = cutout.csp(atlite.cspinstallations.SAM_parabolic_trough, capacity_factor=True)

    assert pt.notnull().all()
    assert (pt >= 0).all()
    assert (pt <= 1).all()

    # Efficiencies <= 1 should lead to the conversion to always be less than perfect
    pt = cutout.csp(atlite.cspinstallations.SAM_parabolic_trough)
    ll = cutout.csp(atlite.cspinstallations.lossless_installation)
    assert (pt <= ll).all()


def solar_thermal_test(cutout):
    """
    Test the atlite.Cutout.solar_thermal function with different settings.
    """
    cap_factor = cutout.solar_thermal()
    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0


def heat_demand_test(cutout):
    """
    Test the atlite.Cutout.heat_demand function with different settings.
    """
    demand = cutout.heat_demand()
    assert demand.notnull().all()
    assert demand.sum() > 0


def soil_temperature_test(cutout):
    """
    Test the atlite.Cutout.soil_temperature function with different settings.
    """
    demand = cutout.soil_temperature()
    assert demand.notnull().all()
    assert demand.sum() > 0


def wind_test(cutout):
    """
    Test the atlite.Cutout.wind function with two different layouts.
    The standard layout proportional to the capacity factors must have a lower
    production than a layout proportionally to the capacity layout squared.
    """
    cap_factor = cutout.wind(atlite.windturbines.Enercon_E101_3000kW)

    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0

    production = cutout.wind(atlite.windturbines.Enercon_E101_3000kW, layout=cap_factor)

    assert production.notnull().all()
    assert production.sum() > 0

    # Now create a better layout with same amount of installed power
    better_layout = (cap_factor**2) / (cap_factor**2).sum() * cap_factor.sum()
    better_production = cutout.wind(
        atlite.windturbines.Enercon_E101_3000kW, layout=better_layout
    )

    assert better_production.sum() > production.sum()

    # now use smooth wind power curve
    production = cutout.wind(
        atlite.windturbines.Enercon_E101_3000kW, layout=cap_factor, smooth=True
    )

    assert production.notnull().all()
    assert production.sum() > 0


def runoff_test(cutout):
    """
    Test the atlite.Cutout.runoff function.

    First check if the total of all capacity factors is not null.
    Then compare the runoff at sites which belong to the lower (altitude) half
    of the map, with the runoff at higher sites. The runoff at the lower sites
    (mostly at sea level) should have a smaller capacity factor total and
    production.
    """
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


def hydro_test(cutout):
    """
    Test the atlite.Cutout.hydro function.
    """
    plants = pd.DataFrame(
        cutout.grid.loc[[0], ["x", "y"]].values, columns=["lon", "lat"]
    )
    basins = gpd.GeoDataFrame(
        dict(
            geometry=[cutout.grid.geometry[0]],
            HYBAS_ID=[0],
            DIST_MAIN=10,
            NEXT_DOWN=None,
        ),
        index=[0],
        crs=cutout.crs,
    )
    ds = cutout.hydro(plants, basins)
    assert ds.sel(plant=0).sum() > 0


def line_rating_test(cutout):
    shapes = [Line([Point(-3, 57), Point(0, 60)])]
    resistance = 0.06 * 1e-3
    i = cutout.line_rating(shapes, resistance)
    assert i.notnull().all().item()


def coefficient_of_performance_test(cutout):
    """
    Test the coefficient_of_performance function.
    """
    cap_factor = cutout.coefficient_of_performance(source="air")
    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0

    cap_factor = cutout.coefficient_of_performance(source="soil")
    assert cap_factor.notnull().all()
    assert cap_factor.sum() > 0


# %% Prepare cutouts to test


TIME = "2013-01-01"
BOUNDS = (-4, 56, 1.5, 62)
SARAH_DIR = os.getenv("SARAH_DIR", "/home/vres/climate-data/sarah_v2")
GEBCO_PATH = os.getenv("GEBCO_PATH", "/home/vres/climate-data/GEBCO_2014_2D.nc")


@pytest.fixture(scope="session")
def cutout_era5(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("era5")
    cutout = Cutout(path=tmp_path / "era5", module="era5", bounds=BOUNDS, time=TIME)
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_coarse(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("era5_coarse")
    cutout = Cutout(
        path=tmp_path / "era5", module="era5", bounds=BOUNDS, time=TIME, dx=0.5, dy=0.7
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_weird_resolution(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("era5_weird_resolution")
    cutout = Cutout(
        path=tmp_path / "era5",
        module="era5",
        bounds=BOUNDS,
        time=TIME,
        dx=0.132,
        dy=0.32,
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_reduced(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("era5_red")
    cutout = Cutout(path=tmp_path / "era5", module="era5", bounds=BOUNDS, time=TIME)
    return cutout


@pytest.fixture(scope="session")
def cutout_sarah(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sarah")
    cutout = Cutout(
        path=tmp_path / "sarah",
        module=["sarah", "era5"],
        bounds=BOUNDS,
        time=TIME,
        sarah_dir=SARAH_DIR,
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_sarah_fine(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sarah_coarse")
    cutout = Cutout(
        path=tmp_path / "sarah",
        module="sarah",
        bounds=BOUNDS,
        time=TIME,
        dx=0.05,
        dy=0.05,
        sarah_dir=SARAH_DIR,
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_sarah_weird_resolution(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sarah_weird_resolution")
    cutout = Cutout(
        path=tmp_path / "sarah",
        module="sarah",
        bounds=BOUNDS,
        time=TIME,
        dx=0.132,
        dy=0.32,
        sarah_dir=SARAH_DIR,
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_gebco(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("gebco")
    cutout = Cutout(
        path=tmp_path / "gebco",
        module="gebco",
        bounds=BOUNDS,
        time=TIME,
        gebco_path=GEBCO_PATH,
    )
    cutout.prepare()
    return cutout


class TestERA5:
    @staticmethod
    def test_data_module_arguments_era5(cutout_era5):
        """
        All data variables should have an attribute to which module thay belong
        """
        for v in cutout_era5.data:
            assert cutout_era5.data.attrs["module"] == "era5"

    @staticmethod
    def test_all_non_na_era5(cutout_era5):
        """Every cells should have data."""
        assert np.isfinite(cutout_era5.data).all()

    @staticmethod
    def test_all_non_na_era5_coarse(cutout_era5_coarse):
        """Every cells should have data."""
        assert np.isfinite(cutout_era5_coarse.data).all()

    @staticmethod
    @pytest.mark.skipif(
        os.name == "nt",
        reason="This test breaks on windows machine on travis"
        " due to unknown reasons.",
    )
    def test_all_non_na_era5_weird_resolution(cutout_era5_weird_resolution):
        """Every cells should have data."""
        assert np.isfinite(cutout_era5_weird_resolution.data).all()

    @staticmethod
    def test_dx_dy_preservation_era5(cutout_era5):
        """The coordinates should be the same after preparation."""
        assert np.allclose(np.diff(cutout_era5.data.x), 0.25)
        assert np.allclose(np.diff(cutout_era5.data.y), 0.25)

    @staticmethod
    def test_dx_dy_preservation_era5_coarse(cutout_era5_coarse):
        """The coordinates should be the same after preparation."""
        assert np.allclose(
            np.diff(cutout_era5_coarse.data.x), cutout_era5_coarse.data.attrs["dx"]
        )
        assert np.allclose(
            np.diff(cutout_era5_coarse.data.y), cutout_era5_coarse.data.attrs["dy"]
        )

    @staticmethod
    @pytest.mark.skipif(
        os.name == "nt",
        reason="This test breaks on windows machine on travis"
        " due to unknown reasons.",
    )
    def test_dx_dy_preservation_era5_weird_resolution(cutout_era5_weird_resolution):
        """The coordinates should be the same after preparation."""
        assert np.allclose(
            np.diff(cutout_era5_weird_resolution.data.x),
            cutout_era5_weird_resolution.data.attrs["dx"],
        )
        assert np.allclose(
            np.diff(cutout_era5_weird_resolution.data.y),
            cutout_era5_weird_resolution.data.attrs["dy"],
        )

    @staticmethod
    def test_compare_with_get_data_era5(cutout_era5, tmp_path):
        """
        The prepared data should be exactly the same as from the low level function
        """
        influx = atlite.datasets.era5.get_data(cutout_era5, "influx", tmpdir=tmp_path)
        assert_allclose(
            influx.influx_toa, cutout_era5.data.influx_toa, atol=1e-5, rtol=1e-5
        )

    @staticmethod
    def test_prepared_features_era5(cutout_era5):
        return prepared_features_test(cutout_era5)

    @staticmethod
    @pytest.mark.skipif(
        sys.platform == "win32", reason="NetCDF update not working on windows"
    )
    def test_update_feature_era5(cutout_era5, cutout_era5_reduced):
        return update_feature_test(cutout_era5, cutout_era5_reduced)

    @staticmethod
    def test_wrong_loading(cutout_era5):
        wrong_recreation(cutout_era5)

    @staticmethod
    def test_pv_era5(cutout_era5):
        return pv_test(cutout_era5)

    @staticmethod
    def test_wind_era5(cutout_era5):
        return wind_test(cutout_era5)

    @staticmethod
    def test_runoff_era5(cutout_era5):
        return runoff_test(cutout_era5)

    @staticmethod
    def test_hydro_era5(cutout_era5):
        return hydro_test(cutout_era5)

    @staticmethod
    def test_solar_thermal_era5(cutout_era5):
        return solar_thermal_test(cutout_era5)

    @staticmethod
    def test_heat_demand_era5(cutout_era5):
        return heat_demand_test(cutout_era5)

    @staticmethod
    def test_soil_temperature_era5(cutout_era5):
        return soil_temperature_test(cutout_era5)

    @staticmethod
    def test_line_rating_era5(cutout_era5):
        return line_rating_test(cutout_era5)


@pytest.mark.skipif(
    not os.path.exists(SARAH_DIR), reason="'sarah_dir' is not a valid path"
)
class TestSarah:
    @staticmethod
    def test_all_non_na_sarah(cutout_sarah):
        """Every cells should have data."""
        assert np.isfinite(cutout_sarah.data).all()

    @staticmethod
    def test_all_non_na_sarah_fine(cutout_sarah_fine):
        """Every cells should have data."""
        assert np.isfinite(cutout_sarah_fine.data).all()

    @staticmethod
    def test_all_non_na_sarah_weird_resolution(cutout_sarah_weird_resolution):
        """Every cells should have data."""
        assert np.isfinite(cutout_sarah_weird_resolution.data).all()

    @staticmethod
    def test_dx_dy_preservation_sarah(cutout_sarah):
        """The coordinates should be the same after preparation."""
        assert np.allclose(np.diff(cutout_sarah.data.x), 0.25)
        assert np.allclose(np.diff(cutout_sarah.data.y), 0.25)

    @staticmethod
    def test_prepared_features_sarah(cutout_sarah):
        return prepared_features_test(cutout_sarah)

    @staticmethod
    def test_merge(cutout_sarah, cutout_era5):
        return merge_test(cutout_sarah, cutout_era5, ["sarah", "era5"])

    @staticmethod
    def test_pv_sarah(cutout_sarah):
        return pv_test(cutout_sarah)

    @staticmethod
    def test_wind_sarah(cutout_sarah):
        return wind_test(cutout_sarah)

    @staticmethod
    def test_runoff_sarah(cutout_sarah):
        return runoff_test(cutout_sarah)


@pytest.mark.skipif(
    not os.path.exists(GEBCO_PATH), reason="'gebco_path' is not a valid path"
)
class TestGebco:
    @staticmethod
    def test_all_non_na_gebco(cutout_gebco):
        """Every cells should have data."""
        assert np.isfinite(cutout_gebco.data).all()
