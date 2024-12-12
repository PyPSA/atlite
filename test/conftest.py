# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

import os
from datetime import date
from pathlib import Path

import pytest
from dateutil.relativedelta import relativedelta

from atlite import Cutout

TIME = "2013-01-01"
BOUNDS = (-4, 56, 1.5, 62)
SARAH_DIR = os.getenv("SARAH_DIR", "/home/vres/climate-data/sarah_v2")
GEBCO_PATH = os.getenv("GEBCO_PATH", "/home/vres/climate-data/GEBCO_2014_2D.nc")


def pytest_addoption(parser):
    parser.addoption(
        "--cache-path",
        action="store",
        default=None,
        help="Specify path for atlite cache files to not use temporary directory",
    )


@pytest.fixture(scope="session", autouse=True)
def cutouts_path(tmp_path_factory, pytestconfig):
    custom_path = pytestconfig.getoption("--cache-path")
    if custom_path:
        path = Path(custom_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    else:
        return tmp_path_factory.mktemp("atlite_cutouts")


@pytest.fixture(scope="session")
def cutout_era5(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5.nc"
    cutout = Cutout(path=tmp_path, module="era5", bounds=BOUNDS, time=TIME)
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_mon(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5_mon.nc"
    cutout = Cutout(path=tmp_path, module="era5", bounds=BOUNDS, time=TIME)
    cutout.prepare(monthly_requests=True, concurrent_requests=False)

    return cutout


@pytest.fixture(scope="session")
def cutout_era5_mon_concurrent(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5_mon_concurrent.nc"
    cutout = Cutout(path=tmp_path, module="era5", bounds=BOUNDS, time=TIME)
    cutout.prepare(monthly_requests=True, concurrent_requests=True)

    return cutout


@pytest.fixture(scope="session")
def cutout_era5_3h_sampling(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5_3h_sampling.nc"
    time = [
        f"{TIME} 00:00",
        f"{TIME} 03:00",
        f"{TIME} 06:00",
        f"{TIME} 09:00",
        f"{TIME} 12:00",
        f"{TIME} 15:00",
        f"{TIME} 18:00",
        f"{TIME} 21:00",
    ]
    cutout = Cutout(path=tmp_path, module="era5", bounds=BOUNDS, time=time)
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_2days_crossing_months(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5_2days_crossing_months.nc"
    time = slice("2013-02-28", "2013-03-01")
    cutout = Cutout(path=tmp_path, module="era5", bounds=BOUNDS, time=time)
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_coarse(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5_coarse.nc"
    cutout = Cutout(
        path=tmp_path, module="era5", bounds=BOUNDS, time=TIME, dx=0.5, dy=0.7
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_weird_resolution(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5_weird_resolution.nc"
    cutout = Cutout(
        path=tmp_path,
        module="era5",
        bounds=BOUNDS,
        time=TIME,
        dx=0.132,
        dy=0.32,
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_reduced(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5_reduced.nc"
    cutout = Cutout(path=tmp_path, module="era5", bounds=BOUNDS, time=TIME)
    return cutout


@pytest.fixture(scope="session")
def cutout_era5_overwrite(cutouts_path, cutout_era5_reduced):
    tmp_path = cutouts_path / "cutout_era5_overwrite.nc"
    cutout = Cutout(path=tmp_path, module="era5", bounds=BOUNDS, time=TIME)
    # cutout.data = cutout.data.drop_vars("influx_direct")
    # cutout.prepare("influx", overwrite=True)
    # TODO Needs to be fixed
    return cutout


@pytest.fixture(scope="session")
def cutout_era5t(cutouts_path):
    tmp_path = cutouts_path / "cutout_era5t.nc"

    today = date.today()
    first_day_this_month = today.replace(day=1)
    first_day_prev_month = first_day_this_month - relativedelta(months=1)
    last_day_second_prev_month = first_day_prev_month - relativedelta(days=1)

    cutout = Cutout(
        path=tmp_path,
        module="era5",
        bounds=BOUNDS,
        time=slice(last_day_second_prev_month, first_day_prev_month),
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_sarah(cutouts_path):
    tmp_path = cutouts_path / "cut_out_sarah.nc"
    cutout = Cutout(
        path=tmp_path,
        module=["sarah", "era5"],
        bounds=BOUNDS,
        time=TIME,
        sarah_dir=SARAH_DIR,
    )
    cutout.prepare()
    return cutout


@pytest.fixture(scope="session")
def cutout_sarah_fine(cutouts_path):
    tmp_path = cutouts_path / "cutout_sarah_fine.nc"
    cutout = Cutout(
        path=tmp_path,
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
def cutout_sarah_weird_resolution(cutouts_path):
    tmp_path = cutouts_path / "cutout_sarah_weird_resolution.nc"
    cutout = Cutout(
        path=tmp_path,
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
def cutout_gebco(cutouts_path):
    tmp_path = cutouts_path / "cutout_gebco.nc"
    cutout = Cutout(
        path=tmp_path,
        module="gebco",
        bounds=BOUNDS,
        time=TIME,
        gebco_path=GEBCO_PATH,
    )
    cutout.prepare()
    return cutout
