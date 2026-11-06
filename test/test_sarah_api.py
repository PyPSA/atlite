# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Tests for downloading SARAH data from the EUMETSAT API (mocked)."""

import io
import sys
import types
import zipfile

import pandas as pd
import pytest
import xarray as xr

from atlite.datasets import sarah


class FakeProduct:
    size = 1024  # kilobytes

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def open(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"{self.name}.nc", b"netcdf-bytes")
        buf.seek(0)
        return buf


class FakeCollection:
    def search(self, dtstart, dtend, title):
        prefix = title.rstrip("*")
        return [FakeProduct(f"{prefix}20190101"), FakeProduct(f"{prefix}20190102")]


@pytest.fixture
def fake_eumdac(monkeypatch):
    class CredentialsFileNotFoundError(Exception):
        pass

    eumdac = types.ModuleType("eumdac")
    eumdac.AccessToken = lambda credentials: credentials
    eumdac.DataStore = lambda token: types.SimpleNamespace(
        get_collection=lambda collection_id: FakeCollection()
    )
    helpers = types.ModuleType("eumdac.cli_helpers")
    helpers.CredentialsFileNotFoundError = CredentialsFileNotFoundError
    helpers.load_credentials = lambda: ("key", "secret")
    eumdac.cli_helpers = helpers
    monkeypatch.setitem(sys.modules, "eumdac", eumdac)
    monkeypatch.setitem(sys.modules, "eumdac.cli_helpers", helpers)
    return helpers


def test_api_time_range():
    time = pd.date_range("2019-01-01", "2019-01-03 12:00", freq="h")
    coords = xr.Dataset(coords={"time": time}).coords
    assert sarah._api_time_range(coords) == (
        "2019-01-01T00:00:00Z",
        "2019-01-03T23:30:00Z",
    )


def test_download_from_api(fake_eumdac, tmp_path, caplog):
    with caplog.at_level("INFO", logger="atlite.datasets.sarah"):
        out = sarah.download_from_api(
            "2019-01-01T00:00:00Z",
            "2019-01-02T23:30:00Z",
            tmp_path / "raw",
            max_workers=2,
        )
    assert "estimated size 0.00 GiB" in caplog.text
    assert "4 products" in caplog.text
    assert sorted(p.name for p in (out / "SID").glob("*.nc")) == [
        "SIDin20190101.nc",
        "SIDin20190102.nc",
    ]
    assert len(list((out / "SIS").glob("*.nc"))) == 2
    assert not list(out.rglob("*.zip"))
    assert not list(out.rglob("*.part"))


def test_missing_credentials(fake_eumdac, tmp_path):
    def raise_missing():
        raise fake_eumdac.CredentialsFileNotFoundError

    fake_eumdac.load_credentials = raise_missing
    with pytest.raises(RuntimeError, match="set-credentials"):
        sarah.download_from_api("a", "b", tmp_path / "raw")


def test_cleanup_removes_raw_files(tmp_path):
    raw = tmp_path / sarah.SARAH_API_SUBDIR / "SID"
    raw.mkdir(parents=True)
    (raw / "x.nc").write_bytes(b"x")
    sarah.cleanup(tmp_path)
    assert not (tmp_path / sarah.SARAH_API_SUBDIR).exists()
