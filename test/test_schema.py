# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Tests for cutout schema versions and their migration."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from atlite import Cutout
from atlite.migrate import main, migrate_cutout
from atlite.schema import (
    CUTOUT_SCHEMA_VERSION,
    SCHEMA_VERSION_ATTR,
    IncompatibleCutoutError,
    OutdatedCutoutWarning,
    read_schema_version,
)


def write_cutout(path, version=None):
    """
    Write a minimal cutout file.

    Returns
    -------
    Path
        ``path``, which has no schema attribute if ``version`` is None.

    """
    attrs = {"module": "era5", "prepared_features": ["height"], "dx": 0.25, "dy": 0.25}
    if version is not None:
        attrs[SCHEMA_VERSION_ATTR] = version
    ds = xr.Dataset(
        {"height": (("y", "x"), np.zeros((2, 3), dtype="float32"))},
        coords={
            "x": [0.0, 0.25, 0.5],
            "y": [50.0, 50.25],
            "time": pd.date_range("2013-01-01", periods=2, freq="h"),
        },
        attrs=attrs,
    )
    ds["height"].attrs.update(module="era5", feature="height")
    ds.to_netcdf(path)
    return path


def test_new_cutout_has_current_version(tmp_path):
    """A newly created cutout carries the current schema version in memory and on disk."""
    cutout = Cutout(
        tmp_path / "new", module="era5", bounds=(0, 50, 1, 51), time="2013-01-01"
    )
    assert cutout.schema_version == CUTOUT_SCHEMA_VERSION
    cutout.to_file()
    assert read_schema_version(cutout.path) == CUTOUT_SCHEMA_VERSION


def test_current_version_loads_without_warning(tmp_path):
    """Loading a cutout with the current schema version emits no outdated warning."""
    path = write_cutout(tmp_path / "current.nc", CUTOUT_SCHEMA_VERSION)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutdatedCutoutWarning)
        Cutout(path)


def test_missing_attribute_is_version_zero(tmp_path):
    """A cutout without the schema version attribute is read as version 0."""
    path = write_cutout(tmp_path / "legacy.nc")
    assert read_schema_version(path) == 0


def test_outdated_version_warns_with_commands(tmp_path):
    """Loading an outdated cutout warns and shows the shell and Python migration commands."""
    path = write_cutout(tmp_path / "legacy.nc")
    output = tmp_path / "legacy-v1.nc"
    with pytest.warns(OutdatedCutoutWarning) as record:
        cutout = Cutout(path)
    assert cutout.schema_version == 0

    message = str(record[0].message)
    assert f'python -m atlite.migrate "{path}" --output "{output}"' in message
    assert "from atlite.migrate import migrate_cutout" in message
    assert f"migrate_cutout({str(path)!r}, output={str(output)!r})" in message


@pytest.mark.parametrize("version", [CUTOUT_SCHEMA_VERSION + 1, -1])
def test_unsupported_version_raises(tmp_path, version):
    """Loading or migrating a cutout with an unsupported schema version raises an error."""
    path = write_cutout(tmp_path / "unsupported.nc", version)
    with pytest.raises(IncompatibleCutoutError):
        Cutout(path)
    with pytest.raises(IncompatibleCutoutError):
        migrate_cutout(path, in_place=True)


def test_migrate_in_place(tmp_path):
    """In-place migration updates the schema version without changing the data."""
    path = write_cutout(tmp_path / "legacy.nc")
    with xr.open_dataset(path) as ds:
        expected = ds.load()

    with warnings.catch_warnings():
        warnings.simplefilter("error", OutdatedCutoutWarning)
        migrate_cutout(path, in_place=True)
    assert read_schema_version(path) == CUTOUT_SCHEMA_VERSION
    with xr.open_dataset(path) as ds:
        xr.testing.assert_identical(ds.drop_attrs(), expected.drop_attrs())

    with warnings.catch_warnings():
        warnings.simplefilter("error", OutdatedCutoutWarning)
        Cutout(path)


def test_migrate_to_output_keeps_original(tmp_path):
    """Migrating to an output file keeps the original and never overwrites an existing output."""
    path = write_cutout(tmp_path / "legacy.nc")
    output = tmp_path / "migrated.nc"

    assert main([str(path), "--output", str(output)]) == 0
    assert read_schema_version(path) == 0
    assert read_schema_version(output) == CUTOUT_SCHEMA_VERSION
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.nc", "migrated.nc"]

    # Existing outputs are never overwritten
    assert main([str(path), "--output", str(output)]) == 1


def test_migrate_current_version_is_noop(tmp_path):
    """Migrating a cutout that already has the current schema version leaves it unchanged."""
    path = write_cutout(tmp_path / "current.nc", CUTOUT_SCHEMA_VERSION)
    migrate_cutout(path, in_place=True)
    assert read_schema_version(path) == CUTOUT_SCHEMA_VERSION


def test_migrate_rejects_non_cutout(tmp_path):
    """Migrating a netCDF file that is not an atlite cutout raises an error."""
    path = tmp_path / "other.nc"
    xr.Dataset({"a": ("x", [1, 2])}).to_netcdf(path)
    with pytest.raises(IncompatibleCutoutError, match="not an atlite cutout"):
        migrate_cutout(path, in_place=True)


@pytest.mark.parametrize("kwargs", [{}, {"output": "out.nc", "in_place": True}])
def test_migrate_requires_output_or_in_place(tmp_path, kwargs):
    """migrate_cutout requires exactly one of an output file or in_place=True."""
    path = write_cutout(tmp_path / "legacy.nc")
    with pytest.raises(ValueError, match="in_place=True"):
        migrate_cutout(path, **kwargs)
    assert read_schema_version(path) == 0


def test_cli_requires_output_or_in_place(tmp_path):
    """The command line requires exactly one of --output or --in-place."""
    path = write_cutout(tmp_path / "legacy.nc")
    with pytest.raises(SystemExit):
        main([str(path)])
    with pytest.raises(SystemExit):
        main([str(path), "--in-place", "-o", "out.nc"])
    assert read_schema_version(path) == 0

    assert main([str(path), "--in-place"]) == 0
    assert read_schema_version(path) == CUTOUT_SCHEMA_VERSION


def test_merge_requires_equal_versions(tmp_path):
    """Merging cutouts with different schema versions raises an error."""
    current = Cutout(write_cutout(tmp_path / "current.nc", CUTOUT_SCHEMA_VERSION))
    with pytest.warns(OutdatedCutoutWarning):
        legacy = Cutout(write_cutout(tmp_path / "legacy.nc"))
    with pytest.raises(ValueError, match="schema versions"):
        current.merge(legacy)


def test_prepare_refuses_outdated_version(tmp_path):
    """Preparing features for an outdated cutout is refused and leaves the file unchanged."""
    path = write_cutout(tmp_path / "legacy.nc")
    with pytest.warns(OutdatedCutoutWarning):
        cutout = Cutout(path)
    with pytest.raises(IncompatibleCutoutError, match="python -m atlite.migrate"):
        cutout.prepare(tmpdir=tmp_path)
    assert read_schema_version(path) == 0


def test_prepare_allows_current_version(tmp_path):
    """Preparing a cutout with the current schema version passes the version check."""
    path = write_cutout(tmp_path / "current.nc", CUTOUT_SCHEMA_VERSION)
    # All features of the minimal cutout are prepared, so nothing is downloaded
    Cutout(path).prepare(tmpdir=tmp_path, features="height")


def test_prepare_skips_check_for_prepared_cutout(tmp_path, monkeypatch):
    """Calling prepare on a fully prepared outdated cutout returns without an error."""
    path = write_cutout(tmp_path / "legacy.nc")
    with pytest.warns(OutdatedCutoutWarning):
        cutout = Cutout(path)
    # Pretend all features are prepared, so prepare returns before any writing
    monkeypatch.setattr(Cutout, "prepared", property(lambda self: True))
    assert cutout.prepare(tmpdir=tmp_path) is cutout


def test_prepare_rejects_schema_attribute_in_source_data(tmp_path, monkeypatch):
    """Source data that sets the schema version attribute makes prepare fail before writing."""
    path = write_cutout(tmp_path / "current.nc", CUTOUT_SCHEMA_VERSION)
    cutout = Cutout(path)

    def get_features(cutout, module, features, **kwargs):
        ds = cutout.data[["height"]].rename(height="wnd100m")
        return ds.assign_attrs({SCHEMA_VERSION_ATTR: CUTOUT_SCHEMA_VERSION})

    monkeypatch.setattr("atlite.data.get_features", get_features)
    with pytest.raises(RuntimeError, match=SCHEMA_VERSION_ATTR):
        cutout.prepare(tmpdir=tmp_path, features="wind")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["current.nc"]
