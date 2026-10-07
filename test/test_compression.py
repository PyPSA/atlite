# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Tests for the compression of prepared cutouts (#508)."""

from pathlib import Path
from typing import Any, Literal

import netCDF4
import numpy as np
import pytest
import xarray as xr

import atlite.data
from atlite import Cutout
from atlite.data import available_features

ZLIB = {"compression": "zlib", "complevel": 9, "shuffle": True}


def _fake_features(cutout, module, features, **kwargs):
    variables = available_features(module).loc[module].loc[list(features)]
    shape = tuple(cutout.data.sizes[d] for d in ("time", "y", "x"))
    values = np.random.default_rng(0).uniform(250, 300, shape)
    values[0, 0, 0] = np.nan
    data = {
        v: xr.Variable(("time", "y", "x"), values, {"module": module, "feature": f})
        for f, v in variables.items()
    }
    return xr.Dataset(data, coords=cutout.data.coords)


@pytest.mark.parametrize(
    ("compression", "codec", "quantization"),
    [(None, "zstd", (14, "BitRound")), (ZLIB, "zlib", None)],
)
def test_prepare_compresses_all_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: dict[str, Any] | None,
    codec: Literal["zstd", "zlib"],
    quantization: tuple[int, str] | None,
) -> None:
    monkeypatch.setattr(atlite.data, "get_features", _fake_features)
    cutout = Cutout(
        tmp_path / "cutout.nc", module="era5", bounds=(0, 50, 1, 51), time="2013-01-01"
    )
    for feature in ["height", "temperature"]:
        cutout.prepare(feature, tmpdir=tmp_path, compression=compression)

    with netCDF4.Dataset(cutout.path) as nc:
        assert nc.variables["height"].filters()[codec]
        assert nc.variables["temperature"].filters()[codec]
        assert nc.variables["temperature"].quantization() == quantization

    expected = _fake_features(cutout, "era5", ["temperature"])
    xr.testing.assert_allclose(
        cutout.data["temperature"], expected["temperature"], rtol=2**-15
    )
