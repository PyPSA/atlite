#!/vsr/bin/env python3

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Created on Mon Oct 18 15:11:42 2021.

@author: fabian
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from atlite import Cutout
from atlite.convert import convert_line_rating, line_azimuth_degrees


def test_ieee_sample_case():
    """
    Test the implementation against the documented results from IEEE standard
    (chapter 4.6).
    """
    ds = {
        "temperature": 313,
        "wnd100m": 0.61,
        "height": 0,
        "wnd_azimuth": 0,
        "influx_direct": 1027,
        "solar_altitude": np.pi / 2,
        "solar_azimuth": np.pi,
    }

    psi = 90  # line azimuth
    D = 0.02814  # line diameter
    Ts = 273 + 100  # max allowed line surface temp
    epsilon = 0.8  # emissivity
    alpha = 0.8  # absorptivity

    R = 9.39e-5  # resistance at 100°C in Ohm/m

    i = convert_line_rating(ds, psi, R, D, Ts, epsilon, alpha)

    assert np.isclose(i, 1025, rtol=0.005)


def test_oeding_and_oswald_sample_case():
    """
    Test the implementation against the documented line parameters documented
    at https://link.springer.com/content/pdf/10.1007%2F978-3-642-19246-3.pdf
    table 9.2, Al 240/40.

    This is the same as the DIN 48204-4/84.

    We do not exactly know at what ambient temperature the DIN is
    calculated. 30 degree is a good guess that fits.
    """
    ds = {
        "temperature": 30 + 273,
        "wnd100m": 0,
        "height": 0,
        "wnd_azimuth": 0,
        "influx_direct": 0,
        "solar_altitude": np.pi / 2,
        "solar_azimuth": np.pi,
    }
    psi = 90  # line azimuth
    D = 0.0218  # line diameter
    Ts = 273 + 80  # max allowed line surface temp
    epsilon = 0.8  # emissivity
    alpha = 0.8  # absorptivity

    R = 0.1188 * 1e-3  # in Ohm/m

    i = convert_line_rating(ds, psi, R, D, Ts, epsilon, alpha)
    assert np.isclose(i, 645, rtol=0.015)


def test_suedkabel_sample_case():
    """
    Test the implementation against the documented line parameters documented
    at https://www.yumpu.com/de/document/read/30614281/kabeldatenblatt-
    typ-2xsfl2y-1x2500-rms-250-220-380-kv assume ambient temperature of 20°C,
    no wind, no sun and max allowed line temperature of 90°C.
    """
    ds = {
        "temperature": 293,
        "wnd100m": 0,
        "height": 0,
        "wnd_azimuth": 0,
        "influx_direct": 0,
        "solar_altitude": np.pi / 2,
        "solar_azimuth": np.pi,
    }
    R = 0.0136 * 1e-3
    psi = 0  # line azimuth

    i = convert_line_rating(ds, psi, R, Ts=363)
    v = 380000  # 220 kV
    s = np.sqrt(3) * i * v / 1e6  # in MW

    assert np.isclose(i, 2460, rtol=0.02)
    assert np.isclose(s, 1619, rtol=0.02)


def test_right_angle_in_different_configuration():
    """Test different configurations of angle difference of 90 degree."""
    ds = {
        "temperature": 313,
        "wnd100m": 0.61,
        "height": 0,
        "wnd_azimuth": 0,
        "influx_direct": 1027,
        "solar_altitude": np.pi / 2,
        "solar_azimuth": np.pi,
    }
    psi = 90  # line azimuth
    D = 0.02814  # line diameter
    Ts = 273 + 100  # max allowed line surface temp
    epsilon = 0.8  # emissivity
    alpha = 0.8  # absorptivity

    R = 9.39e-5  # resistance at 100°C

    expected = convert_line_rating(ds, psi, R, D, Ts, epsilon, alpha)

    psi = 270  # line azimuth
    assert expected == convert_line_rating(ds, psi, R, D, Ts, epsilon, alpha)

    # now set wind angle to 90 degree, line angle to 0 and 180 (preserving right angle)
    ds2 = {**ds, "wnd_azimuth": np.pi / 2}

    psi = 0  # line azimuth
    assert expected == convert_line_rating(ds2, psi, R, D, Ts, epsilon, alpha)

    psi = 180  # line azimuth
    assert expected == convert_line_rating(ds2, psi, R, D, Ts, epsilon, alpha)

    # now set wind angle to 180 degree, line angle to 90 and 270 (preserving right angle)
    ds2 = {**ds, "wnd_azimuth": np.pi}

    psi = 90  # line azimuth
    assert expected == convert_line_rating(ds2, psi, R, D, Ts, epsilon, alpha)

    # exchange psi and wind azimuth
    psi = 270  # line azimuth
    assert expected == convert_line_rating(ds2, psi, R, D, Ts, epsilon, alpha)


def test_angle_increase():
    """Test an increasing angle which should lead to an increasing capacity."""
    ds = {
        "temperature": 313,
        "wnd100m": 0.61,
        "height": 0,
        "wnd_azimuth": 0,
        "influx_direct": 1027,
        "solar_altitude": np.pi / 2,
        "solar_azimuth": np.pi,
    }
    D = 0.02814  # line diameter
    Ts = 273 + 100  # max allowed line surface temp
    epsilon = 0.8  # emissivity
    alpha = 0.8  # absorptivity

    R = 9.39e-5  # resistance at 100°C

    def func(psi):
        return convert_line_rating(ds, psi, R, D, Ts, epsilon, alpha)

    Psi = np.arange(0, 370, 10)
    res = pd.Series([func(psi) for psi in Psi], index=Psi)

    assert (res.iloc[:10].diff().dropna() >= 0).all()
    assert (res.iloc[9:19].diff().dropna() <= 0).all()

    # check point reflection
    assert np.isclose(res.iloc[:19], res.iloc[:17:-1], atol=1e-10, rtol=1e-10).all()
    assert np.isclose(res.iloc[:19], res.iloc[18:], atol=1e-10, rtol=1e-10).all()


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((0.0, 0.0), (0.0, 1.0), 180.0),  # N-pointing line
        ((0.0, 0.0), (0.0, -1.0), 0.0),  # S-pointing line
        ((0.0, 0.0), (1.0, 0.0), -90.0),  # E-pointing
        ((0.0, 0.0), (-1.0, 0.0), 90.0),  # W-pointing
        ((0.0, 0.0), (1.0, 1.0), -135.0),  # NE diagonal
    ],
)
def test_line_azimuth_degrees(start, end, expected):
    """`line_azimuth_degrees` returns degrees consistent with `convert_line_rating`'s `psi`."""
    shape = LineString([Point(*start), Point(*end)])
    assert np.isclose(line_azimuth_degrees(shape), expected)


@pytest.fixture
def cutout(tmp_path):
    rng = np.random.default_rng(0)
    coords = {
        "time": pd.date_range("2013-06-01", periods=24, freq="h"),
        "y": [50.0, 50.25, 50.5],
        "x": [5.0, 5.25, 5.5, 5.75],
    }

    def var(low, high, dims=("time", "y", "x")):
        return (dims, rng.uniform(low, high, [len(coords[d]) for d in dims]))

    ds = xr.Dataset(
        {
            "temperature": var(270, 300),
            "wnd100m": var(0, 15),
            "wnd_azimuth": var(0, 2 * np.pi),
            "influx_direct": var(0, 800),
            "solar_altitude": var(0, 1.2),
            "solar_azimuth": var(0, 2 * np.pi),
            "height": var(0, 500, ("y", "x")),
        },
        coords=coords,
        attrs={"module": "era5"},
    )
    ds["influx_direct"][:, 0, 0] = np.nan
    ds.to_netcdf(tmp_path / "cutout.nc")
    return Cutout(tmp_path / "cutout.nc")


def test_line_rating_is_minimum_over_intersected_cells(cutout):
    shapes = gpd.GeoSeries([
        LineString([(5.0, 50.0), (5.5, 50.5)]),
        LineString([(5.75, 50.25), (5.25, 50.25)]),
        LineString([(9.0, 40.0), (9.5, 40.0)]),
    ])
    R = pd.Series([3e-5, 4e-5, 5e-5])
    res = cutout.line_rating(shapes, R)

    data = cutout.data.stack(spatial=["y", "x"])
    cells = cutout.intersectionmatrix(shapes).tocsr()
    for i, shape in enumerate(shapes):
        if not cells[i].nnz:
            assert res[i].isnull().all()
            continue
        psi = line_azimuth_degrees(shape)
        psi = psi if psi >= 0 else psi + 180
        ds = data.isel(spatial=cells[i].indices)
        expected = convert_line_rating(ds, psi, R[i])
        np.testing.assert_allclose(res[i], expected)


def test_line_rating_without_intersections_is_nan(cutout):
    shapes = gpd.GeoSeries([LineString([(9.0, 40.0), (9.5, 40.0)])])
    res = cutout.line_rating(shapes, 3e-5)
    assert res.shape == (1, 24)
    assert res.isnull().all()
