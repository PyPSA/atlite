#!/vsr/bin/env python3

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Created on Mon Oct 18 15:11:42 2021.

@author: fabian
"""

import numpy as np
import pandas as pd

from atlite.convert import convert_line_rating


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
    """
    Test different configurations of angle difference of 90 degree.
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
    """
    Test an increasing angle which should lead to an increasing capacity.
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
