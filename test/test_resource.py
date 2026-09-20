#!/usr/bin/env python3

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Created on Tue Jun 22 10:46:27 2021.

@author: fabian
"""

import json

import pandas as pd
import pytest

from atlite import resource
from atlite.resource import get_oedb_windturbineconfig, get_windturbineconfig


@pytest.fixture
def oedb_turbines(monkeypatch):
    """Two-row stand-in for the OEDB library, with 'name' != 'turbine_type'.

    In the live OEDB wind turbine library the two columns hold different
    strings for most entries, so a filter has to be applied to the column it
    names. Patching the module-level cache keeps the test offline.
    """
    df = pd.DataFrame({
        "id": [0, 1],
        "manufacturer": ["Enercon", "Nordex"],
        "name": ["E-101/3500 E2", "Gamma Series"],
        "turbine_type": ["E-101/3500", "N131/3600"],
        "has_power_curve": [True, True],
        "power_curve_wind_speeds": [json.dumps([0.0, 25.0])] * 2,
        "power_curve_values": [json.dumps([0.0, 3500.0]), json.dumps([0.0, 3600.0])],
        "hub_height": [149.0, 134.0],
        "source": ["test", "test"],
    })
    monkeypatch.setattr(resource, "_oedb_turbines", df)


def test_oedb_windturbineconfig_turbine_type_alone(oedb_turbines):
    # 'turbine_type' is a documented search parameter and has to work on its own.
    assert get_oedb_windturbineconfig(turbine_type="N131/3600")["name"] == "N131/3600"


def test_oedb_windturbineconfig_turbine_type_narrows(oedb_turbines):
    # A 'turbine_type' matching no turbine must narrow the result to nothing,
    # even when a 'name' that does match is given alongside it.
    with pytest.raises(RuntimeError, match="No turbine found"):
        get_oedb_windturbineconfig(name="E-101/3500", turbine_type="N131/3600")


def test_oedb_windturbineconfig_manufacturer_search(oedb_turbines):
    # Control: the neighbouring search parameters keep working.
    assert get_oedb_windturbineconfig(manufacturer="Nordex")["name"] == "N131/3600"


def test_oedb_windturbineconfig():
    # test int search
    assert get_oedb_windturbineconfig(1)

    # test string search
    assert get_oedb_windturbineconfig("E-101/3500 E2")

    # test string search with param
    assert get_oedb_windturbineconfig("E-101/3500 E2", hub_height=99)


@pytest.mark.parametrize("add_cutout, last_pow", [(True, 0.0), (False, 1.0)])
def test_windturbineconfig_add_cutout(add_cutout, last_pow):
    t = get_windturbineconfig(
        {"V": [0, 25], "POW": [0.0, 1.0], "hub_height": 1.0, "P": 1.0},
        add_cutout_windspeed=add_cutout,
    )
    assert t["POW"][-1] == last_pow
