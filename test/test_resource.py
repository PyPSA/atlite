#!/usr/bin/env python3

# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Created on Tue Jun 22 10:46:27 2021.

@author: fabian
"""

import pytest

from atlite.resource import get_oedb_windturbineconfig, get_windturbineconfig


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
