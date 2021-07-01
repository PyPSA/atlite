#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Created on Tue Jun 22 10:46:27 2021

@author: fabian
"""
from atlite.resource import get_oedb_windturbineconfig
import pytest


def test_oedb_windturbineconfig():

    # test int search
    assert get_oedb_windturbineconfig(1)

    # test string search
    assert get_oedb_windturbineconfig("E-101/3500 E2")

    # test string search with param
    assert get_oedb_windturbineconfig("E-101/3500 E2", hub_height=99)
