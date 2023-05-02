#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2021 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT
"""
Created on Tue Jun 22 10:46:27 2021.

@author: fabian
"""
import pytest

from atlite.resource import get_oedb_windturbineconfig


def test_oedb_windturbineconfig():
    # test int search
    assert get_oedb_windturbineconfig(1)

    # test string search
    assert get_oedb_windturbineconfig("E-101/3500 E2")

    # test string search with param
    assert get_oedb_windturbineconfig("E-101/3500 E2", hub_height=99)
