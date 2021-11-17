# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Atlite helps you to convert weather data into energy systems model data.

Atlite is a free software, xarray-based Python library for converting weather data
(like wind speeds) into energy systems data. It is designed to by lightweight and
work with big weather datasets while keeping the resource requirements especially
on CPU and RAM resources low.
"""

from .cutout import Cutout
from .gis import compute_indicatormatrix, regrid, ExclusionContainer
from .resource import windturbines, solarpanels, cspinstallations

from .version import version as __version__

__author__ = (
    "The Atlite Authors: Gorm Andresen (Aarhus University), "
    "Jonas Hoersch (FIAS/KIT/RLI), "
    "Johannes Hampp (JLUG),"
    "Fabian Hofmann (FIAS)"
    "Tom Brown (FIAS/KIT), "
    "Markus Schlott (FIAS), "
    "David Schlachtberger (FIAS), "
)
__copyright__ = "Copyright 2016-2021 The Atlite Authors"
