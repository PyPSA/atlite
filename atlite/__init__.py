# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
atlite helps you to convert weather data into energy systems model data.

atlite is a free software, xarray-based Python library for converting
weather data (like wind speeds) into energy systems data. It is designed
to by lightweight and work with big weather datasets while keeping the
resource requirements especially on CPU and RAM resources low.
"""

__author__ = (
    "Gorm Andresen, "
    "Jonas Hoersch (FIAS/KIT/RLI), "
    "Johannes Hampp (JLUG),"
    "Fabian Hofmann (FIAS)"
    "Tom Brown (FIAS/KIT), "
    "Markus Schlott (FIAS), "
    "David Schlachtberger (FIAS), "
)

__copyright__ = "Copyright Contributors to atlite"

import re
from importlib.metadata import version

from atlite.cutout import Cutout
from atlite.gis import ExclusionContainer, compute_indicatormatrix, regrid
from atlite.resource import cspinstallations, solarpanels, windturbines

# e.g. "0.17.1" or "0.17.1.dev4+ga3890dc0" (if installed from git)
__version__ = version("atlite")
# e.g. "0.17.0" # TODO, in the network structure it should use the dev version
match = re.match(r"(\d+\.\d+(\.\d+)?)", __version__)
assert match, f"Could not determine release_version of pypsa: {__version__}"
release_version = match.group(0)
assert not __version__.startswith("0.0"), "Could not determine version of atlite."

__all__ = [
    Cutout,
    ExclusionContainer,
    compute_indicatormatrix,
    regrid,
    cspinstallations,
    solarpanels,
    windturbines,
    __version__,
]
