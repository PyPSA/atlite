# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""Tests for the hydro basin helpers."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from atlite.hydro import find_basin


def test_find_basin_outside_raises():
    shapes = gpd.GeoSeries([box(0, 0, 1, 1)], index=[11])
    with pytest.raises(ValueError, match=r"\(5, 5\) is not in any basin"):
        find_basin(shapes, 5, 5)
