# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Photovoltaic modeling functions."""

from __future__ import annotations

from atlite.pv.irradiation import (
    DiffuseHorizontalIrrad,
    TiltedDiffuseIrrad,
    TiltedDirectIrrad,
    TiltedGroundIrrad,
    TiltedIrradiation,
)
from atlite.pv.orientation import (
    SurfaceOrientation,
    get_orientation,
    make_constant,
    make_latitude,
    make_latitude_optimal,
)
from atlite.pv.solar_panel_model import SolarPanelModel
from atlite.pv.solar_position import SolarPosition

__all__: list[str] = [
    "DiffuseHorizontalIrrad",
    "TiltedDirectIrrad",
    "TiltedDiffuseIrrad",
    "TiltedGroundIrrad",
    "TiltedIrradiation",
    "SurfaceOrientation",
    "get_orientation",
    "make_constant",
    "make_latitude",
    "make_latitude_optimal",
    "SolarPanelModel",
    "SolarPosition",
]
