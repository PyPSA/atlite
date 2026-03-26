# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""atlite datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlite.datasets import era5, gebco, sarah

if TYPE_CHECKING:
    from types import ModuleType

modules: dict[str, ModuleType] = {"era5": era5, "sarah": sarah, "gebco": gebco}
