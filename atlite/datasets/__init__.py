# SPDX-FileCopyrightText: Contributors to Atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""
Atlite datasets.
"""

from atlite.datasets import era5, gebco, sarah

modules = {"era5": era5, "sarah": sarah, "gebco": gebco}
