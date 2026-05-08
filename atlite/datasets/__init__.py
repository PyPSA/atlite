# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""
atlite datasets.
"""

from atlite.datasets import era5, era5_ncar, gebco, sarah

modules = {"era5": era5, "era5_ncar": era5_ncar, "sarah": sarah, "gebco": gebco}
