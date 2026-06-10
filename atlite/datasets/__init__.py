# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""atlite datasets."""

from atlite.datasets import era5, era5_edh, gebco, sarah

modules = {
    "era5": era5,
    "era5-edh": era5_edh,
    "sarah": sarah,
    "gebco": gebco,
}
