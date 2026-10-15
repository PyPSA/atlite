# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""atlite datasets."""

from atlite.datasets import era5, era5_corrected, era5_edh, gebco, glofas, sarah

modules = {
    "era5": era5,
    "era5-corrected": era5_corrected,
    "era5-edh": era5_edh,
    "gebco": gebco,
    "glofas": glofas,
    "sarah": sarah,
}
