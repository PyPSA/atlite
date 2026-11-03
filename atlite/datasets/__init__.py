# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""atlite datasets."""

from atlite.datasets import era5, era5_edh, gebco, glofas, mrel_wave, sarah

modules = {
    "era5": era5,
    "era5-edh": era5_edh,
    "gebco": gebco,
    "glofas": glofas,
    "sarah": sarah,
    "mrel_wave": mrel_wave,
}
