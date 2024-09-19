# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT

from atlite.datasets import cosmo_rea6, era5, gebco, sarah

modules = {
    "era5": era5,
    "sarah": sarah,
    "gebco": gebco,
    "cosmo_rea6": cosmo_rea6,
}
