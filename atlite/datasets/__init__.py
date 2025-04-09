# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""
atlite datasets.
"""

from atlite.datasets import (
    era5,
    gebco,
    icon,
    icon_d2,
    icon_eu,
    meteo_forecast,
    meteo_historic,
    meteo_historic_forecast,
    sarah,
)

modules = {
    "era5": era5,
    "sarah": sarah,
    "gebco": gebco,
    "meteo_forecast": meteo_forecast,
    "meteo_historic_forecast": meteo_historic_forecast,
    "meteo_historic": meteo_historic,
    "icon_d2": icon_d2,
    "icon_eu": icon_eu,
    "icon": icon,
}
