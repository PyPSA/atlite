# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2020-2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

linebreak = "\n    "

climatedata = [
    "Temperature",
    "Downward short-wave radiation",
    "Upward short-wave radiation",
    "Wind",
    "Runoff",
    "Surface roughness",
    "Height maps",
    "Soil temperature",
]
processeddata = [
    f"Wind power generation {linebreak}for a given turbine type",
    f"Solar PV power generation {linebreak}for a given panel type",
    "Solar thermal collector heat output",
    "Hydroelectric inflow (simplified)",
    f"Heating demand {linebreak}" "(based on the degree-day approximation)",
]

climatestr = "\n" + "\n\n".join([" ◦ " + s for s in climatedata]) + "\n"

processedstr = "\n" + "\n\n\n".join([" ◦ " + s for s in processeddata]) + "\n"

# defaults for boxes and arrows
kwargs = dict(verticalalignment="center", fontsize=14, color="#545454")
arrowkwargs = dict(
    head_width=0.2,
    width=0.13,
    head_length=0.05,
    edgecolor="white",
    length_includes_head=True,
    color="lightgray",
    alpha=1,
)
y = 0.5

# First arrow
ax.text(
    0.01, y, " Retrieve Data", fontsize=14, color="gray", verticalalignment="center"
)
ax.arrow(0.01, y, 0.14, 0.0, **arrowkwargs)

# First box
ax.text(
    0.17,
    y,
    climatestr,
    **kwargs,
    bbox=dict(facecolor="indianred", alpha=0.5, edgecolor="None", boxstyle="round"),
)

# Second arrow
ax.text(0.5, y, " Process Data", fontsize=14, color="gray", verticalalignment="center")
ax.arrow(0.5, y, 0.14, 0, **arrowkwargs)

# Second Box
ax.text(
    0.66,
    y,
    processedstr,
    **kwargs,
    bbox=dict(facecolor="olivedrab", alpha=0.5, edgecolor="None", boxstyle="round"),
)


fig.tight_layout(pad=0)
fig.savefig("workflow.png", dpi=150)
