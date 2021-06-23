# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2021 The Atlite Authors

# SPDX-License-Identifier: GPL-3.0-or-later

"""
Created on Sun Apr 11 20:16:02 2021

@author: fabian
"""
import atlite
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import pandas as pd
import matplotlib as mpl
from rasterio.plot import show
from matplotlib.gridspec import GridSpec
from pathlib import Path

plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"\setlength{\parindent}{0pt} \setlength{\parskip}{\baselineskip}"
    r"\usepackage{mathtools}",
)
plt.rc("font", family="sans-serif", size=14)
figpath = Path("../figures/")


cutout = atlite.Cutout("cutout")

nutsregions = gpd.read_file("NUTS_RG_60M_2021_4326.geojson")
regions = nutsregions.query('CNTR_CODE == "DE" and LEVL_CODE == 2')
regions = regions.rename_axis("region")
de = gpd.GeoSeries([regions.unary_union], index=["DE"], crs=4326)

# %%

# inspect corine tags
# df = pd.read_html('https://wiki.openstreetmap.org/wiki/Corine_Land_Cover#Tagging')[0]
# tags = df[df.Code.apply(len) > 3].reset_index(drop=True).rename(lambda x: x+1)
corine = "g250_clc06_V18_5.tif"
codes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 31, 32]

# data for first plot
excluder = atlite.ExclusionContainer()
excluder.add_raster(corine, codes, invert=True, crs=3035)
masked, transform = atlite.gis.shape_availability(de.to_crs(excluder.crs), excluder)

# data for second plot
excluder = atlite.ExclusionContainer()
excluder.add_raster(corine, codes, invert=True, crs=3035)
A = cutout.availabilitymatrix(regions, excluder, nprocesses=3, disable_progressbar=True)

# %%
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 9, figure=fig)

projection = ccrs.epsg(excluder.crs)
ax = fig.add_subplot(gs[:, :3], projection=projection)
show(masked, cmap="Greens", transform=transform, ax=ax)
ax.add_feature(cartopy.feature.BORDERS.with_scale("10m"), linewidth=0.5)
ax.add_feature(cartopy.feature.COASTLINE.with_scale("10m"), linewidth=0.5)
ax.set_title("Eligible land (green)", pad=10)
ax.axis("off")

ax = fig.add_subplot(gs[:, 3])
ax.axis("off")
ax.text(0, 0.5, r"$\longrightarrow$", fontsize=35)


regions_i = [0, 20]
for i, r in enumerate(regions_i):
    ax = fig.add_subplot(gs[:, 4 + 2 * i : 6 + 2 * i])
    ds = A.isel(region=r)
    ds.plot(cmap="Greens", ax=ax, vmin=0, vmax=0.7, add_colorbar=False)
    regions.plot(ax=ax, edgecolor="k", color="None", linewidth=0.2)
    regions.iloc[[r]].plot(ax=ax, edgecolor="k", color="None")
    ax.set_title(f"Region {i+1}", fontsize=14)
    ax.axis("off")


ax = fig.add_subplot(gs[:, 8])
ax.text(x=0.5, y=0.5, s=r"$\cdots$", ha="center", fontsize=25)
ax.axis("off")

ax = fig.add_subplot(gs[:, 4:])
ax.set_visible(False)
fig.colorbar(
    mpl.cm.ScalarMappable(cmap="Greens"), ax=ax, orientation="horizontal", aspect=40
)

fig.suptitle("Effective overlap between regions and weather cells", x=0.72, y=0.85)
fig.tight_layout()
fig.savefig(figpath / "land-use-availability.png", bbox_inches="tight")


# %%
# fg = ds.plot(row='region', col_wrap=1, cmap='Greens',
#              gridspec_kws={"wspace":0.4},
#              cbar_kwargs={'label': title, 'aspect':40})
# for i, ax in enumerate(fg.axes.flatten()):
#     regions.plot(ax=ax, edgecolor='k', color='None', linewidth=0.2)
#     regions.iloc[[i]].plot(ax=ax, edgecolor='k', color='None')
#     ax.set_title(f'Region {i+1}')
#     ax.axis('off')

# ax.text(x=11, y=44, s=r'$\cdot$ \\$\cdot$ \\ $\cdot$', ha='center', fontsize=25)

# fg.fig.savefig(figpath/'availability-matrix.png', bbox_inches='tight')
