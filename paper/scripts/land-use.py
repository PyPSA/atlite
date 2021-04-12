#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from rasterio.plot import show
from matplotlib.gridspec import GridSpec
from pathlib import Path

plt.rc('text', usetex=True)
plt.rc('text.latex',
       preamble=r"\setlength{\parindent}{0pt} \setlength{\parskip}{\baselineskip}"
                r"\usepackage{mathtools}")
plt.rc('font', family='sans-serif', size=14)
figpath = Path('../figures/')


cutout = atlite.Cutout('cutout')

nutsregions = gpd.read_file('NUTS_RG_60M_2021_4326.geojson')
regions = nutsregions.query('CNTR_CODE == "DE" and LEVL_CODE == 2')
regions = regions.rename_axis('region')
de = gpd.GeoSeries([regions.unary_union], index=['DE'], crs=4326)

# %%

# inspect corine tags
# df = pd.read_html('https://wiki.openstreetmap.org/wiki/Corine_Land_Cover#Tagging')[0]
# tags = df[df.Code.apply(len) > 3].reset_index(drop=True).rename(lambda x: x+1)
corine = 'g250_clc06_V18_5.tif'
codes =  [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 31, 32]

# data for first plot
excluder = atlite.ExclusionContainer()
excluder.add_raster(corine, codes, invert=True, crs=3035)
masked, transform = atlite.gis.shape_availability(de.to_crs(excluder.crs), excluder)

# data for second plot
excluder = atlite.ExclusionContainer()
excluder.add_raster(corine, codes, invert=True, crs=3035)
A = cutout.availabilitymatrix(regions, excluder, nprocesses=3, disable_progressbar=True)

# %%
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(5, 5, figure=fig)

projection = ccrs.epsg(excluder.crs)
ax = fig.add_subplot(gs[:, :2], projection=projection)
show(masked, cmap='Greens',  transform=transform, ax=ax)
ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), linewidth=.5)
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), linewidth=.5)
ax.set_title('Eligible land (green)', pad=10)
ax.axis('off')

ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
ax.text(0.4, .5, r"$\longrightarrow$", fontsize=35)


ax = fig.add_subplot(gs[:2, 3:])
ds = A.isel(region=0)
ds.plot(cmap='Greens', ax=ax)
regions.plot(ax=ax, edgecolor='k', color='None', linewidth=0.2)
regions.iloc[[0]].plot(ax=ax, edgecolor='k', color='None')
ax.set_title('Region 1')
ax.axis('off')


ax = fig.add_subplot(gs[2:4, 3:])
ds = A.isel(region=1)
ds.plot(cmap='Greens', ax=ax)
regions.plot(ax=ax, edgecolor='k', color='None', linewidth=0.2)
regions.iloc[[1]].plot(ax=ax, edgecolor='k', color='None')
ax.set_title('Region 2')
ax.axis('off')

ax = fig.add_subplot(gs[4, 3:])
ax.text(x=0.5, y=0.8, s=r'$\cdot$ \\$\cdot$ \\ $\cdot$', ha='center', fontsize=25)
ax.axis('off')


fig.supylabel('Effective overlap between regions and weather cells', x=1)
fig.tight_layout()
fig.savefig(figpath/'land-use-availability.png', bbox_inches='tight')


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
