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
from pathlib import Path

plt.rc('text', usetex=True)
plt.rc('text.latex',
       preamble=r"\setlength{\parindent}{0pt} \setlength{\parskip}{\baselineskip}")
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

excluder = atlite.ExclusionContainer()
excluder.add_raster(corine, codes, invert=True, crs=3035)


masked, transform = atlite.gis.shape_availability(de.to_crs(excluder.crs), excluder)

projection = ccrs.epsg(excluder.crs)
fig, ax = plt.subplots(subplot_kw={'projection': projection})
show(masked, cmap='Greens',  transform=transform, ax=ax)
ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), linewidth=.5)
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), linewidth=.5)
ax.axis('off')
fig.savefig(figpath/'availabile-land.png', bbox_inches='tight')

# %%

excluder = atlite.ExclusionContainer()
excluder.add_raster(corine, codes, invert=True, crs=3035)
A = cutout.availabilitymatrix(regions, excluder, nprocesses=3, disable_progressbar=True)

first_regions = regions.index[:2]
title = 'Effective overlap per weather cell'
fg = A.sel(region=first_regions).plot(row='region', col_wrap=1, cmap='Greens',
                                      cbar_kwargs={'label': title, 'aspect':40})
for i, ax in enumerate(fg.axes.flatten()):
    regions.plot(ax=ax, edgecolor='k', color='None', linewidth=0.2)
    regions.iloc[[i]].plot(ax=ax, edgecolor='k', color='None')
    ax.set_title(f'Region {i+1}')
    ax.axis('off')

ax.text(x=11, y=44, s=r'$\cdot$ \\$\cdot$ \\ $\cdot$', ha='center', fontsize=25)

fg.fig.savefig(figpath/'availability-matrix.png', bbox_inches='tight')
