#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:47:13 2021

@author: fabulous
"""

import atlite
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import numpy as np
import cartopy
import xarray as xr
from cartopy.crs import PlateCarree as plate
from pathlib import Path
from matplotlib.gridspec import GridSpec

plt.rc('text', usetex=True)
plt.rc('text.latex',
       preamble=r"\setlength{\parindent}{0pt} \setlength{\parskip}{\baselineskip}")
plt.rc('font', family='serif', size=14)
figpath = Path('../figures/')
figpath.mkdir(exist_ok=True)

# from https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts#nuts21
nutsregions = gpd.read_file('NUTS_RG_60M_2021_4326.geojson')
regions = nutsregions.query('CNTR_CODE == "DE" and LEVL_CODE == 2')
de = regions.unary_union


centroid = regions.total_bounds[[0, 2]].mean(), regions.total_bounds[[1,3]].mean()
projection = ccrs.Orthographic(*centroid)


cutout = atlite.Cutout('cutout', bounds=de.buffer(1).bounds, time='2013-01',
                       module='era5', dx=.5, dy=.5)
# cutout.prepare(features="wind")
cutout_bound = gpd.GeoSeries(cutout.grid.unary_union)


#%% all steps in one
fig = plt.figure(figsize=(10, 12))
gs = GridSpec(6, 6, figure=fig)
title_params = dict(pad=35)

# cutout creation plot
color = 'teal'
ax = fig.add_subplot(gs[:3, :3], projection=projection)
cutout.grid.plot(ax=ax, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.3, transform=plate())
cutout.grid[['x', 'y']].plot.scatter(x='x', y='y', ax=ax, transform=plate(),
                                     color=color, s=1, alpha=1)
cutout_bound.plot(ax=ax, edgecolor='orange', facecolor='None', transform=plate())
ax.set_extent(np.array(de.buffer(2).bounds)[[0,2,1,3]], crs=plate())
ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), edgecolor='gray')
ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), edgecolor='gray')
ax.axis('off')
ax.set_title(r"\begin{center}\textbf{1. Create Cutout} \\(Select spatio-temporal bounds)\end{center}", **title_params)
# ax.set_title(r"\textbf{1. Create Cutout}", **title_params)

# Prepare cutout
df = cutout.data.mean(['x', 'y']).to_dataframe()

ax = fig.add_subplot(gs[0, 3:])
df.wnd100m.plot(ax=ax, color='teal')
ax.set_title(r"\begin{center}\textbf{2. Prepare Cutout} \\(Retrieve data per weather cell)\end{center}", **title_params)
# ax.set_title(r"\textbf{2. Prepare Cutout}", **title_params)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('wnd100m [m/s]')
ax.set_xlabel('')
ax.set_xticklabels([])

ax = fig.add_subplot(gs[1, 3:])
df.roughness.plot(ax=ax, color='orange')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('roughness [m]')
ax.set_xlabel('')


ax = fig.add_subplot(gs[2, 3:])
ax.text(x=.5, y=.8, s=r'\noindent .\\ . \\ .', ha='center', fontsize=30)
ax.axis('off')


# Convert cutout
turbine = atlite.windturbines.Vestas_V112_3MW
cap_per_sqkm = 1.7
layout = cutout.grid.set_index(['x', 'y']).to_crs(3025).area / 1e6 * cap_per_sqkm
layout = xr.DataArray(layout.unstack())
ds, capacity = cutout.wind(turbine, shapes=regions, layout=layout,
                           return_capacity=True, per_unit=True)


ax = fig.add_subplot(gs[3:, :3])
ax.set_title(r"\begin{center}\textbf{3. Convert Cutout} \\ (Calcuate potentials and timeseries per region)\end{center}",
              **title_params)
# ax.set_title(r"\textbf{3. Convert Cutout}", **title_params)
regions.plot(ax=ax, column=capacity.to_series()/1e3, legend=True, cmap='Greens',
              legend_kwds=dict(label='Potential Capcity [GW]',
                               location='right', shrink=0.8)
             )
# cutout.grid.plot(ax=ax, facecolor='None', edgecolor='grey', ls=':')
ax.axis('off')

color = 'steelblue'
ax = fig.add_subplot(gs[3, 3:])
ds.isel(dim_0=0).to_series().plot(ax=ax, color=color)
ax.set_title('Region 1', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Power p.u.')
ax.set_xlabel('')
ax.set_xticklabels([])

ax = fig.add_subplot(gs[4, 3:])
ds.isel(dim_0=1).to_series().plot(ax=ax, color=color)
ax.set_title('Region 2', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Power p.u.')
ax.set_xlabel('')

ax = fig.add_subplot(gs[5, 3:])
ax.text(x=.5, y=.8, s=r'\noindent .\\ . \\ .', ha='center', fontsize=30)
ax.axis('off')


fig.tight_layout()
fig.savefig(figpath/"workflow.png", bbox_inches='tight')





