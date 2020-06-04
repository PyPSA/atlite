#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is dedicated to comparing atlite versions 1 and 2.
It is used with an already existent cutout from pypsa-eur.
Prefereably run it in an separate directory with an subfolder 'figures'

Run it one time with the old version 0.0.2 (or 0.1) of atlite and one time
with the version 0.2.

Ths script creates mutliple figures of power potential of wind, pv, solar t
thermal etc. The time-series and geographical plots can be compared after the
two runs.

@author: fabian
"""

import atlite
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from pathlib import Path


pypsa_eur_dir = Path('/home/fabian/vres/py/pypsa-eur/')
cutout_dir = pypsa_eur_dir / 'cutouts' / 'europe-2013-era5'
shapes_path = pypsa_eur_dir / 'resources' / 'regions_onshore_elec_s_256.geojson'

if atlite.__version__=='0.2':
    tag = 'v2'
    if not Path('pypsa-eur.nc').is_file():
        atlite.utils.migrate_from_cutout_directory(cutout_dir, 'pypsa-eur.nc')
    cutout = atlite.Cutout('pypsa-eur.nc', chunks={'time':'auto'})
else:
    tag = 'v1'
    cutout = atlite.Cutout(cutout_dir)


Shapes = gpd.read_file(shapes_path).set_index('name')
shapes = pd.Series(Shapes.geometry)

def shapes_plot(ds, ax):
    Shapes.assign(ds=ds.sum('time')/Shapes.area).plot(column='ds', ax=ax,
                                                      legend=True)
    fig.tight_layout()

# wind
fig, ax = plt.subplots()
wind = cutout.wind('Enercon_E101_3000kW', shapes=shapes)
shapes_plot(wind, ax=ax)
fig.savefig(f'figures/wind_shapes_{tag}.png')

fig, ax = plt.subplots()
wind.sum('name').plot.line(x='time', ax=ax)
fig.savefig(f'figures/wind_timeseries_total_{tag}.png')

fig, ax = plt.subplots()
wind.sel(name=['AL0 0', 'SE6 8']).plot.line(x='time', ax=ax)
fig.savefig(f'figures/wind_timeseries_sel_{tag}.png')

# pv
fig, ax = plt.subplots()
pv = cutout.pv('CdTe', 'latitude_optimal', shapes=shapes)
shapes_plot(pv, ax=ax)
fig.savefig(f'figures/pv_shapes_{tag}.png')

fig, ax = plt.subplots()
pv.sum('name').plot.line(x='time', ax=ax)
fig.savefig(f'figures/pv_timeseries_total_{tag}.png')

fig, ax = plt.subplots()
pv.sel(name=['AL0 0', 'SE6 8']).plot.line(x='time', ax=ax)
fig.savefig(f'figures/pv_timeseries_sel_{tag}.png')

# thermal
fig, ax = plt.subplots()
solar_thermal = cutout.solar_thermal(shapes=shapes)
shapes_plot(solar_thermal, ax=ax)
fig.savefig(f'figures/solar_thermal_shapes_{tag}.png')

fig, ax = plt.subplots()
solar_thermal.sum('name').plot.line(x='time', ax=ax)
fig.savefig(f'figures/solar_thermal_timeseries_total_{tag}.png')

fig, ax = plt.subplots()
solar_thermal.sel(name=['AL0 0', 'SE6 8']).plot.line(x='time', ax=ax)
fig.savefig(f'figures/solar_thermal_timeseries_sel_{tag}.png')


# runoff
fig, ax = plt.subplots()
runoff = cutout.runoff(shapes=shapes)
shapes_plot(runoff, ax=ax)
fig.savefig(f'figures/runoff_shapes_{tag}.png')

fig, ax = plt.subplots()
runoff.sum('name').plot.line(x='time', ax=ax)
fig.savefig(f'figures/runoff_timeseries_total_{tag}.png')

fig, ax = plt.subplots()
runoff.sel(name=['AL0 0', 'SE6 8']).plot.line(x='time')
fig.savefig(f'figures/runoff_timeseries_sel_{tag}.png')


# heat_demand
fig, ax = plt.subplots()
heat_demand = cutout.heat_demand(shapes=shapes)
shapes_plot(heat_demand, ax=ax)
fig.savefig(f'figures/heat_demand_shapes_{tag}.png')

fig, ax = plt.subplots()
heat_demand.sum('name').plot.line(x='time', ax=ax)
fig.savefig(f'figures/heat_demand_timeseries_total_{tag}.png')

fig, ax = plt.subplots()
heat_demand.sel(name=['AL0 0', 'SE6 8']).plot.line(x='time', ax=ax)
fig.savefig(f'figures/heat_demand_timeseries_sel_{tag}.png')

