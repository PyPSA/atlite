# -*- coding: utf-8 -*-

## Copyright 2016-2017 Jonas Hoersch (RLI)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Renewable Energy Atlas Lite (Atlite)

Light-weight version of Aarhus RE Atlas for converting weather data to power systems data
"""
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

import scipy.sparse as spp
from collections import namedtuple
from shapely.geometry import Point
from six import string_types

from .utils import make_optional_progressbar

import logging
logger = logging.getLogger(__name__)

Basins = namedtuple('Basins', ['plants', 'meta', 'shapes'])

def find_basin(shapes, lon, lat):
    hids = shapes.index[shapes.intersects(Point(lon, lat))]
    if len(hids) > 1:
        logger.warning(f"The point ({lon}, {lat}) is in several basins: {hids}. Assuming the first one.")
    return hids[0]

def find_upstream_basins(meta, hid):
    hids = [hid]
    i = 0
    while i < len(hids):
        hids.extend(meta.index[meta['NEXT_DOWN'] == hids[i]])
        i += 1
    return hids

def determine_basins(plants, hydrobasins, show_progress=True):
    if isinstance(hydrobasins, string_types):
        hydrobasins = gpd.read_file(hydrobasins)

    assert isinstance(hydrobasins, gpd.GeoDataFrame), (
        "hydrobasins should be passed as a filename or a GeoDataFrame, "
        "but received `type(hydrobasins) = {}`".format(type(hydrobasins))
    )

    missing_columns = pd.Index(['HYBAS_ID', 'DIST_MAIN', 'NEXT_DOWN', 'geometry']).difference(hydrobasins.columns)
    assert missing_columns.empty, \
        "Couldn't find the column(s) {} in the hydrobasins dataset.".format(", ".join(missing_columns))

    hydrobasins = hydrobasins.set_index("HYBAS_ID")

    meta = hydrobasins[hydrobasins.columns.difference(('geometry',))]
    shapes = hydrobasins['geometry']

    maybe_progressbar = make_optional_progressbar(show_progress, "Determine upstream basins per plant", len(plants))

    plant_basins = []
    for p in maybe_progressbar(plants.itertuples()):
        hid = find_basin(shapes, p.lon, p.lat)
        plant_basins.append((hid, find_upstream_basins(meta, hid)))
    plant_basins = pd.DataFrame(plant_basins, columns=['hid', 'upstream'], index=plants.index)

    unique_basins = pd.Index(plant_basins['upstream'].sum()).unique().rename("hid")
    return Basins(plant_basins, meta.loc[unique_basins], shapes.loc[unique_basins])

def shift_and_aggregate_runoff_for_plants(basins, runoff, flowspeed=1, show_progress=True):
    inflow = xr.DataArray(np.zeros((len(basins.plants), runoff.indexes["time"].size)),
                          [('plant', basins.plants.index),
                           ('time' , runoff.coords["time"])])

    maybe_progressbar = make_optional_progressbar(show_progress, "Shift and aggregate runoff by plant", len(basins.plants))

    for ppl in maybe_progressbar(basins.plants.itertuples()):
        inflow_plant = inflow.loc[dict(plant=ppl.Index)]
        distances = basins.meta.loc[ppl.upstream, "DIST_MAIN"] - basins.meta.at[ppl.hid, "DIST_MAIN"]
        nhours = (distances * (flowspeed * 3.6) + 0.5).astype(int)

        for b in ppl.upstream:
            inflow_plant += runoff.sel(hid=b).shift(time=nhours.at[b])

    return inflow
