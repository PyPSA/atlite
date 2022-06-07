# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module involving hydro operations in Atlite.
"""

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from collections import namedtuple
from shapely.geometry import Point
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

Basins = namedtuple("Basins", ["plants", "meta", "shapes"])


def find_basin(shapes, lon, lat):
    hids = shapes.index[shapes.intersects(Point(lon, lat))]
    if len(hids) > 1:
        logger.warning(
            f"The point ({lon}, {lat}) is in several basins: {hids}. "
            "Assuming the first one."
        )
    return hids[0]


def find_upstream_basins(meta, hid):
    hids = [hid]
    i = 0
    while i < len(hids):
        hids.extend(meta.index[meta["NEXT_DOWN"] == hids[i]])
        i += 1
    return hids


def determine_basins(plants, hydrobasins, show_progress=True):
    if isinstance(hydrobasins, str):
        hydrobasins = gpd.read_file(hydrobasins)

    assert isinstance(hydrobasins, gpd.GeoDataFrame), (
        "hydrobasins should be passed as a filename or a GeoDataFrame, "
        "but received `type(hydrobasins) = {}`".format(type(hydrobasins))
    )

    missing_columns = pd.Index(
        ["HYBAS_ID", "DIST_MAIN", "NEXT_DOWN", "geometry"]
    ).difference(hydrobasins.columns)
    assert (
        missing_columns.empty
    ), "Couldn't find the column(s) {} in the hydrobasins dataset.".format(
        ", ".join(missing_columns)
    )

    hydrobasins = hydrobasins.set_index("HYBAS_ID")

    meta = hydrobasins[hydrobasins.columns.difference(("geometry",))]
    shapes = hydrobasins["geometry"]

    plant_basins = []
    for p in tqdm(
        plants.itertuples(),
        disable=not show_progress,
        desc="Determine upstream basins per plant",
    ):
        hid = find_basin(shapes, p.lon, p.lat)
        plant_basins.append((hid, find_upstream_basins(meta, hid)))
    plant_basins = pd.DataFrame(
        plant_basins, columns=["hid", "upstream"], index=plants.index
    )

    unique_basins = pd.Index(plant_basins["upstream"].sum()).unique().rename("hid")
    return Basins(plant_basins, meta.loc[unique_basins], shapes.loc[unique_basins])


def shift_and_aggregate_runoff_for_plants(
    basins, runoff, flowspeed=1, show_progress=True
):
    inflow = xr.DataArray(
        np.zeros((len(basins.plants), runoff.indexes["time"].size)),
        [("plant", basins.plants.index), runoff.coords["time"]],
    )

    for ppl in tqdm(
        basins.plants.itertuples(),
        disable=not show_progress,
        desc="Shift and aggregate runoff by plant",
    ):
        inflow_plant = inflow.loc[dict(plant=ppl.Index)]
        distances = (
            basins.meta.loc[ppl.upstream, "DIST_MAIN"]
            - basins.meta.at[ppl.hid, "DIST_MAIN"]
        )
        nhours = (distances * (flowspeed * 3.6) + 0.5).astype(int)

        for b in ppl.upstream:
            inflow_plant += runoff.sel(hid=b).roll(time=nhours.at[b])

    return inflow
