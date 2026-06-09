# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Module involving hydro operations in atlite."""

from __future__ import annotations

import logging
from collections import namedtuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from tqdm import tqdm

logger = logging.getLogger(__name__)

Basins = namedtuple("Basins", ["plants", "meta", "shapes"])


def find_basin(
    shapes: gpd.GeoSeries,
    lon: float,
    lat: float,
) -> int:
    """
    Find the basin containing a point.

    Parameters
    ----------
    shapes : geopandas.GeoSeries
        Basin geometries indexed by basin id.
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.

    Returns
    -------
    int
        Basin id containing the point.
    """
    hids = shapes.index[shapes.intersects(Point(lon, lat))]
    if len(hids) > 1:
        logger.warning(
            "The point (%s, %s) is in several basins: %s. Assuming the first one.",
            lon,
            lat,
            hids,
        )
    return int(hids[0])


def find_upstream_basins(
    meta: pd.DataFrame,
    hid: int,
) -> list[int]:
    """
    Collect all upstream basins of a basin.

    Parameters
    ----------
    meta : pandas.DataFrame
        Basin metadata with a ``NEXT_DOWN`` column.
    hid : int
        Basin id from which to start.

    Returns
    -------
    list[int]
        Basin ids including the selected basin and all upstream basins.
    """
    hids = [hid]
    i = 0
    while i < len(hids):
        hids.extend(meta.index[meta["NEXT_DOWN"] == hids[i]])
        i += 1
    return hids


def determine_basins(
    plants: pd.DataFrame,
    hydrobasins: str | gpd.GeoDataFrame,
    show_progress: bool = False,
) -> Basins:
    """
    Determine local and upstream basins for hydro plants.

    Parameters
    ----------
    plants : pandas.DataFrame
        Plant table with ``lon`` and ``lat`` columns.
    hydrobasins : str or geopandas.GeoDataFrame
        HydroBASINS data source or loaded basin geometries.
    show_progress : bool, default False
        Whether to show a progress bar.

    Returns
    -------
    Basins
        Basin assignments, metadata, and geometries for the plants.
    """
    if isinstance(hydrobasins, str):
        hydrobasins = gpd.read_file(hydrobasins)

    assert isinstance(hydrobasins, gpd.GeoDataFrame), (
        "hydrobasins should be passed as a filename or a GeoDataFrame, "
        f"but received `type(hydrobasins) = {type(hydrobasins)}`"
    )

    missing_columns = pd.Index([
        "HYBAS_ID",
        "DIST_MAIN",
        "NEXT_DOWN",
        "geometry",
    ]).difference(hydrobasins.columns)
    assert missing_columns.empty, (
        "Couldn't find the column(s) {} in the hydrobasins dataset.".format(
            ", ".join(missing_columns)
        )
    )

    hydrobasins = hydrobasins.set_index("HYBAS_ID")

    meta = hydrobasins[hydrobasins.columns.difference(("geometry",))]
    shapes = hydrobasins["geometry"]

    plant_basins: list[tuple[int, list[int]]] = []
    for p in tqdm(
        plants.itertuples(),
        disable=not show_progress,
        desc="Determine upstream basins per plant",
    ):
        hid = find_basin(shapes, p.lon, p.lat)
        plant_basins.append((hid, find_upstream_basins(meta, hid)))
    plant_basins_df = pd.DataFrame(
        plant_basins, columns=["hid", "upstream"], index=plants.index
    )

    unique_basins = pd.Index(plant_basins_df["upstream"].sum()).unique().rename("hid")
    return Basins(plant_basins_df, meta.loc[unique_basins], shapes.loc[unique_basins])


def shift_and_aggregate_runoff_for_plants(
    basins: Basins,
    runoff: xr.DataArray,
    flowspeed: float = 1,
    show_progress: bool = False,
) -> xr.DataArray:
    """
    Shift basin runoff in time and aggregate it per plant.

    Parameters
    ----------
    basins : Basins
        Basin mappings and metadata for the plants.
    runoff : xarray.DataArray
        Runoff time series indexed by ``hid`` and ``time``.
    flowspeed : float, default 1
        Flow speed in m/s used to convert distance to travel time.
    show_progress : bool, default False
        Whether to show a progress bar.

    Returns
    -------
    xarray.DataArray
        Plant inflow time series indexed by ``plant`` and ``time``.
    """
    inflow: xr.DataArray = xr.DataArray(
        np.zeros((len(basins.plants), runoff.indexes["time"].size)),
        [("plant", basins.plants.index), runoff.coords["time"]],
    )

    for ppl in tqdm(
        basins.plants.itertuples(),
        disable=not show_progress,
        desc="Shift and aggregate runoff by plant",
    ):
        inflow_plant: xr.DataArray = inflow.loc[{"plant": ppl.Index}]
        distances: pd.Series = (
            basins.meta.loc[ppl.upstream, "DIST_MAIN"]
            - basins.meta.at[ppl.hid, "DIST_MAIN"]
        )
        nhours: pd.Series = (distances / (flowspeed * 3.6) + 0.5).astype(int)

        for b in ppl.upstream:
            inflow_plant += runoff.sel(hid=b).roll(time=nhours.at[b])

    return inflow
