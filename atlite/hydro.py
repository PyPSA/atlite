# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Module involving hydro operations in atlite.
"""

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


def determine_basins(plants, hydrobasins, show_progress=False):
    if isinstance(hydrobasins, str):
        hydrobasins = gpd.read_file(hydrobasins)

    assert isinstance(hydrobasins, gpd.GeoDataFrame), (
        "hydrobasins should be passed as a filename or a GeoDataFrame, "
        f"but received `type(hydrobasins) = {type(hydrobasins)}`"
    )

    missing_columns = pd.Index(
        ["HYBAS_ID", "DIST_MAIN", "NEXT_DOWN", "geometry"]
    ).difference(hydrobasins.columns)
    assert missing_columns.empty, (
        "Couldn't find the column(s) {} in the hydrobasins dataset.".format(
            ", ".join(missing_columns)
        )
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
    basins, runoff, flowspeed=1, show_progress=False
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
        nhours = (distances / (flowspeed * 3.6) + 0.5).astype(int)

        for b in ppl.upstream:
            inflow_plant += runoff.sel(hid=b).roll(time=nhours.at[b])

    return inflow


def _hydro_from_runoff(
    cutout,
    plants,
    hydrobasins,
    flowspeed=1,
    weight_with_height=False,
    show_progress=False,
    **kwargs,
):
    """
    Compute inflow time-series for `plants` by aggregating over catchment
    basins from `hydrobasins` (ERA5 runoff-based computation).

    Parameters
    ----------
    plants : pd.DataFrame
        Run-of-river plants or dams with lon, lat columns.
    hydrobasins : str|gpd.GeoDataFrame
        Filename or GeoDataFrame of one level of the HydroBASINS dataset.
    flowspeed : float
        Average speed of water flows to estimate the water travel time from
        basin to plant (default: 1 m/s).
    weight_with_height : bool
        Whether surface runoff should be weighted by potential height (probably
        better for coarser resolution).
    show_progress : bool
        Whether to display progressbars.

    References
    ----------
    [1] Liu, Hailiang, et al. "A validated high-resolution hydro power
    time-series model for energy systems analysis." arXiv preprint
    arXiv:1901.08476 (2019).

    [2] Lehner, B., Grill G. (2013): Global river hydrography and network
    routing: baseline data and new approaches to study the world’s large river
    systems. Hydrological Processes, 27(15): 2171–2186. Data is available at
    www.hydrosheds.org.

    """
    basins = determine_basins(plants, hydrobasins, show_progress=show_progress)

    matrix = cutout.indicatormatrix(basins.shapes)
    # compute the average surface runoff in each basin
    # Fix NaN and Inf values to 0.0 to avoid numerical issues
    matrix_normalized = np.nan_to_num(
        matrix / matrix.sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0
    )
    runoff = cutout.runoff(
        matrix=matrix_normalized,
        index=basins.shapes.index,
        weight_with_height=weight_with_height,
        show_progress=show_progress,
        **kwargs,
    )
    # The hydrological parameters are in units of "m of water per day" and so
    # they should be multiplied by 1000 and the basin area to convert to m3
    # d-1 = m3 h-1 / 24
    runoff *= xr.DataArray(basins.shapes.to_crs(dict(proj="cea")).area)

    return shift_and_aggregate_runoff_for_plants(
        basins, runoff, flowspeed, show_progress
    )


def _hydro_from_discharge(
    cutout,
    plants,
    time=None,
):
    """
    Get inflow time-series for `plants` from GLOFAS discharge by snapping each
    plant to the nearest grid cell that holds data and interpolating onto the
    target time index.

    Parameters
    ----------
    plants : pd.DataFrame
        Run-of-river plants or dams with lon, lat columns.
    time : pd.DatetimeIndex, optional
        Time index to interpolate the plant inflow onto. Defaults to the cutout's
        own time index.
    """
    if time is None:
        time = cutout.coords["time"]
    discharge = cutout.data.discharge
    # snap plants to GLOFAS cells with data (cutout grid points may be all-NaN)
    present = discharge.isel(time=0).notnull()
    discharge = discharge.isel(
        x=np.flatnonzero(present.any("y").values),
        y=np.flatnonzero(present.any("x").values),
    )
    x = xr.DataArray(plants["lon"].values, dims="plant", coords={"plant": plants.index})
    y = xr.DataArray(plants["lat"].values, dims="plant", coords={"plant": plants.index})
    inflow = discharge.sel(x=x, y=y, method="nearest").compute()
    inflow = inflow.dropna("time", how="all").interp(time=time)
    inflow = inflow.ffill("time").bfill("time")
    return inflow.transpose("plant", "time")
