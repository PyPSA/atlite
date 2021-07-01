# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module containing specific operations for creating cutouts from the CORDEX dataset.

DEPRECATED
----------

The cordex dataset module has not been ported to Atlite v0.2, yet. Use Atlite v0.0.2 to use it,
for the time being!
"""

import pandas as pd
import numpy as np
import xarray as xr
import pyproj
from itertools import groupby
from operator import itemgetter
import os
import glob

# Model and CRS Settings
model = "MPI-M-MPI-ESM-LR"

crs = 4326  # TODO
# something like the following is correct
# crs = pyproj.crs.DerivedGeographicCRS(4326, pcrs.coordinate_operation.RotatedLatitudeLongitudeConversion(??))
# RotProj(dict(proj='ob_tran', o_proj='latlong', lon_0=180, o_lon_p=-162, o_lat_p=39.25))


def rename_and_clean_coords(ds):
    ds = ds.rename({"rlon": "x", "rlat": "y"})
    # drop some coordinates and variables we do not use
    ds = ds.drop(
        (set(ds.coords) | set(ds.data_vars)) & {"bnds", "height", "rotated_pole"}
    )
    return ds


def prepare_data_cordex(fn, year, months, oldname, newname, xs, ys):
    with xr.open_dataset(fn) as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.rename({oldname: newname})
        ds = ds.sel(x=xs, y=ys)

        if newname in {"influx", "outflux"}:
            # shift averaged data to beginning of bin
            ds = ds.assign_coords(
                time=(
                    pd.to_datetime(ds.coords["time"].values) - pd.Timedelta(hours=1.5)
                )
            )
        elif newname in {"runoff"}:
            # shift and fill 6hr average data to beginning of 3hr bins
            t = pd.to_datetime(ds.coords["time"].values)
            ds = ds.reindex(method="bfill", time=(t - pd.Timedelta(hours=3.0)).union(t))

        for m in months:
            yield (year, m), ds.sel(time="{}-{}".format(year, m))


def prepare_static_data_cordex(fn, year, months, oldname, newname, xs, ys):
    with xr.open_dataset(fn) as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.rename({oldname: newname})
        ds = ds.sel(x=xs, y=ys)

        for m in months:
            yield (year, m), ds


def prepare_weather_types_cordex(fn, year, months, oldname, newname, xs, ys):
    with xr.open_dataset(fn) as ds:
        ds = ds.rename({oldname: newname})
        for m in months:
            yield (year, m), ds.sel(time="{}-{}".format(year, m))


def prepare_meta_cordex(
    xs, ys, year, month, template, height_config, module, model=model
):
    fn = next(glob.iglob(template.format(year=year, model=model)))
    with xr.open_dataset(fn) as ds:
        ds = rename_and_clean_coords(ds)
        ds = ds.coords.to_dataset()
        meta = ds.sel(time="{}-{}".format(year, month), x=xs, y=ys).load()

    xs = ds["x"].values
    ys = ds["y"].values

    height_config = height_config.copy()
    height_tasks_func = height_config.pop("tasks_func")
    (height_task,) = height_tasks_func(
        xs, ys, [(year, month)], meta_attrs={}, **height_config
    )
    height_prepare_func = height_task.pop("prepare_func")
    _, ds = height_prepare_func(**height_task)[0]

    meta["height"] = ds["height"]

    return meta


def tasks_yearly_cordex(
    xs, ys, yearmonths, prepare_func, template, oldname, newname, meta_attrs
):
    model = meta_attrs["model"]

    if not isinstance(xs, slice):
        first, second, last = xs.values[[0, 1, -1]]
        xs = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))
    if not isinstance(ys, slice):
        first, second, last = ys.values[[0, 1, -1]]
        ys = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))

    return [
        dict(
            prepare_func=prepare_func,
            xs=xs,
            ys=ys,
            oldname=oldname,
            newname=newname,
            fn=next(glob.iglob(template.format(year=year, model=model))),
            year=year,
            months=list(map(itemgetter(1), yearmonths)),
        )
        for year, yearmonths in groupby(yearmonths, itemgetter(0))
    ]


weather_data_config = {
    "influx": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_data_cordex,
        oldname="rsds",
        newname="influx",
        template=os.path.join(
            config.cordex_dir, "{model}", "influx", "rsds_*_{year}*.nc"
        ),
    ),
    "outflux": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_data_cordex,
        oldname="rsus",
        newname="outflux",
        template=os.path.join(
            config.cordex_dir, "{model}", "outflux", "rsus_*_{year}*.nc"
        ),
    ),
    "temperature": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_data_cordex,
        oldname="tas",
        newname="temperature",
        template=os.path.join(
            config.cordex_dir, "{model}", "temperature", "tas_*_{year}*.nc"
        ),
    ),
    "humidity": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_data_cordex,
        oldname="hurs",
        newname="humidity",
        template=os.path.join(
            config.cordex_dir, "{model}", "humidity", "hurs_*_{year}*.nc"
        ),
    ),
    "wnd10m": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_data_cordex,
        oldname="sfcWind",
        newname="wnd10m",
        template=os.path.join(
            config.cordex_dir, "{model}", "wind", "sfcWind_*_{year}*.nc"
        ),
    ),
    "roughness": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_static_data_cordex,
        oldname="rlst",
        newname="roughness",
        template=os.path.join(config.cordex_dir, "{model}", "roughness", "rlst_*.nc"),
    ),
    "runoff": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_data_cordex,
        oldname="mrro",
        newname="runoff",
        template=os.path.join(
            config.cordex_dir, "{model}", "runoff", "mrro_*_{year}*.nc"
        ),
    ),
    "height": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_static_data_cordex,
        oldname="orog",
        newname="height",
        template=os.path.join(config.cordex_dir, "{model}", "altitude", "orog_*.nc"),
    ),
    "CWT": dict(
        tasks_func=tasks_yearly_cordex,
        prepare_func=prepare_weather_types_cordex,
        oldname="CWT",
        newname="CWT",
        template=os.path.join(
            config.cordex_dir, "{model}", "weather_types", "CWT_*_{year}*.nc"
        ),
    ),
}

meta_data_config = dict(
    prepare_func=prepare_meta_cordex,
    template=os.path.join(
        config.cordex_dir, "{model}", "temperature", "tas_*_{year}*.nc"
    ),
    height_config=weather_data_config["height"],
)
