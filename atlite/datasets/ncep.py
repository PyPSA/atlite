# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module containing specific operations for creating cutouts from the NCEP dataset.

DEPRECATED
----------

The ncep dataset module has not been ported to Atlite v0.2, yet. Use Atlite v0.0.2 to use it,
for the time being!
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import glob
import tempfile
import subprocess
import shutil

engine = "pynio"
crs = 4326


def convert_lons_lats_ncep(ds, xs, ys):
    if not isinstance(xs, slice):
        first, second, last = np.asarray(xs)[[0, 1, -1]]
        xs = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))
    if not isinstance(ys, slice):
        first, second, last = np.asarray(ys)[[0, 1, -1]]
        ys = slice(first - 0.1 * (second - first), last + 0.1 * (second - first))

    ds = ds.sel(lat_0=ys)

    # Lons should go from -180. to +180.
    if len(ds.coords["lon_0"].sel(lon_0=slice(xs.start + 360.0, xs.stop + 360.0))):
        ds = xr.concat(
            [ds.sel(lon_0=slice(xs.start + 360.0, xs.stop + 360.0)), ds.sel(lon_0=xs)],
            dim="lon_0",
        )
        ds = ds.assign_coords(
            lon_0=np.where(
                ds.coords["lon_0"].values <= 180,
                ds.coords["lon_0"].values,
                ds.coords["lon_0"].values - 360.0,
            )
        )
    else:
        ds = ds.sel(lon_0=xs)

    ds = ds.rename({"lon_0": "x", "lat_0": "y"})
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds


def convert_time_hourly_ncep(ds, drop_time_vars=True):
    # Combine initial_time0 and forecast_time0
    ds = ds.stack(time=("initial_time0_hours", "forecast_time0")).assign_coords(
        time=np.ravel(
            ds.coords["initial_time0_hours"]
            + ds.coords["forecast_time0"]
            - pd.Timedelta("1h")
        )
    )
    if drop_time_vars:
        ds = ds.drop(["initial_time0", "initial_time0_encoded"])
    return ds


def convert_unaverage_ncep(ds):
    # the fields ending in _avg contain averages which have to be unaveraged by using
    # \begin{equation}
    # \tilde x_1 = x_1 \quad \tilde x_i = i \cdot x_i - (i - 1) \cdot x_{i-1} \quad \forall i > 1
    # \end{equation}

    def unaverage(da, dim="forecast_time0"):
        coords = da.coords[dim]
        y = da * xr.DataArray(
            np.arange(1, len(coords) + 1), dims=[dim], coords={dim: coords}
        )
        return y - y.shift(**{dim: 1}).fillna(0.0)

    for k, da in ds.items():
        if k.endswith("_avg"):
            ds[k[: -len("_avg")]] = unaverage(da)
            ds = ds.drop(k)

    return ds


def convert_unaccumulate_ncep(ds):
    # the fields ending in _acc contain values that are accumulated over the
    # forecast_time which have to be unaccumulated by using:
    # \begin{equation}
    # \tilde x_1 = x_1
    # \tilde x_i = x_i - x_{i-1} \forall 1 < i <= 6
    # \end{equation}
    # Source:
    # http://rda.ucar.edu/datasets/ds094.1/#docs/FAQs_hrly_timeseries.html

    def unaccumulate(da, dim="forecast_time0"):
        return da - da.shift(**{dim: 1}).fillna(0.0)

    for k, da in ds.items():
        if k.endswith("_acc"):
            ds[k[: -len("_acc")]] = unaccumulate(da)
            ds = ds.drop(k)

    return ds


def convert_clip_lower(ds, variable, a_min, value):
    """
    Set values of `variable` that are below `a_min` to `value`.
    Similar to `numpy.clip`.
    """
    ds[variable] = ds[variable].where(ds[variable] > a_min).fillna(value)
    return ds


def prepare_wnd10m_ncep(fn, yearmonth, xs, ys, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds)
        ds["wnd10m"] = np.sqrt(
            ds["VGRD_P0_L103_GGA0"] ** 2 + ds["UGRD_P0_L103_GGA0"] ** 2
        )
        ds = ds.drop(["VGRD_P0_L103_GGA0", "UGRD_P0_L103_GGA0"])
        yield yearmonth, ds


def prepare_influx_ncep(fn, yearmonth, xs, ys, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_unaverage_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"DSWRF_P8_L1_GGA0": "influx"})
        # clipping random fluctuations around zero
        ds = convert_clip_lower(ds, "influx", a_min=0.1, value=0.0)
        yield yearmonth, ds


def prepare_outflux_ncep(fn, yearmonth, xs, ys, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_unaverage_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"USWRF_P8_L1_GGA0": "outflux"})
        # clipping random fluctuations around zero
        ds = convert_clip_lower(ds, "outflux", a_min=3.0, value=0.0)
        yield yearmonth, ds


def prepare_temperature_ncep(fn, yearmonth, xs, ys, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"TMP_P0_L103_GGA0": "temperature"})
        yield yearmonth, ds


def prepare_soil_temperature_ncep(fn, yearmonth, xs, ys, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"TMP_P0_2L106_GGA0": "soil temperature"})
        yield yearmonth, ds


def prepare_runoff_ncep(fn, yearmonth, xs, ys, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        # runoff has missing values: set nans to 0
        ds = ds.fillna(0.0)
        ds = convert_unaccumulate_ncep(ds)
        ds = convert_time_hourly_ncep(ds)

        ds = ds.rename({"WATR_P8_L1_GGA0": "runoff"})
        yield yearmonth, ds


def prepare_height_ncep(fn, xs, ys, yearmonths, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = ds.rename({"HGT_P0_L105_GGA0": "height"})
        for ym in yearmonths:
            yield ym, ds


def prepare_roughness_ncep(fn, yearmonth, xs, ys, engine=engine):
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = ds.rename({"SFCR_P8_L1_GGA0": "roughness"})
        ds = ds.assign_coords(year=yearmonth[0]).assign_coords(month=yearmonth[1])
        yield yearmonth, ds


def prepare_meta_ncep(
    xs, ys, year, month, template, height_config, module, engine=engine
):
    fn = next(glob.iglob(template.format(year=year, month=month)))
    with xr.open_dataset(fn, engine=engine) as ds:
        ds = ds.coords.to_dataset()
        ds = convert_lons_lats_ncep(ds, xs, ys)
        ds = convert_time_hourly_ncep(ds, drop_time_vars=False)
        meta = ds.load()

    xs = ds["x"].values
    ys = ds["y"].values

    height_config = height_config.copy()
    height_tasks_func = height_config.pop("tasks_func")
    (height_task,) = height_tasks_func(
        xs, ys, [(year, month)], meta_attrs={}, **height_config
    )
    height_prepare_func = height_task.pop("prepare_func")
    _, ds = next(height_prepare_func(**height_task))

    meta["height"] = ds["height"]

    return meta


def tasks_monthly_ncep(xs, ys, yearmonths, prepare_func, template, meta_attrs):
    return [
        dict(
            prepare_func=prepare_func,
            xs=xs,
            ys=ys,
            fn=next(glob.iglob(template.format(year=ym[0], month=ym[1]))),
            engine=engine,
            yearmonth=ym,
        )
        for ym in yearmonths
    ]


def tasks_height_ncep(
    xs, ys, yearmonths, prepare_func, template, meta_attrs, **extra_args
):
    return [
        dict(
            prepare_func=prepare_func,
            xs=xs,
            ys=ys,
            yearmonths=yearmonths,
            fn=next(glob.iglob(template)),
            **extra_args
        )
    ]


weather_data_config = {
    "influx": dict(
        tasks_func=tasks_monthly_ncep,
        prepare_func=prepare_influx_ncep,
        template=os.path.join(config.ncep_dir, "{year}{month:0>2}/dswsfc.*.grb2"),
    ),
    "outflux": dict(
        tasks_func=tasks_monthly_ncep,
        prepare_func=prepare_outflux_ncep,
        template=os.path.join(config.ncep_dir, "{year}{month:0>2}/uswsfc.*.grb2"),
    ),
    "temperature": dict(
        tasks_func=tasks_monthly_ncep,
        prepare_func=prepare_temperature_ncep,
        template=os.path.join(config.ncep_dir, "{year}{month:0>2}/tmp2m.*.grb2"),
    ),
    "soil temperature": dict(
        tasks_func=tasks_monthly_ncep,
        prepare_func=prepare_soil_temperature_ncep,
        template=os.path.join(config.ncep_dir, "{year}{month:0>2}/soilt1.*.grb2"),
    ),
    "wnd10m": dict(
        tasks_func=tasks_monthly_ncep,
        prepare_func=prepare_wnd10m_ncep,
        template=os.path.join(config.ncep_dir, "{year}{month:0>2}/wnd10m.*.grb2"),
    ),
    "runoff": dict(
        tasks_func=tasks_monthly_ncep,
        prepare_func=prepare_runoff_ncep,
        template=os.path.join(config.ncep_dir, "{year}{month:0>2}/runoff.*.grb2"),
    ),
    "roughness": dict(
        tasks_func=tasks_monthly_ncep,
        prepare_func=prepare_roughness_ncep,
        template=os.path.join(config.ncep_dir, "{year}{month:0>2}/flxf.gdas.*.grb2"),
    ),
    "height": dict(
        tasks_func=tasks_height_ncep,
        prepare_func=prepare_height_ncep,
        template=os.path.join(config.ncep_dir, "height/cdas1.20130101.splgrbanl.grb2"),
    ),
}

meta_data_config = dict(
    prepare_func=prepare_meta_ncep,
    template=os.path.join(config.ncep_dir, "{year}{month:0>2}/tmp2m.*.grb2"),
    height_config=weather_data_config["height"],
)
