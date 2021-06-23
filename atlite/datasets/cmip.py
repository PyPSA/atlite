"""
Module for downloading and preparing data from the ESGF servers 
to be used in atlite


"""

import xarray as xr
from pyesgf.search import SearchConnection

import logging
import yaml
import dask
from pathlib import Path
import pkg_resources
from ..gis import maybe_swap_spatial_dims
import numpy as np

# Null context for running a with statements wihout any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib

    @contextlib.contextmanager
    def nullcontext():
        yield


logger = logging.getLogger(__name__)

features = {
    "wind": ["wnd10m"],
    "influx": ["influx", "outflux"],
    "temperature": ["temperature"],
    "runoff": ["runoff"],
}
CMIP_SETUP_FILE = Path(pkg_resources.resource_filename(__name__, "cmip.yml"))


crs = 4326

static_features = {"height"}

dask.config.set({"array.slicing.split_large_chunks": True})


def search_ESGF(esgf_params, url="https://esgf-data.dkrz.de/esg-search"):
    conn = SearchConnection(url, distrib=True)
    ctx = conn.new_context(latest=True, **esgf_params)

    if ctx.hit_count == 0:
        ctx = ctx.constrain(frequency=esgf_params["frequency"] + "Pt")
        if ctx.hit_count == 0:
            raise (ValueError("No results found in the ESGF_database"))
    latest = ctx.search()[0]
    return latest.file_context().search()


def get_data_runoff(esgf_params, cutout, **retrieval_params):
    """Get runoff data for given retrieval parameters"""
    coords = cutout.coords
    ds = retrieve_data(
        esgf_params,
        coords,
        variables=["mrro"],
        **retrieval_params,
    )
    ds = _rename_and_fix_coords(ds, cutout.dt)
    ds = ds.rename({"mrro": "runoff"})
    return ds


def get_data_influx(esgf_params, cutout, **retrieval_params):
    """Get influx data for given retrieval parameters."""
    coords = cutout.coords
    ds = retrieve_data(
        esgf_params,
        coords,
        variables=["rsds", "rsus"],
        **retrieval_params,
    )

    ds = _rename_and_fix_coords(ds, cutout.dt)

    ds = ds.rename({"rsds": "influx", "rsus": "outflux"})

    return ds


def get_data_temperature(esgf_params, cutout, **retrieval_params):
    """Get temperature for given retrieval parameters."""
    coords = cutout.coords
    ds = retrieve_data(esgf_params, coords, variables=["tas"], **retrieval_params)

    ds = _rename_and_fix_coords(ds, cutout.dt)
    ds = ds.rename({"tas": "temperature"})
    ds = ds.drop_vars("height")

    return ds


def get_data_wind(esgf_params, cutout, **retrieval_params):
    """Get wind for given retrieval parameters"""

    coords = cutout.coords
    ds = retrieve_data(esgf_params, coords, ["sfcWind"], **retrieval_params)
    ds = _rename_and_fix_coords(ds, cutout.dt)
    ds = ds.rename({"sfcWind": "wnd{:0d}m".format(int(ds.sfcWind.height.values))})
    ds = ds.drop_vars("height")
    return ds


def retrieve_data(
    esgf_params, coords, variables, chunks=None, tmpdir=None, lock=None, **updates
):
    """
    Download data from egsf database

    """
    time = coords["time"].to_index()
    years = time.year.unique()
    dsets = []
    for variable in variables:
        esgf_params["variable"] = variable
        search_results = search_ESGF(esgf_params)
        files = [
            f.opendap_url
            for f in search_results
            if int(f.opendap_url.split("_")[-1][:4]) in years
        ]

        dsets.append(xr.open_mfdataset(files, chunks=chunks, concat_dim=["time"]))
    ds = xr.merge(dsets)

    ds.attrs = {**ds.attrs, **esgf_params}

    return ds


def _rename_and_fix_coords(ds, dt, add_lon_lat=True, add_ctime=False):
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and longitude
    columns as 'lat' and 'lon'.

    CMIP specifics; shif the longitude from 0..360 to -180..180. In addition
    CMIP sometimes specify the time in the center of the output intervall this shifted
    to the beginning.
    """
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    ds.lon.attrs["valid_max"] = 180
    ds.lon.attrs["valid_min"] = -180
    ds = ds.sortby("lon")

    ds = ds.rename({"lon": "x", "lat": "y"})
    # round coords since cds coords are float32 which would lead to mismatches
    # ds = ds.assign_coords(
    #     x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    # )

    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    if add_ctime:
        ds = ds.assign_coords(ctime=ds.coords["time"])

    # shift averaged data to beginning of bin
    ds = ds.assign_coords(time=ds.coords["time"].dt.floor(dt))

    return ds


def get_data(cutout, feature, tmpdir, lock=None, **creation_parameters):
    """
    Retrieve data from the ESGF CMIP database.

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.era5.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    **creation_parameters :
        Additional keyword arguments. The only effective argument is 'sanitize'
        (default True) which sets sanitization of the data on or off.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

    """
    coords = cutout.coords

    # sanitize = creation_parameters.get("sanitize", True)

    if cutout.esgf_params == None:
        with open(CMIP_SETUP_FILE, "r") as f:
            cmip_params = yaml.safe_load(f)

        model = creation_parameters.get("model")
        if model:
            try:
                esgf_params = cmip_params[model]
            except:
                KeyError(f"{model} not reconized, update cmip.yml")
        else:
            raise (ValueError("Model not specified"))
    else:
        esgf_params = cutout.esgf_params
    if esgf_params.get("frequency") == None:
        if cutout.dt == "H":
            freq = "h"
        elif cutout.dt == "3H":
            freq = "3hr"
        elif cutout.dt == "6H":
            freq = "6hr"
        elif cutout.dt == "D":
            freq = "day"
        elif cutout.dt == "M":
            freq = "mon"
        elif cutout.dt == "Y":
            freq = "year"
        else:
            raise (ValueError(f"{cutout.dt} not valid time frequency in CMIP"))

    retrieval_params = {"chunks": cutout.chunks, "tmpdir": tmpdir, "lock": lock}

    func = globals().get(f"get_data_{feature}")
    # sanitize_func = globals().get(f"sanitize_{feature}")

    logger.info(f"Requesting data for feature {feature}...")
    ds = func(esgf_params, cutout, **retrieval_params)
    ds = ds.sel(time=coords["time"])
    bounds = cutout.bounds
    ds = ds.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[1], bounds[3]))
    ds = ds.interp({"x": cutout.data.x, "y": cutout.data.y})
    return ds
