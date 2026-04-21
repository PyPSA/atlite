# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""All functions for converting weather data into energy system model data."""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import absolute, arccos, cos, maximum, mod, radians, sin, sqrt
from dask.diagnostics import ProgressBar
from numpy import pi
from scipy.sparse import csr_matrix

from atlite import csp as cspm
from atlite import hydro as hydrom
from atlite import wind as windm
from atlite.aggregate import aggregate_matrix
from atlite.gis import spdiag
from atlite.pv.irradiation import TiltedIrradiation
from atlite.pv.orientation import SurfaceOrientation, get_orientation
from atlite.pv.solar_panel_model import SolarPanelModel
from atlite.pv.solar_position import SolarPosition
from atlite.resource import (
    get_cspinstallationconfig,
    get_solarpanelconfig,
    get_windturbineconfig,
    windturbine_smooth,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from atlite._types import (
        ClearskyModel,
        ConvertResult,
        HeatPumpSource,
        IrradiationType,
        OrientationName,
        TrackingType,
        TrigonModel,
    )
    from atlite.cutout import Cutout
    from atlite.resource import CSPConfig, PanelConfig, TurbineConfig


def _aggregate_time(
    da: xr.DataArray, method: Literal["sum", "mean"] | None
) -> xr.DataArray:
    if method == "sum":
        return da.sum("time", keep_attrs=True)
    if method == "mean":
        return da.mean("time", keep_attrs=True)
    return da


def convert_and_aggregate(
    cutout: Cutout,
    convert_func: Callable[..., Any],
    matrix: Any = None,
    index: Any = None,
    layout: Any = None,
    shapes: Any = None,
    shapes_crs: int = 4326,
    per_unit: bool = False,
    return_capacity: bool = False,
    aggregate_time: Literal["sum", "mean", "legacy"] | None = "legacy",
    capacity_factor: bool = False,
    capacity_factor_timeseries: bool = False,
    show_progress: bool = False,
    dask_kwargs: dict[str, Any] | None = None,
    **convert_kwds: Any,
) -> ConvertResult:
    """
    Convert and aggregate a weather-based renewable generation time-series.

    This is a gateway function called by the individual time-series
    generation functions like ``pv`` and ``wind``. All parameters documented
    here are also available from those functions.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    convert_func : callable
        Callback like convert_wind, convert_pv.
    matrix : N x S - xr.DataArray or sp.sparse.csr_matrix or None
        If given, it is used to aggregate the grid cells to buses.
        N is the number of buses, S the number of spatial coordinates, in the
        order of ``cutout.grid``.
    index : pd.Index
        Index of Buses.
    layout : X x Y - xr.DataArray
        The capacity to be build in each of the ``grid_cells``.
    shapes : list or pd.Series of shapely.geometry.Polygon
        If given, matrix is constructed as indicatormatrix of the polygons, its
        index determines the bus index on the time-series.
    shapes_crs : pyproj.CRS or compatible
        If different to the map crs of the cutout, the shapes are
        transformed to match cutout.crs (defaults to EPSG:4326).
    per_unit : boolean
        Returns the time-series in per-unit units, instead of in MW (defaults
        to False).
    return_capacity : boolean
        Additionally returns the installed capacity at each bus corresponding
        to ``layout`` (defaults to False).
    aggregate_time : "sum", "mean", "legacy", or None
        Controls temporal aggregation of results. ``"sum"`` sums over time,
        ``"mean"`` averages over time, ``None`` returns full timeseries.
        ``"legacy"`` (default) preserves historical behavior: time-summed
        without spatial aggregation and full timeseries with spatial
        aggregation; this option is deprecated and will be removed in a
        future release. Replaces the deprecated ``capacity_factor`` and
        ``capacity_factor_timeseries`` parameters.
    capacity_factor : boolean
        Deprecated. Use ``aggregate_time="mean"`` instead.
    capacity_factor_timeseries : boolean
        Deprecated. Use ``aggregate_time=None`` instead (which is the default).
    show_progress : boolean, default False
        Whether to show a progress bar.
    dask_kwargs : dict, default {}
        Dict with keyword arguments passed to ``dask.compute``.
    **convert_kwds : Any
        Additional keyword arguments passed to ``convert_func``.

    Returns
    -------
    resource : xr.DataArray
        The return value depends on which arguments are provided:

        **With aggregation** (``matrix``, ``shapes``, or ``layout`` given):
        Time-series of renewable generation aggregated to buses, with
        dimensions ``(bus, time)``. If ``aggregate_time`` is set, the time
        dimension is reduced accordingly.

        **Without aggregation** (none of the above given):

        - ``aggregate_time=None``: per-cell timeseries ``(time, y, x)``.
        - ``aggregate_time="mean"``: time-averaged per cell ``(y, x)``.
        - ``aggregate_time="sum"``: time-summed per cell ``(y, x)``.

        Legacy behavior (deprecated):

        - ``aggregate_time="legacy"``: historical context-dependent default.
        - ``capacity_factor_timeseries=True``: equivalent to
          ``aggregate_time=None``.
        - ``capacity_factor=True``: equivalent to ``aggregate_time="mean"``.

    units : xr.DataArray (optional)
        The installed units per bus in MW corresponding to ``layout``
        (only if ``return_capacity`` is True).

    Raises
    ------
    ValueError
        If deprecated parameters conflict or invalid arguments are provided.

    See Also
    --------
    wind : Generate wind generation time-series.
    pv : Generate solar PV generation time-series.

    """
    if aggregate_time not in ("sum", "mean", "legacy", None):
        raise ValueError(
            f"aggregate_time must be 'sum', 'mean', 'legacy', or None, "
            f"got {aggregate_time!r}"
        )

    if aggregate_time == "legacy":
        warnings.warn(
            "aggregate_time='legacy' is deprecated and will be removed in a "
            "future release. Pass 'sum', 'mean', or None explicitly.",
            FutureWarning,
            stacklevel=2,
        )

    if capacity_factor or capacity_factor_timeseries:
        if aggregate_time != "legacy":
            raise ValueError(
                "Cannot use 'aggregate_time' together with deprecated "
                "'capacity_factor' or 'capacity_factor_timeseries'."
            )
        if capacity_factor:
            warnings.warn(
                "capacity_factor is deprecated. Use aggregate_time='mean' instead.",
                FutureWarning,
                stacklevel=2,
            )
            aggregate_time = "mean"
        if capacity_factor_timeseries:
            warnings.warn(
                "capacity_factor_timeseries is deprecated. "
                "Use aggregate_time=None instead.",
                FutureWarning,
                stacklevel=2,
            )
            aggregate_time = None

    func_name = convert_func.__name__.replace("convert_", "")
    logger.info("Convert and aggregate '%s'.", func_name)
    da = convert_func(cutout.data, **convert_kwds)

    dask_kwargs = dask_kwargs or {}
    no_args = all(v is None for v in [layout, shapes, matrix])

    if no_args:
        if per_unit or return_capacity:
            raise ValueError(
                "One of `matrix`, `shapes` and `layout` must be "
                "given for `per_unit` or `return_capacity`"
            )

        agg = "sum" if aggregate_time == "legacy" else aggregate_time
        res = _aggregate_time(da, agg)
        return maybe_progressbar(res, show_progress, **dask_kwargs)

    if matrix is not None:
        if shapes is not None:
            raise ValueError(
                "Passing matrix and shapes is ambiguous. Pass only one of them."
            )

        if isinstance(matrix, xr.DataArray):
            coords = matrix.indexes[matrix.dims[1]].to_frame(index=False)
            if not np.array_equal(coords[["x", "y"]], cutout.grid[["x", "y"]]):
                raise ValueError(
                    "Matrix spatial coordinates not aligned with cutout spatial "
                    "coordinates."
                )

            if index is None:
                index = matrix

        if not matrix.ndim == 2:
            raise ValueError("Matrix not 2-dimensional.")

        matrix = csr_matrix(matrix)

    if shapes is not None:
        geoseries_like = (pd.Series, gpd.GeoDataFrame, gpd.GeoSeries)
        if isinstance(shapes, geoseries_like) and index is None:
            index = shapes.index

        matrix = cutout.indicatormatrix(shapes, shapes_crs)

    if layout is not None:
        assert isinstance(layout, xr.DataArray)
        layout = layout.reindex_like(cutout.data).stack(spatial=["y", "x"])

        if matrix is None:
            matrix = csr_matrix(layout.expand_dims("new"))
        else:
            matrix = csr_matrix(matrix) * spdiag(layout)

    # From here on, matrix is defined and ensured to be a csr matrix.
    if index is None:
        index = pd.RangeIndex(matrix.shape[0])

    results = aggregate_matrix(da, matrix=matrix, index=index)

    if per_unit or return_capacity:
        caps = matrix.sum(-1)
        capacity = xr.DataArray(np.asarray(caps).flatten(), [index])
        capacity.attrs["units"] = "MW"

    if per_unit:
        results = (results / capacity.where(capacity != 0)).fillna(0.0)
        results.attrs["units"] = "p.u."
    else:
        results.attrs["units"] = "MW"

    if aggregate_time != "legacy":
        results = _aggregate_time(results, aggregate_time)

    if return_capacity:
        return maybe_progressbar(results, show_progress, **dask_kwargs), capacity
    return maybe_progressbar(results, show_progress, **dask_kwargs)


def maybe_progressbar(
    ds: xr.Dataset | xr.DataArray, show_progress: bool, **kwargs: Any
) -> xr.Dataset | xr.DataArray:
    """
    Load a dataset or data array, optionally showing a dask progress bar.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Object backed by dask arrays.
    show_progress : bool
        Whether to display a progress bar while loading.
    **kwargs
        Keyword arguments passed to ``load``.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Loaded object.
    """
    if show_progress:
        with ProgressBar(minimum=2):
            ds.load(**kwargs)
    else:
        ds.load(**kwargs)
    return ds


# temperature
def convert_temperature(ds: xr.Dataset) -> xr.DataArray:
    """
    Convert ambient air temperature from Kelvin to degrees Celsius.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``temperature``.

    Returns
    -------
    xr.DataArray
        Ambient temperature in degrees Celsius.
    """
    # Temperature is in Kelvin
    return ds["temperature"] - 273.15


def temperature(cutout: Cutout, **params: Any) -> ConvertResult:
    """
    Return ambient air temperature converted from Kelvin to degrees Celsius.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Ambient temperature in °C.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.
    """
    return cutout.convert_and_aggregate(convert_func=convert_temperature, **params)


# soil temperature
def convert_soil_temperature(ds: xr.Dataset) -> xr.DataArray:
    """
    Convert soil temperature from Kelvin to degrees Celsius.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``soil temperature``.

    Returns
    -------
    xr.DataArray
        Soil temperature in degrees Celsius with missing values filled by zero.
    """
    # Temperature is in Kelvin

    # There are nans where there is sea; by setting them
    # to zero we guarantee they do not contribute when multiplied
    # by matrix in atlite/aggregate.py
    return (ds["soil temperature"] - 273.15).fillna(0.0)


def soil_temperature(cutout: Cutout, **params: Any) -> ConvertResult:
    """
    Return soil temperature converted from Kelvin to degrees Celsius.

    Sea grid cells, where soil temperature is undefined, are filled with 0.0
    so they do not contribute during spatial aggregation.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Soil temperature in °C.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.
    """
    return cutout.convert_and_aggregate(convert_func=convert_soil_temperature, **params)


# dewpoint temperature
def convert_dewpoint_temperature(ds: xr.Dataset) -> xr.DataArray:
    """
    Convert dew point temperature from Kelvin to degrees Celsius.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``dewpoint temperature``.

    Returns
    -------
    xr.DataArray
        Dew point temperature in degrees Celsius.
    """
    # Temperature is in Kelvin
    return ds["dewpoint temperature"] - 273.15


def dewpoint_temperature(cutout: Cutout, **params: Any) -> ConvertResult:
    """
    Return dew point temperature converted from Kelvin to degrees Celsius.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Dew point temperature in °C.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.
    """
    return cutout.convert_and_aggregate(
        convert_func=convert_dewpoint_temperature, **params
    )


def convert_coefficient_of_performance(
    ds: xr.Dataset,
    source: HeatPumpSource,
    sink_T: float,
    c0: float | None,
    c1: float | None,
    c2: float | None,
) -> xr.DataArray:
    """
    Convert source temperatures to heat pump COP values.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the required temperature variables.
    source : {"air", "soil"}
        Heat source used for the heat pump.
    sink_T : float
        Sink temperature in degrees Celsius.
    c0, c1, c2 : float or None
        Quadratic regression coefficients. If ``None``, source-specific
        defaults are used.

    Returns
    -------
    xr.DataArray
        Coefficient of performance for each time step and grid cell.
    """
    assert source in ["air", "soil"], NotImplementedError(
        "'source' must be one of  ['air', 'soil']"
    )

    if source == "air":
        source_T = convert_temperature(ds)
        if c0 is None:
            c0 = 6.81
        if c1 is None:
            c1 = -0.121
        if c2 is None:
            c2 = 0.000630
    elif source == "soil":
        source_T = convert_soil_temperature(ds)
        if c0 is None:
            c0 = 8.77
        if c1 is None:
            c1 = -0.150
        if c2 is None:
            c2 = 0.000734

    delta_T = sink_T - source_T

    return c0 + c1 * delta_T + c2 * delta_T**2


def coefficient_of_performance(
    cutout: Cutout,
    source: HeatPumpSource = "air",
    sink_T: float = 55.0,
    c0: float | None = None,
    c1: float | None = None,
    c2: float | None = None,
    **params: Any,
) -> ConvertResult:
    """
    Convert temperature to heat pump coefficient of performance (COP).

    The COP is modelled as a quadratic function of the temperature difference
    ``dT = sink_T - source_T``: ``COP = c0 + c1 * dT + c2 * dT**2``.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    source : {"air", "soil"}
        Heat source type. Default coefficients per source:

        - ``"air"``:  ``c0=6.81, c1=-0.121, c2=0.000630``
        - ``"soil"``: ``c0=8.77, c1=-0.150, c2=0.000734``
    sink_T : float, default 55.0
        Heat sink temperature in °C.
    c0 : float or None
        Constant regression coefficient. If ``None``, uses source default.
    c1 : float or None
        Linear regression coefficient. If ``None``, uses source default.
    c2 : float or None
        Quadratic regression coefficient. If ``None``, uses source default.
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Coefficient of performance time-series (dimensionless).

    See Also
    --------
    heat_demand : Compute heating degree-day demand.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.

    References
    ----------
    [1] Staffell, Brett, Brandon, Hawkes, A review of domestic heat pumps,
    Energy & Environmental Science (2012), 5, 9291-9306,
    https://doi.org/10.1039/C2EE22653G.
    """
    return cutout.convert_and_aggregate(
        convert_func=convert_coefficient_of_performance,
        source=source,
        sink_T=sink_T,
        c0=c0,
        c1=c1,
        c2=c2,
        **params,
    )


# heat demand
def convert_heat_demand(
    ds: xr.Dataset,
    threshold: float,
    a: float,
    constant: float,
    hour_shift: float,
) -> xr.DataArray:
    """
    Convert ambient temperature to daily heat demand by degree days.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``temperature``.
    threshold : float
        Heating threshold temperature in degrees Celsius.
    a : float
        Linear scaling factor.
    constant : float
        Constant demand component added to the result.
    hour_shift : float
        Time shift in hours applied before daily averaging.

    Returns
    -------
    xr.DataArray
        Daily heat demand in degree-day-like units.

    Notes
    -----
    The formula is ``max(0, a * (threshold - T_daily_mean)) + constant``
    where ``T_daily_mean`` is the daily-averaged temperature.
    """
    # Temperature is in Kelvin; take daily average
    T = ds["temperature"]
    T = T.assign_coords(
        time=(T.coords["time"] + np.timedelta64(dt.timedelta(hours=hour_shift)))
    )

    T = T.resample(time="1D").mean(dim="time")
    threshold += 273.15
    heat_demand = a * (threshold - T)

    heat_demand = heat_demand.clip(min=0.0)

    return (constant + heat_demand).rename("heat_demand")


def heat_demand(
    cutout: Cutout,
    threshold: float = 15.0,
    a: float = 1.0,
    constant: float = 0.0,
    hour_shift: float = 0.0,
    **params: Any,
) -> ConvertResult:
    """
    Convert outside temperature into daily heat demand using degree-day approximation.

    The formula is ``max(0, a * (threshold - T_daily_mean)) + constant``
    where ``T_daily_mean`` is the daily-averaged temperature. Output is in
    degree-day-like units (scaled by *a*).

    Since "daily average temperature" means different things in different time
    zones, you can provide *hour_shift* to redefine when the day starts.
    E.g. for Moscow in winter ``hour_shift=4``, for New York ``hour_shift=-5``.
    The shift applies uniformly across all grid cells and times.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    threshold : float, default 15.0
        Outside temperature in °C above which there is no heat demand.
    a : float, default 1.0
        Linear factor relating heat demand to outside temperature.
    constant : float, default 0.0
        Constant part of heat demand independent of outside temperature
        (e.g. water heating).
    hour_shift : float, default 0.0
        Time shift in hours relative to UTC for daily averaging.
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Daily heat demand time-series in degree-day-like units.

    Warnings
    --------
    Because the original data is provided per month, at month boundaries
    there is untidiness when using a time shift. The resulting array will
    have duplicate indices for parts of the day at each boundary. You must
    re-average these based on the number of hours in each month.

    See Also
    --------
    cooling_demand : Degree-day cooling demand.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.
    """
    return cutout.convert_and_aggregate(
        convert_func=convert_heat_demand,
        threshold=threshold,
        a=a,
        constant=constant,
        hour_shift=hour_shift,
        **params,
    )


# cooling demand
def convert_cooling_demand(
    ds: xr.Dataset,
    threshold: float,
    a: float,
    constant: float,
    hour_shift: float,
) -> xr.DataArray:
    """
    Convert ambient temperature to daily cooling demand by degree days.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``temperature``.
    threshold : float
        Cooling threshold temperature in degrees Celsius.
    a : float
        Linear scaling factor.
    constant : float
        Constant demand component added to the result.
    hour_shift : float
        Time shift in hours applied before daily averaging.

    Returns
    -------
    xr.DataArray
        Daily cooling demand in degree-day-like units.

    Notes
    -----
    The formula is ``max(0, a * (T_daily_mean - threshold)) + constant``
    where ``T_daily_mean`` is the daily-averaged temperature.
    """
    # Temperature is in Kelvin; take daily average
    T = ds["temperature"]
    T = T.assign_coords(
        time=(T.coords["time"] + np.timedelta64(dt.timedelta(hours=hour_shift)))
    )

    T = T.resample(time="1D").mean(dim="time")
    threshold += 273.15
    cooling_demand = a * (T - threshold)

    cooling_demand = cooling_demand.clip(min=0.0)

    return (constant + cooling_demand).rename("cooling_demand")


def cooling_demand(
    cutout: Cutout,
    threshold: float = 23.0,
    a: float = 1.0,
    constant: float = 0.0,
    hour_shift: float = 0.0,
    **params: Any,
) -> ConvertResult:
    """
    Convert outside temperature into daily cooling demand using degree-day approximation.

    The formula is ``max(0, a * (T_daily_mean - threshold)) + constant``
    where ``T_daily_mean`` is the daily-averaged temperature. Output is in
    degree-day-like units (scaled by *a*).

    Since "daily average temperature" means different things in different time
    zones, you can provide *hour_shift* to redefine when the day starts.
    E.g. for Moscow in summer ``hour_shift=3``, for New York ``hour_shift=-4``.
    The shift applies uniformly across all grid cells and times.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    threshold : float, default 23.0
        Outside temperature in °C below which there is no cooling demand.
        The default follows European computational practices (UK Met Office
        uses 22 °C, European Commission uses 24 °C).
    a : float, default 1.0
        Linear factor relating cooling demand to outside temperature.
    constant : float, default 0.0
        Constant part of cooling demand independent of outside temperature
        (e.g. ventilation).
    hour_shift : float, default 0.0
        Time shift in hours relative to UTC for daily averaging.
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Daily cooling demand time-series in degree-day-like units.

    Warnings
    --------
    Because the original data is provided per month, at month boundaries
    there is untidiness when using a time shift. The resulting array will
    have duplicate indices for parts of the day at each boundary. You must
    re-average these based on the number of hours in each month.

    See Also
    --------
    heat_demand : Degree-day heating demand.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.
    """
    return cutout.convert_and_aggregate(
        convert_func=convert_cooling_demand,
        threshold=threshold,
        a=a,
        constant=constant,
        hour_shift=hour_shift,
        **params,
    )


# solar thermal collectors
def convert_solar_thermal(
    ds: xr.Dataset,
    orientation: Callable,
    trigon_model: TrigonModel,
    clearsky_model: ClearskyModel | None,
    c0: float,
    c1: float,
    t_store: float,
) -> xr.DataArray:
    """
    Convert weather data to solar thermal collector output.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing radiation and temperature variables.
    orientation : callable
        Surface orientation callback.
    trigon_model : str
        Trigonometric irradiation model.
    clearsky_model : str or None
        Clear-sky model used for diffuse irradiation.
    c0, c1 : float
        Collector efficiency parameters.
    t_store : float
        Storage temperature in degrees Celsius.

    Returns
    -------
    xr.DataArray
        Specific solar thermal output in W/m².

    Notes
    -----
    Collector efficiency is ``eta = c0 - c1 * (T_store - T_amb) / G`` where
    *G* is the tilted irradiation. Output is ``max(0, G * eta)``.
    """
    # convert storage temperature to Kelvin in line with reanalysis data
    t_store += 273.15

    # Downward shortwave radiation flux is in W/m^2
    # http://rda.ucar.edu/datasets/ds094.0/#metadata/detailed.html?_do=y
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation)
    irradiation = TiltedIrradiation(
        ds, solar_position, surface_orientation, trigon_model, clearsky_model
    )

    # overall efficiency; can be negative, so need to remove negative values
    # below
    eta = c0 - c1 * (
        (t_store - ds["temperature"]) / irradiation.where(irradiation != 0)
    ).fillna(0)

    output = irradiation * eta

    return output.where(output > 0.0, 0.0)


def solar_thermal(
    cutout: Cutout,
    orientation: OrientationName | dict[str, float] | Callable | None = None,
    trigon_model: TrigonModel = "simple",
    clearsky_model: ClearskyModel = "simple",
    c0: float = 0.8,
    c1: float = 3.0,
    t_store: float = 80.0,
    **params: Any,
) -> ConvertResult:
    """
    Convert radiation and temperature into solar thermal collector time series.

    Collector efficiency is ``eta = c0 - c1 * (T_store - T_amb) / G``.
    Mathematical model and defaults for *c0*, *c1* based on [1].

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    orientation : dict, str, or callable, optional
        Panel orientation. A dict with ``'slope'`` and ``'azimuth'`` keys
        in degrees, the string ``'latitude_optimal'``, or a callable with
        the same signature as callbacks from
        ``atlite.pv.orientation.make_*``. Default: ``{'slope': 45.0,
        'azimuth': 180.0}``.
    trigon_model : {"simple", "perez"}, default "simple"
        Trigonometric model for tilted irradiation decomposition.
    clearsky_model : {"simple", "enhanced"} or None, default "simple"
        Clear-sky model for diffuse irradiation. ``'enhanced'`` also uses
        ambient temperature and relative humidity.
    c0 : float, default 0.8
        Optical efficiency parameter.
    c1 : float, default 3.0
        Thermal loss coefficient in W/(m² K).
    t_store : float, default 80.0
        Storage temperature in °C.
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Solar thermal generation time-series.

    See Also
    --------
    pv : Photovoltaic generation.
    irradiation : Tilted surface irradiation.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.

    References
    ----------
    [1] Henning and Palzer, Renewable and Sustainable Energy Reviews 30
    (2014) 1003-1018
    """
    if orientation is None:
        orientation = {"slope": 45.0, "azimuth": 180.0}
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(
        convert_func=convert_solar_thermal,
        orientation=orientation,
        trigon_model=trigon_model,
        clearsky_model=clearsky_model,
        c0=c0,
        c1=c1,
        t_store=t_store,
        **params,
    )


# wind
def convert_wind(
    ds: xr.Dataset,
    turbine: TurbineConfig,
    interpolation_method: Literal["logarithmic", "power"],
) -> xr.Dataset | xr.DataArray:
    """
    Convert wind speeds to turbine-specific generation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing wind speed data.
    turbine : TurbineConfig
        Turbine configuration with power curve and hub height.
    interpolation_method : {"logarithmic", "power"}
        Method used to extrapolate wind speed to hub height.

    Returns
    -------
    xr.DataArray
        Wind power output as specific yield per unit of installed capacity.
    """
    V, POW, hub_height, P = itemgetter("V", "POW", "hub_height", "P")(turbine)

    wnd_hub = windm.extrapolate_wind_speed(
        ds, to_height=hub_height, method=interpolation_method
    )

    def apply_power_curve(da):
        return np.interp(da, V, POW / P)

    da = xr.apply_ufunc(
        apply_power_curve,
        wnd_hub,
        input_core_dims=[[]],
        output_core_dims=[[]],
        output_dtypes=[wnd_hub.dtype],
        dask="parallelized",
    )
    assert isinstance(da, xr.DataArray)

    da.attrs["units"] = "MWh/MWp"
    return da.rename("specific generation")


def wind(
    cutout: Cutout,
    turbine: str | Path | dict[str, Any],
    smooth: bool | dict[str, Any] = False,
    add_cutout_windspeed: bool = False,
    interpolation_method: Literal["logarithmic", "power"] = "logarithmic",
    **params: Any,
) -> ConvertResult:
    """
    Generate wind generation time-series.

    Extrapolates wind speed to hub height (using logarithmic or power law) and
    evaluates the power curve.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    turbine : str or dict
        A turbineconfig dictionary with the keys 'hub_height' for the
        hub height and 'V', 'POW' defining the power curve.
        Alternatively a str refering to a local or remote turbine configuration
        as accepted by atlite.resource.get_windturbineconfig(). Locally stored turbine
        configurations can also be modified with this function. E.g. to setup a different hub
        height from the one used in the yaml file,one would write
                "turbine=get_windturbineconfig(“NREL_ReferenceTurbine_5MW_offshore”)|{“hub_height”:120}"
    smooth : bool or dict
        If True smooth power curve with a gaussian kernel as
        determined for the Danish wind fleet to Delta_v = 1.27 and
        sigma = 2.29. A dict allows to tune these values.
    add_cutout_windspeed : bool
        If True and in case the power curve does not end with a zero, will add zero power
        output at the highest wind speed in the power curve. If False, a warning will be
        raised if the power curve does not have a cut-out wind speed. The default is
        False.
    interpolation_method : {"logarithmic", "power"}
        Law to interpolate wind speed to turbine hub height. Refer to
        :py:func:`atlite.wind.extrapolate_wind_speed`.
    **params : Any
        Additional keyword arguments passed to `convert_and_aggregate`.

    Returns
    -------
    resource : xr.DataArray
        Wind generation time-series. Without aggregation, values are capacity
        factors (MWh/MWp). With aggregation and ``per_unit=False``, values are
        in MW. See :py:func:`convert_and_aggregate` for details.

    See Also
    --------
    pv : Photovoltaic generation.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.

    References
    ----------
    .. [1] Andresen G B, Søndergaard A A and Greiner M 2015 Energy 93, Part 1
       1074 – 1088. doi:10.1016/j.energy.2015.09.071

    Examples
    --------
    Aggregate wind generation to bus regions:

    >>> wind = cutout.wind(turbine="Vestas_V112_3MW", matrix=matrix,
    ...                    index=buses, per_unit=True)

    Get per-cell capacity factor time series (no aggregation):

    >>> cf = cutout.wind(turbine="Vestas_V112_3MW",
    ...                  aggregate_time=None)
    >>> cf.dims
    ('time', 'y', 'x')
    >>> location_cf = cf.sel(x=6.9, y=53.1, method="nearest")

    """
    turbine_config = get_windturbineconfig(
        turbine, add_cutout_windspeed=add_cutout_windspeed
    )

    if smooth:
        turbine_config = windturbine_smooth(turbine_config, params=smooth)

    return cutout.convert_and_aggregate(
        convert_func=convert_wind,
        turbine=turbine_config,
        interpolation_method=interpolation_method,
        **params,
    )


# irradiation
def convert_irradiation(
    ds: xr.Dataset,
    orientation: Callable,
    tracking: TrackingType | None = None,
    irradiation: IrradiationType = "total",
    trigon_model: TrigonModel = "simple",
    clearsky_model: ClearskyModel | None = "simple",
) -> xr.DataArray:
    """
    Convert weather data to irradiation on a tilted surface.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing radiation and meteorological variables.
    orientation : callable
        Surface orientation callback.
    tracking : {None, "horizontal", "tilted_horizontal", "vertical", "dual"}, optional
        Tracking mode of the surface.
    irradiation : {"total", "direct", "diffuse", "ground"}, default "total"
        Irradiation component to return.
    trigon_model : str, default "simple"
        Trigonometric irradiation model.
    clearsky_model : str or None, default "simple"
        Clear-sky model used for diffuse irradiation.

    Returns
    -------
    xr.DataArray
        Tilted surface irradiation in W/m².
    """
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation, tracking)
    return TiltedIrradiation(
        ds,
        solar_position,
        surface_orientation,
        trigon_model=trigon_model,
        clearsky_model=clearsky_model,
        tracking=tracking,
        irradiation=irradiation,
    )


def irradiation(
    cutout: Cutout,
    orientation: OrientationName | dict[str, float] | Callable,
    irradiation: IrradiationType = "total",
    tracking: TrackingType | None = None,
    clearsky_model: ClearskyModel | None = None,
    **params: Any,
) -> ConvertResult:
    """
    Calculate irradiation on a tilted surface.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    orientation : str, dict or callback
        Panel orientation can be chosen from either
        'latitude_optimal', a constant orientation {'slope': 0.0,
        'azimuth': 0.0} or a callback function with the same signature
        as the callbacks generated by the
        'atlite.pv.orientation.make_*' functions.
    irradiation : str
        The irradiation quantity to be returned. Defaults to "total" for total
        combined irradiation. Other options include "direct" for direct irradiation,
        "diffuse" for diffuse irradation, and "ground" for irradiation reflected
        by the ground via albedo. NOTE: "ground" irradiation is not calculated
        by all `trigon_model` options in the `convert_irradiation` method,
        so use with caution!
    tracking : None or str:
        None for no tracking, default
        'horizontal' for 1-axis horizontal tracking
        'tilted_horizontal' for 1-axis horizontal tracking with tilted axis
        'vertical' for 1-axis vertical tracking
        'dual' for 2-axis tracking
    clearsky_model : str or None
        Either the 'simple' or the 'enhanced' Reindl clearsky
        model. The default choice of None will choose dependending on
        data availability, since the 'enhanced' model also
        incorporates ambient air temperature and relative humidity.
    **params : Any
        Additional keyword arguments passed to `convert_and_aggregate`.

    Returns
    -------
    irradiation : xr.DataArray
        Irradiation on the tilted surface in W/m².

    See Also
    --------
    pv : Photovoltaic generation.
    solar_thermal : Solar thermal collector output.

    Notes
    -----
    The ``trigon_model`` is fixed to ``'simple'`` internally.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.

    References
    ----------
    [1] D.T. Reindl, W.A. Beckman, and J.A. Duffie. Diffuse fraction correla-
    tions. Solar Energy, 45(1):1 – 7, 1990.
    """
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(
        convert_func=convert_irradiation,
        orientation=orientation,
        tracking=tracking,
        irradiation=irradiation,
        clearsky_model=clearsky_model,
        **params,
    )


# solar PV
def convert_pv(
    ds: xr.Dataset,
    panel: dict[str, Any],
    orientation: Callable,
    tracking: TrackingType,
    trigon_model: TrigonModel = "simple",
    clearsky_model: ClearskyModel | None = "simple",
) -> xr.DataArray:
    """
    Convert weather data to photovoltaic specific generation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing radiation and temperature variables.
    panel : dict
        Solar panel configuration.
    orientation : callable
        Surface orientation callback.
    tracking : {None, "horizontal", "tilted_horizontal", "vertical", "dual"}
        Tracking mode of the panel.
    trigon_model : str, default "simple"
        Trigonometric irradiation model.
    clearsky_model : str or None, default "simple"
        Clear-sky model used for diffuse irradiation.

    Returns
    -------
    xr.DataArray
        PV power output as capacity factors (unitless, 0–1).
    """
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation, tracking)
    irradiation = TiltedIrradiation(
        ds,
        solar_position,
        surface_orientation,
        trigon_model=trigon_model,
        clearsky_model=clearsky_model,
        tracking=tracking,
    )
    return SolarPanelModel(ds, irradiation, panel)


def pv(
    cutout: Cutout,
    panel: str | PanelConfig,
    orientation: OrientationName | dict[str, float] | Callable,
    tracking: TrackingType | None = None,
    clearsky_model: ClearskyModel | None = None,
    **params: Any,
) -> ConvertResult:
    """
    Convert radiation and temperature into PV generation time series.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    panel : str or dict
        Panel config dictionary with the parameters for the electrical
        model in [3]. Alternatively, a name accepted by
        :py:func:`atlite.resource.get_solarpanelconfig`.
    orientation : str, dict or callback
        Panel orientation can be chosen from either
        'latitude_optimal', a constant orientation {'slope': 0.0,
        'azimuth': 0.0} or a callback function with the same signature
        as the callbacks generated by the
        'atlite.pv.orientation.make_*' functions.
    tracking : None or str:
        None for no tracking, default
        'horizontal' for 1-axis horizontal tracking
        'tilted_horizontal' for 1-axis horizontal tracking with tilted axis
        'vertical' for 1-axis vertical tracking
        'dual' for 2-axis tracking
    clearsky_model : str or None
        Either the 'simple' or the 'enhanced' Reindl clearsky
        model. The default choice of None will choose dependending on
        data availability, since the 'enhanced' model also
        incorporates ambient air temperature and relative humidity.
    **params : Any
        Additional keyword arguments passed to `convert_and_aggregate`.

    Returns
    -------
    pv : xr.DataArray
        PV generation time-series. Without aggregation, values are capacity
        factors (unitless, 0–1). With aggregation and ``per_unit=False``,
        values are in MW. See :py:func:`convert_and_aggregate` for details.

    See Also
    --------
    wind : Wind generation.
    irradiation : Tilted surface irradiation.
    solar_thermal : Solar thermal collector output.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.

    References
    ----------
    [1] Soteris A. Kalogirou. Solar Energy Engineering: Processes and Systems,
    pages 49–117,469–516. Academic Press, 2009. ISBN 0123745012.

    [2] D.T. Reindl, W.A. Beckman, and J.A. Duffie. Diffuse fraction correla-
    tions. Solar Energy, 45(1):1 – 7, 1990.

    [3] Hans Georg Beyer, Gerd Heilscher and Stefan Bofinger. A Robust Model
    for the MPP Performance of Different Types of PV-Modules Applied for
    the Performance Check of Grid Connected Systems, Freiburg, June 2004.
    Eurosun (ISES Europe Solar Congress).


    Examples
    --------
    Aggregate PV generation to bus regions:

    >>> pv = cutout.pv(panel="CSi", orientation="latitude_optimal",
    ...                matrix=matrix, index=buses, per_unit=True)

    Get per-cell capacity factor time series (no aggregation):

    >>> cf = cutout.pv(panel="CSi", orientation="latitude_optimal",
    ...                aggregate_time=None)
    >>> location_cf = cf.sel(x=6.9, y=53.1, method="nearest")
    """
    if isinstance(panel, (str | Path)):
        panel = get_solarpanelconfig(panel)
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(
        convert_func=convert_pv,
        panel=panel,
        orientation=orientation,
        tracking=tracking,
        clearsky_model=clearsky_model,
        **params,
    )


# solar CSP
def convert_csp(ds, installation):
    """
    Convert direct solar radiation to CSP specific generation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing direct radiation variables.
    installation : dict
        CSP installation configuration.

    Returns
    -------
    xr.DataArray
        CSP output as specific yield (kWh/kW_ref), clipped to [0, 1].

    Raises
    ------
    ValueError
        If the CSP technology option is not recognized.
    """
    solar_position = SolarPosition(ds)

    tech = installation["technology"]
    if tech == "parabolic trough":
        irradiation = ds["influx_direct"]
    elif tech == "solar tower":
        irradiation = cspm.calculate_dni(ds, solar_position)
    else:
        raise ValueError(f'Unknown CSP technology option "{tech}".')

    # Determine solar_position dependend efficiency for each grid cell and time step
    efficiency = installation["efficiency"].interp(
        altitude=solar_position["altitude"], azimuth=solar_position["azimuth"]
    )

    # Thermal system output
    da = efficiency * irradiation

    # output relative to reference irradiance
    da /= installation["r_irradiance"]

    # Limit output to max of reference irradiance
    da = da.clip(max=1.0)

    # Fill NaNs originating from DNI or solar positions outside efficiency bounds
    da = da.fillna(0.0)

    da.attrs["units"] = "kWh/kW_ref"
    return da.rename("specific generation")


def csp(
    cutout: Cutout,
    installation: str | CSPConfig,
    technology: Literal["parabolic trough", "solar tower"] | None = None,
    **params: Any,
) -> ConvertResult:
    """
    Convert direct radiation into CSP generation time series.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    installation : str or xr.DataArray
        CSP installation details determining the solar field efficiency
        dependent on the local solar position. Can be a name accepted by
        :py:func:`atlite.resource.get_cspinstallationconfig` or an
        ``xr.DataArray`` with ``'azimuth'`` (rad) and ``'altitude'`` (rad)
        coordinates and an ``'efficiency'`` (p.u.) entry.
    technology : {"parabolic trough", "solar tower"} or None
        Overwrite CSP technology from the installation configuration.
        ``'parabolic trough'`` uses direct horizontal irradiance (DHI),
        ``'solar tower'`` uses direct normal irradiance (DNI).
    **params
        Additional keyword arguments passed to `convert_and_aggregate`.

    Returns
    -------
    csp : xr.DataArray
        CSP generation time-series in specific yield (kWh/kW_ref), clipped
        to [0, 1]. See :py:func:`convert_and_aggregate` for details on
        aggregation behaviour.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.

    References
    ----------
    [1] Tobias Hirsch (ed.). SolarPACES Guideline for Bankable STE Yield Assessment,
    IEA Technology Collaboration Programme SolarPACES, 2017.
    URL: https://www.solarpaces.org/csp-research-tasks/task-annexes-iea/task-i-solar-thermal-electric-systems/solarpaces-guideline-for-bankable-ste-yield-assessment/

    [2] Tobias Hirsch (ed.). CSPBankability Project Report, DLR, 2017.
    URL: https://www.dlr.de/sf/en/desktopdefault.aspx/tabid-11126/19467_read-48251/

    """
    if isinstance(installation, (str | Path)):
        installation = get_cspinstallationconfig(installation)

    # Overwrite technology
    if technology is not None:
        installation["technology"] = technology

    return cutout.convert_and_aggregate(
        convert_func=convert_csp,
        installation=installation,
        **params,
    )


# hydro
def convert_runoff(ds, weight_with_height=True):
    """
    Convert runoff data, optionally weighting by surface height.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``runoff`` and, if needed, ``height``.
    weight_with_height : bool, default True
        Whether to weight runoff by terrain height.

    Returns
    -------
    xr.DataArray
        Runoff field, optionally weighted by surface height.
    """
    runoff = ds["runoff"]

    if weight_with_height:
        runoff = runoff * ds["height"]

    return runoff


def runoff(
    cutout,
    smooth=None,
    lower_threshold_quantile=None,
    normalize_using_yearly=None,
    **params,
):
    """
    Compute aggregated surface runoff with optional smoothing and normalization.

    Parameters
    ----------
    cutout : atlite.Cutout
        Cutout providing weather data with runoff variables.
    smooth : bool or int, optional
        If ``True``, apply a rolling mean with the default window of
        ``24 * 7`` time steps. If an integer, use it as the rolling window
        size. Default ``None`` (no smoothing).
    lower_threshold_quantile : bool or float, optional
        If ``True``, use the default quantile ``5e-3``. If a float, set
        values below that quantile to zero. Default ``None`` (no
        thresholding).
    normalize_using_yearly : pd.Series, optional
        Annual country totals used to scale ``countries``-indexed results over
        overlapping full years. One factor per country is derived from the
        summed reference values across the overlap.
    **params
        Additional keyword arguments passed to ``convert_and_aggregate()``,
        including ``weight_with_height`` for the underlying runoff
        conversion.

    Returns
    -------
    xr.DataArray or tuple[xr.DataArray, xr.DataArray]
        Runoff output from ``convert_and_aggregate``. Smoothing also supports
        the tuple return form used with ``return_capacity=True``. Thresholding
        and normalization are only supported for ``xr.DataArray`` results.

    See Also
    --------
    convert_and_aggregate : General conversion/aggregation arguments.
    """
    result = cutout.convert_and_aggregate(convert_func=convert_runoff, **params)

    if smooth is not None:
        if smooth is True:
            smooth = 24 * 7
        if "return_capacity" in params:
            result = result[0].rolling(time=smooth, min_periods=1).mean(), result[1]
        else:
            result = result.rolling(time=smooth, min_periods=1).mean()

    if lower_threshold_quantile is not None:
        if lower_threshold_quantile is True:
            lower_threshold_quantile = 5e-3
        lower_threshold = pd.Series(result.values.ravel()).quantile(
            lower_threshold_quantile
        )
        result = result.where(result >= lower_threshold, 0.0)

    if normalize_using_yearly is not None:
        normalize_using_yearly_i = normalize_using_yearly.index
        if isinstance(normalize_using_yearly_i, pd.DatetimeIndex):
            normalize_using_yearly_i = normalize_using_yearly_i.year
        else:
            normalize_using_yearly_i = normalize_using_yearly_i.astype(int)

        years = (
            pd
            .Series(pd.to_datetime(result.coords["time"].values).year)
            .value_counts()
            .loc[lambda x: x > 8700]
            .index.intersection(normalize_using_yearly_i)
        )
        assert len(years), "Need at least a full year of data (more is better)"
        years_overlap = slice(str(min(years)), str(max(years)))

        dim = result.dims[1 - result.get_axis_num("time")]
        result *= (
            xr.DataArray(normalize_using_yearly.loc[years_overlap].sum(), dims=[dim])
            / result.sel(time=years_overlap).sum("time")
        ).reindex(countries=result.coords["countries"])

    return result


def hydro(
    cutout,
    plants,
    hydrobasins,
    flowspeed=1,
    weight_with_height=False,
    show_progress=False,
    **kwargs,
):
    """
    Compute inflow time series for plants by aggregating over catchment basins.

    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
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
    **kwargs
        Additional keyword arguments passed to `convert_and_aggregate`.

    Returns
    -------
    xr.DataArray
        Inflow time-series for each plant.

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
    basins = hydrom.determine_basins(plants, hydrobasins, show_progress=show_progress)

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
    runoff *= xr.DataArray(basins.shapes.to_crs({"proj": "cea"}).area)

    return hydrom.shift_and_aggregate_runoff_for_plants(
        basins, runoff, flowspeed, show_progress
    )


def convert_line_rating(
    ds, psi, R, D=0.028, Ts=373, epsilon=0.6, alpha=0.6, per_unit=False
):
    """
    Convert weather data to dynamic line rating time series.

    The formulation is based on:
    [1] "IEEE Std 738™-2012 (Revision of IEEE Std 738-2006/Incorporates IEEE Std
        738-2012/Cor 1-2013), IEEE Standard for Calculating the Current-Temperature
        Relationship of Bare Overhead Conductors," p. 72.

    Simplifications:
    1. Wind speed is taken at 100 m above ground, whereas transmission lines are
       typically at 50-60 m.
    2. Solar heat influx is set proportional to solar short wave influx.
    3. Incidence angle of the solar heat influx is assumed to be 90°.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for the cells intersecting a line.
    psi : float
        Line azimuth in degrees clockwise from north.
    R : float
        Conductor resistance in Ω/m at temperature *Ts*.
    D : float, default 0.028
        Conductor diameter in meters.
    Ts : float, default 373
        Maximum conductor surface temperature in Kelvin.
    epsilon : float, default 0.6
        Conductor emissivity (dimensionless).
    alpha : float, default 0.6
        Conductor absorptivity (dimensionless).
    per_unit : bool, default False
        Unused compatibility parameter.

    Returns
    -------
    xr.DataArray or numpy.ndarray
        Maximum current per time step in ampere. When *ds* is an
        ``xr.Dataset`` the result is aggregated across intersecting cells
        via ``.min("spatial")``.
    """
    Ta = ds["temperature"]
    Tfilm = (Ta + Ts) / 2
    T0 = 273.15

    # 1. Convective Loss, at first forced convection
    V = ds["wnd100m"]  # typically ironmen are about 40-60 meters high
    mu = (1.458e-6 * Tfilm**1.5) / (
        Tfilm + 383.4 - T0
    )  # Dynamic viscosity of air (13a)
    H = ds["height"]
    rho = (1.293 - 1.525e-4 * H + 6.379e-9 * H**2) / (
        1 + 0.00367 * (Tfilm - T0)
    )  # (14a)

    reynold = D * V * rho / mu

    k = (
        2.424e-2 + 7.477e-5 * (Tfilm - T0) - 4.407e-9 * (Tfilm - T0) ** 2
    )  # thermal conductivity
    anglediff = ds["wnd_azimuth"] - radians(psi)
    Phi = absolute(mod(anglediff + pi / 2, pi) - pi / 2)
    K = (
        1.194 - cos(Phi) + 0.194 * cos(2 * Phi) + 0.368 * sin(2 * Phi)
    )  # wind direction factor

    Tdiff = Ts - Ta
    qcf1 = K * (1.01 + 1.347 * reynold**0.52) * k * Tdiff  # (3a) in [1]
    qcf2 = K * 0.754 * reynold**0.6 * k * Tdiff  # (3b) in [1]

    qcf = maximum(qcf1, qcf2)

    #  natural convection
    qcn = 3.645 * sqrt(rho) * D**0.75 * Tdiff**1.25

    # convection loss is the max between forced and natural
    qc = maximum(qcf, qcn)

    # 2. Radiated Loss
    qr = 17.8 * D * epsilon * ((Ts / 100) ** 4 - (Ta / 100) ** 4)

    # 3. Solar Radiance Heat Gain
    Q = ds["influx_direct"]  # assumption, this is short wave and not heat influx
    A = D * 1  # projected area of conductor in square meters

    if isinstance(ds, dict):
        altitude = ds["solar_altitude"]
        azimuth = ds["solar_azimuth"]
    else:
        sp = SolarPosition(ds)
        altitude = sp["altitude"]
        azimuth = sp["azimuth"]
    Phi_s = arccos(cos(altitude) * cos(azimuth - radians(psi)))

    qs = alpha * Q * A * sin(Phi_s)

    Imax = sqrt((qc + qr - qs) / R)
    return Imax.min("spatial") if isinstance(Imax, xr.DataArray) else Imax


def line_rating(
    cutout, shapes, line_resistance, show_progress=False, dask_kwargs=None, **params
):
    """
    Create a dynamic line rating time series based on the IEEE-738 standard.

    [1].

    The steady-state capacity is derived from the balance between heat
    losses due to radiation and convection, and heat gains due to solar influx
    and conductur resistance. For more information on assumptions and modifications
    see ``convert_line_rating``.


    [1]“IEEE Std 738™-2012 (Revision of IEEE Std 738-2006/Incorporates IEEE Std
        738-2012/Cor 1-2013), IEEE Standard for Calculating the Current-Temperature
        Relationship of Bare Overhead Conductors,” p. 72.


    Parameters
    ----------
    cutout : atlite.Cutout
        The cutout to process.
    shapes : geopandas.GeoSeries
        Line shapes of the lines.
    line_resistance : float/series
        Resistance of the lines in Ohm/meter. Alternatively in p.u. system in
        Ohm/1000km (see example below).
    show_progress : boolean, default False
        Whether to show a progress bar.
    dask_kwargs : dict, default {}
        Dict with keyword arguments passed to `dask.compute`.
    D : float, default 0.028
        Conductor diameter in meters.
    Ts : float, default 373
        Maximum allowed conductor surface temperature in Kelvin.
    epsilon : float, default 0.6
        Conductor emissivity (dimensionless).
    alpha : float, default 0.6
        Conductor absorptivity (dimensionless).
    **params : Any
        Additional keyword arguments passed to
        :py:func:`convert_line_rating`.

    Returns
    -------
    xr.DataArray
        Thermal current limit time-series with dimensions
        ``(time, lines)`` in ampere.

    Note
    ----
    This function also accepts all keyword arguments of
    :py:func:`convert_and_aggregate`.

    Example
    -------

    >>> import pypsa
    >>> import xarray as xr
    >>> import atlite
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point, LineString as Line

    >>> n = pypsa.examples.scigrid_de()
    >>> n.calculate_dependent_values()
    >>> x = n.buses.x
    >>> y = n.buses.y
    >>> buses = n.lines[["bus0", "bus1"]].values
    >>> shapes = [Line([Point(x[b0], y[b0]), Point(x[b1], y[b1])]) for (b0, b1) in buses]
    >>> shapes = gpd.GeoSeries(shapes, index=n.lines.index)

    >>> cutout = atlite.Cutout('test', x=slice(x.min(), x.max()), y=slice(y.min(), y.max()),
                            time='2020-01-01', module='era5', dx=1, dy=1)
    >>> cutout.prepare()

    >>> i = cutout.line_rating(shapes, n.lines.r/n.lines.length)
    >>> v = xr.DataArray(n.lines.v_nom, dims='name')
    >>> s = np.sqrt(3) * i * v / 1e3 # in MW

    """
    if dask_kwargs is None:
        dask_kwargs = {}
    if not isinstance(shapes, gpd.GeoSeries):
        shapes = gpd.GeoSeries(shapes).rename_axis("dim_0")

    I = cutout.intersectionmatrix(shapes)
    rows, cols = I.nonzero()

    data = cutout.data.stack(spatial=["y", "x"])

    def get_azimuth(shape):
        """
        Return the line azimuth in degrees from its end points.

        Parameters
        ----------
        shape : shapely.geometry.base.BaseGeometry
            Line geometry.

        Returns
        -------
        float
            Azimuth angle in degrees computed from the line end points.
        """
        coords = np.array(shape.coords)
        start = coords[0]
        end = coords[-1]
        return np.degrees(np.arctan2(start[0] - end[0], start[1] - end[1]))

    azimuth = shapes.apply(get_azimuth)
    azimuth = azimuth.where(azimuth >= 0, azimuth + 180.0)

    params.setdefault("D", 0.028)
    params.setdefault("Ts", 373)
    params.setdefault("epsilon", 0.6)
    params.setdefault("alpha", 0.6)

    df = pd.DataFrame({"psi": azimuth, "R": line_resistance}).assign(**params)

    assert df.notnull().all().all(), "Nan values encountered."
    assert df.columns.equals(pd.Index(["psi", "R", "D", "Ts", "epsilon", "alpha"]))

    dummy = xr.DataArray(np.full(len(data.time), np.nan), coords=(data.time,))
    res = []
    for i in range(len(df)):
        cells_i = cols[rows == i]
        if cells_i.size:
            ds = data.isel(spatial=cells_i)
            res.append(delayed(convert_line_rating)(ds, *df.iloc[i].values))
        else:
            res.append(dummy)
    if show_progress:
        with ProgressBar(minimum=2):
            (computed,) = compute(res, **dask_kwargs)
    else:
        (computed,) = compute(res, **dask_kwargs)

    return xr.concat(computed, dim=df.index).assign_attrs(units="A")
