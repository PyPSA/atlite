# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

from warnings import warn

import pandas as pd
import xarray as xr
from dask.array import arccos, arcsin, arctan2, cos, radians, sin
from numpy import pi


def SolarPosition(ds, time_shift="0H"):
    """
    Compute solar azimuth and altitude.

    Solar altitude errors are up to 1.5 deg during sun-rise and set, but at
    0.05-0.1 deg during daytime.

    Parameters
    ----------
    ds: xr.DataSet
        DataSet for which the solar positions are calculated.
    time_shift: str or pandas.TimeDelta (optional)
        Time shift to apply before the solar position calculations. Useful
        for datasets representing aggregate data (e.g. ERA5) instead of
        instantenous data (e.g. SARAH). Must be parseable by pandas.to_timedelta().
        Default: "0H"

    References
    ----------
    [1] Michalsky, J. J., The astronomical almanac’s algorithm for approximate
    solar position (1950–2050), Solar Energy, 40(3), 227–235 (1988).
    [2] Sproul, A. B., Derivation of the solar geometric relationships using
    vector analysis, Renewable Energy, 32(7), 1187–1205 (2007).
    [3] Kalogirou, Solar Energy Engineering (2009).

    More accurate algorithms would be
    ---------------------------------
    [4] I. Reda and A. Andreas, Solar position algorithm for solar
    radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
    [5] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
    solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007.
    [6] Blanc, P., & Wald, L., The SG2 algorithm for a fast and accurate
    computation of the position of the sun for multi-decadal time period, Solar
    Energy, 86(10), 3072–3083 (2012).

    The unfortunately quite computationally intensive SPA algorithm [4,5] has
    been implemented using numba or plain numpy for a single location at
    https://github.com/pvlib/pvlib-python/blob/master/pvlib/spa.py.

    """
    # Act like a getter if these return variables are already in ds
    rvs = {
        "solar_azimuth",
        "solar_altitude",
    }

    if rvs.issubset(set(ds.data_vars)):
        return ds[rvs].rename({v: v.replace("solar_", "") for v in rvs})

    warn(
        """The calculation method and handling of solar position variables will change.
    The solar position will in the future be a permanent variables of a cutout.
    Recreate your cutout to remove this warning and permanently include the solar position variables into your cutout.""",
        DeprecationWarning,
    )

    # up to h and dec from [1]

    time_shift = pd.to_timedelta(time_shift)

    t = ds.indexes["time"] + time_shift
    n = xr.DataArray(t.to_julian_date(), coords=ds["time"].coords) - 2451545.0
    hour = (ds["time"] + time_shift).dt.hour
    minute = (ds["time"] + time_shift).dt.minute

    # Operations make new DataArray eager; reconvert to lazy dask arrays
    chunks = ds.chunksizes.get("time", "auto")
    if n.ndim == 1:
        chunks = chunks[0]
    n = n.chunk(chunks)
    hour = hour.chunk(chunks)
    minute = minute.chunk(chunks)

    L = 280.460 + 0.9856474 * n  # mean longitude (deg)
    g = radians(357.528 + 0.9856003 * n)  # mean anomaly (rad)
    l = radians(L + 1.915 * sin(g) + 0.020 * sin(2 * g))  # ecliptic long. (rad)
    ep = radians(23.439 - 4e-7 * n)  # obliquity of the ecliptic (rad)

    ra = arctan2(cos(ep) * sin(l), cos(l))  # right ascencion (rad)
    lmst = (6.697375 + (hour + minute / 60.0) + 0.0657098242 * n) * 15.0 + ds[
        "lon"
    ]  # local mean sidereal time (deg)
    h = (radians(lmst) - ra + pi) % (2 * pi) - pi  # hour angle (rad)

    dec = arcsin(sin(ep) * sin(l))  # declination (rad)

    # alt and az from [2]
    lat = radians(ds["lat"])
    # Clip before arcsin to prevent values < -1. from rounding errors; can
    # cause NaNs later
    alt = arcsin(
        (sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(h)).clip(min=-1.0, max=1.0)
    ).rename("altitude")
    alt.attrs["time shift"] = f"{time_shift}"
    alt.attrs["units"] = "rad"

    az = arccos(
        ((sin(dec) * cos(lat) - cos(dec) * sin(lat) * cos(h)) / cos(alt)).clip(
            min=-1.0, max=1.0
        )
    )
    az = az.where(h <= 0, 2 * pi - az).rename("azimuth")
    az.attrs["time shift"] = f"{time_shift}"
    az.attrs["units"] = "rad"

    vars = {da.name: da for da in [alt, az]}
    solar_position = xr.Dataset(vars)

    return solar_position
