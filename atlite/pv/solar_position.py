# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xarray as xr

def SolarPosition(ds):
    """
    Compute solar azimuth and altitude

    Solar altitude errors are up to 1.5 deg during sun-rise and set, but at
    0.05-0.1 deg during daytime.

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

    # up to h and dec from [1]

    t = ds.indexes['time']
    n = xr.DataArray(t.to_julian_date(), [ds.indexes['time']]) - 2451545.0

    L = 280.460 + 0.9856474 * n # mean longitude (deg)
    g = np.deg2rad(357.528 + 0.9856003 * n) # mean anomaly (rad)
    l = np.deg2rad(L + 1.915 * np.sin(g) + 0.020 * np.sin(2*g)) # ecliptic long. (rad)
    ep = np.deg2rad(23.439 - 4e-7 * n) # obliquity of the ecliptic (rad)

    ra = np.arctan2(np.cos(ep) * np.sin(l), np.cos(l)) # right ascencion (rad)
    lmst = (6.697375 + (ds['time.hour'] + ds['time.minute'] / 60.0) +
            0.0657098242 * n) * 15. + ds['lon'] # local mean sidereal time (deg)
    h = (np.deg2rad(lmst) - ra + np.pi) % (2*np.pi) - np.pi # hour angle (rad)

    dec = np.arcsin(np.sin(ep) * np.sin(l))            # declination (rad)

    # alt and az from [2]
    lat = np.deg2rad(ds['lat'])
    # Clip alt before arcsin to prevent values < -1. from rounding errors; can cause NaNs later
    alt = (np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(h))
    alt = np.arcsin(alt.clip(min=-1., max=1.)).rename('altitude')

    az = np.arccos(((np.sin(dec)*np.cos(lat) - np.cos(dec)*np.sin(lat)*np.cos(h))/np.cos(alt)).clip(min=-1., max=1.))
    az = az.where(h <= 0, 2*np.pi - az).rename('azimuth')

    if 'influx_toa' in ds:
        atmospheric_insolation = ds['influx_toa'].rename('atmospheric insolation')
    else:
        # [3]
        atmospheric_insolation = (1366.1 * (1+0.033*np.cos(g)) * np.sin(alt)).rename('atmospheric insolation')

    solar_position = xr.Dataset({da.name: da
                                 for da in [alt,
                                            az,
                                            atmospheric_insolation]})
    return solar_position
