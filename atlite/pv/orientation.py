# -*- coding: utf-8 -*-
import sys
import numpy as np
import xarray as xr

def get_orientation(name, **params):
    '''
    Definitions:
        -`slope` is the angle between ground and panel.
        -`azimuth` is the clockwise angle from North.
            i.e. azimuth = 180 faces exactly South
    '''
    if isinstance(name, dict):
        params = name
        name = params.pop('name', 'constant')
    return getattr(sys.modules[__name__], 'make_{}'.format(name))(**params)

def make_latitude_optimal():
    """
    Returns an optimal tilt angle for the given ``lat``, assuming that
    the panel is facing towards the equator, using a simple method from [1].

    This method only works for latitudes between 0 and 50. For higher
    latitudes, a static 40 degree angle is returned.

    These results should be used with caution, but there is some
    evidence that tilt angle may not be that important [2].

    Function and documentation has been adapted from gsee [3].

    [1] http://www.solarpaneltilt.com/#fixed
    [2] http://dx.doi.org/10.1016/j.solener.2010.12.014
    [3] https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    """

    def latitude_optimal(lon, lat, solar_position):
        if (lat < 0).any():
            raise NotImplementedError('Not implemented for negative latitudes')

        slope = np.empty_like(lat.values)

        below_25 = lat.values <= np.deg2rad(25)
        below_50 = lat.values <= np.deg2rad(50)

        slope[below_25] = 0.87 * lat.values[below_25]
        slope[~below_25 & below_50] = 0.76 * lat.values[~below_25 & below_50] + np.deg2rad(0.31)
        slope[~below_50] = np.deg2rad(40.)

        return dict(slope=xr.DataArray(slope, coords=lat.coords), azimuth=180.)

    return latitude_optimal

def make_constant(slope, azimuth):
    slope = np.deg2rad(slope)
    azimuth = np.deg2rad(azimuth)

    def constant(lon, lat, solar_position):
        return dict(slope=slope, azimuth=azimuth)
    return constant

def SurfaceOrientation(ds, solar_position, orientation):
    """
    Compute cos(incidence) for slope and panel azimuth

    References
    ----------
    [1] Sproul, A. B., Derivation of the solar geometric relationships using
    vector analysis, Renewable Energy, 32(7), 1187–1205 (2007).
    """

    lon = np.deg2rad(ds['lon'])
    lat = np.deg2rad(ds['lat'])

    orientation = orientation(lon, lat, solar_position)
    surface_slope = orientation['slope']
    surface_azimuth = orientation['azimuth']

    sun_altitude = solar_position['altitude']
    sun_azimuth = solar_position['azimuth']

    cosincidence = (np.sin(surface_slope) * np.cos(sun_altitude) *
                    np.cos(surface_azimuth - sun_azimuth) +
                    np.cos(surface_slope) * np.sin(sun_altitude))

    # fixup incidence angle: if the panel is badly oriented and the sun shines
    # on the back of the panel (incidence angle > 90degree), the irradiation
    # would be negative instead of 0; this is prevented here.
    cosincidence.values[cosincidence.values < 0.] = 0.

    return xr.Dataset({'cosincidence': cosincidence,
                       'slope': surface_slope,
                       'azimuth': surface_azimuth})
