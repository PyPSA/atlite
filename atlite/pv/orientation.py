import sys
import numpy as np
import xarray as xr

def get_orientation(name, **params):
    '''
    Definitions:
        -`slope` is the angle between ground and panel.
        -`azimuth` is the clockwise angle from SOUTH.
            i.e. azimuth=0 faces exactly South,
                        =90 faces West
                        =-45 faces South-East
    '''
    if isinstance(name, dict):
        params = name
        name = params.pop('name', 'constant')
    return getattr(sys.modules[__name__], 'make_{}'.format(name))(**params)

def make_latitude_optimal():
    def latitude_optimal(lon, lat, solar_position):
        return dict(slope=lat, azimuth=0.)
    return latitude_optimal

def make_constant(slope, azimuth):
    slope = np.deg2rad(slope)
    azimuth = np.deg2rad(azimuth)

    def constant(lon, lat, solar_position):
        return dict(slope=slope, azimuth=azimuth)
    return constant

def SurfaceOrientation(ds, solar_position, orientation):
    lon = np.deg2rad(ds['lon'])
    lat = np.deg2rad(ds['lat'])

    orientation = orientation(lon, lat, solar_position)
    surface_slope = orientation['slope']
    surface_azimuth = orientation['azimuth']

    declination = solar_position['declination']
    hour_angle = solar_position['hour angle']

    theta = np.arccos( np.sin(lat)*np.sin(declination)*np.cos(surface_slope) \
                        - np.cos(lat)*np.sin(declination)*np.sin(surface_slope)*np.cos(surface_azimuth) \
                        + np.cos(lat)*np.cos(declination)*np.cos(hour_angle)*np.cos(surface_slope) \
                        + np.sin(lat)*np.cos(declination)*np.cos(hour_angle)*np.sin(surface_slope)*np.cos(surface_azimuth) \
                        + np.cos(declination)*np.sin(hour_angle)*np.sin(surface_slope)*np.sin(surface_azimuth) )

    return xr.Dataset({'incidence': theta,
                       'slope': surface_slope,
                       'azimuth': surface_azimuth})
