## Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Renewable Energy Atlas Lite (Atlite)

Light-weight version of Aarhus RE Atlas for converting weather data to power systems data
"""

from __future__ import absolute_import

import numpy as np
import pyproj as pp

from .config import lon_0, o_lon_p, o_lat_p

# lon, lat are geographic coords; the inverse transforms from geographic to projected coords
def to_projected_plane(lon, lat, lon_0=lon_0, o_lon_p=o_lon_p, o_lat_p=o_lat_p, inverse=True):
    
    projection = pp.Proj(proj = 'ob_tran', o_proj = 'longlat',
                         lon_0 = lon_0, o_lon_p = o_lon_p, o_lat_p = o_lat_p)
    
    return projection(lon*np.pi/180, lat*np.pi/180, inverse=inverse)
