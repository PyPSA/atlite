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
import pandas as pd
import scipy as sp, scipy.sparse
from collections import OrderedDict
from six import string_types, iteritems
from six.moves import map, range
from itertools import product
from functools import partial
import pyproj
from shapely.prepared import prep
from shapely.ops import transform

def spdiag(v):
    N = len(v)
    inds = np.arange(N+1, dtype=np.int32)
    return sp.sparse.csr_matrix((v, inds[:-1], inds), (N, N))

class RotProj(pyproj.Proj):
    def __call__(self, x, y, inverse=False, **kw):
        if inverse:
            gx, gy = super(RotProj, self).__call__(x, y,
                                                   inverse=False, **kw)
            return np.rad2deg(gx), np.rad2deg(gy)
        else:
            return super(RotProj, self).__call__(np.deg2rad(x),
                                                 np.deg2rad(y),
                                                 inverse=True, **kw)

def as_projection(p):
    if isinstance(p, pyproj.Proj):
        return p
    elif isinstance(p, string_types):
        return pyproj.Proj(dict(proj=p))
    else:
        return pyproj.Proj(p)

def reproject(shapes, p1, p2):
    """
    Project a collection of `shapes` from one projection `p1` to
    another projection `p2`

    Projections can be given as strings or instances of pyproj.Proj.
    Special care is taken for the case where the final projection is
    of type rotated pole as handled by RotProj.
    """

    if p1 == p2:
        return shapes

    if isinstance(p2, RotProj):
        shapes = reproject(shapes, p1, 'latlong')
        reproject_points = p2
    else:
        reproject_points = partial(pyproj.transform, as_projection(p1), as_projection(p2))

    def reproject_shape(shape):
        return transform(reproject_points, shape)

    if isinstance(shapes, pd.Series):
        return shapes.map(reproject_shape)
    elif isinstance(shapes, dict):
        return OrderedDict((k, reproject_shape(v)) for k, v in iteritems(shapes))
    else:
        return list(map(reproject_shape, shapes))

def compute_indicatormatrix(orig, dest, orig_proj='latlong', dest_proj='latlong'):
    """
    Compute the indicatormatrix

    The indicatormatrix I[i,j] is a sparse representation of the ratio
    of the area in orig[j] lying in dest[i], where orig and dest are
    collections of polygons, i.e.

    A value of I[i,j] = 1 indicates that the shape orig[j] is fully
    contained in shape dest[j].

    Note that the polygons must be in the same crs.

    Parameters
    ---------
    orig : Collection of shapely polygons
    dest : Collection of shapely polygons

    Returns
    -------
    I : sp.sparse.lil_matrix
      Indicatormatrix
    """

    dest = reproject(dest, dest_proj, orig_proj)
    dest_prepped = list(map(prep, dest))

    indicator = sp.sparse.lil_matrix((len(dest), len(orig)), dtype=np.float)
    for i,j in product(range(len(dest)), range(len(orig))):
        if dest_prepped[i].intersects(orig[j]):
            area = dest[i].intersection(orig[j]).area
            indicator[i,j] = area/orig[j].area

    return indicator
