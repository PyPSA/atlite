import numpy as np
from six import string_types
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

    orig_prepped = list(map(prep, orig))
    dest = reproject(dest, dest_proj, orig_proj)

    indicator = sparse.lil_matrix((len(dest), len(orig)), dtype=np.float)
    for i,j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest[i]):
            area = orig[j].intersection(dest[i]).area
            indicator[i,j] = area/orig[j].area

    return indicator
