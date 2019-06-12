import os
import weakref
from functools import wraps
from six import string_types
import pandas as pd
from pandas.core.resample import TimeGrouper
import xarray as xr
import ast
import dask
from tempfile import mkstemp

import logging
logger = logging.getLogger(__name__)

from .utils import receive

def literal_eval_creation_parameters(node_or_string):
    """
    Safely evaluate an expression node or a string containing a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, bytes, numbers, tuples, lists, dicts,
    sets, booleans, slices and None.

    Variant: of ast.literal_eval
    """
    if isinstance(node_or_string, str):
        node_or_string = ast.parse(node_or_string, mode='eval')
    if isinstance(node_or_string, ast.Expression):
        node_or_string = node_or_string.body
    def _convert(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, (ast.Str, ast.Bytes)):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, ast.List):
            return list(map(_convert, node.elts))
        elif isinstance(node, ast.Set):
            return set(map(_convert, node.elts))
        elif isinstance(node, ast.Dict):
            return dict((_convert(k), _convert(v))
                        for k, v in zip(node.keys, node.values))
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _convert(node.operand)
            if isinstance(operand, ast._NUM_TYPES):
                if isinstance(node.op, ast.UAdd):
                    return + operand
                else:
                    return - operand
        elif isinstance(node, ast.Call) and node.func.id == 'slice':
            return slice(*map(_convert, node.args))

        raise ValueError('malformed creation parameters: ' + repr(node))

    return _convert(node_or_string)

def _get_creation_parameters(data):
    return literal_eval_creation_parameters(data.attrs['creation_parameters'])

def requires_coords(f):
    @wraps(f)
    def wrapper(cutout, *args, **kwargs):
        if not cutout.data.coords:
            creation_parameters = _get_creation_parameters(cutout.data)
            cutout.data = cutout.data.merge(cutout.dataset_module.get_coords(**creation_parameters))
        return f(cutout, *args, **kwargs)
    return wrapper


class requires_windowed(object):
    def __init__(self, features, windows=None, allow_dask=False):
        self.features = features
        self.windows = windows
        self.allow_dask = allow_dask

    def __call__(self, f):
        @wraps(f)
        def wrapper(cutout, *args, **kwargs):
            features = kwargs.pop('features', self.features)
            windows_params = kwargs.pop('windows', self.windows)
            windows = create_windows(cutout, features, windows_params, self.allow_dask)

            return f(cutout, *args, windows=windows, **kwargs)

        return wrapper

def create_windows(cutout, features, windows_params, allow_dask):
    features = set(features if features is not None else cutout.available_features)
    missing_features = features - set(cutout.data.attrs.get('prepared_features', []))

    if missing_features:
        logger.error(f"The following features need to be prepared first: {', '.join(missing_features)}. Will try anyway!")

    return Windows(cutout, features, windows_params, allow_dask)

class Windows(object):
    def __init__(self, cutout, features, params=None, allow_dask=False):
        group_kws = {}
        if params is None:
            group_kws['grouper'] = TimeGrouper(freq="M")
        elif isinstance(params, string_types):
            group_kws['grouper'] = TimeGrouper(freq=params)
        elif isinstance(params, int):
            group_kws['bins'] = params
        elif isinstance(params, (pd.Index, np.array)):
            group_kws['bins'] = params
        elif isinstance(params, dict):
            group_kws.update(params)
        else:
            raise RuntimeError(f"Type of `params` (`{type(params)}`) is unsupported")

        dataset_vars = sum((cutout.dataset_module.features[f] for f in features), [])
        vars = cutout.data.data_vars.keys() & dataset_vars
        self.data = cutout.data[list(vars)]
        self.group_kws = group_kws

        if self.data.chunks is None or allow_dask:
            self.maybe_instantiate = lambda it: it
        else:
            self.maybe_instantiate = lambda it: (ds.load() for ds in it)

        self.groupby = xr.core.groupby.DatasetGroupBy(self.data, self.data.coords['time'], **self.group_kws)

    def __iter__(self):
        return self.maybe_instantiate(self.groupby._iter_grouped())

    def __len__(self):
        return len(self.groupby)

def get_missing_data(cutout, features, monthly=False):
    creation_parameters = _get_creation_parameters(cutout.data)
    timeindex = cutout.coords.indexes['time']
    datasets = []
    for feature in features:
        if feature in cutout.dataset_module.static_features:
            creation_parameters.pop('month', None)
            feature_data = cutout.dataset_module.get_data(
                cutout.data.coords,
                timeindex[0],
                feature,
                **creation_parameters
            )
        else:
            feature_data = []
            for date in pd.date_range(timeindex[0], timeindex[-1], freq="MS" if monthly else "YS"):
                if monthly:
                    creation_parameters['month'] = date.month
                feature_data.append(
                    cutout.dataset_module.get_data(
                        cutout.data.coords,
                        date,
                        feature,
                        **creation_parameters
                    ))
            feature_data = dask.delayed(xr.concat)(feature_data, dim='time')

        datasets.append(feature_data)

    datasets, = dask.compute(datasets)

    return xr.merge(datasets, compat='identical')

@requires_coords
def cutout_prepare(cutout, features=None, monthly=False, overwrite=False):
    """
    Prepare all or a given set of `features`

    Download `features` in yearly or monthly slices and merge them into the
    cutout data.
    """
    if cutout.is_view:
        assert features is None, f"It's not possible to add features to a view, use `cutout.prepare()` to save it to {cutout.cutout_fn} first."
        assert not os.path.exists(cutout.cutout_fn) or overwrite, f"Not overwriting {cutout.cutout_fn} with a view, unless `overwrite=True`."

        ds = cutout.data
        if 'prepared_features' not in ds.attrs:
            logger.warn("Using empty `prepared_features`!")
            ds.attrs['prepared_features'] = []
    else:
        features = set(features if features is not None else cutout.available_features)
        missing_features = features - cutout.prepared_features

        ds = get_missing_data(cutout, missing_features, monthly)

        # Merge with existing cutout
        ds = xr.merge([cutout.data, ds])
        ds.attrs.update(cutout.data.attrs)
        ds.attrs['prepared_features'].extend(missing_features)

    # Write to a temporary file in the same directory first and then move back,
    # because we might still want to load data from the original file in the process
    directory, filename = os.path.split(cutout.cutout_fn)
    fd, target = mkstemp(suffix=filename, dir=directory)
    os.close(fd)

    ds.to_netcdf(target)
    ds.close()

    if os.path.exists(cutout.cutout_fn):
        os.unlink(cutout.cutout_fn)
    os.rename(target, cutout.cutout_fn)

    # Re-open
    cutout.data = xr.open_dataset(cutout.cutout_fn)
    prepared_features = cutout.data.attrs.get('prepared_features')
    if not isinstance(prepared_features, list):
        cutout.data.attrs['prepared_features'] = [prepared_features]

    # Save spatial index
    sindex_fn = os.path.join(cutout.cutout_dir, cutout.name + ".sindex.pickle")
    cutout._grid_cells.to_file(sindex_fn)
