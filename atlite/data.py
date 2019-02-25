from functools import wraps
from ast import literal_eval
from six import string_types
from pandas.core.resample import TimeGrouper
from dask import delayed
import xarray as xr

import logging
logger = logging.getLogger(__name__)

from .utils import receive
from .config import features as available_features

def _get_creation_parameters(data):
    return literal_eval(data.attrs['creation_parameters'])

def requires_coords(f):
    @wraps(f)
    def wrapper(cutout):
        if not cutout.data.coords:
            creation_parameters = _get_creation_parameters(cutout.data)
            cutout.data.coords = cutout.dataset_module.get_coords(**creation_parameters)
        return f(cutout)
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
    features = set(features if features is not None else available_features)
    missing_features = features - set(cutout.data.attrs.get('prepared_features', []))

    if not missing_features:
        return Windows(cutout.data, features, windows_params, allow_dask)
    else:
        logger.warn(f"Sideloading features {', '.join(missing_features)}")
        missing_data = get_missing_data(cutout, missing_features, allow_dask)
        return SideloadWindows(cutout.data, features, missing_data, windows_params, allow_dask)

def get_missing_data(cutout, features, allow_dask):
    creation_parameters = _get_creation_parameters(cutout.data)
    timeindex = cutout.index['time']
    for date in pd.daterange(timeindex[0], timeindex[-1], freq="MS"):
        monthdata = [delayed(cutout.dataset_module.get_data)(cutout.data.coords, s, date, **creation_parameters)
                     for s in features]
        yield (date, delayed(xr.concat)(monthdata, compat='identical'))

class Windows(object):
    def __init__(self, data, features, params=None, allow_dask=False):
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

        vars = data.data_vars.keys() & sum((available_features[f] for f in features), [])
        self.data = data[list(vars)]
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



class SideloadWindows(object):
    def __init__(self, data, features, missing_data, params=None, allow_dask=False):
        assert params is None or params == 'M', (
            f"We only support monthly chunks for data side-loading for now. "
             " Use prepare to save at least the features {', '.join(features)} into a cutout file."
        )

        vars = data.data_vars.keys() & sum((available_features[f] for f in features), [])
        self.data = data[list(vars)]

        self.missing_data = missing_data

    def __iter__(self):
        # TODO can we do something with allow_dask here?
        for date, missing_data_task in self.missing_data:
            with receive(missing_data_task.compute()) as missing_ds:
                yield xr.concat([self.data.sel(time=date.strftime("%Y-%m")), missing_ds],
                                compat="identical").load()

    def __len__(self):
        return None
