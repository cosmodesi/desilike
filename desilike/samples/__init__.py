"""Classes and functions dedicated to handling samples drawn from likelihood."""

import glob

import numpy as np

from ..parameter import Samples
from .chain import Chain
from .profiles import Profiles, ParameterBestFit, ParameterCovariance, ParameterContours
from . import diagnostics, utils
from .utils import BaseClass


__all__ = ['Samples', 'Chain', 'Profiles', 'ParameterBestFit', 'ParameterCovariance', 'ParameterContours', 'diagnostics']


def load_source(source, choice=None, cov=None, burnin=None, params=None, default=False, return_type=None):
    if not utils.is_sequence(source) and not isinstance(source, np.ndarray): fns = [source]
    else: fns = source

    sources = []
    for fn in fns:
        if isinstance(fn, str):
            sources += [BaseClass.load(ff) for ff in glob.glob(fn)]
        else:
            sources.append(fn)

    if burnin is not None:
        sources = [source.remove_burnin(burnin) if hasattr(source, 'remove_burnin') else source for source in sources]

    if choice is not None or cov is not None:
        if not all(type(source) is type(sources[0]) for source in sources):
            raise ValueError('Sources must be of same type for "choice / cov"')
        if all(source is None for source in sources):
            source = {}
        elif hasattr(sources[0], 'concatenate'):
            source = sources[0].concatenate(sources)
        else:
            source = np.array(sources)

    toret = []
    if choice is not None:
        if not isinstance(choice, dict):
            choice = {}
        if hasattr(source, 'bestfit'):
            source = source.bestfit
        tmp = {}
        if params is not None:
            if isinstance(source, np.ndarray):
                tmp = list(source)
                size = len(params)
                if len(tmp) != size:
                    raise ValueError('Provide a 1D array of size {:d} for params = {} (found {})'.format(size, params, len(tmp)))
            else:
                params_in_source = [param for param in params if param in source]
                if params_in_source:
                    tmp = source.choice(params=params_in_source, return_type='dict', **choice)
                params_not_in_source = [param for param in params if param not in params_in_source]
                for param in params_not_in_source:
                    tmp[str(param)] = (param.value if default is False else default)
                tmp = [tmp[str(param)] for param in params]
            source = ParameterBestFit(tmp, params=params)
        if source:
            tmp = source.choice(params=source.params(), return_type=return_type, **choice)
        toret.append(tmp)

    if cov is not None:
        if hasattr(source, 'covariance'):
            source = source.covariance
        tmp = None
        if params is not None:
            if isinstance(source, np.ndarray):
                tmp = source
                sizes = np.ones(tmp.shape[0], dtype='i')
                shape = (len(params),) * 2
                if tmp.shape != shape:
                    raise ValueError('Provide a 2D array of shape {} for params = {} (found {})'.format(shape, params, tmp.shape))
            else:
                params_in_source = [param for param in params if param in source]
                if params_in_source:
                    cov = source.cov(params=params_in_source, return_type=None)
                    params = [cov._params[param] if params in params_in_source else param for param in params]
                params_not_in_source = [param for param in params if param not in params_in_source]
                sizes = [param.size if param in params_not_in_source else cov._sizes[params_in_source.index(param)] for param in params]
                tmp = np.zeros((len(sizes),) * 2, dtype='f8')
                cumsizes = np.cumsum([0] + sizes)
                if params_in_source:
                    idx = [params.index(param) for param in params_in_source]
                    index = np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx])
                    tmp[np.ix_(index, index)] = cov._value
                if params_not_in_source:
                    idx = [params.index(param) for param in params_not_in_source]
                    indices = np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx])
                    indices = (indices,) * 2
                    if default is False:
                        tmp[indices] = [param.proposal**2 if param.proposal is not None else np.nan for param in params_not_in_source]
                    else:
                        tmp[indices] = default
            source = ParameterCovariance(tmp, params=params)
        if source:
            tmp = source.cov(return_type=return_type)
        toret.append(tmp)

    if len(toret) == 0:
        return sources
    if len(toret) == 1:
        return toret[0]
    return tuple(toret)
