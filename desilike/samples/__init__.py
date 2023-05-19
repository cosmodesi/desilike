"""Classes and functions dedicated to handling samples drawn from likelihood."""

import glob

import numpy as np

from ..parameter import ParameterCollection, Samples
from .chain import Chain
from .profiles import Profiles, ParameterBestFit, ParameterCovariance, ParameterProfiles, ParameterContours, ParameterGrid
from . import diagnostics, utils
from .utils import BaseClass, path_types


__all__ = ['Samples', 'Chain', 'Profiles', 'ParameterBestFit', 'ParameterCovariance', 'ParameterContours', 'ParameterProfiles', 'ParameterGrid', 'diagnostics']


def load_source(source, choice=None, cov=None, burnin=None, params=None, default=False, return_type=None):
    """
    Internal function that from a source (:class:`Chain`, :class:`Profiles`, :class:`ParameterCovariance`, or path to these objects),
    return best fit, mean, or covariance matrix.

    Parameters
    ----------
    source : str, Path, :class:`Chain`, :class:`Profiles`, :class:`ParameterCovariance`
        Source to take best fit / mean / covariance from: chain, profiles, covariance, or path to such objects.

    choice : dict, bool, default=None
        If not ``None``, extract best fit {'index': 'argmax'} or mean {'index': 'mean'} from source.

    cov : bool, default=None
        If ``True``, return covariance.

    burnin : float, int, default=None
        If input is chains, remove burnin:
        if between 0 and 1, remove that fraction of samples;
        else, remove ``burnin`` first points.

    params : list, ParameterCollection, default=None
        Parameters to compute best fit / mean / covariance for. Defaults to all parameters.

    default : bool, default=False
        Default value for best fit / mean / covariance in case it is not in ``source``.

    return_type : str, default=None
        If ``choice`` is desired and ``return_type`` is:
        - 'dict' : return dictionary mapping parameter names to best fit / mean.
        - 'nparray' : return array of parameter best fits / means.
        - ``None``: return :class:`ParameterBestFit` instance.

        If ``cov`` is desired and ``return_type`` is:
        - 'nparray' : return matrix array
        - ``None``: return :class:`ParameterCovariance` instance.
    """
    is_not_sequence = not utils.is_sequence(source) and not isinstance(source, np.ndarray)
    if is_not_sequence: fns = [source]
    else: fns = source

    sources = []
    for fn in fns:
        if isinstance(fn, path_types):
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
        elif is_not_sequence and len(sources) == 1:
            source = sources[0]
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
                params_in_source = source.params(name=[str(param) for param in params]) if source else []
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
        if hasattr(source, 'to_fisher'):  # Chain, Profiles
            source = source.to_fisher(params=params)
        if hasattr(source, 'covariance'):  # LikelihoodFisher
            source = source.covariance(params=params, return_type=None)
        if hasattr(source, 'to_covariance'):  # ParameterPrecision
            source = source.to_covariance(params=params, return_type=None)
        tmp = None
        if params is not None:
            if isinstance(source, np.ndarray):
                tmp = source
                sizes = np.ones(tmp.shape[0], dtype='i')
                shape = (len(params),) * 2
                if tmp.shape != shape:
                    raise ValueError('Provide a 2D array of shape {} for params = {} (found {})'.format(shape, params, tmp.shape))
            else:
                params_in_source = source.params(name=[str(param) for param in params]) if source else []
                if params_in_source:
                    cov = source.view(params=params_in_source, return_type=None)
                    params = [cov._params[param] if param in params_in_source else param for param in params]
                params = ParameterCollection(params)
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
                        tmp[indices] = [getattr(param, 'proposal', np.nan)**2 for param in params_not_in_source]
                    else:
                        tmp[indices] = default
            source = ParameterCovariance(tmp, params=params)
        if source:
            tmp = source.view(return_type=return_type)
        toret.append(tmp)

    if len(toret) == 0:
        return sources
    if len(toret) == 1:
        return toret[0]
    return tuple(toret)
