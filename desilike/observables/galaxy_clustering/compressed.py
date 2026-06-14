"""
Compressed (summary-statistic) observables for galaxy clustering.

Each class wraps one Extractor Calculator and presents its scalar outputs
as a flat data/theory vector so that a Gaussian likelihood can be attached.

Classes
-------
BaseCompressionObservable
    Base class: stores extractor dep, formats data, assembles flattheory in __call__.
BAOCompressionObservable
    Compare BAO distance measurements to :class:`BAOExtractor` predictions.
    Measurable parameters: ``DH_over_rd``, ``DM_over_rd``, ``DV_over_rd``,
    ``DH_over_DM``, ``qpar``, ``qper``, ``qiso``, ``qap``.
BAOPhaseShiftCompressionObservable
    Compare BAO + N_eff phase-shift measurements to :class:`BAOPhaseShiftExtractor` predictions.
    Measurable parameters: same as BAO plus ``N_eff``, ``baoshift``.
TurnOverCompressionObservable
    Compare turn-over measurements to :class:`TurnOverExtractor` predictions.
    Measurable parameters: ``DV_times_kTO``, ``DH_over_DM``, ``kTO``, ``pkTO_dd``,
    ``qto``, ``qap``.
"""

import numpy as np
import jax.numpy as jnp
import lsstypes as types

from ...base import Calculator
from ...theories.galaxy_clustering.template import BAOExtractor, BAOPhaseShiftExtractor, TurnOverExtractor


def _format_compression_data(data, covariance, parameters):
    """Return (data, flatdata, parameters, covariance) from flexible inputs.

    Parameters
    ----------
    data : None, array-like, or lsstypes.ObservableLike
        Measured values.  If None, zeros are used (requires *parameters*).
        If a plain array, *parameters* must be provided.
    covariance : None, array-like, or lsstypes.CovarianceMatrix
        Covariance matrix.  1-D input is treated as a diagonal.
    parameters : list of str or None
        Parameter names.  Required when *data* is not an lsstypes object.

    Returns
    -------
    data : lsstypes.ObservableTree
    flatdata : numpy array, shape (n,)
    parameters : list of str
    covariance : lsstypes.CovarianceMatrix or None
    """
    if isinstance(data, types.ObservableLike):
        obs = data
        parameters = list(obs.parameters)
        flatdata = obs.value()
    else:
        if parameters is None:
            raise ValueError('When data is an array or None, provide parameters')
        parameters = list(parameters)
        if data is None:
            flatdata = np.zeros(len(parameters), dtype='f8')
        else:
            flatdata = np.asarray(data, dtype='f8').ravel()
        leaves = [types.ObservableLeaf(value=np.atleast_1d(v)) for v in flatdata]
        obs = types.ObservableTree(leaves, parameters=parameters)

    cov = None
    if covariance is not None:
        if isinstance(covariance, types.CovarianceMatrix):
            cov = covariance
        else:
            cov_arr = np.asarray(covariance, dtype='f8')
            if cov_arr.ndim == 1:
                cov_arr = np.diag(cov_arr)
            if cov_arr.ndim != 2 or cov_arr.shape[0] != len(flatdata):
                raise ValueError(f'covariance shape {cov_arr.shape} does not match data size {len(flatdata)}')
            cov = types.CovarianceMatrix(value=cov_arr,
                                         observable=obs.clone(value=np.zeros_like(flatdata)))
    return obs, flatdata, parameters, cov


class BaseCompressionObservable(Calculator):
    """Base class for compressed observables.

    Wraps a single Extractor Calculator.  At each call, assembles
    ``flattheory`` by reading the requested parameter attributes from the extractor.

    Parameters
    ----------
    extractor : Calculator
        Extractor that computes the scalar observables.
    data : None, array-like, or lsstypes.ObservableLike, default=None
        Measured values.  Set to zeros when None (requires *parameters*).
    covariance : None, array-like, or lsstypes.CovarianceMatrix, default=None
        Covariance matrix passed to the likelihood.
    parameters : list of str, default=None
        Names of extractor attributes to compare to data.
        Required when *data* is not an lsstypes object.
    name : str, default='compressed'
        Observable name used to match covariance matrices in multi-observable likelihoods.

    Attributes set by ``__call__``
    --------------------------------
    flattheory : JAX array, shape (n,)
        Theory prediction vector in the same order as *parameters*.
    """

    def __init__(self, extractor, data=None, covariance=None, parameters=None, name='compressed'):
        self.extractor = extractor
        self.name = str(name)
        self.data, self.flatdata, self.parameters, self.covariance = _format_compression_data(
            data=data, covariance=covariance, parameters=parameters)

    def __call__(self):
        self.flattheory = jnp.array([getattr(self.extractor, param) for param in self.parameters])
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


class BAOCompressionObservable(BaseCompressionObservable):
    r"""Compare BAO distance measurements to theory predictions.

    Wraps :class:`BAOExtractor`.  Valid parameter names are
    ``'DH_over_rd'``, ``'DM_over_rd'``, ``'DV_over_rd'``, ``'DH_over_DM'``,
    ``'qpar'``, ``'qper'``, ``'qiso'``, ``'qap'``.

    Parameters
    ----------
    data : None, array-like, or lsstypes.ObservableLike, default=None
        Measured BAO values.
    covariance : None, array-like, or lsstypes.CovarianceMatrix, default=None
        Covariance matrix.
    parameters : list of str, default=None
        Parameter names to compare; required when *data* is not an lsstypes object.
    name : str, default='bao'
        Observable name.
    z : float, default=1.
        Effective redshift forwarded to :class:`BAOExtractor`.
    eta : float, default=1./3.
        DV exponent forwarded to :class:`BAOExtractor`.
    fiducial : str or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology forwarded to :class:`BAOExtractor`.
    cosmo : PrimordialCosmology, optional
        Cosmology provider forwarded to :class:`BAOExtractor`.
    """

    def __init__(self, data=None, covariance=None, parameters=None, name='bao', **kwargs):
        super().__init__(extractor=BAOExtractor(**kwargs),
                         data=data, covariance=covariance, parameters=parameters, name=name)


class BAOPhaseShiftCompressionObservable(BaseCompressionObservable):
    r"""Compare BAO + phase-shift measurements to theory predictions.

    Wraps :class:`BAOPhaseShiftExtractor`.  Valid parameter names are the same as
    :class:`BAOCompressionObservable` plus ``'N_eff'`` and ``'baoshift'``.

    Reference
    ---------
    https://arxiv.org/abs/1803.10741

    Parameters
    ----------
    data : None, array-like, or lsstypes.ObservableLike, default=None
        Measured values.
    covariance : None, array-like, or lsstypes.CovarianceMatrix, default=None
        Covariance matrix.
    parameters : list of str, default=None
        Parameter names; required when *data* is not an lsstypes object.
    name : str, default='baoshift'
        Observable name.
    z : float, default=1.
        Effective redshift.
    eta : float, default=1./3.
        DV exponent.
    fiducial : str or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology.
    cosmo : PrimordialCosmology, optional
        Cosmology provider.
    """

    def __init__(self, data=None, covariance=None, parameters=None, name='baoshift', **kwargs):
        super().__init__(extractor=BAOPhaseShiftExtractor(**kwargs),
                         data=data, covariance=covariance, parameters=parameters, name=name)


class TurnOverCompressionObservable(BaseCompressionObservable):
    r"""Compare turn-over measurements to theory predictions.

    Wraps :class:`TurnOverExtractor`.  Valid parameter names are
    ``'DV_times_kTO'``, ``'DH_over_DM'``, ``'kTO'``, ``'pkTO_dd'``,
    ``'qto'``, ``'qap'``.

    Reference
    ---------
    https://arxiv.org/abs/2302.07484

    Parameters
    ----------
    data : None, array-like, or lsstypes.ObservableLike, default=None
        Measured values.
    covariance : None, array-like, or lsstypes.CovarianceMatrix, default=None
        Covariance matrix.
    parameters : list of str, default=None
        Parameter names; required when *data* is not an lsstypes object.
    name : str, default='turnover'
        Observable name.
    z : float, default=1.
        Effective redshift.
    eta : float, default=1./3.
        DV exponent.
    fiducial : str or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology.
    cosmo : PrimordialCosmology, optional
        Cosmology provider.
    """

    def __init__(self, data=None, covariance=None, parameters=None, name='turnover', **kwargs):
        super().__init__(extractor=TurnOverExtractor(**kwargs),
                         data=data, covariance=covariance, parameters=parameters, name=name)
