"""
Gaussian likelihood for galaxy clustering observables.

Classes
-------
ObservablesGaussianLikelihood
    Gaussian likelihood that combines one or more observables with a covariance matrix.
"""

import numpy as np
import jax.numpy as jnp
import lsstypes as types

from ..base import GaussianLikelihood, get_params


class ObservablesGaussianLikelihood(GaussianLikelihood):
    r"""
    Gaussian likelihood combining one or more observables.

    Computes ``logpdf = -½ r @ precision @ r`` where
    ``r = flatdata - flattheory`` and ``flattheory`` concatenates
    ``flattheory`` from all observables.

    Parameters
    ----------
    observables : Calculator or list of Calculator
        Observables exposing ``flatdata``, ``data`` (lsstypes), and (after
        calling) ``flattheory``.
    covariance : array or lsstypes.CovarianceMatrix, default=None
        Covariance matrix (2-D array or diagonal) or a lsstypes
        ``CovarianceMatrix`` that is matched to the joint observable. If None
        and a single observable carries a ``covariance`` attribute, that is used.
    scale_covariance : float, default=1.
        The precision is divided by this factor.
    correct_covariance : dict, default=None
        Optional correction for a covariance estimated from simulations.
        Pass ``dict(correction='hartlap2007', nobs=<int>)`` to apply the Hartlap
        2007 factor ``(nobs - nbins - 2) / (nobs - 1)`` to the precision.
        Pass ``dict(correction='percival2014', nobs=<int>)`` to apply the
        Percival 2014 factor ``(1 + B(nbins - nparams)) / (1 + A + B(nparams + 1))``
        where ``nparams`` is determined at compile time from ``get_params(self).select(input=True)``.
        Corrections can be combined, e.g. ``correction='hartlap2007+percival2014'``.
    precision : array, default=None
        Precision matrix to use directly (mutually exclusive with
        ``covariance``).
    """

    def __init__(self, observables, covariance=None, scale_covariance=1.,
                 correct_covariance=None, precision=None):

        correct_covariance = {'correction': '', 'nobs': None} if correct_covariance is None else correct_covariance
        assert isinstance(correct_covariance, dict) , "correct_covariance should be a dict with keys 'correction' and 'nobs' or None"

        # Nodes (the observable dependencies) and all derived data/covariance live in __init__.
        if not isinstance(observables, (list, tuple)):
            observables = [observables]
        self.observables = list(observables)

        if covariance is None:
            if len(self.observables) == 1 and getattr(self.observables[0], 'covariance', None) is not None:
                covariance = self.observables[0].covariance
                if correct_covariance['correction']:
                    correct_covariance.setdefault('nobs', getattr(covariance, 'nobs', None))
                covariance = covariance.clone(observable=types.ObservableTree([covariance.observable], observables=[self.observables[0].name]))
            elif precision is None:
                raise ValueError('Observables must have their own covariance if global covariance or precision matrix not provided')
        data = [observable.data for observable in self.observables]
        # Build joint lsstypes data tree so covariance matching works.
        data = [observable.data for observable in self.observables]
        self.data = types.ObservableTree(data, observables=[observable.name for observable in self.observables])
        self.flatdata = self.data.value()

        def check_matrix(matrix, name):
            matrix = np.atleast_2d(matrix).copy()
            if matrix.shape != (matrix.shape[0],) * 2:
                raise ValueError('{} must be a square matrix, but found shape {}'.format(name, matrix.shape))
            mshape = '({0}, {0})'.format(matrix.shape[0])
            shape = '({0}, {0})'.format(self.flatdata.size)
            shape_obs = '({0}, {0})'.format(' + '.join([str(obs.flatdata.size) for obs in self.observables]))
            if matrix.shape[0] != self.flatdata.size:
                raise ValueError('based on provided observables, {} expected to be a matrix of shape {} = {}, but found {}'.format(name, shape, shape_obs, mshape))
            return matrix

        self.precision = check_matrix(precision, 'precision') if precision is not None else None

        self.covariance = None
        if isinstance(covariance, types.CovarianceMatrix):
            self.covariance = covariance.at.observable.match(self.data)
        elif covariance is not None:
            covariance = check_matrix(covariance, 'covariance')
            self.covariance = types.CovarianceMatrix(observable=self.data.clone(value=0. * self.data.value()), value=covariance)
        for observable in self.observables:
            observable.covariance = self.covariance.at.observable.get(observables=observable.name)

        if self.precision is None and self.covariance is None:
            raise ValueError('if precision is not provided, provide covariance')
        self.correct_covariance = correct_covariance
        if self.correct_covariance['correction'] and self.correct_covariance['nobs'] is None:
            raise ValueError(f'provide nobs to apply correction {self.correct_covariance["correction"]}')
        valid = ['', 'hartlap2007', 'percival2014', 'hartlap2007+percival2014']
        if self.correct_covariance['correction'] not in valid:
            raise ValueError(f"correct_covariance 'correction' must be one of {valid}, got {self.correct_covariance['correction']!r}")
        self.hartlap2007_factor = 1.
        self.percival2014_factor = 1.

    def __post_init__(self, observables, covariance=None, scale_covariance=1.,
                      correct_covariance=None, precision=None):
        if self.precision is None:
            self.precision = self.covariance.inv(level=1) / scale_covariance
        if 'hartlap2007' in self.correct_covariance['correction']:
            nbins = self.precision.shape[0]
            nobs = self.correct_covariance['nobs']
            self.hartlap2007_factor = (nobs - nbins - 2.) / (nobs - 1.)
            self.logger.info(f'Covariance matrix with {nbins:d} points built from {nobs:d} observations.')
            self.logger.info(f'...resulting in a Hartlap 2007 factor of {self.hartlap2007_factor:.4f}.')
            self.precision = self.precision * self.hartlap2007_factor
        if 'percival2014' in self.correct_covariance['correction']:
            nbins = self.precision.shape[0]
            nobs = self.correct_covariance['nobs']
            nparams = len(get_params(self).select(input=True, varied=True))
            A = 2. / ((nobs - nbins - 1.) * (nobs - nbins - 4.))
            B = (nobs - nbins - 2.) / ((nobs - nbins - 1.) * (nobs - nbins - 4.))
            self.percival2014_factor = (1. + B * (nbins - nparams)) / (1. + A + B * (nparams + 1.))
            self.logger.info(f'Covariance matrix with {nbins:d} points and {nparams:d} parameters built from {nobs:d} observations.')
            self.logger.info(f'...resulting in a Percival 2014 factor of {self.percival2014_factor:.4f}.')
            self.precision = self.precision / self.percival2014_factor

    def __call__(self):
        self.flattheory = jnp.concatenate([obs.flattheory for obs in self.observables])
        return super().__call__()
