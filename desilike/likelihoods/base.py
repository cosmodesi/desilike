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

from ..base import GaussianLikelihood


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
        Optional Hartlap correction. Pass
        ``dict(correction='hartlap', nobs=<int>)`` to apply
        ``(nobs - nbins - 2) / (nobs - 1)`` to the precision.
    precision : array, default=None
        Precision matrix to use directly (mutually exclusive with
        ``covariance``).
    """

    def __init__(self, observables, covariance=None, scale_covariance=1.,
                 correct_covariance=None, precision=None):
        # Nodes (the observable dependencies) and all derived data/covariance live in __init__.
        if not isinstance(observables, (list, tuple)):
            observables = [observables]
        self.observables = list(observables)

        # Build joint lsstypes data tree so covariance matching works.
        obs_data = [obs.data for obs in self.observables]
        obs_names = [obs.name for obs in self.observables]
        self._data = types.ObservableTree(obs_data, observables=obs_names)
        self.flatdata = self._data.value()

        # Resolve covariance from observable if not provided.
        if covariance is None and precision is None:
            if len(self.observables) == 1 and getattr(self.observables[0], 'covariance', None) is not None:
                obs_cov = self.observables[0].covariance
                # Wrap in named tree so matching works.
                covariance = obs_cov.clone(
                    observable=types.ObservableTree([obs_cov.observable],
                                                    observables=[self.observables[0].name]))
            else:
                raise ValueError('provide covariance or precision')

        if precision is not None:
            self.precision = np.atleast_2d(np.asarray(precision, dtype='f8'))
        else:
            if isinstance(covariance, types.CovarianceMatrix):
                try:
                    cov_arr = covariance.at.observable.match(self._data).value()
                except (AssertionError, KeyError, IndexError):
                    cov_arr = covariance.value()
            else:
                cov_arr = np.atleast_2d(np.asarray(covariance, dtype='f8'))
            self.precision = np.linalg.inv(cov_arr) / float(scale_covariance)

        if correct_covariance is not None:
            correction = correct_covariance.get('correction', '')
            if 'hartlap' in correction:
                nobs = int(correct_covariance['nobs'])
                nbins = self.precision.shape[0]
                self.precision = self.precision * (nobs - nbins - 2.) / (nobs - 1.)

        self.covariance = np.linalg.inv(self.precision)

    def __call__(self):
        self.flattheory = jnp.concatenate([obs.flattheory for obs in self.observables])
        return super().__call__()
