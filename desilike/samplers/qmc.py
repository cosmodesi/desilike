"""Quasi-Monte Carlo sampling kernel and utilities."""

import logging
import warnings

import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import Sobol, Halton, LatinHypercube

from .base import StaticKernel


class KroneckerSequence(qmc.QMCEngine):
    """A quasi-random sequence based on the inverse golden ratio."""

    def __init__(self, d, seed=0.5):
        """
        Parameters
        ----------
        d : int
            Dimensionality.
        seed : float, optional
            Starting sample for the sequence in each dimension.  Default is 0.5.
        """
        super().__init__(d=d)
        self.seed = float(seed)
        phi = 1.0
        while np.abs(phi**(self.d + 1) - phi - 1) > 1e-12:
            phi -= (phi**(self.d + 1) - phi - 1) / ((self.d + 1) * phi**self.d - 1)
        self.alpha = np.array([phi**(-(1 + d)) for d in range(self.d)])

    def _random(self, n=1, *, workers=1):
        """Return ``n`` quasi-random samples."""
        idx = np.arange(self.num_generated + 1, self.num_generated + n + 1)
        samples = (self.seed + np.outer(idx, self.alpha)) % 1.
        self.num_generated += n
        if self.num_generated < np.amax(1 / self.alpha):
            warnings.warn(f'Kronecker sequence does not fill space with less '
                          f'{int(np.amax(1 / self.alpha))} samples.')
        return samples


ENGINES = dict(sobol=Sobol, halton=Halton, lhs=LatinHypercube,
               kronecker=KroneckerSequence)


class QMC(StaticKernel):
    """Evaluate the posterior on a quasi-random sequence.

    Supported engines: ``'sobol'``, ``'halton'``, ``'lhs'``, ``'kronecker'``
    (the default).
    """

    logger = logging.getLogger('QMC')

    def get_samples(self, varied_params, size=1000, engine='kronecker', **kwargs):
        """Return QMC sample points in original parameter space.

        Parameters
        ----------
        varied_params : VariableCollection
        size : int, optional
            Number of samples.  Default is 1000.
        engine : str, optional
            QMC engine name.  Default is ``'kronecker'``.
        **kwargs : dict
            Extra keyword arguments forwarded to the engine constructor.

        Returns
        -------
        numpy.ndarray, shape ``(size, ndim)``
        """
        lower, upper = [], []
        for param in varied_params:
            limits = param.prior.limits if param.prior is not None else (None, None)
            if limits is None or not (np.isfinite(limits[0]) and np.isfinite(limits[1])):
                raise ValueError(f'Provide finite limits for {param.name}.')
            lower.append(float(limits[0]))
            upper.append(float(limits[1]))

        if engine not in ENGINES:
            raise ValueError(f"'engine' must be one of {list(ENGINES.keys())}. Received '{engine}'.")

        self.engine = ENGINES[engine](d=len(varied_params), **kwargs)
        return qmc.scale(self.engine.random(n=size), lower, upper)
