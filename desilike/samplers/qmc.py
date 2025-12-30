"""Module implementing quasi-random sequences with reduced variance."""

import warnings

import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import Sobol, Halton, LatinHypercube

from .base import StaticSampler
from desilike.parameter import ParameterPriorError


class KroneckerSequence(qmc.QMCEngine):
    """A quasi-random sequence based on the inverse golden ratio."""

    def __init__(self, d, seed=0.5):
        """Initialize the sequence.

        Parameters
        ----------
        d : int
            Dimensionality.
        seed : float, optional
            Starting sample for the sequence in each dimension. Default is 0.5.

        """
        super().__init__(d=d)
        self.seed = float(seed)
        phi = 1.0
        # Use Newton's method to solve phi**(d+1) - phi - 1 = 0.
        while np.abs(phi**(self.d + 1) - phi - 1) > 1e-12:
            phi -= (phi**(self.d + 1) - phi - 1) / (
                (self.d + 1) * phi**self.d - 1)
        self.alpha = np.array([phi**(-(1 + d)) for d in range(self.d)])

    def _random(self, n=1, *, workers=1):
        """Get samples.

        Parameters
        ----------
        n : int
            Number of samples.
        workers : optional
            Ignored.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_dim)
            Samples.

        """
        i = np.arange(self.num_generated + 1, self.num_generated + n + 1)
        samples = (self.seed + np.outer(i, self.alpha)) % 1.
        self.num_generated += n
        if self.num_generated < np.amax(1 / self.alpha):
            warnings.warn(f"Kronecker sequence does not fill space with less "
                          f"{int(np.amax(1 / self.alpha))} samples.")
        return samples


ENGINES = dict(sobol=Sobol, halton=Halton, lhs=LatinHypercube,
               kronecker=KroneckerSequence)


class QMCSampler(StaticSampler):
    """Quasi Monte-Carlo (QMC) sequences implemented in :mod:`scipy.qmc`.

    This module also implements Kronecker sequences.

    .. rubric:: References
    - https://docs.scipy.org/doc/scipy/reference/stats.qmc.html

    """

    def get_samples(self, size=1000, engine='kronecker', **kwargs):
        """Get samples from a QMC sequence.

        Parameters
        ----------
        size : dict or int, optional
            Size of the sequence. Default is 1000.

        engine : str, optional
            Engine to use. Choices are 'sobol', 'halton', 'lhs', 'kronecker'.
            Default is 'kronecker'.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_dim)
            Grid to be evaluated.
        """
        lower, upper = [], []
        for param in self.likelihood.varied_params:
            if param.limits is None:
                raise ParameterPriorError(
                    f"Provide a limit for {param.name}.")
            lower.append(param.limits[0])
            upper.append(param.limits[1])

        if engine not in ENGINES:
            raise ValueError(f"'engine' must be in {list(ENGINES.keys())}. "
                             f"Received {engine}.")

        self.engine = ENGINES[engine](
            d=len(self.likelihood.varied_params), **kwargs)

        return qmc.scale(self.engine.random(n=size), lower, upper)
