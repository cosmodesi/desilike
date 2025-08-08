"""Module implementing the nautilus sampler."""
import sys

try:
    import nautilus
    NAUTILUS_INSTALLED = True
except ModuleNotFoundError:
    NAUTILUS_INSTALLED = False
import numpy as np

from .base import (prior_transform, compute_likelihood, update_kwargs,
                   PopulationSampler)
from desilike.samples import Chain


class NautilusSampler(PopulationSampler):
    """Class for the nautilus sampler.

    Reference
    ---------
    - https://github.com/johannesulf/nautilus
    - https://doi.org/10.1093/mnras/stad2441

    """

    def __init__(self, likelihood, rng=None, save_fn=None, mpicomm=None,
                 **kwargs):
        """Initialize the nautilus sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.

        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.

        save_fn : str, Path, optional
            Save samples to this location. Default is ``None``.

        mpicomm : mpi.COMM_WORLD, optional
            MPI communicator. If ``None``, defaults to ``likelihood``'s
            :attr:`BaseLikelihood.mpicomm`. Default is ``None``.

        kwargs: dict, optional
            Extra keyword arguments passed to nautilus during initialization.

        """
        if not NAUTILUS_INSTALLED:
            raise ImportError("The 'nautilus-sampler' package is required but "
                              "not installed.")

        super().__init__(likelihood, rng=rng, save_fn=save_fn, mpicomm=mpicomm)

        kwargs = update_kwargs(
            kwargs, 'nautilus', pass_dict=False,
            filepath=self.path('sampler', 'hdf5'), pool=self.pool,
            seed=self.rng.integers(2**32 - 1))

        if self.mpicomm.rank == 0:
            self.sampler = nautilus.Sampler(
                prior_transform, compute_likelihood, self.n_dim, **kwargs)
        else:
            self.sampler = None

    def run_sampler(self, **kwargs):
        """Run the nautilus sampler.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments passed to nautilus' ``run`` method.

        Returns
        -------
        Chain
            Sampler results.

        """
        self.sampler.run(**kwargs)

        points, log_w, log_l = self.sampler.posterior()
        chain = [points[..., i] for i in range(self.n_dim)]
        chain.append(log_w)
        chain.append(np.exp(log_w - self.sampler.log_z))
        chain.append(log_l)
        return Chain(chain, params=self.likelihood.varied_params +
                     ['logweight', 'aweight', 'loglikelihood'])
