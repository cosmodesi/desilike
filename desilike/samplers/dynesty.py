"""Module implementing the dynesty samplers."""

try:
    import dynesty
    DYNESTY_INSTALLED = True
except ModuleNotFoundError:
    DYNESTY_INSTALLED = False
import numpy as np

from .base import (prior_transform, compute_likelihood, update_kwargs,
                   PopulationSampler)
from desilike.samples import Chain


class DynestySampler(PopulationSampler):
    """Class for the dynesty samplers.

    Reference
    ---------
    - https://github.com/joshspeagle/dynesty
    - https://doi.org/10.1093/mnras/staa278

    """

    def __init__(self, likelihood, rng=None, save_fn=None, mpicomm=None,
                 dynamic=True, **kwargs):
        """Initialize the dynesty sampler.

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

        dynamic : boolean, optional
            If True, use ``dynesty.DynamicPopulationSampler`` instead of
            ``dynesty.PopulationSampler``. Default is True.

        kwargs: dict, optional
            Extra keyword arguments passed to dynesty during initialization.

        """
        if not DYNESTY_INSTALLED:
            raise ImportError("The 'dynesty' package is required but not "
                              "installed.")

        super().__init__(likelihood, rng=rng, save_fn=save_fn, mpicomm=mpicomm)

        kwargs = update_kwargs(kwargs, 'dynesty', pool=self.pool,
                               rstate=self.rng)

        if not dynamic and self.save_fn is not None:
            raise ValueError("dynesty does not support checkpointing for the "
                             "static sampler.")

        if self.mpicomm.rank == 0:
            args = (compute_likelihood, prior_transform, self.n_dim)
            if dynamic:
                self.sampler = dynesty.DynamicNestedSampler(
                    *args, **kwargs)
            else:
                self.sampler = dynesty.NestedSampler(*args, **kwargs)
        else:
            self.sampler = None

    def run_sampler(self, **kwargs):
        """Run the dynesty sampler.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments passed to dynesty's ``run_nested`` method.

        Returns
        -------
        Chain
            Sampler results.

        """
        kwargs = update_kwargs(kwargs, 'dynesty', checkpoint_file=self.save_fn)

        self.sampler.run_nested(**kwargs)

        results = self.sampler.results
        chain = [results.samples[..., i] for i in range(self.n_dim)]
        chain.append(results.logwt)
        chain.append(np.exp(results.logwt - results.logz[-1]))
        return Chain(chain, params=self.likelihood.varied_params +
                     ['logweight', 'aweight'])
