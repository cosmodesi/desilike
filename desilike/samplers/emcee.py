"""Module implementing the ``emcee`` sampler."""

try:
    import emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False
import numpy as np

from .base import _update_parameters, MarkovChainSampler


class EmceeSampler(MarkovChainSampler):
    """Wrapper for the affine-invariant ensemble sampler ``emcee``.

    .. rubric:: References

    - `emcee repo <https://github.com/dfm/emcee>`_
    - `emcee docs <https://emcee.readthedocs.io>`_
    - `emcee paper <https://doi.org/10.1086/670067>`_

    """

    ensemble = True

    def __init__(self, likelihood, n_walkers=10, chains=None, rng=None,
                 directory=None, **kwargs):
        """Initialize the ``emcee`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_walkers : int, optional
            Number of walkers. Note that each walker produces a chain but
            different chains are not strictly independent. Default is 10.
        chains : list of desilike.samples.Chain, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        **kwargs
            Extra keyword arguments passed to ``emcee`` during initialization.

        """
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_walkers, chains=chains,
                         rng=rng, directory=directory)

        if self.mpicomm.rank == 0:
            kwargs = _update_parameters(
                kwargs, 'emcee', nwalkers=n_walkers, ndim=self.n_dim,
                log_prob_fn=self._compute_posterior, pool=self.pool, args=None,
                kwargs=None, vectorize=False)
            self.sampler = emcee.EnsembleSampler(**kwargs)

    def _run(self, n_steps):
        """Run the ``emcee`` sampler.

        Parameters
        ----------
        n_steps: int
            Number of steps to take.

        """
        samples, derived, log_post = self._state

        initial_state = emcee.State(
            samples, blobs=derived, log_prob=log_post,
            random_state=np.random.RandomState(
                self.rng.integers(2**32 - 1)).get_state())

        samples = np.zeros((self.n_chains, n_steps, self.n_dim))
        derived = np.zeros((self.n_chains, n_steps, self.n_derived))
        log_post = np.zeros((self.n_chains, n_steps))
        for i, state in enumerate(self.sampler.sample(
                initial_state, iterations=n_steps, store=False)):
            samples[:, i, :] = state.coords
            derived[:, i, :] = state.blobs.reshape(self.n_chains, -1)
            log_post[:, i] = state.log_prob

        self._extend(samples, derived, log_post)
