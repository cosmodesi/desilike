"""Module implementing the ``emcee`` sampler."""

try:
    import emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False
import numpy as np

from .base import update_parameters, EnsembleSampler


class EmceeSampler(EnsembleSampler):
    """Wrapper for the affine-invariant ensemble sampler ``emcee``.

    .. rubric:: References
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665

    """

    def __init__(self, likelihood, n_chains=1, chains=None, rng=None,
                 directory=None, nwalkers=None, **kwargs):
        """Initialize the ``emcee`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.Chain, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        nwalkers : int, optional
            Number of walkers, defaults to :attr:`Chain.shape[1]` of input chains, if any,
            else ``2 * max((int(2.5 * ndim) + 1) // 2, 2)``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``emcee`` during initialization.

        """
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not "
                              "installed.")

        super().__init__(likelihood, n_chains=n_chains, chains=chains, rng=rng,
                         directory=directory, nwalkers=nwalkers)
        if self.nwalkers is None:
            self.nwalkers = 2 * max((int(2.5 * self.n_dim) + 1) // 2, 2)
        if self.pool.main:
            kwargs = update_parameters(
                kwargs, 'emcee', ndim=self.n_dim,
                log_prob_fn=self.compute_posterior, pool=self.pool, args=None,
                kwargs=None, vectorize=False, nwalkers=self.nwalkers)
            self.sampler = emcee.EnsembleSampler(**kwargs)

    def run_sampler(self, n_steps):
        """Run the ``emcee`` sampler.

        Parameters
        ----------
        n_steps: int
            Number of steps to take.

        """
        if self.pool.main:
            samples, derived, log_post = self.state

            initial_state = emcee.State(
                samples, blobs=derived, log_prob=log_post,
                random_state=np.random.RandomState(
                    self.rng.integers(2**32 - 1)).get_state())

            samples = np.zeros((n_steps, self.nwalkers, self.n_dim))
            derived = np.zeros((n_steps, self.nwalkers, self.n_derived))
            log_post = np.zeros((n_steps, self.nwalkers))
            for i, state in enumerate(self.sampler.sample(
                    initial_state, iterations=n_steps, store=False)):
                samples[i, :, :] = state.coords
                derived[i, :, :] = state.blobs.reshape(self.nwalkers, -1)
                log_post[i, :] = state.log_prob

            self.extend(samples, derived, log_post)
            self.pool.stop_wait()
        else:
            self.pool.wait()
