"""Module implementing the ``emcee`` sampler."""

try:
    import emcee
    EMCEE_INSTALLED = True
except ModuleNotFoundError:
    EMCEE_INSTALLED = False
import numpy as np

from .base import update_kwargs, EnsembleSampler


class EmceeSampler(EnsembleSampler):
    """Wrapper for the affine-invariant ensemble sampler ``emcee``.

    .. rubric:: References
    - https://github.com/dfm/emcee
    - https://arxiv.org/abs/1202.3665

    """

    def __init__(self, posterior, nchains=1, chains=None, rng=None,
                 directory=None, nwalkers=None, rescale=False, covariance=None, **kwargs):
        """Initialize the ``emcee`` sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        nchains : int, optional
            Number of independent chains. Default is 1.
        chains : list of desilike.samples.MCSamples, optional
            If given, continue the chains. In that case, we will ignore what
            was read from disk. Default is ``None``.
        rng : numpy.random.Generator, int, or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.
        nwalkers : int, optional
            Number of walkers, defaults to :attr:`MCSamples.shape[1]` of input chains, if any,
            else ``2 * max((int(2.5 * ndim) + 1) // 2, 2)``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``emcee`` during initialization.

        """
        if not EMCEE_INSTALLED:
            raise ImportError("The 'emcee' package is required but not "
                              "installed.")

        super().__init__(posterior, nchains=nchains, chains=chains, rng=rng,
                         directory=directory, nwalkers=nwalkers,
                         rescale=rescale, covariance=covariance)
        if self.nwalkers is None:
            self.nwalkers = 2 * max((int(2.5 * self.ndim) + 1) // 2, 2)
        if self.pool.main:
            # emcee treats tuple returns as (log_prob, blobs).  When there are
            # no derived parameters, return plain scalars so emcee never
            # allocates a blob array (mixing scalar and tuple returns within
            # one run causes an internal boolean-index mismatch in emcee).
            # compute_posterior is batched: it takes an (N, ndim) batch and
            # returns a list of N (log_post, derived) tuples.
            if self.n_derived:
                log_prob_fn = self.compute_posterior
            else:
                log_prob_fn = lambda batch: [result[0] for result in self.compute_posterior(batch)]
            kwargs = update_kwargs(
                kwargs, 'emcee', ndim=self.ndim,
                log_prob_fn=log_prob_fn, pool=self.pool, args=None,
                kwargs=None, vectorize=False, nwalkers=self.nwalkers)
            self.sampler = emcee.EnsembleSampler(**kwargs)

    def adapt_sampler(self, steps):
        """No-op: emcee does not support explicit adaptation."""

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
                samples,
                blobs=derived if self.n_derived else None,
                log_prob=log_post,
                random_state=np.random.RandomState(
                    self.rng.integers(2**32 - 1)).get_state())

            samples  = np.zeros((n_steps, self.nwalkers, self.ndim))
            derived  = np.zeros((n_steps, self.nwalkers, self.n_derived))
            log_post = np.zeros((n_steps, self.nwalkers))
            for i, state in enumerate(self.sampler.sample(
                    initial_state, iterations=n_steps, store=False)):
                samples[i, :, :] = state.coords
                if self.n_derived:
                    derived[i, :, :] = np.asarray(state.blobs).reshape(self.nwalkers, -1)
                log_post[i, :] = state.log_prob

            self.extend(samples, derived, log_post)
            self.pool.stop_wait()
        else:
            self.pool.wait()
