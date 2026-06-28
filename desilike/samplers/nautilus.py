"""Nautilus importance nested sampling kernel."""

import logging

import numpy as np

try:
    import nautilus as _nautilus
    NAUTILUS_INSTALLED = True
except ModuleNotFoundError:
    NAUTILUS_INSTALLED = False

from .base import PopulationKernel, update_kwargs


class Nautilus(PopulationKernel):
    """Importance nested sampler via ``nautilus``.

    Requires two MPI pools: one for likelihood evaluations and one for
    internal sampler tasks.  The secondary pool is created lazily on the
    first :meth:`run` call.

    .. rubric:: References
    - https://github.com/johannesulf/nautilus
    - https://doi.org/10.1093/mnras/stad2441
    """

    logger = logging.getLogger('Nautilus')

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Extra keyword arguments forwarded to ``nautilus.Sampler``.
        """
        self._kwargs = kwargs
        self._sampler = None
        self._pool_sampler = None
        self._initialized = False

    def reset_state(self):
        self._sampler = None
        self._initialized = False

    @classmethod
    def install(cls, installer):
        installer.pip('nautilus-sampler')

    def init(self, likelihood, prior, rng, **context):
        _, self._likelihood_logpdf_with_derived = likelihood
        self._prior_logpdf, self._prior_ppf, _ = prior
        self._rng = rng
        self._pool = context['pool']
        self._ndim = context['ndim']
        self._output_dir = context.get('output_dir')

    def run(self, **kwargs):
        if not NAUTILUS_INSTALLED:
            raise ImportError("The 'nautilus-sampler' package is required but not installed.")

        from desilike.samplers.pool import make_pool, wait_many

        # Create the secondary pool on all MPI ranks (needed for wait_many on workers).
        if self._pool_sampler is None:
            self._pool_sampler = make_pool(self._pool.comm, batch_size=0)

        if self._pool.main:
            if not self._initialized:
                init_kwargs = update_kwargs(
                    dict(**self._kwargs), 'nautilus',
                    prior=self._prior_ppf, likelihood=self._likelihood_logpdf_with_derived,
                    n_dim=self._ndim, pass_dict=False,
                    filepath=None if self._output_dir is None else self._output_dir / 'nautilus.h5',
                    pool=(self._pool, self._pool_sampler),
                    seed=self._rng.integers(2**32))
                self._sampler = _nautilus.Sampler(**init_kwargs)
                self._pool.stop_wait()   # release workers from init-time pool.wait()
                self._initialized = True

            self._sampler.run(**kwargs)
            samples, log_w, log_l, blobs = self._sampler.posterior(return_blobs=True)
            blobs = blobs.reshape(len(samples), -1)
            log_prior = np.array(list(self._pool.map(self._prior_logpdf, samples)))
            self._pool.stop_wait()
            self._pool_sampler.stop_wait()
            self.logger.info('Finished sampling.')
            return samples, blobs, dict(aweight=np.exp(log_w), logposterior=log_l + log_prior)
        else:
            if not self._initialized:
                self._initialized = True
                self._pool.wait()   # serve during nautilus.Sampler.__init__
            wait_many([self._pool, self._pool_sampler])
            return None
