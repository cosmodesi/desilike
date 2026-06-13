"""Dynesty nested sampling kernel."""

import logging

import numpy as np

try:
    import dynesty as _dynesty
    DYNESTY_INSTALLED = True
except ModuleNotFoundError:
    DYNESTY_INSTALLED = False

from .base import NestedKernel, update_kwargs


class Dynesty(NestedKernel):
    """Nested sampler via ``dynesty``.

    .. rubric:: References
    - https://github.com/joshspeagle/dynesty
    - https://doi.org/10.1093/mnras/staa278
    """

    logger = logging.getLogger('Dynesty')
    # dynesty is not vectorized; the pool must call loglikelihood once per particle.
    _batch_size = 0

    def __init__(self, dynamic=True, **kwargs):
        """
        Parameters
        ----------
        dynamic : bool
            If ``True`` (default), use ``dynesty.DynamicNestedSampler``;
            otherwise use the static ``dynesty.NestedSampler``.
        **kwargs
            Extra keyword arguments forwarded to the dynesty sampler constructor.
        """
        self.dynamic = dynamic
        self._kwargs = kwargs
        self._sampler = None

    def run(self, likelihood_logpdf, prior,
            pool, rng, ndim, directory=None, n_derived=0, params=None, **kwargs):
        if not DYNESTY_INSTALLED:
            raise ImportError("The 'dynesty' package is required but not installed.")

        if not self.dynamic and directory is not None:
            raise ValueError("dynesty does not support checkpointing for the static sampler.")

        prior_logpdf, prior_rvs, prior_ppf = prior

        if pool.main:
            if self._sampler is None:
                sampler_cls = (_dynesty.DynamicNestedSampler if self.dynamic
                               else _dynesty.NestedSampler)
                if directory is not None:
                    try:
                        self._sampler = sampler_cls.restore(str(directory / 'dynesty.pkl'))
                        self._sampler.loglikelihood.loglikelihood = likelihood_logpdf
                        self._sampler.prior_transform = prior_ppf
                    except (FileNotFoundError, ValueError):
                        pass
                if self._sampler is None:
                    init_kwargs = update_kwargs(
                        dict(**self._kwargs), 'dynesty',
                        loglikelihood=likelihood_logpdf, prior_transform=prior_ppf,
                        ndim=ndim, blob=True, pool=pool, rstate=rng)
                    self._sampler = sampler_cls(**init_kwargs)

            checkpoint_file = None if directory is None else str(directory / 'dynesty.pkl')
            run_kwargs = update_kwargs(kwargs, 'dynesty', checkpoint_file=checkpoint_file)
            self._sampler.run_nested(**run_kwargs)
            results = self._sampler.results
            log_prior = np.array(list(pool.map(prior_logpdf, results.samples)))
            pool.stop_wait()
            self.logger.info('Finished sampling.')
            return results.samples, results['blob'], dict(
                aweight=results.importance_weights(),
                logposterior=results.logl + log_prior)
        pool.wait()
        return None
