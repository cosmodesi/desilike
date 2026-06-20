"""Dynesty nested sampling kernel."""

import logging

import numpy as np

try:
    import dynesty as _dynesty
    DYNESTY_INSTALLED = True
except ModuleNotFoundError:
    DYNESTY_INSTALLED = False

from .base import PopulationKernel, update_kwargs


class Dynesty(PopulationKernel):
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

    @classmethod
    def install(cls, installer):
        installer.pip('dynesty')

    def init(self, likelihood, prior, rng, **context):
        _, self._likelihood_logpdf_with_derived = likelihood
        self._prior_logpdf, self._prior_ppf, _ = prior
        self._rng = rng
        self._pool = context['pool']
        self._ndim = context['ndim']
        self._output_dir = context.get('output_dir')

    def run(self, **kwargs):
        if not DYNESTY_INSTALLED:
            raise ImportError("The 'dynesty' package is required but not installed.")

        if not self.dynamic and self._output_dir is not None:
            raise ValueError("dynesty does not support checkpointing for the static sampler.")

        if self._pool.main:
            if self._sampler is None:
                sampler_cls = (_dynesty.DynamicNestedSampler if self.dynamic
                               else _dynesty.NestedSampler)
                if self._output_dir is not None:
                    try:
                        self._sampler = sampler_cls.restore(str(self._output_dir / 'dynesty.pkl'))
                        self._sampler.loglikelihood.loglikelihood = self._likelihood_logpdf_with_derived
                        self._sampler.prior_transform = self._prior_ppf
                    except (FileNotFoundError, ValueError):
                        pass
                if self._sampler is None:
                    init_kwargs = update_kwargs(
                        dict(**self._kwargs), 'dynesty',
                        loglikelihood=self._likelihood_logpdf_with_derived,
                        prior_transform=self._prior_ppf,
                        ndim=self._ndim, blob=True,
                        pool=self._pool, rstate=self._rng)
                    self._sampler = sampler_cls(**init_kwargs)

            checkpoint_file = None if self._output_dir is None else str(self._output_dir / 'dynesty.pkl')
            run_kwargs = update_kwargs(kwargs, 'dynesty', checkpoint_file=checkpoint_file)
            self._sampler.run_nested(**run_kwargs)
            results = self._sampler.results
            log_prior = np.array(list(self._pool.map(self._prior_logpdf, results.samples)))
            self._pool.stop_wait()
            self.logger.info('Finished sampling.')
            return results.samples, results['blob'], dict(
                aweight=results.importance_weights(),
                logposterior=results.logl + log_prior)
        self._pool.wait()
        return None
