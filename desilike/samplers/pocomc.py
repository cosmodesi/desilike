"""PocoMC preconditioned Monte Carlo kernel."""

import logging

import numpy as np

try:
    import pocomc as _pocomc
    POCOMC_INSTALLED = True
except ModuleNotFoundError:
    POCOMC_INSTALLED = False

from .base import PopulationKernel, update_kwargs


class _Prior:
    """Prior wrapper for ``pocoMC`` built from prior callables."""

    def __init__(self, prior_logpdf, prior_rvs, prior_ppf, ndim):
        self._logpdf = prior_logpdf
        self._rvs = prior_rvs
        lo = prior_ppf(np.zeros((1, ndim), dtype='f4')).ravel()
        hi = prior_ppf(np.ones((1, ndim), dtype='f4')).ravel()
        self._bounds = np.column_stack([lo, hi])   # (ndim, 2)
        self._ndim = ndim

    def logpdf(self, x):
        return np.asarray([result for result in self._logpdf(x)])

    def rvs(self, size=1):
        return self._rvs(size)

    @property
    def bounds(self):
        return self._bounds

    @property
    def dim(self):
        return self._ndim


_CLEAR_BEFORE_SAVE = ('log_likelihood', 'log_prior', 'sample_prior', 'prior', 'pool', 'distribute', 'save_state')


def _patch_save_state(pocomc_sampler):
    """Monkey-patch ``pocoMC``'s ``save_state`` to null unpicklable attributes before dumping."""
    _original_save_state = pocomc_sampler.save_state

    def _save_state_no_likelihood(path):
        saved = {attr: getattr(pocomc_sampler, attr, None) for attr in _CLEAR_BEFORE_SAVE}
        for attr in _CLEAR_BEFORE_SAVE:
            setattr(pocomc_sampler, attr, None)
        try:
            _original_save_state(path)
        finally:
            for attr, val in saved.items():
                setattr(pocomc_sampler, attr, val)

    pocomc_sampler.save_state = _save_state_no_likelihood
    return _save_state_no_likelihood


class PocoMC(PopulationKernel):
    """Preconditioned Monte Carlo sampler via ``pocomc``.

    .. rubric:: References
    - https://github.com/minaskar/pocomc
    - https://doi.org/10.21105/joss.04634
    - https://doi.org/10.1093/mnras/stac2272
    """

    logger = logging.getLogger('PocoMC')

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Extra keyword arguments forwarded to ``pocomc.Sampler``.
        """
        self._kwargs = kwargs
        self._sampler = None

    def init(self, likelihood, prior, rng, **context):
        _, self._likelihood_logpdf_with_derived = likelihood
        self._prior_logpdf, self._prior_rvs, self._prior_ppf = prior
        self._rng = rng
        self._pool = context['pool']
        self._ndim = context['ndim']
        self._output_dir = context.get('output_dir')

    def run(self, **kwargs):
        if not POCOMC_INSTALLED:
            raise ImportError("The 'pocomc' package is required but not installed.")

        if self._pool.main:
            if self._sampler is None:
                prior_obj = _Prior(self._prior_logpdf, self._prior_rvs, self._prior_ppf, self._ndim)
                init_kwargs = update_kwargs(
                    dict(**self._kwargs), 'pocoMC',
                    prior=prior_obj, likelihood=self._likelihood_logpdf_with_derived,
                    n_dim=self._ndim, pool=self._pool,
                    output_dir=self._output_dir,
                    random_state=self._rng.integers(2**32 - 1))
                self._sampler = _pocomc.Sampler(**init_kwargs)

                _patch_save_state(self._sampler)

                # Restore checkpoint if available.
                if self._output_dir is not None:
                    filepath_max = None
                    state_max = -1
                    for filepath in self._output_dir.glob('pmc_*.state'):
                        state = str(filepath.stem).split('_')[1]
                        if state == 'final':
                            filepath_max = filepath
                            break
                        state = int(state)
                        if state > state_max:
                            state_max = state
                            filepath_max = filepath
                    if filepath_max is not None:
                        saved = {attr: getattr(self._sampler, attr, None)
                                 for attr in _CLEAR_BEFORE_SAVE}
                        self._sampler.load_state(filepath_max)
                        for attr, val in saved.items():
                            setattr(self._sampler, attr, val)
                        _patch_save_state(self._sampler)

            run_kwargs = update_kwargs(
                kwargs, 'pocoMC',
                resume_state_path=None,
                save_every=1 if self._output_dir is not None else None)
            self._sampler.run(**run_kwargs)

            samples, weights, logl, logp, blobs = self._sampler.posterior(return_blobs=True)
            blobs = blobs.reshape(len(samples), -1)

            self._pool.stop_wait()
            self.logger.info('Finished sampling.')
            return samples, blobs, dict(aweight=weights, logposterior=logl + logp)
        self._pool.wait()
        return None
