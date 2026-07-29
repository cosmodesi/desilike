"""Module implementing the dynesty samplers."""

try:
    import dynesty
    DYNESTY_INSTALLED = True
except ModuleNotFoundError:
    DYNESTY_INSTALLED = False

from .base import PopulationSampler, _update_parameters


class DynestySampler(PopulationSampler):
    """Wrapper for ``dynesty`` nested samplers.

    .. rubric:: References

    - `dynesty repo <https://github.com/joshspeagle/dynesty>`_
    - `dynesty docs <https://dynesty.readthedocs.io>`_
    - `dynesty paper <https://doi.org/10.1093/mnras/staa278>`_

    """

    def __init__(self, likelihood, dynamic=True, rng=None, directory=None,
                 **kwargs):
        """Initialize the ``dynesty`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        dynamic : boolean, optional
            If ``True``, use ``dynesty.DynamicPopulationSampler`` instead of
            ``dynesty.PopulationSampler``. Default is ``True``.
        rng : numpy.random.Generator, int or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        **kwargs
            Extra keyword arguments passed to ``dynesty`` during
            initialization.

        Raises
        ------
        ImportError
            If ``dynesty`` is not installed.
        ValueError
            If ``directory`` is not ``None`` but ``dynamic`` is Flalse.

        """
        if not DYNESTY_INSTALLED:
            msg = "The 'dynesty' package is required but not installed."
            raise ImportError(msg)

        super().__init__(likelihood, rng=rng, directory=directory)

        if not dynamic and self.directory is not None:
            msg = "dynesty static samplers do not support checkpointing."
            raise ValueError(msg)

        if self.pool.main:
            sampler_cls = (dynesty.DynamicNestedSampler if dynamic else
                           dynesty.NestedSampler)
            if self.directory is not None:
                try:
                    self.sampler = sampler_cls.restore(str(
                        self.directory / 'dynesty.pkl'))
                    self.sampler.loglikelihood.loglikelihood =\
                        self._compute_likelihood
                    self.sampler.prior_transform = self._prior_transform
                except (FileNotFoundError, ValueError):
                    pass
            if not hasattr(self, 'sampler'):
                kwargs = _update_parameters(
                    kwargs, 'dynesty', loglikelihood=self._compute_likelihood,
                    prior_transform=self._prior_transform, ndim=self.n_dim,
                    blob=True, pool=self.pool, rstate=self.rng)
                self.sampler = sampler_cls(**kwargs)

    def _run(self, **kwargs):
        """Run the ``dynesty`` sampler.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments passed to ``dynesty``'s ``run_nested``
            method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Samples of varied parameters.
        derived : numpy.ndarray
            Samples of derived parameters.
        weights : numpy.ndarray
            Weights for the samples.

        """
        checkpoint_file = None if self.directory is None else str(
            self.directory / 'dynesty.pkl')
        kwargs = _update_parameters(
            kwargs, 'dynesty', checkpoint_file=checkpoint_file)

        self.sampler.run_nested(**kwargs)
        results = self.sampler.results

        return results.samples, results['blob'], {
            'log_weight': results['logwt'], 'log_likelihood': results['logl']}
