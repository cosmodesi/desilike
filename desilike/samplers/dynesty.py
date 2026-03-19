"""Module implementing the dynesty samplers."""

try:
    import dynesty
    DYNESTY_INSTALLED = True
except ModuleNotFoundError:
    DYNESTY_INSTALLED = False

from .base import update_parameters, PopulationSampler


class DynestySampler(PopulationSampler):
    """Wrapper for ``dynesty`` nested samplers.

    .. rubric:: References
    - https://github.com/joshspeagle/dynesty
    - https://doi.org/10.1093/mnras/staa278

    """

    def __init__(self, likelihood, dynamic=True, rng=None, directory=None,
                 **kwargs):
        """Initialize the ``dynesty`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        dynamic : boolean, optional
            If True, use ``dynesty.DynamicPopulationSampler`` instead of
            ``dynesty.PopulationSampler``. Default is True.
        rng : numpy.random.Generator, int or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``dynesty`` during
            initialization.

        """
        if not DYNESTY_INSTALLED:
            raise ImportError("The 'dynesty' package is required but not "
                              "installed.")

        super().__init__(likelihood, rng=rng, directory=directory)

        if not dynamic and self.directory is not None:
            raise ValueError("dynesty does not support checkpointing for the "
                             "static sampler.")

        if self.mpicomm.rank == 0:
            sampler_cls = (dynesty.DynamicNestedSampler if dynamic else
                           dynesty.NestedSampler)
            if self.directory is not None:
                try:
                    self.sampler = sampler_cls.restore(str(
                        self.directory / 'dynesty.pkl'))
                    self.sampler.loglikelihood.loglikelihood =\
                        self.compute_likelihood
                    self.sampler.prior_transform = self.prior_transform
                except (FileNotFoundError, ValueError):
                    pass
            if not hasattr(self, 'sampler'):
                kwargs = update_parameters(
                    kwargs, 'dynesty', loglikelihood=self.compute_likelihood,
                    prior_transform=self.prior_transform, ndim=self.n_dim,
                    blob=True, pool=self.pool, rstate=self.rng)
                self.sampler = sampler_cls(**kwargs)

    def run_sampler(self, **kwargs):
        """Run the ``dynesty`` sampler.

        Parameters
        ----------
        **kwargs: dict, optional
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
        kwargs = update_parameters(
            kwargs, 'dynesty', checkpoint_file=checkpoint_file)

        self.sampler.run_nested(**kwargs)
        results = self.sampler.results

        return results.samples, results['blob'], dict(
            log_weight=results['logwt'], log_likelihood=results['logl'])
