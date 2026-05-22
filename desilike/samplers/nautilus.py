"""Module implementing the nautilus sampler."""
try:
    import nautilus
    NAUTILUS_INSTALLED = True
except ModuleNotFoundError:
    NAUTILUS_INSTALLED = False
import numpy as np

from .base import update_parameters, PopulationSampler


class NautilusSampler(PopulationSampler):
    """Wrapper for ``nautilus`` importance nested sampling.

    .. rubric:: References
    - https://github.com/johannesulf/nautilus
    - https://doi.org/10.1093/mnras/stad2441

    """

    def __init__(self, likelihood, rng=None, directory=None, **kwargs):
        """Initialize the ``nautilus`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.Generator, int or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        **kwargs: dict, optional
            Extra keyword arguments passed to ``nautilus`` during
            initialization.

        """
        if not NAUTILUS_INSTALLED:
            raise ImportError("The 'nautilus-sampler' package is required but "
                              "not installed.")

        super().__init__(likelihood, rng=rng, directory=directory)

        if self.mpicomm.rank == 0:
            kwargs = update_parameters(
                kwargs, 'nautilus', prior=self.prior_transform,
                likelihood=self.compute_likelihood, n_dim=self.n_dim,
                pass_dict=False,
                filepath=None if self.directory is None else self.directory /
                'nautilus.hdf5', pool=self.pool, seed=self.rng.integers(2**32))
            self.sampler = nautilus.Sampler(**kwargs)

    def run_sampler(self, **kwargs):
        """Run the ``nautilus`` sampler.

        Parameters
        ----------
        **kwargs: dict, optional
            Extra keyword arguments passed to ``nautilus``'s ``run`` method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Samples of varied parameters.
        derived : numpy.ndarray
            Samples of derived parameters.
        extras : dict
            Extra parameters such as weights.

        """
        self.sampler.run(**kwargs)
        samples, log_w, log_l, blobs = self.sampler.posterior(
            return_blobs=True)
        return samples, blobs.reshape(len(samples), -1), dict(
            aweight=np.exp(log_w))
