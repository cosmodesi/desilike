"""Module implementing the nautilus sampler."""
try:
    import nautilus
    NAUTILUS_INSTALLED = True
except ModuleNotFoundError:
    NAUTILUS_INSTALLED = False
import numpy as np

from .base import update_kwargs, PopulationSampler


class NautilusSampler(PopulationSampler):
    """Class for the nautilus sampler.

    Reference
    ---------
    - https://github.com/johannesulf/nautilus
    - https://doi.org/10.1093/mnras/stad2441

    """

    def __init__(self, likelihood, rng=None, directory=None, **kwargs):
        """Initialize the nautilus sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to nautilus during initialization.

        """
        if not NAUTILUS_INSTALLED:
            raise ImportError("The 'nautilus-sampler' package is required but "
                              "not installed.")

        super().__init__(likelihood, rng=rng, directory=directory)

        kwargs = update_kwargs(
            kwargs, 'nautilus', pass_dict=False,
            filepath=None if self.directory is None else self.directory /
            'nautilus.hdf5', pool=self.pool, seed=self.rng.integers(2**32))

        if self.mpicomm.rank == 0:
            self.sampler = nautilus.Sampler(
                self.prior_transform, self.compute_likelihood, self.n_dim,
                **kwargs)
        else:
            self.sampler = None

    def run_sampler(self, **kwargs):
        """Run the nautilus sampler.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments passed to nautilus' ``run`` method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Sampler results.
        extras : dict
            Extra parameters such as weights and derived parameters.

        """
        self.sampler.run(**kwargs)
        samples, log_w, log_l, blobs = self.sampler.posterior(
            return_blobs=True)
        extras = dict(aweight=np.exp(log_w), loglikelihood=log_l)
        extras.update(dict(zip(self.params.keys()[self.n_dim:],
                               np.atleast_2d(blobs.T))))
        return samples, extras
