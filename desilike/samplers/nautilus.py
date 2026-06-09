"""Module implementing the nautilus sampler."""
try:
    import nautilus
    NAUTILUS_INSTALLED = True
except ModuleNotFoundError:
    NAUTILUS_INSTALLED = False
import numpy as np

from .base import update_kwargs, PopulationSampler
from .pool import _SerialPool


class NautilusSampler(PopulationSampler):
    """Wrapper for ``nautilus`` importance nested sampling.

    .. rubric:: References
    - https://github.com/johannesulf/nautilus
    - https://doi.org/10.1093/mnras/stad2441

    """

    def __init__(self, posterior, rng=None, directory=None, rescale=False, covariance=None,
                 batch_size=None, **kwargs):
        """Initialize the ``nautilus`` sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
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

        super().__init__(posterior, rng=rng, directory=directory,
                         rescale=rescale, covariance=covariance, batch_size=batch_size)

        # nautilus accepts pool=(pool_likelihood, pool_sampler).
        # pool_likelihood uses self.pool (MPI-aware, batched) for likelihood calls.
        # pool_sampler must be a plain serial pool: using a second MPIPool on the
        # same communicator would interfere with workers blocked in self.pool.wait(),
        # causing them to receive nautilus's internal messages and corrupt the
        # function registry.
        pool_sampler = _SerialPool(batch_size=0)
        if self.pool.main:
            kwargs = update_kwargs(
                kwargs, 'nautilus', prior=self.prior_transform,
                likelihood=self.compute_likelihood, n_dim=self.ndim,
                pass_dict=False,
                filepath=None if self.directory is None else self.directory /
                'nautilus.h5', pool=(self.pool, pool_sampler),
                seed=self.rng.integers(2**32))
            self.sampler = nautilus.Sampler(**kwargs)
            self.pool.stop_wait()
        else:
            self.pool.wait()

    def run_sampler(self, **kwargs):
        """Run the ``nautilus`` sampler.

        Parameters
        ----------
        **kwargs: dict, optional
            Extra keyword arguments passed to ``nautilus``'s ``run`` method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, ndim)
            Samples of varied parameters.
        derived : numpy.ndarray
            Samples of derived parameters.
        extras : dict
            Extra parameters such as weights.

        """
        self.sampler.run(**kwargs)
        samples, log_w, log_l, blobs = self.sampler.posterior(
            return_blobs=True)
        # Compute log-prior for each sample; log-posterior = log-posterior +
        # log-prior.
        log_prior = np.array(list(self.pool.map(self.compute_prior, samples)))
        self.logger.info('Finished sampling.')
        return samples, blobs.reshape(len(samples), -1), dict(aweight=np.exp(log_w), logposterior=log_l + log_prior)
