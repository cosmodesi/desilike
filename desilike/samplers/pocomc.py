"""Module implementing the pocoMC samplers."""

import numpy as np
try:
    import pocomc
    POCOMC_INSTALLED = True
except ModuleNotFoundError:
    POCOMC_INSTALLED = False

from .base import update_kwargs, PopulationSampler


class Prior(object):
    """Prior distribution for PocoMC."""

    def __init__(self, params, random_state=None):
        self.dists = [param.prior for param in params]
        self.random_state = random_state

    def logpdf(self, x):
        """Logarithm of the prior distribution."""
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.dists):
            logp += dist(x[:, i])
        return logp

    def rvs(self, size=1):
        """Sample from the pior."""
        samples = []
        for dist in self.dists:
            samples.append(dist.sample(
                size=size, random_state=self.random_state))
        return np.transpose(samples)

    @property
    def bounds(self):
        """Bounds of the prior distribution."""
        bounds = []
        for dist in self.dists:
            bounds.append(dist.limits)
        return np.array(bounds).astype(float)

    @property
    def dim(self):
        """Dimensionality of the prior."""
        return len(self.dists)


class PocoMCSampler(PopulationSampler):
    """Class for the pocoMC sampler.

    Reference
    ---------
    - https://github.com/minaskar/pocomc
    - https://arxiv.org/abs/2207.05652
    - https://arxiv.org/abs/2207.05660

    """

    def __init__(self, likelihood, rng=None, directory=None, **kwargs):
        """Initialize the PocoMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.RandomState or int, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        kwargs: dict, optional
            Extra keyword arguments passed to pocoMC during initialization.

        """
        if not POCOMC_INSTALLED:
            raise ImportError("The 'pocomc' package is required but not "
                              "installed.")

        super().__init__(likelihood, rng=rng, directory=directory)

        kwargs = update_kwargs(
            kwargs, 'pocoMC', pool=self.pool, output_dir=self.directory,
            random_state=self.rng.integers(2**32 - 1))

        if self.mpicomm.rank == 0:
            prior = Prior(self.likelihood.varied_params)
            self.sampler = pocomc.Sampler(
                prior, self.compute_likelihood, **kwargs)

            # Try to read existing sampler state, if available.
            if self.directory is not None:
                filepath_max = None
                state_max = -1
                for filepath in self.directory.glob('pmc_*.state'):
                    state = str(filepath.stem).split('_')[1]
                    if state == 'final':
                        filepath_max = filepath
                        break
                    state = int(state)
                    if state > state_max:
                        state_max = state
                        filepath_max = filepath
                if filepath_max is not None:
                    # PocoMC tries to read the pickled likelihood. However,
                    # that's not stored on disk.
                    log_likelihood = self.sampler.log_likelihood
                    self.sampler.load_state(filepath_max)
                    self.sampler.log_likelihood = log_likelihood
        else:
            self.sampler = None

    def run_sampler(self, **kwargs):
        """Run the pocoMC sampler.

        Parameters
        ----------
        kwargs: dict, optional
            Extra keyword arguments passed to pocoMC's ``run`` method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Sampler results.
        extras : dict
            Extra parameters such as weights and derived parameters.

        """
        kwargs = update_kwargs(kwargs, 'pocoMC', resume_state_path=None,
                               save_every=1 if self.directory is not None else
                               None)

        self.sampler.run(**kwargs)
        samples, weights, logl, logp, blobs = self.sampler.posterior(
            return_blobs=True)
        extras = dict(aweight=weights, loglikelihood=logl, logposterior=logp)
        extras.update(dict(zip(self.params.keys()[self.n_dim:],
                               np.atleast_2d(blobs.T))))
        return samples, extras
