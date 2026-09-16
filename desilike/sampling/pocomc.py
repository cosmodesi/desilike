"""Module implementing the pocoMC samplers."""

import numpy as np

try:
    import pocomc
    POCOMC_INSTALLED = True
except ModuleNotFoundError:
    POCOMC_INSTALLED = False

from .base import PopulationSampler, _update_parameters


class Prior:
    """Prior distribution for ``pocoMC``."""

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
    """Class for the ``pocoMC`` preconditioned Monte Carlo sampling.

    .. rubric:: References

    - `pocoMC repo <https://github.com/minaskar/pocomc>`_
    - `pocoMC docs <https://pocomc.readthedocs.io>`_
    - `pocoMC paper A <https://doi.org/10.21105/joss.04634>`_
    - `pocoMC paper B <https://doi.org/10.1093/mnras/stac2272>`_

    """

    def __init__(self, likelihood, rng=None, directory=None, **kwargs):
        """Initialize the ``PocoMC`` sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        rng : numpy.random.Generator, int or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        **kwargs
            Extra keyword arguments passed to pocoMC during initialization.

        Raises
        ------
        ImportError
            If ``pocomc`` is not installed.

        """
        if not POCOMC_INSTALLED:
            msg = "The 'pocomc' package is required but not installed."
            raise ImportError(msg)

        super().__init__(likelihood, rng=rng, directory=directory)

        if self.mpicomm.rank == 0:
            prior = Prior(self.likelihood.varied_params)
            kwargs = _update_parameters(
                kwargs, 'pocoMC', prior=prior,
                likelihood=self._compute_likelihood, n_dim=self.n_dim,
                pool=self.pool, output_dir=self.directory,
                random_state=self.rng.integers(2**32 - 1))
            self.sampler = pocomc.Sampler(**kwargs)

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

    def _run(self, **kwargs):
        """Run the ``pocoMC`` sampler.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments passed to ``pocoMC``'s ``run`` method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, n_dim)
            Samples of varied parameters.
        derived : numpy.ndarray
            Samples of derived parameters.
        extras : dict
            Extra parameters such as weights.

        """
        kwargs = _update_parameters(
            kwargs, 'pocoMC', resume_state_path=None,
            save_every=1 if self.directory is not None else None)

        self.sampler.run(**kwargs)
        samples, weights, logl, logp, blobs = self.sampler.posterior(
            return_blobs=True)
        extras = {'log_weight': np.log(weights), 'log_posterior': logp,
                  'log_likelihood': logl}

        return samples, blobs, extras
