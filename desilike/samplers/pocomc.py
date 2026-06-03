"""Module implementing the pocoMC samplers."""

import numpy as np
import jax
try:
    import pocomc
    POCOMC_INSTALLED = True
except ModuleNotFoundError:
    POCOMC_INSTALLED = False

from .base import update_kwargs, PopulationSampler


class Prior(object):
    """Prior distribution for ``pocoMC``."""

    def __init__(self, params, rng=None):
        self.dists = [param.prior for param in params]
        self._rng = rng if rng is not None else np.random.default_rng()

    def logpdf(self, x):
        """Logarithm of the prior distribution."""
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.dists):
            logp += np.asarray(dist.logpdf(x[:, i]))
        return logp

    def rvs(self, size=1):
        """Sample from the prior."""
        samples = []
        for dist in self.dists:
            key = jax.random.PRNGKey(int(self._rng.integers(2**32)))
            samples.append(np.asarray(dist.sample(key, shape=(size,))))
        return np.column_stack(samples)

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
    - https://github.com/minaskar/pocomc
    - https://doi.org/10.21105/joss.04634
    - https://doi.org/10.1093/mnras/stac2272

    """

    def __init__(self, posterior, rng=None, directory=None, **kwargs):
        """Initialize the ``PocoMC`` sampler.

        Parameters
        ----------
        posterior : CompiledGraph
            Compiled pipeline returning the log-posterior.
        rng : numpy.random.Generator, int or None, optional
            Random number generator. Default is ``None``.
        directory : str, Path, optional
            Save samples to this location. Default is ``None``.
        **kwargs: dict, optional
            Extra keyword arguments passed to pocoMC during initialization.

        """
        if not POCOMC_INSTALLED:
            raise ImportError("The 'pocomc' package is required but not "
                              "installed.")

        super().__init__(posterior, rng=rng, directory=directory)

        if self.mpicomm.rank == 0:
            prior = Prior(self.varied_params, rng=self.rng)

            # pocomc maps the likelihood over particles through ``self.pool``.
            # The pool is always vectorized, so compute_likelihood is batched:
            # it takes an (N, ndim) batch and returns a list of N
            # (log_l, derived) tuples.
            #
            # When there are derived parameters, return ``(log_l, derived)``
            # so pocomc stores blobs.  When there are none, return plain
            # scalars — pocomc checks ``len(result) > 1`` to detect blobs and a
            # scalar result skips that path entirely.
            if self.n_derived:
                _likelihood_fn = self.compute_likelihood
            else:
                _likelihood_fn = lambda batch: [result[0] for result in self.compute_likelihood(batch)]

            kwargs = update_kwargs(
                kwargs, 'pocoMC', prior=prior,
                likelihood=_likelihood_fn, n_dim=self.ndim,
                pool=self.pool, output_dir=self.directory,
                random_state=self.rng.integers(2**32 - 1))
            self.sampler = pocomc.Sampler(**kwargs)

            # pocomc's save_state serialises its entire __dict__ with dill.
            # Our likelihood function captures ``self`` (PocoMCSampler) which
            # holds a JAX-JIT-compiled callable and MPI objects that cannot be
            # pickled.  We monkey-patch save_state to strip ``log_likelihood``
            # before dumping and restore it afterwards, mirroring how load_state
            # is already handled below.
            _original_save_state = self.sampler.save_state

            def _save_state_no_likelihood(path):
                saved_log_likelihood = self.sampler.log_likelihood
                self.sampler.log_likelihood = None
                try:
                    _original_save_state(path)
                finally:
                    self.sampler.log_likelihood = saved_log_likelihood

            self.sampler.save_state = _save_state_no_likelihood

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
                    # that's not stored on disk — restore it manually.
                    log_likelihood = self.sampler.log_likelihood
                    self.sampler.load_state(filepath_max)
                    self.sampler.log_likelihood = log_likelihood
                    # Re-apply our save_state patch after load_state, which
                    # replaces the method with the one from the pickled state.
                    self.sampler.save_state = _save_state_no_likelihood

    def run_sampler(self, **kwargs):
        """Run the ``pocoMC`` sampler.

        Parameters
        ----------
        **kwargs: dict, optional
            Extra keyword arguments passed to ``pocoMC``'s ``run`` method.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, ndim)
            Samples of varied parameters.
        derived : numpy.ndarray
            Samples of derived parameters.
        extras : dict
            Extra parameters such as weights.

        """
        kwargs = update_kwargs(
            kwargs, 'pocoMC', resume_state_path=None,
            save_every=1 if self.directory is not None else None)

        self.sampler.run(**kwargs)
        if self.n_derived:
            samples, weights, logl, logp, blobs = self.sampler.posterior(
                return_blobs=True)
            blobs = blobs.reshape(len(samples), -1)
        else:
            samples, weights, logl, logp = self.sampler.posterior(
                return_blobs=False)
            blobs = np.empty((len(samples), 0))
        # pocomc returns log_likelihood as ``logl`` and log_prior as ``logp``;
        # the log-posterior is their sum.
        extras = dict(aweight=weights, logposterior=logl + logp)

        return samples, blobs, extras
