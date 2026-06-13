"""Metropolis-Hastings kernel and utilities."""

import sys
import logging

import numpy as np
import jax

from .base import Kernel


class FastSlowProposer:
    """Proposer sampling fast and slow parameter spaces separately."""

    def __init__(self, cov, fast=None, rng=None):
        """Initialize the proposal distribution.

        Parameters
        ----------
        cov : numpy.ndarray
            Covariance matrix used to whiten parameter space.
        fast : list, optional
            List of dimensions that are considered fast.
        rng : numpy.random.Generator, optional
            Random number generator. Default is ``None``.

        """
        self.rng = rng

        self.n_dim = len(cov)
        if fast is None:
            fast = []
        is_fast = np.isin(np.arange(self.n_dim), fast)
        self.n_fast = np.sum(is_fast)
        self.n_slow = self.n_dim - self.n_fast

        self.sort = np.argsort(is_fast)
        self.unsort = np.argsort(self.sort)
        self.L = np.linalg.cholesky(cov[:, self.sort][self.sort, :])

    def _propose(self, n_dim, k):
        r"""Generate :math:`k \cdot n` random orthogonal vectors.

        Parameters
        ----------
        n_dim : int
            Number of dimensions :math:`n`.
        k : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of shape (k, n_dim, n_dim)
            :math:`k \cdot n` :math:`n`-dimensional orthogonal vectors drawn
            from a unit normal. All vectors within each of the :math:`k` sets
            are orthogonal to each other.

        """
        m = self.rng.standard_normal((k, n_dim, n_dim))
        d = np.sqrt(self.rng.chisquare(n_dim, size=(k, n_dim))) * (
            self.rng.choice([-1, +1], (k, n_dim)))
        Q = np.linalg.qr(m)[0]
        return Q * d[:, :, np.newaxis]

    def propose_fast(self, k):
        r"""Generate random vectors along the fast parameter directions.

        Parameters
        ----------
        k : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of shape (k, n_fast, n_dim)
            :math:`k \cdot n_\mathrm{fast}` :math:`n`-dimensional vectors. All
            vectors are 0 along slow dimensions.

        """
        m_fast = np.zeros((k, self.n_fast, self.n_dim))
        if self.n_fast == 0:
            return np.zeros((k, 0, self.n_dim))
        m_fast[:, :, self.n_slow:] = self._propose(self.n_fast, k)
        return (m_fast @ self.L.T)[:, :, self.unsort]

    def propose_slow(self, k):
        r"""Generate random vectors along the slow parameter directions.

        Parameters
        ----------
        k : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of shape (k, n_slow, n_dim)
            :math:`k \cdot n_\mathrm{slow}` :math:`n`-dimensional vectors.

        """
        m_slow = np.zeros((k, self.n_slow, self.n_dim))
        if self.n_slow == 0:
            return m_slow
        m_slow[:, :, :self.n_slow] = self._propose(self.n_slow, k)
        return (m_slow @ self.L.T)[:, :, self.unsort]


class StandAloneMetropolisHastingsSampler:
    """A Metropolis-Hastings sampler with fast-slow decomposition.

    Note that this is a from-scratch reimplementation of this algorithm. Also,
    this class works outside of ``desilike``.

    .. rubric:: References
    - https://arxiv.org/abs/1304.4473

    """

    def __init__(self, posterior, fast=[], f_fast=1, f_drag=0, pool=None,
                 rng=np.random.default_rng()):
        """Initialize the sampler.

        Parameters
        ----------
        posterior : callable
            Logarithm of the posterior.
        fast : list, optional
            List of dimensions that are considered fast.
        f_fast : int, optional
            Oversampling factor of fast parameters. The default is 1 which
            implies not oversampling.
        f_drag : int, optional
            Factor for dragging of fast parameters. The default is 0, i.e., no
            dragging.
        pool : object
            Pool used for distributing the posterior computation.
        rng : numpy.random.Generator, optional
            NumPy random number generator used for seeding.

        Raises
        ------
        Valuerror
            If `f_fast` is smaller than 1 or `f_drag` is smaller than 0.

        """
        self.posterior = posterior
        self.fast = fast
        self.f_fast = int(f_fast)
        if self.f_fast < 1:
            raise ValueError("'f_fast' cannot be smaller than 1.")
        self.f_drag = int(f_drag)
        if self.f_drag < 0:
            raise ValueError("'f_drag' cannot be smaller than 1.")
        if pool is None:
            self.map = map
        else:
            self.map = pool.map
        self.rng = rng

    def update(self, pos=None, log_p=None, blobs=None, cov=None):
        """Update the sampler's starting position and/or proposal.

        Parameters
        ----------
        pos : numpy.ndarray of shape (nchains, n_dim) or None, optional
            Starting position(s) of the chains.
        log_p : numpy.ndarray of shape (nchains) or None, optional
            Logarith of the posterior of the starting position(s). If not
            provided, these values are computed.
        blobs : numpy.ndarray of shape (nchains, ...) or None, optional
            Blobs for the starting positions.
        cov : numpy.ndarray or None, optional
            Covariance matrix used to whiten parameter space.

        """
        if pos is not None:
            self.pos = np.array(pos, dtype=float)
            self.nchains = len(pos)

            if log_p is None or blobs is None:
                log_p, blobs = self.compute_posterior(self.pos)

            self.log_p = np.array(log_p)
            self.blobs = np.array(blobs)

            self.counter = 0
            self.proposal_fast = []
            self.proposal_slow = []

        if cov is not None:
            self.proposer = FastSlowProposer(
                cov * 2.38**2 / np.sqrt(len(cov)), fast=self.fast,
                rng=self.rng)

    def compute_posterior(self, points):
        """Compute the natural logarithm of the posterior.

        Parameters
        ----------
        points : numpy.ndarray of shape (n_points, n_dim)
            Points for which to compute the posterior.

        Returns
        -------
        log_p : np.ndarray of shape (n_points, )
            Natural logarithm of the posterior.
        blobs : np.ndarray of shape (n_points, ...)
            Blobs associated with the posterior function.

        """
        results = list(self.map(self.posterior, points))
        if isinstance(results[0], tuple):
            log_p = np.array([r[0] for r in results])
            blobs = np.array([r[1] for r in results])
        else:
            log_p = np.array(results)
            blobs = np.zeros((len(points), 0))
        return log_p, blobs

    def propose_fast(self):
        """Propose a fast-parameter step.

        Returns
        -------
        step_fast : numpy.ndararay of shape (nchains, n_dim)
            Fast-parameter steps where slow parameters are unchanged.

        """
        if len(self.proposal_fast) == 0:
            self.proposal_fast = list(np.transpose(self.proposer.propose_fast(
                self.nchains), axes=[1, 0, 2]))
        return self.proposal_fast.pop()

    def propose_slow(self):
        """Propose a slow-parameter step.

        Returns
        -------
        step_slow : numpy.ndararay of shape (nchains, n_dim)
            Slow-parameter steps.

        """
        if len(self.proposal_slow) == 0:
            self.proposal_slow = list(np.transpose(self.proposer.propose_slow(
                self.nchains), axes=[1, 0, 2]))
        proposal_drag = []
        for _ in range(self.f_drag):
            proposal_drag += list(np.transpose(self.proposer.propose_fast(
                self.nchains), axes=[1, 0, 2]))
        return self.proposal_slow.pop(), proposal_drag

    def make_one_step(self):
        """Advance all chains by one step.

        Returns
        -------
        pos : numpy.ndarray of shape (nchains, n_dim)
            New positions in parameter space.
        blobs : np.ndarray of shape (nchains, ...)
            Blobs associated with the posterior function.
        log_p : numpy.ndarray of shape (nchains)
            Logarithm of the posterior.

        """
        n_cycle = self.proposer.n_fast * self.f_fast + self.proposer.n_slow
        if self.counter % n_cycle < self.proposer.n_fast * self.f_fast:
            step, steps_drag = self.propose_fast(), []
        else:
            step, steps_drag = self.propose_slow()
        self.counter += 1

        # First, assume we do a regular step.
        pos_prop = self.pos + step
        log_p_prop, blobs_prop = self.compute_posterior(pos_prop)
        p_accept = np.exp(log_p_prop - self.log_p)

        # If applicable, do a dragging step, instead.
        if len(steps_drag) > 0:
            # The following is described in section III of 1304.4473.
            n = len(steps_drag) + 1

            # We will use a slightly different notation than in the paper.
            # In particular, x represents the change in the fast parameters,
            # not the fast parameters themselves.
            y_new = pos_prop
            y_old = self.pos.copy()
            x = [np.zeros(self.pos.shape)]
            log_p_new = [log_p_prop]
            log_p_old = [self.log_p]

            # Run a mini MCMC chain on x, the fast parameter.
            for i, step in enumerate(steps_drag, start=1):
                log_p_new_prop, blobs_prop = self.compute_posterior(
                    y_new + x[-1] + step)
                log_p_old_prop = self.compute_posterior(
                    y_old + x[-1] + step)[0]
                p_accept = np.exp(
                    ((n - i) * log_p_old_prop + i * log_p_new_prop -
                     (n - i) * log_p_old[-1] - i * log_p_new[-1]) / n)
                accept = self.rng.random(size=self.nchains) < p_accept
                x.append(np.where(accept[:, None], x[-1] + step, x[-1]))
                log_p_new.append(
                    np.where(accept, log_p_new_prop, log_p_new[-1]))
                log_p_old.append(
                    np.where(accept, log_p_old_prop, log_p_old[-1]))

            pos_prop = y_new + x[-1]
            log_p_prop = log_p_new[-1]
            p_accept = np.exp(np.mean(log_p_new, axis=0) -
                              np.mean(log_p_old, axis=0))

        accept = self.rng.random(size=self.nchains) < p_accept
        self.pos = np.where(accept[:, None], pos_prop, self.pos)
        self.log_p = np.where(accept, log_p_prop, self.log_p)
        self.blobs = np.where(accept[:, None], blobs_prop, self.blobs)

        return self.pos.copy(), self.blobs.copy(), self.log_p.copy()

    def make_n_steps(self, n_steps):
        """Advance all chains by :math:`n` steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to take.

        Returns
        -------
        chains : numpy.ndarray of shape (nchains, n_steps, n_dim)
            Positions in parameter space.
        blobs : numpy.ndarray of shape (nchains, n_steps, ...)
            Blobs returned from the posterior.
        log_p : numpy.ndarray of shape (nchains, n_steps)
            Logarithm of the posterior.

        """
        results = [self.make_one_step() for _ in range(n_steps)]
        chains = np.stack([r[0] for r in results], axis=1)
        blobs = np.stack([r[1] for r in results], axis=1)
        log_p = np.stack([r[2] for r in results], axis=1)
        return chains, blobs, log_p


class MetropolisHastings(Kernel):
    """Metropolis-Hastings sampler with fast-slow decomposition.

    .. rubric:: References
    - https://arxiv.org/abs/1304.4473
    """

    logger = logging.getLogger('MetropolisHastings')
    _sampler_cls = 'MCMCSampler'

    def __init__(self, f_fast=1, f_drag=0, fast=None, covariance=None):
        """
        Parameters
        ----------
        f_fast : int
            Fast-parameter oversampling factor.  Default is 1 (no oversampling).
        f_drag : int
            Dragging factor for fast parameters.  Default is 0 (no dragging).
        fast : list of str or None
            Names of parameters considered "fast".  Default is ``None`` (none).
        covariance : array_like or None
            Initial proposal covariance in *rescaled* parameter space.  When
            ``None``, the identity matrix is used (suitable when the sampler
            uses ``rescale=True`` or the posterior is already ~unit-variance).
        """
        self.f_fast = f_fast
        self.f_drag = f_drag
        self.fast = list(fast) if fast is not None else []
        self.covariance = covariance

    def init(self, posterior_logpdf, rng, **context):
        ndim = context['ndim']
        param_shapes = context['param_shapes']

        # Map fast parameter names to flat column indices.
        flat_fast_indices = []
        col = 0
        for name, shape in param_shapes.items():
            size = int(np.prod(shape)) if shape else 1
            if name in self.fast:
                flat_fast_indices.extend(range(col, col + size))
            col += size

        def _posterior(flat):
            return float(posterior_logpdf(jax.numpy.asarray(flat)[None])[0])

        self._standalone = StandAloneMetropolisHastingsSampler(
            _posterior, fast=flat_fast_indices,
            f_fast=self.f_fast, f_drag=self.f_drag, rng=rng)

        self._posterior = _posterior
        self._ndim = ndim
        self._cov = np.asarray(self.covariance) if self.covariance is not None else np.eye(ndim)
        self._initialized = False
        self._adaptation_steps = 0
        self._accumulated_samples = []
        self._total_steps = 0

    def adapt(self, initial_position=None, **kwargs):
        """Store the adaptation horizon; proposal covariance is updated inline during :meth:`run`."""
        self._adaptation_steps = int(kwargs.get('steps', 0))

    def run(self, n_steps, initial_position=None):
        if not self._initialized:
            # initial_position is a flat (ndim,) array in rescaled space
            initial_flat = np.asarray(initial_position).ravel()
            initial_log_p = self._posterior(initial_flat)
            self._standalone.update(
                pos=initial_flat[None, :],
                log_p=np.array([initial_log_p]),
                blobs=np.zeros((1, 0)),
                cov=self._cov)
            self._initialized = True
        chains, _blobs, log_p = self._standalone.make_n_steps(n_steps)
        # chains: (1, n_steps, ndim); log_p: (1, n_steps)
        samples  = chains[0]    # (n_steps, ndim)
        log_post = log_p[0]     # (n_steps,)

        self._total_steps += n_steps

        # Adapt the proposal covariance from accumulated samples while within
        # the adaptation window.
        if self._total_steps < self._adaptation_steps:
            self._accumulated_samples.append(samples)
            all_samps = np.concatenate(self._accumulated_samples, axis=0)
            if len(all_samps) > self._ndim:
                try:
                    self._standalone.update(cov=np.cov(all_samps.T))
                except np.linalg.LinAlgError:
                    pass

        return samples, None, {'logposterior': log_post}
