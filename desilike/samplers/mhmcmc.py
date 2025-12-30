"""Module implementing a Metropolis-Hastings sampler.

Note that the implemenation here is independent of the one in cobaya.
"""

import numpy as np

from .base import MarkovChainSampler


class FastSlowProposer:
    """Proposer sampling fast and slow parameter spaces separately."""

    def __init__(self, cov, fast=[], rng=np.random.default_rng()):
        """Initialize the proposal distribution.

        Parameters
        ----------
        cov : numpy.ndarray
            Covariance matrix used to whiten parameter space.
        fast : list, optional
            List of dimensions that are considered fast.
        rng : numpy.random.Generator, optional
            NumPy random number generator used for seeding.

        """
        self.rng = rng

        self.n_dim = len(cov)
        is_fast = np.isin(np.arange(self.n_dim), fast)
        self.n_fast = np.sum(is_fast)
        self.n_slow = self.n_dim - self.n_fast

        self.sort = np.argsort(is_fast)
        self.unsort = np.argsort(self.sort)
        self.L = np.linalg.cholesky(cov[:, self.sort][self.sort, :])

    def _propose(self, n_dim):
        """Generate :math:`n` random :math:`n`-dimensional orthogonal vectors.

        Parameters
        ----------
        n_dim : int
            Number of dimensions :math:`n`.

        Returns
        -------
        numpy.ndarray of shape (n_dim, n_dim)
            :math:`n` :math:`n`-dimensional orthogonal vectors drawn from a
            unit normal. All vectors are orthogonal to each other.

        """
        m = self.rng.standard_normal((n_dim, n_dim))
        d = np.linalg.norm(m, axis=1)
        Q = (np.linalg.qr(m.T)[0])
        return (Q * d * 2 * (self.rng.integers(0, 2, n_dim) - 0.5)).T

    def propose_fast(self):
        r"""Generate random vectors along the fast parameter directions.

        Returns
        -------
        numpy.ndarray of shape (n_fast, n_dim)
            :math:`n_\mathrm{fast}` :math:`n`-dimensional vectors. All vectors
            are 0 along slow dimensions.

        """
        if self.n_fast == 0:
            return np.zeros((0, self.n_dim))
        m_fast = np.hstack((np.zeros((self.n_fast, self.n_slow)),
                            self._propose(self.n_fast)))
        return (self.L @ m_fast.T).T[:, self.unsort]

    def propose_slow(self):
        r"""Generate random vectors along the slow parameter directions.

        Returns
        -------
        numpy.ndarray of shape (n_fast, n_dim)
            :math:`n_\mathrm{slow}` :math:`n`-dimensional vectors.

        """
        if self.n_slow == 0:
            return np.zeros((0, self.n_dim))
        m_slow = np.hstack((self._propose(self.n_slow),
                            np.zeros((self.n_slow, self.n_fast))))
        return (self.L @ m_slow.T).T[:, self.unsort]


class StandAloneMetropolisHastingsSampler():
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

    def update(self, pos=None, log_p=None, cov=None):
        """Update the sampler's starting position and/or proposal.

        Parameters
        ----------
        pos : numpy.ndarray of shape (n_chains, n_dim) or None, optional
            Starting position(s) of the chains.
        log_p : numpy.ndarray of shape (n_chains) or None, optional
            Logarith of the posterior of the starting position(s). If not
            provided, these values are computed.
        cov : numpy.ndarray or None, optional
            Covariance matrix used to whiten parameter space.

        """
        if pos is not None:
            self.pos = np.array(pos, dtype=float)
            self.n_chains = len(pos)

            if log_p is None:
                self.log_p = self.compute_posterior(self.pos)
            else:
                self.log_p = np.array(log_p)

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
        point : numpy.ndarray of shape (n_points, n_dim)
            Points for which to compute the likelihood.

        Returns
        -------
        log_p : float
            Natural logarithm of the posterior.

        """
        return np.array(list(self.map(self.posterior, points)))

    def propose_fast(self):
        """Propose a fast-parameter step.

        Returns
        -------
        step_fast : numpy.ndararay of shape (n_chains, n_dim)
            Fast-parameter steps where slow parameters are unchanged.

        """
        if len(self.proposal_fast) == 0:
            self.proposal_fast = list(np.stack([
                self.proposer.propose_fast() for _ in range(self.n_chains)],
                axis=1))
        return self.proposal_fast.pop()

    def propose_slow(self):
        """Propose a slow-parameter step.

        Returns
        -------
        step_slow : numpy.ndararay of shape (n_chains, n_dim)
            Slow-parameter steps.

        """
        if len(self.proposal_slow) == 0:
            self.proposal_slow = list(np.stack([
                self.proposer.propose_slow() for _ in range(self.n_chains)],
                axis=1))
        proposal_drag = []
        for _ in range(self.f_drag):
            proposal_drag += list(np.stack([
                self.proposer.propose_fast() for _ in range(self.n_chains)],
                axis=1))
        return self.proposal_slow.pop(), proposal_drag

    def make_one_step(self):
        """Advance all chains by one step.

        Returns
        -------
        pos : numpy.ndarray of shape (n_chains, n_dim)
            New positions in parameter space.
        log_p : numpy.ndarray of shape (n_chains)
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
        log_p_prop = np.array(list(self.map(self.posterior, pos_prop)))
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
                log_p_new_prop = self.compute_posterior(
                    y_new + x[-1] + step)
                log_p_old_prop = self.compute_posterior(
                    y_old + x[-1] + step)
                p_accept = np.exp(
                    ((n - i) * log_p_old_prop + i * log_p_new_prop -
                     (n - i) * log_p_old[-1] - i * log_p_new[-1]) / n)
                accept = self.rng.random(size=self.n_chains) < p_accept
                x.append(np.where(accept[:, None], x[-1] + step, x[-1]))
                log_p_new.append(
                    np.where(accept, log_p_new_prop, log_p_new[-1]))
                log_p_old.append(
                    np.where(accept, log_p_old_prop, log_p_old[-1]))

            pos_prop = y_new + x[-1]
            log_p_prop = log_p_new[-1]
            p_accept = np.exp(np.mean(log_p_new, axis=0) -
                              np.mean(log_p_old, axis=0))

        accept = self.rng.random(size=self.n_chains) < p_accept
        self.pos = np.where(accept[:, None], pos_prop, self.pos)
        self.log_p = np.where(accept, log_p_prop, self.log_p)

        return self.pos.copy(), self.log_p.copy()

    def make_n_steps(self, n_steps):
        """Advance all chains by :math:`n` steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to take.

        Returns
        -------
        chains : numpy.ndarray of shape (n_chains, n_steps, n_dim)
            Positiions in parameter space.
        log_p : numpy.ndarray of shape (n_chains, n_steps)
            Logarithm of the posterior.

        """
        results = [self.make_one_step() for _ in range(n_steps)]
        chains = np.stack([r[0] for r in results], axis=1)
        log_p = np.stack([r[1] for r in results], axis=1)
        return chains, log_p


class MetropolisHastingsSampler(MarkovChainSampler):
    """A Metropolis-Hastings sampler with fast-slow decomposition.

    Note that this is a from-scratch reimplementation of this algorithm.

    .. rubric:: References
    - https://arxiv.org/abs/1304.4473

    """

    def __init__(self, likelihood, n_chains=4, cov=None, f_fast=1, f_drag=0,
                 fast=[], rng=None, directory=None):
        """Initialize the Metropolis-Hastings sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood to sample.
        n_chains : int, optional
            Number of chains. Default is 4.
        cov : numpy.ndarray or None, optional
            Covariance matrix estimate used to whiten parameter space. If None,
            the sampler will use each parameter's proposal scale.
        f_fast : int, optional
            Oversampling factor of fast parameters. The default is 1 which
            implies not oversampling.
        f_drag : int, optional
            Factor for dragging of fast parameters. The default is 0, i.e., no
            dragging.
        fast : list, optional
            List of dimensions that are considered fast.
        rng : numpy.random.RandomState, int, or None, optional
            Random number generator for seeding. If ``None``, no seed is used.
            Default is ``None``.
        directory : str, Path, or None, optional
            Save samples to this location. Default is ``None``.

        """
        super().__init__(likelihood, n_chains=n_chains, rng=rng,
                         directory=directory)

        self.sampler = StandAloneMetropolisHastingsSampler(
            self.compute_posterior, pool=self.pool, rng=self.rng)
        if cov is None:
            n_dim = len(self.likelihood.varied_params)
            cov = np.zeros((n_dim, n_dim))
            for i, param in enumerate(self.likelihood.varied_params):
                cov[i, i] = param.proposal**2
        self.sampler.update(cov=cov)

    def run_sampler(self, steps):
        """Run the Metropolis-Hastings sampler.

        Parameters
        ----------
        steps: int
            Number of steps to take.

        """
        if not hasattr(self.sampler, 'pos'):
            self.sampler.update(
                pos=self.chains[:, -1, :], log_p=self.log_post[:, -1])

        chains, log_post = self.sampler.make_n_steps(steps)
        self.chains = np.concatenate([self.chains, chains], axis=1)
        self.log_post = np.concatenate([self.log_post, log_post], axis=1)

        if len(self.chains[0]) < self.learn_steps:
            self.sampler.update(cov=np.cov(np.stack(self.chains)))
