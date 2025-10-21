"""Module implementing a Metropolis-Hastings sampler.

Note that the implemenation here is independent of the one in cobaya.
"""

import numpy as np


class FastSlowProposal:
    """Proposal sampling fast and slow parameter spaces separately."""

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
            :math:`n_\mathrm{fast}` orthogonal vectors. All proposals are set
            to 0 along slow dimensions.

        """
        m_fast = np.hstack((np.zeros((self.n_fast, self.n_slow)),
                            self._propose(self.n_fast)))
        return (self.L @ m_fast.T).T[:, self.unsort]

    def propose_slow(self):
        r"""Generate random vectors along the slow parameter directions.

        Returns
        -------
        numpy.ndarray of shape (n_fast, n_dim)
            :math:`n_\mathrm{slow}` orthogonal vectors.

        """
        m_slow = np.hstack((self._propose(self.n_slow),
                            np.zeros((self.n_slow, self.n_fast))))
        return (self.L @ m_slow.T).T[:, self.unsort]


class SimpleMetropolisHastingsSampler():
    """A simple Metropolis-Hastings sampler with fast-slow decomposition.

    This sampler only contructs a single chain and is not capable of
    parallelization.
    """

    def __init__(self, posterior, pos, cov, f_fast=1, f_drag=0, fast=[],
                 thin_by=1, rng=np.random.default_rng()):
        """Initialize the sampler.

        Parameters
        ----------
        posterior : function
            Logarithm of the posterior.
        pos : numpy.ndarray
            Starting position.
        cov : numpy.ndarray
            Covariance matrix used to whiten parameter space.
        f_fast : int, optional
            Oversampling factor of fast parameters. The default is 1 which
            implies not oversampling.
        f_drag : int, optional
            Factor for dragging of fast parameters. The default is 0, i.e., no
            dragging.
        fast : list, optional
            List of dimensions that are considered fast.
        thin_by : int
            Thin the chain by this factor. Default is 1, i.e., no thinning.
        rng : numpy.random.Generator, optional
            NumPy random number generator used for seeding.

        Raises
        ------
        Valuerror
            If `f_fast` is smaller than 1 or `f_drag` is smaller than 0.

        """
        self.posterior = posterior
        self.prop = FastSlowProposal(cov, fast=fast, rng=rng)
        self.rng = rng
        self.pos = pos
        self.log_p = self.posterior(pos)
        self.chain = [pos]
        self.thin_by = int(thin_by)
        self.count = 0
        self.f_fast = int(f_fast)
        if self.f_fast < 1:
            raise ValueError("'f_fast' cannot be smaller than 1.")
        self.f_drag = int(f_drag)
        if self.f_drag < 0:
            raise ValueError("'f_drag' cannot be smaller than 1.")

    def make_cycle(self):

        steps_cycle = np.vstack(
            [self.prop.propose_fast() for _ in range(self.f_fast)] +
            [self.prop.propose_slow()])
        slow_cycle = ([False] * self.prop.n_fast * self.f_fast +
                      [True] * self.prop.n_slow)

        for step, slow in zip(steps_cycle, slow_cycle):

            pos = self.pos + step

            if True:  # not slow or self.f_drag == 0:
                log_p = self.posterior(pos)
            else:
                steps_drag = np.vstack(
                    [self.prop.propose_fast()] * self.f_drag)

            self.propose(pos, log_p)

    def propose(self, pos, log_p):
        if self.rng.random() < np.exp(log_p - self.log_p):
            self.log_p = log_p
            self.pos = pos
        self.count += 1
        if self.count % self.thin_by == 0:
            self.chain.append(self.pos)
