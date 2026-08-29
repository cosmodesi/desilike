"""Tempered sequential Monte Carlo, bridging a proposal to the posterior."""

import logging
import time

import numpy as np

from ..samples import diagnostics
from .base import PopulationKernel


class SMC(PopulationKernel):
    r"""Gradient-free tempered sequential Monte Carlo.

    The tempered path is geometric between the proposal :math:`q` (the beta = 0
    distribution, see the ``proposal`` argument of
    :func:`~desilike.samplers.base.Sampler`) and the posterior :math:`p`:

    .. math:: \log \pi_\beta = \log q + \beta (\log p - \log q), \quad \beta: 0 \to 1.

    With the default proposal -- the prior -- this is the usual prior-to-posterior
    annealing. Its real use is with a proposal that is *already posterior-shaped*: give
    it an emulated posterior and a chain sampled under it
    (:class:`~desilike.samplers.proposals.SamplesProposal`), and the distance the sampler
    has to cover is no longer prior-to-posterior but the emulator error alone. The
    exponent then rides on ``log p - log q``, whose scatter sigma sets the whole cost: the
    schedule needs roughly ``sigma / 0.47`` temperatures, against the ``exp(sigma^2)``
    weight collapse that kills plain importance sampling
    (:class:`~desilike.samplers.importance.Importance`) over the same gap.

    Each temperature is: choose ``dbeta`` so the incremental effective sample size holds
    at ``target_ess`` -> systematic resampling -> differential-evolution Metropolis
    rejuvenation. The DE proposal ``theta' = theta + gamma (theta_j - theta_k) + eps``
    needs no gradients and no tuned covariance -- the particle cloud supplies scale and
    correlations -- and every operation is a fixed-size batched call.

    When the proposal is a density in its own right rather than the prior, rejuvenation
    uses **delayed acceptance**: a proposal is first screened on the cheap ``q'/q``, and
    the expensive posterior is evaluated only for the survivors, accepted on
    ``exp(beta (Delta' - Delta))``. The product of the two stages is exactly the plain
    Metropolis acceptance for :math:`\pi_\beta`, *whatever* the quality of the surrogate,
    so invariance is untouched and only efficiency depends on the emulator.

    ``log(Z_p / Z_q)`` accumulates over the schedule as a free byproduct and is returned
    in ``samples.attrs['logevidence']``, together with the full per-temperature history.
    Read that history: every diagnosis of a bridge that went wrong is in it.

    The run is not checkpointed: ``output_dir`` receives the final samples, not the
    intermediate temperatures, so an interrupted run restarts from beta = 0.

    .. rubric:: References
    - Del Moral, Doucet & Jasra, https://doi.org/10.1111/j.1467-9868.2006.00553.x
    - Ter Braak, https://doi.org/10.1007/s11222-006-8769-1
    - Christen & Fox, https://doi.org/10.1198/106186005X76983
    """

    logger = logging.getLogger('SMC')

    # Pad every pooled call up to a multiple of this many points, so the vmapped
    # posterior compiles for exactly one batch shape however many points are live.
    _batch_size = 256

    def __init__(self, nparticles=1024, target_ess=0.8, target_moves=2., max_steps=20,
                 target_accept=0.3, delayed=None, jitter=1e-4, mode_jump_every=10,
                 max_temperatures=200):
        """
        Parameters
        ----------
        nparticles : int, optional
            Number of particles. Default is 1024.
        target_ess : float, optional
            Fraction of ``nparticles`` the incremental effective sample size is held at
            when choosing each temperature step. Default is 0.8; 0.5 roughly halves the
            number of temperatures and is acceptable when the posterior is cheap.
        target_moves : float, optional
            Rejuvenation stops once particles have accepted this many moves on average.
            Default is 2.
        max_steps : int, optional
            Hard cap on rejuvenation steps per temperature. Default is 20.
        target_accept : float, optional
            Acceptance rate the DE step size is tuned toward, within each temperature.
            Default is 0.3.
        delayed : bool or None, optional
            Use delayed acceptance during rejuvenation. ``None`` (default) enables it
            whenever a proposal is set, and disables it for the plain prior -- where
            screening on the prior would cost an evaluation and screen out nothing.
        jitter : float, optional
            Per-dimension jitter added to DE proposals, in units of the particle standard
            deviation. Breaks the exact degeneracy of duplicated particles. Default 1e-4.
        mode_jump_every : int, optional
            Every n-th rejuvenation step uses ``gamma = 1`` instead of the tuned step size,
            so particles can jump between modes. Default is 10.
        max_temperatures : int, optional
            Give up after this many temperatures. Default is 200.
        """
        self.nparticles = int(nparticles)
        self.target_ess = float(target_ess)
        self.target_moves = float(target_moves)
        self.max_steps = int(max_steps)
        self.target_accept = float(target_accept)
        self.delayed = delayed
        self.jitter = float(jitter)
        self.mode_jump_every = int(mode_jump_every)
        self.max_temperatures = int(max_temperatures)
        self.history = []

    def reset_state(self):
        self.history = []

    def init(self, likelihood, prior, rng, **context):
        _, self._likelihood_logpdf_with_derived = likelihood
        self._prior_logpdf, _, self._prior_rvs, self._prior_bounds = prior
        self._rng = rng
        self._pool = context['pool']
        self._ndim = context['ndim']
        self._has_proposal = context.get('proposal', None) is not None
        self._output_dir = context.get('output_dir')

    # ── pooled evaluation ─────────────────────────────────────────────────────

    def _pad(self, points):
        """Pad *points* up to a multiple of the per-rank batch size.

        The pooled evaluators are ``jax.jit(jax.vmap(...))``, so every distinct batch
        length costs a compilation. Delayed acceptance hands them a different number of
        survivors at every step, so the batch is padded (with copies of its last row,
        dropped on return) to a shape the pool always chunks identically.
        """
        points = np.asarray(points)
        # Match the granularity the pool actually chunks at, so every chunk it hands the
        # compiled function is full. batch_size = 0 means one call per point: nothing to pad to.
        batch_size = getattr(self._pool, 'batch_size', None)
        if batch_size == 0:
            return points, len(points)
        granularity = int(batch_size or self._batch_size) * getattr(self._pool, 'size', 1)
        npoints = len(points)
        npadded = int(np.ceil(npoints / granularity)) * granularity
        if npadded == npoints:
            return points, npoints
        return np.concatenate([points, np.repeat(points[-1:], npadded - npoints, axis=0)]), npoints

    def _log_proposal(self, points):
        """Return the log-proposal density (beta = 0 distribution) at *points*."""
        padded, npoints = self._pad(points)
        return np.asarray(self._pool.map(self._prior_logpdf, padded))[:npoints]

    def _delta(self, points):
        """Return ``(log_posterior - log_proposal, derived)`` at *points*.

        This is the exponent the schedule tempers, and the expensive call: the posterior
        is evaluated here and nowhere else.
        """
        padded, npoints = self._pad(points)
        results = self._pool.map(self._likelihood_logpdf_with_derived, padded)
        delta = np.array([result[0] for result in results[:npoints]])
        derived = np.array([result[1] for result in results[:npoints]])
        # Count the padding too: it is evaluated, and hiding it would flatter the cost.
        self._nevaluations += len(padded)
        return delta, derived

    # ── schedule ──────────────────────────────────────────────────────────────

    def _choose_dbeta(self, beta, delta):
        """Largest step whose incremental ESS still reaches ``target_ess``, by bisection."""
        def ess_fraction(dbeta):
            return diagnostics.kish_ess(dbeta * delta) / delta.size

        remaining = 1. - beta
        if ess_fraction(remaining) >= self.target_ess:
            return remaining
        low, high = 0., remaining
        for _ in range(60):
            mid = 0.5 * (low + high)
            if ess_fraction(mid) >= self.target_ess:
                low = mid
            else:
                high = mid
        if low <= 0.:
            raise RuntimeError(
                'SMC cannot make any progress: even an infinitesimal temperature step drops the '
                'effective sample size below the target. The proposal is far too narrow, or the '
                'log-posterior differences are not finite.')
        return low

    # ── rejuvenation ──────────────────────────────────────────────────────────

    def _rejuvenate(self, beta, state, delayed):
        """Move the particles at fixed *beta*, in place, until they have mixed.

        Returns the number of steps taken, the mean number of accepted moves per particle,
        the acceptance rate, and the number of posterior evaluations spent.
        """
        particles, log_proposal, delta, derived = state
        nparticles = len(particles)
        own = np.arange(nparticles)
        gamma0 = 2.38 / np.sqrt(2. * self._ndim)
        moves = np.zeros(nparticles)
        accepted = proposed = nevaluations = 0
        step = 0
        while step < self.max_steps and np.mean(moves) < self.target_moves:
            # Standard DE-MC: the tuned gamma most steps, gamma = 1 every mode_jump_every-th
            # so particles can jump between modes.
            mode_jump = self.mode_jump_every and (step + 1) % self.mode_jump_every == 0
            gamma = 1. if mode_jump else gamma0 * self._gamma_scale
            first = self._rng.integers(1, nparticles, size=nparticles)
            second = self._rng.integers(1, nparticles - 1, size=nparticles)
            second[second >= first] += 1
            jitter = self.jitter * np.std(particles, axis=0) * self._rng.standard_normal(particles.shape)
            candidates = particles + gamma * (particles[(own + first) % nparticles]
                                              - particles[(own + second) % nparticles]) + jitter

            log_proposal_new = self._log_proposal(candidates)
            delta_new = np.full(nparticles, np.nan)
            derived_new = np.zeros_like(derived)
            accept = np.zeros(nparticles, dtype=bool)
            if delayed:
                # Stage 1: screen on the cheap proposal density alone.
                with np.errstate(invalid='ignore'):
                    log_ratio = log_proposal_new - log_proposal
                survives = np.log(self._rng.random(nparticles)) < np.nan_to_num(log_ratio, nan=-np.inf)
                surviving = np.flatnonzero(survives)
                if surviving.size:
                    # Stage 2: the posterior is evaluated only for stage-1 survivors. The
                    # product of the two acceptance probabilities is the plain Metropolis
                    # one for pi_beta, for any surrogate quality.
                    delta_sub, derived_sub = self._delta(candidates[surviving])
                    nevaluations += surviving.size
                    delta_new[surviving] = delta_sub
                    derived_new[surviving] = derived_sub
                    with np.errstate(invalid='ignore'):
                        log_ratio = beta * (delta_new[surviving] - delta[surviving])
                    accept[surviving] = (np.log(self._rng.random(surviving.size))
                                         < np.nan_to_num(log_ratio, nan=-np.inf))
            else:
                delta_new, derived_new = self._delta(candidates)
                nevaluations += nparticles
                with np.errstate(invalid='ignore'):
                    log_ratio = (log_proposal_new + beta * delta_new) - (log_proposal + beta * delta)
                accept = np.log(self._rng.random(nparticles)) < np.nan_to_num(log_ratio, nan=-np.inf)

            particles[accept] = candidates[accept]
            log_proposal[accept] = log_proposal_new[accept]
            delta[accept] = delta_new[accept]
            derived[accept] = derived_new[accept]
            moves += accept
            accepted += int(accept.sum())
            proposed += nparticles
            # Robbins-Monro step-size control *within* the temperature; mode-jump steps are
            # excluded, their gamma being fixed at 1 by construction.
            if not mode_jump:
                self._gamma_scale = float(np.clip(
                    self._gamma_scale * np.exp(accept.mean() - self.target_accept), 0.1, 4.))
            step += 1
        return step, float(np.mean(moves)), accepted / max(proposed, 1), nevaluations

    # ── driver ────────────────────────────────────────────────────────────────

    def run(self, nparticles=None, **kwargs):
        """Run the tempered bridge from the proposal to the posterior.

        Parameters
        ----------
        nparticles : int or None, optional
            Overrides the number of particles given at construction.

        Returns
        -------
        particles : numpy.ndarray, shape ``(nparticles, ndim)``
            Equal-weight samples of the posterior, in rescaled space. **Main process
            only**; workers return ``None``.
        derived : numpy.ndarray
            Derived-parameter values.
        extras : dict
            ``aweight``, ``logposterior``, and ``attrs`` carrying ``logevidence``, the
            per-temperature ``history``, and the evaluation counts.
        """
        if kwargs:
            raise TypeError(f'SMC.run got unexpected keyword arguments {sorted(kwargs)}.')
        if not self._pool.main:
            self._pool.wait()
            return None

        nparticles = self.nparticles if nparticles is None else int(nparticles)
        if nparticles < 8:
            raise ValueError(f'nparticles = {nparticles} is too small: the differential-evolution '
                             'proposal draws its scale and correlations from the particle cloud.')
        delayed = self._has_proposal if self.delayed is None else bool(self.delayed)
        self._nevaluations = 0
        self._gamma_scale = 1.
        self.history = []

        start = time.time()
        particles = np.array(self._prior_rvs(nparticles, self._rng), dtype='f8')
        log_proposal = self._log_proposal(particles)
        delta, derived = self._delta(particles)
        finite = np.isfinite(delta) & np.isfinite(log_proposal)
        if not finite.all():
            raise ValueError(f'{int((~finite).sum())} of {nparticles} initial particles have a '
                             'non-finite log-posterior or log-proposal: the proposal puts mass where '
                             'the posterior is undefined. Truncate it to the prior support.')
        self.logger.info('Initialized %d particles in %.1f s; sigma(log p - log q) = %.3f, '
                         'delayed acceptance %s.', nparticles, time.time() - start, np.std(delta),
                         'on' if delayed else 'off')

        beta = 0.
        run_start = time.time()
        for temperature in range(self.max_temperatures):
            dbeta = self._choose_dbeta(beta, delta)
            beta += dbeta
            log_weights = dbeta * delta
            ess = diagnostics.kish_ess(log_weights)
            # log Z_beta / Z_{beta - dbeta}: the reweighting gives it away for free.
            shift = log_weights.max()
            log_evidence_increment = float(np.log(np.mean(np.exp(log_weights - shift))) + shift)

            index = diagnostics.systematic_resample(np.exp(log_weights - shift), nparticles, self._rng)
            particles, log_proposal = particles[index], log_proposal[index]
            delta, derived = delta[index], derived[index]
            unique_fraction = len(np.unique(index)) / nparticles

            step_start = time.time()
            state = (particles, log_proposal, delta, derived)
            nsteps, mean_moves, acceptance, nevaluations = self._rejuvenate(beta, state, delayed)

            entry = dict(temperature=temperature, beta=float(beta), dbeta=float(dbeta),
                         ess_fraction=ess / nparticles, unique_fraction=unique_fraction,
                         nsteps=nsteps, mean_moves=mean_moves, acceptance=acceptance,
                         gamma_scale=self._gamma_scale, nevaluations=nevaluations,
                         log_evidence_increment=log_evidence_increment,
                         delta_std=float(np.std(delta)), seconds=time.time() - step_start)
            self.history.append(entry)
            self.logger.info('[%3d] beta %.5f (+%.5f)  ESS %.3f  unique %.3f  steps %d  moves %.2f  '
                             'accept %.3f  gamma x%.2f  evaluations %d  sigma(dlogP) %.3f  %.1f s',
                             temperature, beta, dbeta, ess / nparticles, unique_fraction, nsteps,
                             mean_moves, acceptance, self._gamma_scale, nevaluations,
                             entry['delta_std'], entry['seconds'])
            if beta >= 1.:
                break
        else:
            self.logger.warning('beta = %.5f after %d temperatures; stopping unconverged.',
                                beta, self.max_temperatures)

        log_evidence = float(np.sum([entry['log_evidence_increment'] for entry in self.history]))
        self.logger.info('Finished sampling: %d temperatures, %d posterior evaluations, %.1f s, '
                         'log(Z_posterior / Z_proposal) = %.3f', len(self.history),
                         self._nevaluations, time.time() - run_start, log_evidence)

        self._pool.stop_wait()
        extras = dict(aweight=np.ones(nparticles), logposterior=log_proposal + delta,
                      attrs=dict(logevidence=log_evidence, beta=float(beta),
                                 nevaluations=self._nevaluations, history=list(self.history)))
        return particles, derived, extras
