"""OptaxProfiler — fully JAX-native gradient-descent profiler.

The inner optimisation loop uses ``jax.lax.while_loop`` so the entire
run (including early-stopping) compiles to a single XLA programme and
benefits from JIT, vmap, and GPU/TPU acceleration.
"""

from __future__ import annotations

import logging

import numpy as np
import jax
import jax.numpy as jnp

from ..samples import Profiles
from .base import BaseProfiler, ProfilerState, _build_best_from_x


def _make_lr_schedule(base_lr: float, num_steps: int):
    """Cosine decay with a 10 % linear warm-up."""
    import optax
    warmup_steps = max(1, int(0.1 * num_steps))
    cosine_steps = max(1, num_steps - warmup_steps)
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0., base_lr, warmup_steps),
            optax.cosine_decay_schedule(base_lr, cosine_steps),
        ],
        boundaries=[warmup_steps],
    )


class OptaxProfiler(BaseProfiler):
    """Fully JAX-native profiler using `optax <https://github.com/google-deepmind/optax>`_.

    The inner loop is expressed as ``jax.lax.while_loop`` so the whole
    optimisation compiles to a single XLA programme.  This enables JIT,
    GPU acceleration, and (in principle) vmap over independent starts.

    Gradient is always required — ``_probe_jax`` must succeed.  If JAX is
    unavailable or jit fails the profiler raises at construction time
    (during ``_ensure_jax_probed``).

    Parameters
    ----------
    method : str
        Optax optimiser name, e.g. ``'adam'``, ``'adamw'``, ``'bfgs'``.
        Any attribute of the ``optax`` module that returns an
        :class:`optax.GradientTransformation` is accepted.
    *args, **kwargs
        Forwarded to :class:`BaseProfiler`.
    """

    logger = logging.getLogger('OptaxProfiler')

    name = 'optax'
    with_gradient = True

    def __init__(self, *args, method='adam', **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    @classmethod
    def install(cls, installer):
        installer.pip('optax')

    def _maximize_one(
        self,
        state: ProfilerState,
        max_iterations: int = 10_000,
        learning_rate: float = 1e-2,
        learning_rate_scheduling: bool = True,
        patience: int = 200,
        xtol: float = 1e-4,
        **kwargs,
    ):
        import optax

        chi2_fn = state.chi2_fn
        x0 = jnp.asarray(state.start, dtype=jnp.float64)

        if self._grad_chi2 is None:
            raise RuntimeError(
                'OptaxProfiler requires a JAX-differentiable likelihood. '
                'JAX grad probe failed during initialisation.'
            )

        # ── build optimiser ───────────────────────────────────────────────
        if learning_rate_scheduling:
            lr_fn = _make_lr_schedule(learning_rate, max_iterations)
        else:
            lr_fn = learning_rate
        tx = getattr(optax, self.method)(lr_fn)

        # ── while_loop body & condition ───────────────────────────────────
        # carry = (params, opt_state, best_params, best_loss,
        #          patience_counter, epoch, prev_test_loss)
        init_opt_state = tx.init(x0)
        INF = jnp.asarray(jnp.inf, dtype=jnp.float64)

        @jax.jit
        def body(carry):
            params, opt_state, best_params, best_loss, patience_ctr, epoch, test_loss = carry
            loss, grads = jax.value_and_grad(chi2_fn)(params)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            improved = loss < best_loss
            best_params  = jnp.where(improved, params,    best_params)
            best_loss    = jnp.where(improved, loss,      best_loss)
            # patience resets when the loss improves *and* the absolute
            # change relative to the checkpoint exceeds xtol
            big_step = jnp.abs(loss - test_loss) > xtol
            patience_ctr = jnp.where(improved & big_step, 0,       patience_ctr + 1)
            test_loss    = jnp.where(improved & big_step, best_loss, test_loss)
            return params, opt_state, best_params, best_loss, patience_ctr, epoch + 1, test_loss

        def cond(carry):
            _, _, _, _, patience_ctr, epoch, _ = carry
            return (patience_ctr < patience) & (epoch < max_iterations)

        init_carry = (x0, init_opt_state, x0, INF, jnp.int32(0), jnp.int32(0), INF)
        _, _, best_params, best_loss, _, _, _ = jax.lax.while_loop(cond, body, init_carry)

        best_params = np.asarray(best_params)
        logpost     = float(-0.5 * float(best_loss))

        profiles = Profiles()
        profiles.best = _build_best_from_x(best_params, logpost, state.varied_params)
        return profiles
