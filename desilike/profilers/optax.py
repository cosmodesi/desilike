"""Optax profiler kernel — fully JAX-native gradient-descent optimisation.

The inner optimisation loop uses ``jax.lax.while_loop`` so the entire
run (including early-stopping) compiles to a single XLA programme.
"""

from __future__ import annotations

import logging

import numpy as np
import jax
import jax.numpy as jnp

from .base import Kernel, ProfilerState


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


class Optax(Kernel):
    """Fully JAX-native optimisation kernel using `optax <https://github.com/google-deepmind/optax>`_.

    The inner loop is expressed as ``jax.lax.while_loop`` so the whole
    optimisation compiles to a single XLA programme, enabling JIT and
    GPU acceleration.

    Gradient is always required.

    Parameters
    ----------
    method : str
        Optax optimiser name, e.g. ``'adam'``, ``'adamw'``, ``'bfgs'``.
        Any attribute of the ``optax`` module that returns a
        :class:`optax.GradientTransformation` is accepted.
    """

    logger = logging.getLogger('Optax')

    with_gradient = True

    def __init__(self, method='adam'):
        self.method = method

    @classmethod
    def install(cls, installer):
        installer.pip('optax')

    def init(self):
        try:
            import optax  # noqa: F401
        except ImportError:
            raise ImportError("'optax' is required but not installed. Run: pip install optax")

    def run(
        self,
        state: ProfilerState,
        chi2,
        grad=None,
        max_iterations: int = 10_000,
        learning_rate: float = 1e-2,
        learning_rate_scheduling: bool = True,
        patience: int = 200,
        xtol: float = 1e-4,
        **kwargs,
    ) -> ProfilerState:
        import optax

        if grad is None:
            raise RuntimeError(
                'Optax requires a JAX-differentiable likelihood (grad=None was passed).'
            )

        x0 = jnp.asarray(state.start, dtype=jnp.float64)

        if learning_rate_scheduling:
            lr_fn = _make_lr_schedule(learning_rate, max_iterations)
        else:
            lr_fn = learning_rate
        tx = getattr(optax, self.method)(lr_fn)

        # carry = (params, opt_state, best_params, best_loss,
        #          patience_counter, epoch, prev_test_loss)
        init_opt_state = tx.init(x0)
        INF = jnp.asarray(jnp.inf, dtype=jnp.float64)

        @jax.jit
        def body(carry):
            params, opt_state, best_params, best_loss, patience_ctr, epoch, test_loss = carry
            loss, grads = jax.value_and_grad(chi2)(params)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            improved = loss < best_loss
            best_params  = jnp.where(improved, params,    best_params)
            best_loss    = jnp.where(improved, loss,      best_loss)
            big_step     = jnp.abs(loss - test_loss) > xtol
            patience_ctr = jnp.where(improved & big_step, 0,         patience_ctr + 1)
            test_loss    = jnp.where(improved & big_step, best_loss,  test_loss)
            return params, opt_state, best_params, best_loss, patience_ctr, epoch + 1, test_loss

        def cond(carry):
            _, _, _, _, patience_ctr, epoch, _ = carry
            return (patience_ctr < patience) & (epoch < max_iterations)

        init_carry = (x0, init_opt_state, x0, INF, jnp.int32(0), jnp.int32(0), INF)
        _, _, best_params, best_loss, _, _, _ = jax.lax.while_loop(cond, body, init_carry)

        state.logpdf = np.array(-0.5 * best_loss)
        state.best = np.asarray(best_params)
        return state
