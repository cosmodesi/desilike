"""Conditioning transforms for sampler and profiler parameter spaces.

A conditioner maps the space a kernel moves in (``z``) to the original parameter space (``x``)
in two stages::

    z  --affine-->  y  --transforms-->  x

``transforms`` are bijections that remove bounds or reparameterize groups of parameters
(:class:`Logit`, :class:`Log`, :class:`Reparameterization`); the affine stage centres, scales
and optionally whitens in the unconstrained space ``y``. With no transforms the conditioner is
the affine map alone, as it always was.

Kernels see ``z``. Samplers add :meth:`AffineConditioner.log_abs_det_jacobian` to densities
evaluated in ``z``, so the target stays the posterior in ``x``; profilers do not (the best fit
is then independent of the transform) and map covariances back with :meth:`AffineConditioner.jacobian`.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .parameter import _cumsize_params, _flat_to_dict, _dict_to_flat


class Transform:
    """Bijection ``y -> x`` acting on a group of parameters.

    Parameters
    ----------
    params : str or list of str
        Names of the parameters the transform acts on.
    """

    #: ``True`` when each output coordinate depends on its own input coordinate only.
    elementwise = True

    def __init__(self, params):
        self.params = [params] if isinstance(params, str) else list(params)

    def init(self, varied_params, param_slices):
        """Resolve flat indices (and anything else read from the parameters). Called by the conditioner."""
        missing = [name for name in self.params if name not in param_slices]
        if missing:
            raise ValueError(f'{type(self).__name__}: parameters {missing} are not varied')
        self.indices = np.concatenate([np.arange(param_slices[name].start, param_slices[name].stop) for name in self.params])

    def forward(self, y):
        """Unconstrained ``y`` (..., n) → original ``x`` (..., n)."""
        raise NotImplementedError

    def inverse(self, x):
        """Original ``x`` (..., n) → unconstrained ``y`` (..., n)."""
        raise NotImplementedError

    def log_abs_det_jacobian(self, y):
        """``log|det dx/dy|`` (...,)."""
        raise NotImplementedError

    def bounds(self, low, high):
        """Map original-space bounds (n,) to unconstrained-space bounds (n,)."""
        return np.full_like(low, -np.inf), np.full_like(high, np.inf)


def _prior_limits(varied_params, names, param_slices, size):
    """Flat ``(low, high)`` arrays of the prior limits of *names*."""
    low, high = np.full(size, -np.inf), np.full(size, np.inf)
    offset = 0
    for name in names:
        param = varied_params[name]
        slc = param_slices[name]
        nsize = slc.stop - slc.start
        if param.prior is not None:
            low[offset:offset + nsize], high[offset:offset + nsize] = param.prior.limits
        offset += nsize
    return low, high


class Logit(Transform):
    """``x = low + (high - low) * sigmoid(y)``: maps the real line onto ``(low, high)``.

    Parameters
    ----------
    params : str or list of str
        Parameter names.
    limits : tuple, dict, or None
        ``(low, high)`` for all *params*, ``{name: (low, high)}``, or ``None`` to use each
        parameter's prior limits (which must then be finite).
    """

    def __init__(self, params, limits=None):
        super().__init__(params)
        self.limits = limits

    def init(self, varied_params, param_slices):
        super().init(varied_params, param_slices)
        size = self.indices.size
        low, high = _prior_limits(varied_params, self.params, param_slices, size)
        if self.limits is not None:
            offset = 0
            for name in self.params:
                slc = param_slices[name]
                nsize = slc.stop - slc.start
                limits = self.limits[name] if isinstance(self.limits, dict) else self.limits
                low[offset:offset + nsize], high[offset:offset + nsize] = limits
                offset += nsize
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError(f'Logit on {self.params} needs finite limits, got low={low}, high={high}')
        self._low, self._width = low, high - low

    def forward(self, y):
        return self._low + self._width * jax.nn.sigmoid(y)

    def inverse(self, x):
        unit = (jnp.asarray(x) - self._low) / self._width
        return jnp.log(unit) - jnp.log1p(-unit)

    def log_abs_det_jacobian(self, y):
        return jnp.sum(jnp.log(self._width) + jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y), axis=-1)


class Log(Transform):
    """``x = low + exp(y)`` (or ``x = high - exp(y)``): maps the real line onto a half-line.

    Parameters
    ----------
    params : str or list of str
        Parameter names.
    side : {'low', 'high'}
        Which limit is the finite one.
    limit : float or None
        The finite limit; ``None`` takes it from each parameter's prior.
    """

    def __init__(self, params, side='low', limit=None):
        super().__init__(params)
        if side not in ('low', 'high'):
            raise ValueError(f"side must be 'low' or 'high', not {side!r}")
        self.side, self.limit = side, limit

    def init(self, varied_params, param_slices):
        super().init(varied_params, param_slices)
        low, high = _prior_limits(varied_params, self.params, param_slices, self.indices.size)
        edge = low if self.side == 'low' else high
        if self.limit is not None:
            edge = np.full(self.indices.size, float(self.limit))
        if not np.all(np.isfinite(edge)):
            raise ValueError(f'Log on {self.params} needs a finite {self.side} limit, got {edge}')
        self._edge, self._sign = edge, (1. if self.side == 'low' else -1.)

    def forward(self, y):
        return self._edge + self._sign * jnp.exp(y)

    def inverse(self, x):
        return jnp.log(self._sign * (jnp.asarray(x) - self._edge))

    def log_abs_det_jacobian(self, y):
        return jnp.sum(y, axis=-1)


class Reparameterization(Transform):
    """User-defined bijection on a group of parameters, e.g. ``(w0, wa) -> (w0, log(-0.5 - w0 - wa))``.

    Parameters
    ----------
    params : list of str
        Parameter names, in the order of the last axis of the arrays *forward* / *inverse* take.
    forward : callable
        Unconstrained ``y`` (..., n) → original ``x`` (..., n); JAX-traceable.
    inverse : callable
        Original ``x`` (..., n) → unconstrained ``y`` (..., n).
    log_abs_det_jacobian : callable, optional
        ``y`` (..., n) → ``log|det dx/dy|`` (...,). Default: computed with ``jax.jacfwd``.
    """

    elementwise = False

    def __init__(self, params, forward, inverse, log_abs_det_jacobian=None):
        super().__init__(params)
        self._forward, self._inverse, self._logdet = forward, inverse, log_abs_det_jacobian

    def forward(self, y):
        return self._forward(y)

    def inverse(self, x):
        return self._inverse(x)

    def log_abs_det_jacobian(self, y):
        if self._logdet is not None:
            return self._logdet(y)
        y = jnp.asarray(y)

        def one(point):
            return jnp.linalg.slogdet(jax.jacfwd(self._forward)(point))[1]

        flat = y.reshape((-1, y.shape[-1]))
        return jax.vmap(one)(flat).reshape(y.shape[:-1])


class AffineConditioner:
    """Conditioning transform: optional bijections, then centre-and-scale with optional Cholesky whitening.

    Parameters
    ----------
    covariance : Covariance, array_like, or None
        Covariance matrix used to set the scale, in original parameter space.  When ``None``
        and *rescale* is ``True`` or ``'diag'``, each parameter's ``ref.std()`` is used.
    rescale : bool or {'diag', 'full'}
        ``False`` (default): no scaling or centering.
        ``True`` or ``'diag'``: diagonal scaling — from *covariance* diagonal when
        given, from each parameter's ``ref.std()`` otherwise.
        ``'full'``: Cholesky whitening from *covariance* when it is non-diagonal;
        falls back to diagonal scaling when it is diagonal.
        With *transforms*, scales are carried to the unconstrained space through the
        transforms' Jacobian at the centre.
    transforms : list of Transform, optional
        Bijections from the unconstrained space to the original parameters.  A parameter
        may appear in one transform only.
    bounded : {'logit'} or None
        ``'logit'``: add :class:`Logit` for every varied parameter whose prior has two finite
        limits and :class:`Log` for those with one, unless *transforms* already covers it.
    """

    def __init__(self, covariance=None, rescale=False, transforms=None, bounded=None):
        self.covariance = covariance
        self.rescale = rescale
        self.transforms = list(transforms or [])
        if bounded not in (None, 'logit'):
            raise ValueError(f"bounded must be None or 'logit', not {bounded!r}")
        self.bounded = bounded

    # ── setup ────────────────────────────────────────────────────────────────

    def init(self, varied_params):
        """Configure from the varied parameter collection.  Called once by the sampler/profiler."""
        self._varied_params = varied_params
        cumsize = _cumsize_params(varied_params)
        param_slices = {param.name: slice(cumsize[i], cumsize[i + 1]) for i, param in enumerate(varied_params)}
        flat_size = int(cumsize[-1]) if len(cumsize) else 0

        transforms = list(self.transforms)
        covered = [name for transform in transforms for name in transform.params]
        duplicates = sorted({name for name in covered if covered.count(name) > 1})
        if duplicates:
            raise ValueError(f'parameters {duplicates} appear in more than one transform')
        if self.bounded == 'logit':
            for param in varied_params:
                if param.name in covered or param.prior is None:
                    continue
                low, high = param.prior.limits
                if np.isfinite(low) and np.isfinite(high):
                    transforms.append(Logit(param.name))
                elif np.isfinite(low):
                    transforms.append(Log(param.name, side='low'))
                elif np.isfinite(high):
                    transforms.append(Log(param.name, side='high'))
        for transform in transforms:
            transform.init(varied_params, param_slices)
        self._transforms = transforms
        self._flat_size = flat_size

        center_x = []
        for param in varied_params:
            center = np.asarray(param.value if param.value is not None else param.ref.center()).ravel()
            if center.size == 1 and param.size > 1:
                center = np.full(param.size, float(center[0]))
            center_x.append(center.astype('f8'))
        center_x = np.concatenate(center_x) if center_x else np.array([], dtype='f8')
        self._loc = np.asarray(self._transforms_inverse(center_x), dtype='f8')
        if self._transforms and not np.all(np.isfinite(self._loc)):
            bad = [param.name for param in varied_params if not np.all(np.isfinite(self._loc[param_slices[param.name]]))]
            raise ValueError(f'the centre of {bad} is on or outside their transform bounds; set a value or ref inside them')
        # dy/dx at the centre carries original-space scales to the unconstrained space
        inv_jac = np.asarray(jax.jacfwd(self._transforms_inverse)(jnp.asarray(center_x))) if (self._transforms and flat_size) else None

        self._L = self._L_inv = None
        if self.rescale:
            C_full = None
            if hasattr(self.covariance, 'select') and hasattr(self.covariance, 'value'):
                # Covariance object (desilike.samples.Covariance)
                C_full = np.zeros((flat_size, flat_size), dtype='f8')
                in_cov_indices, params_in_cov = [], []
                for i, param in enumerate(varied_params):
                    if param.name in self.covariance:
                        in_cov_indices.extend(range(cumsize[i], cumsize[i + 1]))
                        params_in_cov.append(param)
                if params_in_cov:
                    sub = self.covariance.select(params_in_cov).value
                    ix = np.ix_(in_cov_indices, in_cov_indices)
                    C_full[ix] = sub
                for i, param in enumerate(varied_params):
                    if param.name not in self.covariance:
                        std = param.ref.std()
                        if std is None or not np.isfinite(std) or std <= 0.:
                            raise ValueError(
                                f'Parameter {param.name!r}: cannot determine scale from '
                                f'ref.std()={std!r}.')
                        for k in range(cumsize[i], cumsize[i + 1]):
                            C_full[k, k] = float(std) ** 2
            elif self.covariance is not None:
                C_full = np.asarray(self.covariance)
            if C_full is None:
                stds = []
                for param in varied_params:
                    std = param.ref.std()
                    if std is None or not np.isfinite(std) or std <= 0.:
                        raise ValueError(
                            f'Parameter {param.name!r}: cannot determine scale from '
                            f'ref.std()={std!r}.')
                    stds.append(np.full(param.size, float(std), dtype='f8'))
                C_full = np.diag(np.concatenate(stds) ** 2) if stds else np.zeros((0, 0))
                diagonal_only = True
            else:
                diagonal_only = False
            if inv_jac is not None:
                C_full = inv_jac @ C_full @ inv_jac.T
            self._scale = np.sqrt(np.diag(C_full))
            if not diagonal_only and self.rescale != 'diag' and np.any(C_full != np.diag(np.diag(C_full))):
                _L = np.linalg.cholesky(C_full)
                self._L = jnp.array(_L)
                self._L_inv = jnp.array(np.linalg.inv(_L))
        else:
            self._scale = np.ones(flat_size, dtype='f8')
        self._index_to_transform = {}
        for transform_idx, transform in enumerate(self._transforms):
            for position, flat_idx in enumerate(transform.indices):
                self._index_to_transform[int(flat_idx)] = (transform_idx, position)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def is_mixing(self):
        """``True`` when the transform mixes parameter dimensions (Cholesky whitening or a group reparameterization)."""
        return self._L is not None or any(not transform.elementwise for transform in self._transforms)

    @property
    def is_linear(self):
        """``True`` when there are no (nonlinear) transforms: the conditioner is affine."""
        return not self._transforms

    # ── transforms stage ─────────────────────────────────────────────────────

    def _transforms_forward(self, y):
        x = jnp.asarray(y)
        for transform in self._transforms:
            x = x.at[..., transform.indices].set(transform.forward(x[..., transform.indices]))
        return x

    def _transforms_inverse(self, x):
        y = jnp.asarray(x)
        for transform in self._transforms[::-1]:
            y = y.at[..., transform.indices].set(transform.inverse(y[..., transform.indices]))
        return y

    def _affine_forward(self, z):
        if self._L is not None:
            return jnp.asarray(z) @ self._L.T + self._loc
        return jnp.asarray(z) * self._scale + self._loc

    def _affine_inverse(self, y):
        if self._L is not None:
            return (jnp.asarray(y) - self._loc) @ self._L_inv.T
        return (jnp.asarray(y) - self._loc) / self._scale

    # ── public maps ──────────────────────────────────────────────────────────

    def forward(self, x):
        """Conditioned → original space.  Accepts and returns a flat array or a ``{name: value}`` dict.

        JAX-traceable for array input.  Broadcasts over leading axes.
        """
        if isinstance(x, dict):
            return _flat_to_dict(self.forward(_dict_to_flat(x, self._varied_params)), self._varied_params)
        y = self._affine_forward(x)
        return self._transforms_forward(y) if self._transforms else y

    def inverse(self, x):
        """Original → conditioned space.  Accepts and returns a flat array or a ``{name: value}`` dict.

        JAX-traceable for array input.  Broadcasts over leading axes.
        """
        if isinstance(x, dict):
            return _flat_to_dict(self.inverse(_dict_to_flat(x, self._varied_params)), self._varied_params)
        y = self._transforms_inverse(x) if self._transforms else jnp.asarray(x)
        return self._affine_inverse(y)

    def log_abs_det_jacobian(self, z):
        """``log|det dx/dz|`` of the transforms stage, at conditioned points *z* (...,).

        The affine stage's constant is left out, as it always has been: it does not change
        any sampler's target, only the normalisation.  Zero without transforms.
        """
        z = jnp.asarray(z)
        if not self._transforms:
            return jnp.zeros(z.shape[:-1])
        y = self._affine_forward(z)
        return sum(transform.log_abs_det_jacobian(y[..., transform.indices]) for transform in self._transforms)

    def jacobian(self, z):
        """``dx/dz`` at a single conditioned point *z* (ndim,) → (ndim, ndim)."""
        z = jnp.asarray(z, dtype='f8')
        if not self._transforms:
            if self._L is not None:
                return np.asarray(self._L)
            return np.diag(self._scale)
        return np.asarray(jax.jacfwd(self.forward)(z))

    def forward_coordinate(self, values, flat_idx):
        """Map conditioned values of one flat coordinate to original values (non-mixing conditioners only)."""
        if self.is_mixing:
            raise ValueError('forward_coordinate is not defined for a mixing conditioner')
        y = np.asarray(values) * self._scale[flat_idx] + self._loc[flat_idx]
        return self._coordinate_transform(y, flat_idx, 'forward')

    def inverse_coordinate(self, values, flat_idx):
        """Map original values of one flat coordinate to conditioned values (non-mixing conditioners only)."""
        if self.is_mixing:
            raise ValueError('inverse_coordinate is not defined for a mixing conditioner')
        y = self._coordinate_transform(np.asarray(values, dtype='f8'), flat_idx, 'inverse')
        return (y - self._loc[flat_idx]) / self._scale[flat_idx]

    def _coordinate_transform(self, values, flat_idx, direction):
        entry = self._index_to_transform.get(int(flat_idx))
        if entry is None:
            return values
        transform_idx, position = entry
        transform = self._transforms[transform_idx]
        # evaluate the elementwise transform with every other coordinate of its group at the centre
        base = np.asarray(self._loc[transform.indices] if direction == 'forward' else self._transforms_forward(self._loc)[transform.indices])
        values = np.asarray(values, dtype='f8')
        full = np.broadcast_to(base, values.shape + base.shape).copy()
        full[..., position] = values
        mapped = transform.forward(full) if direction == 'forward' else transform.inverse(full)
        return np.asarray(mapped)[..., position]

    def covariance_to_original(self, cov, z):
        """Map a conditioned-space covariance at *z* to original space: ``J C Jᵀ``."""
        jac = self.jacobian(z)
        return jac @ np.asarray(cov) @ jac.T

    def prior_bounds(self):
        """Return ``(ndim, 2)`` lower/upper prior bounds in conditioned space."""
        lo_orig = np.full(self._flat_size, -np.inf)
        hi_orig = np.full(self._flat_size, np.inf)
        cumsize = _cumsize_params(self._varied_params)
        for i, param in enumerate(self._varied_params):
            if param.prior is not None:
                lo_orig[cumsize[i]:cumsize[i + 1]], hi_orig[cumsize[i]:cumsize[i + 1]] = param.prior.limits
        for transform in self._transforms:
            lo_orig[transform.indices], hi_orig[transform.indices] = transform.bounds(lo_orig[transform.indices], hi_orig[transform.indices])
        if self._L_inv is None:
            return np.column_stack([(lo_orig - self._loc) / self._scale,
                                    (hi_orig - self._loc) / self._scale])
        delta_lo = lo_orig - self._loc
        delta_hi = hi_orig - self._loc
        B_pos = np.maximum(self._L_inv, 0.)
        B_neg = np.minimum(self._L_inv, 0.)
        lo = (np.where(B_pos == 0., 0., B_pos * delta_lo) + np.where(B_neg == 0., 0., B_neg * delta_hi)).sum(axis=-1)
        hi = (np.where(B_pos == 0., 0., B_pos * delta_hi) + np.where(B_neg == 0., 0., B_neg * delta_lo)).sum(axis=-1)
        return np.column_stack([lo, hi])


#: The conditioner is no longer affine-only; this is its general name.
Conditioner = AffineConditioner
