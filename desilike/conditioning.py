"""Conditioning transforms for sampler and profiler parameter spaces."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .parameter import _cumsize_params


class AffineConditioner:
    """Affine conditioning transform: centre-and-scale, optionally with Cholesky whitening.

    Parameters
    ----------
    covariance : Covariance, array_like, or None
        Covariance matrix used to set the scale.  When ``None`` and *rescale* is
        ``True`` or ``'diag'``, each parameter's ``ref.std()`` is used.
    rescale : bool or {'diag', 'full'}
        ``False`` (default): identity transform (scale = 1, no centering).
        ``True`` or ``'diag'``: diagonal scaling — from *covariance* diagonal when
        given, from each parameter's ``ref.std()`` otherwise.
        ``'full'``: Cholesky whitening from *covariance* when it is non-diagonal;
        falls back to diagonal scaling when it is diagonal.
    """

    def __init__(self, covariance=None, rescale=False):
        self.covariance = covariance
        self.rescale = rescale

    def init(self, varied_params):
        """Configure from the varied parameter collection.  Called once by the sampler/profiler."""
        self._varied_params = varied_params

        loc_parts = []
        for param in varied_params:
            center = np.asarray(
                param.value if param.value is not None else param.ref.center()).ravel()
            if center.size == 1 and param.size > 1:
                center = np.full(param.size, float(center[0]))
            loc_parts.append(center.astype('f8'))
        self._loc = np.concatenate(loc_parts) if loc_parts else np.array([], dtype='f8')

        flat_size = self._loc.size
        self._L = self._L_inv = None

        if self.rescale:
            C_full = None
            if hasattr(self.covariance, 'select') and hasattr(self.covariance, 'value'):
                # Covariance object (desilike.samples.Covariance)
                C_full = np.zeros((flat_size, flat_size), dtype='f8')
                cumsize = _cumsize_params(varied_params)
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

            if C_full is not None:
                self._scale = np.sqrt(np.diag(C_full))
                if self.rescale != 'diag' and np.any(C_full != np.diag(np.diag(C_full))):
                    _L = np.linalg.cholesky(C_full)
                    self._L = jnp.array(_L)
                    self._L_inv = jnp.array(np.linalg.inv(_L))
            else:
                scale_parts = []
                for param in varied_params:
                    std = param.ref.std()
                    if std is None or not np.isfinite(std) or std <= 0.:
                        raise ValueError(
                            f'Parameter {param.name!r}: cannot determine scale from '
                            f'ref.std()={std!r}.')
                    scale_parts.append(np.full(param.size, float(std), dtype='f8'))
                self._scale = np.concatenate(scale_parts) if scale_parts else np.array([], dtype='f8')
        else:
            self._scale = np.ones(flat_size, dtype='f8')

    @property
    def is_mixing(self):
        """``True`` when the transform mixes parameter dimensions (Cholesky whitening)."""
        return self._L is not None

    def forward(self, x):
        """Conditioned → original space.  Accepts and returns a flat array or a ``{name: value}`` dict.

        JAX-traceable for array input.  Broadcasts over leading axes.
        """
        if isinstance(x, dict):
            return self._forward_dict(x)
        if self._L is not None:
            return jnp.asarray(x) @ self._L.T + self._loc
        return jnp.asarray(x) * self._scale + self._loc

    def inverse(self, x):
        """Original → conditioned space.  Accepts and returns a flat array or a ``{name: value}`` dict.

        JAX-traceable for array input.  Broadcasts over leading axes.
        """
        if isinstance(x, dict):
            return self._inverse_dict(x)
        if self._L is not None:
            return (jnp.asarray(x) - self._loc) @ self._L_inv.T
        return (jnp.asarray(x) - self._loc) / self._scale

    def _forward_dict(self, sample):
        if self._L is not None:
            flat = jnp.concatenate([
                jnp.atleast_1d(jnp.ravel(jnp.asarray(sample[param.name])))
                for param in self._varied_params])
            flat = flat @ self._L.T + self._loc
            result = {}
            cumsize = _cumsize_params(self._varied_params)
            for i, param in enumerate(self._varied_params):
                v = flat[cumsize[i]:cumsize[i + 1]]
                result[param.name] = v.reshape(param.shape) if param.shape else v[0]
            return result
        result = {}
        cumsize = _cumsize_params(self._varied_params)
        for i, param in enumerate(self._varied_params):
            sl = slice(cumsize[i], cumsize[i + 1])
            v = jnp.ravel(jnp.asarray(sample[param.name])) * self._scale[sl] + self._loc[sl]
            result[param.name] = v.reshape(param.shape) if param.shape else v[0]
        return result

    def _inverse_dict(self, sample):
        if self._L is not None:
            flat = jnp.concatenate([
                jnp.atleast_1d(jnp.ravel(jnp.asarray(sample[param.name])))
                for param in self._varied_params])
            flat = (flat - self._loc) @ self._L_inv.T
            result = {}
            cumsize = _cumsize_params(self._varied_params)
            for i, param in enumerate(self._varied_params):
                v = flat[cumsize[i]:cumsize[i + 1]]
                result[param.name] = v.reshape(param.shape) if param.shape else v[0]
            return result
        result = {}
        cumsize = _cumsize_params(self._varied_params)
        for i, param in enumerate(self._varied_params):
            sl = slice(cumsize[i], cumsize[i + 1])
            v = (jnp.ravel(jnp.asarray(sample[param.name])) - self._loc[sl]) / self._scale[sl]
            result[param.name] = v.reshape(param.shape) if param.shape else v[0]
        return result

    def prior_bounds(self):
        """Return ``(ndim, 2)`` lower/upper prior bounds in conditioned space."""
        lo_orig = np.full(self._loc.size, -np.inf)
        hi_orig = np.full(self._loc.size, np.inf)
        cumsize = _cumsize_params(self._varied_params)
        for i, param in enumerate(self._varied_params):
            if param.prior is not None:
                lo_orig[cumsize[i]:cumsize[i + 1]], hi_orig[cumsize[i]:cumsize[i + 1]] = param.prior.limits
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
