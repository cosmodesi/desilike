"""Density-split galaxy clustering theory models."""

import numpy as np

from desilike import BaseCalculator
from desilike.jax import numpy as jnp
from desilike.jax import interp1d

from .base import ProjectToMultipoles
from .full_shape import BasePTPowerSpectrumMultipoles
from .power_template import StandardPowerSpectrumTemplate


_QUANTILES = (1, 2, 3, 4, 5)


def _normalize_quantiles(quantiles):
    if np.ndim(quantiles) == 0:
        quantiles = (quantiles,)
    quantiles = tuple(int(quantile) for quantile in quantiles)
    invalid = [quantile for quantile in quantiles if quantile not in _QUANTILES]
    if invalid:
        raise ValueError('quantiles must be drawn from {}; found {}'.format(_QUANTILES, invalid))
    if len(set(quantiles)) != len(quantiles):
        raise ValueError('quantiles must be unique')
    return quantiles


def _gaussian_smoothing(k, radius):
    return jnp.exp(-0.5 * (k * radius)**2)


class DensitySplitTracerPowerSpectrumMultipoles(BaseCalculator):
    r"""
    Tree-level density-split quantile-galaxy cross-power spectrum multipoles.

    The redshift-space model is

    .. math::

        P_{qg}(k, \mu) = W_R(k) (b_q + \beta_q f \mu^2)
                         (b_1 + f \mu^2) P_\mathrm{L}(k).

    ``k`` is expressed in ``h / Mpc`` and ``smoothing_radius`` in ``Mpc / h``.
    """

    config_fn = 'density_split.yaml'

    def initialize(self, k=None, ells=(0, 2, 4), quantiles=_QUANTILES, template=None, z=None, mu=20, smoothing_radius=10.):
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)
        self.quantiles = _normalize_quantiles(quantiles)
        self.smoothing_radius = float(smoothing_radius)
        if self.smoothing_radius < 0.:
            raise ValueError('smoothing_radius must be non-negative')

        keep = ['b1']
        keep += ['bq{:d}'.format(quantile) for quantile in self.quantiles]
        keep += ['beta{:d}'.format(quantile) for quantile in self.quantiles]
        self.init.params = self.init.params.select(basename=keep)

        if template is None:
            template = StandardPowerSpectrumTemplate()
        BasePTPowerSpectrumMultipoles._set_template(self, template=template, z=z)
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu

    def calculate(self, b1=2., **params):
        self.z = self.template.z
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pk = interp1d(jnp.log10(kap), jnp.log10(self.template.k), self.template.pk_dd, method='cubic')
        window = _gaussian_smoothing(kap, self.smoothing_radius)
        f = self.template.f
        mu2 = muap**2

        power = []
        for quantile in self.quantiles:
            bq = params['bq{:d}'.format(quantile)]
            beta = params['beta{:d}'.format(quantile)]
            pkmu = jac * window * (bq + beta * f * mu2) * (b1 + f * mu2) * pk
            power.append(self.to_poles(pkmu))
        self.power = jnp.stack(power, axis=0)

    def get(self):
        return self.power

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'quantiles', 'smoothing_radius', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
