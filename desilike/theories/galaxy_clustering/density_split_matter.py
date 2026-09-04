"""Focused, JAX-native density-split--matter power-spectrum model."""

import numpy as np
import jax.numpy as jnp

from ...base import Calculator
from ...parameter import Parameter, VariableCollection
from ..primordial_cosmology import CosmoprimoCosmology


_QUANTILES = (1, 2, 3, 4, 5)


def _normalize_quantiles(quantiles):
    quantiles = tuple(int(quantile) for quantile in quantiles)
    if not quantiles or len(set(quantiles)) != len(quantiles):
        raise ValueError('quantiles must be non-empty and unique')
    if any(quantile not in _QUANTILES for quantile in quantiles):
        raise ValueError(f'quantiles must be drawn from {_QUANTILES}')
    return quantiles


def _gaussian_quantile_coefficients(nquantiles=5):
    """Return mean standardized-Gaussian density in each equal-probability bin."""
    from statistics import NormalDist

    normal = NormalDist()
    edges = [-np.inf] + [normal.inv_cdf(index / nquantiles)
                         for index in range(1, nquantiles)] + [np.inf]
    norm = np.sqrt(2. * np.pi)
    coefficients = {}
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        phi_low = 0. if not np.isfinite(low) else np.exp(-0.5 * low**2) / norm
        phi_high = 0. if not np.isfinite(high) else np.exp(-0.5 * high**2) / norm
        coefficients[index] = {'c1': nquantiles * (phi_low - phi_high)}
    return coefficients


class DensitySplitMatterPowerSpectrumMultipoles(Calculator):
    r"""Linear density-split--matter spectrum with exact quantile responses.

    The cosmology provider may be :class:`CosmoprimoCosmology`, an
    :class:`~desilike.theories.primordial_cosmology.ACECosmology` using MAPSE,
    or any compatible provider.  Alternatively, ``matter`` may supply precomputed
    Kaiser multipoles.  The output has shape ``(nquantiles, nells, nk)`` and is
    available as both ``power`` and ``poles``.
    """

    def __init__(self, k=None, z=0., ells=(0, 2, 4), quantiles=_QUANTILES,
                 smoothing_radius=10., rsd=True, cosmo=None, matter=None,
                 engine='class', fiducial='DESI', params=None):
        self.quantiles = _normalize_quantiles(quantiles)
        coefficients = _gaussian_quantile_coefficients()
        defaults = VariableCollection([
            Parameter(f'c1q{quantile}', value=coefficients[quantile]['c1'],
                      prior=dict(dist='norm', loc=coefficients[quantile]['c1'], scale=2.),
                      ref=dict(dist='norm', loc=coefficients[quantile]['c1'], scale=0.25),
                      fixed=False, latex=rf'c_{{1,{quantile}}}')
            for quantile in self.quantiles
        ])
        if params is not None:
            defaults = defaults + VariableCollection(params)
        self.response_params = []
        for param in defaults:
            setattr(self, param.basename, param)
            self.response_params.append(param)

        if matter is not None and cosmo is not None:
            raise ValueError('provide either matter or cosmo, not both')
        self.matter = matter
        self.cosmo = (CosmoprimoCosmology(engine=engine, fiducial=fiducial)
                      if matter is None and cosmo is None else cosmo)

        if k is None:
            k = getattr(matter, 'k', None)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)
        self.ells = tuple(int(ell) for ell in ells)
        self.smoothing_radius = float(smoothing_radius)
        self.rsd = bool(rsd)

    def __post_init__(self, **kwargs):
        if (self.k.ndim != 1 or not self.k.size or not np.isfinite(self.k).all()
                or np.any(self.k <= 0.) or np.any(np.diff(self.k) <= 0.)):
            raise ValueError('k must be a non-empty, finite, positive, strictly increasing one-dimensional array')
        if not np.isfinite(self.z) or self.z < 0.:
            raise ValueError('z must be finite and non-negative')
        if not self.ells or len(set(self.ells)) != len(self.ells):
            raise ValueError('ells must be non-empty and unique')
        if any(ell not in (0, 2, 4) for ell in self.ells):
            raise ValueError('ells must be drawn from (0, 2, 4)')
        if not np.isfinite(self.smoothing_radius) or self.smoothing_radius < 0.:
            raise ValueError('smoothing_radius must be finite and non-negative')
        if not self.rsd and self.ells != (0,):
            raise ValueError('real-space density-split matter power supports only ell=0')

        if self.matter is not None:
            if (not np.array_equal(self.k, self.matter.k)
                    or self.ells != tuple(self.matter.ells)
                    or self.z != float(self.matter.z)
                    or self.rsd != bool(self.matter.rsd)):
                raise ValueError('matter kernel grid, multipoles, redshift and RSD must match')
        else:
            self.cosmo.add_requirements({
                'fourier.pk': [{'of': 'delta_cb', 'z': self.z, 'k': self.k}],
                'fourier.sigma8_z': [
                    {'of': 'delta_cb', 'z': self.z},
                    {'of': 'theta_cb', 'z': self.z},
                ],
            })

    def __call__(self):
        if self.matter is not None:
            matter_power = jnp.asarray(self.matter.power)
            shape = (len(self.ells), self.k.size)
            if matter_power.shape not in (shape, (shape[0] * shape[1],)):
                raise ValueError('matter power must have shape (nells, nk) or be flattened in ell-major order')
            matter_power = matter_power.reshape(shape)
        else:
            fourier = self.cosmo.get_fourier()
            linear_power = fourier.pk(of='delta_cb', z=self.z, k=self.k)
            if self.rsd:
                sigma8 = fourier.sigma8_z(of='delta_cb', z=self.z)
                fsigma8 = fourier.sigma8_z(of='theta_cb', z=self.z)
                f = fsigma8 / sigma8
                factors = {0: 1. + 2. * f / 3. + f**2 / 5.,
                           2: 4. * f / 3. + 4. * f**2 / 7.,
                           4: 8. * f**2 / 35.}
                matter_power = jnp.stack([factors[ell] * linear_power for ell in self.ells])
            else:
                matter_power = linear_power[None, :]
        window = jnp.exp(-0.5 * (jnp.asarray(self.k) * self.smoothing_radius)**2)
        responses = jnp.stack([param.value for param in self.response_params])
        self.power = responses[:, None, None] * matter_power[None, :, :] * window[None, None, :]
        self.poles = self.power
        return self.power

    def tree_flatten(self):
        return [self.power], {'k': self.k, 'z': self.z, 'ells': self.ells,
                              'quantiles': self.quantiles, 'rsd': self.rsd,
                              'smoothing_radius': self.smoothing_radius}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.power = obj.poles = children[0]
        for name, value in aux.items():
            setattr(obj, name, value)
        return obj
