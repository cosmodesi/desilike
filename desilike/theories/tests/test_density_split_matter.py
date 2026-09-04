import builtins

import numpy as np
import pytest

from desilike import BaseCalculator, PipelineError
from desilike.jax import numpy as jnp
from desilike.theories import Cosmoprimo
from desilike.theories.galaxy_clustering import DensitySplitMatterPowerSpectrumMultipoles
from desilike.theories.galaxy_clustering.density_split import _gaussian_quantile_coefficients


def make_cosmo():
    return Cosmoprimo(fiducial='DESI', massive_neutrino=False)


class MatterKernel(BaseCalculator):
    _params = {'amplitude': {'value': 1., 'fixed': False}}

    def initialize(self, k=(0.02, 0.04), z=0.5, ells=(0, 2, 4), rsd=True,
                   flattened=False, bad_shape=False):
        self.k = np.asarray(k)
        self.z, self.ells, self.rsd = z, tuple(ells), rsd
        self.flattened, self.bad_shape = flattened, bad_shape

    def calculate(self, amplitude=1.):
        self.power = amplitude * jnp.arange(1., len(self.ells) * len(self.k) + 1.).reshape(len(self.ells), -1)
        if self.flattened:
            self.power = self.power.ravel()
        if self.bad_shape:
            self.power = self.power[None, ...]


@pytest.mark.parametrize('flattened', [False, True])
@pytest.mark.parametrize('rsd, ells', [(False, (0,)), (True, (4, 0, 2))])
def test_injected_matter_owns_cosmology_dependency(monkeypatch, flattened, rsd, ells):
    from desilike.theories.galaxy_clustering import density_split

    def forbidden_cosmo(*args, **kwargs):
        raise AssertionError('injected matter must not create a cosmology')

    monkeypatch.setattr(density_split, 'Cosmoprimo', forbidden_cosmo)
    kernel = MatterKernel(ells=ells, rsd=rsd, flattened=flattened)
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        matter=kernel, z=0.5, ells=ells, rsd=rsd, quantiles=(5, 3, 1), smoothing_radius=12.)
    power = np.asarray(theory(amplitude=1.7, c1q5=2., c1q3=0.4, c1q1=-1.))
    expected = (np.array([2., 0.4, -1.])[:, None, None]
                * np.asarray(kernel.power).reshape(len(ells), -1)[None, ...]
                * np.exp(-0.5 * (kernel.k * 12.)**2))
    np.testing.assert_allclose(power, expected, rtol=2e-15)
    np.testing.assert_array_equal(theory.k, kernel.k)
    assert theory.matter is kernel
    assert theory.cosmo is None
    assert set(theory.varied_params.names()) == {'amplitude', 'c1q5', 'c1q3', 'c1q1'}
    assert set(theory.__getstate__()) == {'k', 'z', 'ells', 'quantiles', 'smoothing_radius', 'rsd', 'power'}


@pytest.mark.parametrize('kwargs', [
    {'k': [0.03, 0.04]}, {'z': 1.}, {'ells': (0, 2)},
    {'ells': (0,), 'rsd': False},
])
def test_injected_matter_rejects_metadata_mismatch(kwargs):
    options = dict(z=0.5, matter=MatterKernel())
    options.update(kwargs)
    theory = DensitySplitMatterPowerSpectrumMultipoles(**options)
    with pytest.raises(PipelineError) as exc:
        theory()
    assert 'matter kernel grid, multipoles, redshift and RSD must match' in str(exc.value.__cause__)


def test_injected_matter_rejects_ambiguous_cosmology_and_bad_shape():
    theory = DensitySplitMatterPowerSpectrumMultipoles(matter=MatterKernel(), cosmo=make_cosmo())
    with pytest.raises(PipelineError) as exc:
        theory()
    assert 'provide either matter or cosmo, not both' in str(exc.value.__cause__)
    theory = DensitySplitMatterPowerSpectrumMultipoles(matter=MatterKernel(bad_shape=True), z=0.5)
    with pytest.raises(ValueError, match='matter power must have shape'):
        theory()


def test_matter_dependency_can_replace_default_exact_cosmology():
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        k=[0.02, 0.04], z=0.5, ells=(0,), rsd=False, quantiles=(3,))
    theory(c1q3=1.)
    assert 'h' in theory.all_params.names()
    kernel = MatterKernel(ells=(0,), rsd=False)
    theory.init.update(matter=kernel)
    theory(amplitude=1.2, c1q3=0.7)
    assert theory.cosmo is None
    assert set(theory.varied_params.names()) == {'amplitude', 'c1q3'}
    theory.init.update(matter=None)
    theory(c1q3=1.)
    assert isinstance(theory.cosmo, Cosmoprimo)
    assert 'amplitude' not in theory.all_params.names()


def test_density_split_matter_matches_analytic_kaiser():
    k = np.linspace(0.02, 0.08, 4)
    z = 0.5
    radius = 10.
    quantiles = (1, 2, 3, 4, 5)
    responses = {f'c1q{quantile}': float(quantile) for quantile in quantiles}
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        k=k, z=z, ells=(0, 2, 4), quantiles=quantiles,
        smoothing_radius=radius, cosmo=make_cosmo(),
    )
    power = np.asarray(theory(**responses))

    fourier = theory.cosmo.cosmo.get_fourier()
    linear = fourier.pk_interpolator(of='delta_cb').to_1d(z=z)(k)
    growth_rate = (
        fourier.sigma8_z(z, of='theta_cb')
        / fourier.sigma8_z(z, of='delta_cb')
    )
    factors = np.array([
        1. + 2. * growth_rate / 3. + growth_rate**2 / 5.,
        4. * growth_rate / 3. + 4. * growth_rate**2 / 7.,
        8. * growth_rate**2 / 35.,
    ])
    window = np.exp(-0.5 * (k * radius)**2)
    expected = (
        np.arange(1., 6.)[:, None, None]
        * factors[None, :, None]
        * linear[None, None, :]
        * window[None, None, :]
    )
    assert power.shape == (5, 3, k.size)
    assert np.isfinite(power).all()
    assert np.allclose(power, expected, rtol=2e-13, atol=0.)


def test_density_split_matter_smoothing_and_independent_middle_quantile():
    k = np.linspace(0.02, 0.08, 4)
    kwargs = dict(k=k, z=0.5, ells=(0,), quantiles=(1, 3), rsd=False)
    unsmoothed = DensitySplitMatterPowerSpectrumMultipoles(
        smoothing_radius=0., cosmo=make_cosmo(), **kwargs
    )(c1q1=0., c1q3=2.)
    smoothed = DensitySplitMatterPowerSpectrumMultipoles(
        smoothing_radius=10., cosmo=make_cosmo(), **kwargs
    )(c1q1=0., c1q3=2.)
    unsmoothed, smoothed = np.asarray(unsmoothed), np.asarray(smoothed)

    assert np.all(unsmoothed[0] == 0.)
    assert np.any(unsmoothed[1] != 0.)
    assert np.allclose(
        smoothed[1] / unsmoothed[1],
        np.exp(-0.5 * (k * 10.)**2)[None, :],
        rtol=2e-13,
        atol=0.,
    )


def test_density_split_matter_selected_quantiles_and_parameters():
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        k=[0.02, 0.04], quantiles=(5, 2), ells=(4, 0), cosmo=make_cosmo()
    )
    power = np.asarray(theory(c1q5=1., c1q2=-1.))
    assert theory.quantiles == (5, 2)
    assert theory.ells == (4, 0)
    assert power.shape == (2, 2, 2)
    assert set(theory.runtime_info.params.basenames()) == {'c1q5', 'c1q2'}


def test_density_split_matter_default_parameters_and_cosmology():
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        k=[0.02, 0.04], ells=(0,), quantiles=(1, 3, 5)
    )
    power = np.asarray(theory())
    coefficients = _gaussian_quantile_coefficients()

    assert isinstance(theory.cosmo, Cosmoprimo)
    assert power.shape == (3, 1, 2)
    for quantile in theory.quantiles:
        param = theory.runtime_info.params[f'c1q{quantile}']
        assert param.value == coefficients[quantile]['c1']
        assert not param.fixed


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'k': []}, 'k must be'),
        ({'k': [0.02, 0.01]}, 'k must be'),
        ({'k': [0.01, np.nan]}, 'k must be'),
        ({'ells': (0, 6)}, 'ells must be drawn'),
        ({'ells': (0, 0)}, 'ells must be non-empty and unique'),
        ({'smoothing_radius': -1.}, 'smoothing_radius'),
        ({'z': -0.5}, 'z must be'),
        ({'rsd': False, 'ells': (0, 2)}, 'real-space'),
    ],
)
def test_density_split_matter_rejects_invalid_initialization(kwargs, message):
    theory = DensitySplitMatterPowerSpectrumMultipoles(cosmo=make_cosmo(), **kwargs)
    with pytest.raises(PipelineError) as exc:
        theory()
    assert message in str(exc.value.__cause__)


@pytest.mark.parametrize('quantiles', [(1, 1), (0,), (6,)])
def test_density_split_matter_rejects_invalid_quantiles(quantiles):
    with pytest.raises(ValueError, match='quantiles'):
        DensitySplitMatterPowerSpectrumMultipoles(quantiles=quantiles)


def test_density_split_matter_uses_injected_cosmology_without_folps(monkeypatch):
    cosmo = make_cosmo()
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        k=[0.02, 0.04], ells=(0,), quantiles=(1,), cosmo=cosmo
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == 'folps' or name.startswith('folps.'):
            raise AssertionError('minimal density-split matter model imported FOLPS')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    power = theory(c1q1=1.)
    assert theory.cosmo is cosmo
    assert np.isfinite(power).all()
