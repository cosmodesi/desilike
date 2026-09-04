import numpy as np
import pytest

from desilike import compile
from desilike.theories.galaxy_clustering import DensitySplitMatterPowerSpectrumMultipoles


@pytest.mark.parametrize('rsd, ells', [(False, (0,)), (True, (4, 0, 2))])
def test_density_split_matter_compiled_graph(rsd, ells):
    k = np.array([0.02, 0.04, 0.08])
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        k=k, z=0.5, ells=ells, quantiles=(5, 3, 1), rsd=rsd,
        smoothing_radius=12., engine='eisenstein_hu')
    run = compile(theory)
    power = np.asarray(run({'c1q5': 2., 'c1q3': 0.4, 'c1q1': -1.}))

    assert power.shape == (3, len(ells), k.size)
    assert np.isfinite(power).all()
    np.testing.assert_allclose(power[0] / power[1], 5.)
    np.testing.assert_allclose(power[2] / power[1], -2.5)
    assert theory.poles is theory.power


def test_density_split_matter_smoothing():
    k = np.array([0.02, 0.04, 0.08])
    options = dict(k=k, z=0.5, ells=(0,), quantiles=(3,), rsd=False,
                   engine='eisenstein_hu')
    plain = compile(DensitySplitMatterPowerSpectrumMultipoles(
        smoothing_radius=0., **options))({'c1q3': 1.})
    smooth = compile(DensitySplitMatterPowerSpectrumMultipoles(
        smoothing_radius=10., **options))({'c1q3': 1.})
    np.testing.assert_allclose(
        np.asarray(smooth / plain)[0, 0], np.exp(-0.5 * (10. * k)**2),
        rtol=2e-6)


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'k': []}, 'k must be'),
        ({'k': [0.02, 0.01]}, 'k must be'),
        ({'ells': (0, 6)}, 'ells must be drawn'),
        ({'smoothing_radius': -1.}, 'smoothing_radius'),
        ({'z': -0.5}, 'z must be'),
        ({'rsd': False, 'ells': (0, 2)}, 'real-space'),
    ],
)
def test_density_split_matter_validation(kwargs, message):
    theory = DensitySplitMatterPowerSpectrumMultipoles(
        engine='eisenstein_hu', **kwargs)
    with pytest.raises(ValueError, match=message):
        compile(theory)


@pytest.mark.parametrize('quantiles', [(1, 1), (0,), (6,)])
def test_density_split_matter_invalid_quantiles(quantiles):
    with pytest.raises(ValueError, match='quantiles'):
        DensitySplitMatterPowerSpectrumMultipoles(quantiles=quantiles)
