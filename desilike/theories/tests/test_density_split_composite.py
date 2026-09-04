import numpy as np
import pytest

from desilike import jax as desilike_jax


def _toy_loop_inputs():
    kt = np.logspace(-4, 1, 64)
    pklin = 1e4 * kt / (1. + (kt / 0.2)**2)
    k = np.asarray([[0.05, 0.1]])
    mu = np.asarray([[0.2, 0.8]])
    kwargs = dict(smoothing_radius=10., smoothing_kernel='gaussian', nq=6, nx=4, nphi=4, qmax=1.)
    return k, mu, kt, pklin, kwargs


@pytest.mark.skipif(desilike_jax.jax is None, reason='jax is not available')
def test_composite_p2_jax_outputs_are_finite_and_deterministic():
    from desilike.theories.galaxy_clustering.density_split import (
        composite_p2_moments,
        contract_p2_moments,
    )

    k, mu, kt, pklin, kwargs = _toy_loop_inputs()
    moments = composite_p2_moments(k, mu, kt, pklin, 0.75, **kwargs)
    repeated = composite_p2_moments(k, mu, kt, pklin, 0.75, **kwargs)
    p2g = contract_p2_moments(moments, b1=2., b2=0.5, bs=-0.3)

    assert moments.shape == (6, 3) + k.shape
    assert p2g.shape == k.shape
    assert np.isfinite(np.asarray(moments)).all()
    assert np.isfinite(np.asarray(p2g)).all()
    assert np.allclose(np.asarray(moments), np.asarray(repeated), rtol=0., atol=0.)
    assert np.any(np.abs(np.asarray(p2g)) > 0.)


@pytest.mark.skipif(desilike_jax.jax is None, reason='jax is not available')
def test_composite_p2_jax_jit_and_vmap():
    from desilike.theories.galaxy_clustering.density_split import composite_p2_moments

    jax = desilike_jax.jax
    jnp = desilike_jax.numpy
    k, mu, kt, pklin, kwargs = _toy_loop_inputs()
    k, mu, kt, pklin = (jnp.asarray(array) for array in (k, mu, kt, pklin))

    def get_moments(pklin_input):
        return composite_p2_moments(k, mu, kt, pklin_input, 0.75, **kwargs)

    compiled = jax.jit(get_moments)
    moments = compiled(pklin)
    scaled = jax.jit(jax.vmap(lambda scale: get_moments(scale * pklin)))(jnp.asarray([0.8, 1., 1.2]))

    assert np.isfinite(np.asarray(moments)).all()
    assert np.isfinite(np.asarray(scaled)).all()
    assert scaled.shape == (3,) + moments.shape


def test_composite_p2_requires_jax_backend():
    from desilike.base import PipelineError
    from desilike.theories.galaxy_clustering import DensitySplitTracerPowerSpectrumMultipoles

    theory = DensitySplitTracerPowerSpectrumMultipoles(k=np.linspace(0.01, 0.02, 2), ells=(0,), quantiles=(1,),
                                                       model='1-loop', backend='numpy')
    with pytest.raises(PipelineError) as exc:
        theory.runtime_info.initialize()
    assert "requires backend='jax'" in str(exc.value.__cause__)
