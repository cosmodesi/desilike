"""
Prototype validation for the JAX MG emulator (Wall-1 of the jit/vmap effort).

Confirms:
  1. JaxMgPkEmulator reproduces the numpy MgPkEmulator bit-for-bit;
  2. the numpy emulator is NOT jax-vmap-able (the original blocker), while the
     JAX emulator IS jit- and vmap-able;
  3. swapping it clears "Wall 1": the emulator+provider arithmetic traces under
     jit(vmap), and the remaining failure moves into Kfuncs_to_tables (the ODE
     machinery -- "Wall 2").

Skipped unless the trained emulators are present on disk.

Run: python -m pytest tests/test_mg_emulator_jax.py -q
"""
import os
import numpy as np
import pytest

PLIN = "/pscratch/sd/p/prakharb/training_isitgr_plin_pnw_mg_binned_500000/plin"
PNW = "/pscratch/sd/p/prakharb/training_isitgr_plin_pnw_mg_binned_500000_pnw_restart/pnw"
SCAL = "/pscratch/sd/p/prakharb/training_isitgr_plin_pnw_mg_binned_500000/scalars"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PLIN), reason="trained MG emulators not available")

ARGS = dict(z=0.706, ln10As=3.044, ns=0.9649, H0=67.36, ombh2=0.02237, omch2=0.12,
            mu1=1.2, mu2=0.9, mu3=1.1, mu4=1.0, Sigma1=1.1, Sigma2=0.9, Sigma3=1.0, Sigma4=1.0)
ORDER = ['z', 'ln10As', 'ns', 'H0', 'ombh2', 'omch2',
         'mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']


def _emus():
    from desilike.theories.galaxy_clustering.pklin_mg_emulator import MgPkEmulator
    from desilike.theories.galaxy_clustering.pklin_mg_emulator_jax import JaxMgPkEmulator
    kw = dict(path_plin=PLIN, path_pnw=PNW, path_scalars=SCAL)
    return MgPkEmulator(**kw), JaxMgPkEmulator(**kw)


def test_bit_for_bit():
    emu_np, emu_jx = _emus()
    kn, pn, wn, sn = emu_np.predict_all(**ARGS)
    kj, pj, wj, sj = emu_jx.predict_all(*[ARGS[k] for k in ORDER])
    sn_arr = np.array([sn[n] for n in emu_np.SCALAR_NAMES])
    assert np.allclose(np.asarray(pj), pn, rtol=1e-11, atol=0)
    assert np.allclose(np.asarray(wj), wn, rtol=1e-11, atol=0)
    assert np.allclose(np.asarray(sj), sn_arr, rtol=1e-11, atol=0)


def test_numpy_emulator_not_vmappable():
    import jax
    emu_np, _ = _emus()

    def call(vec):
        return emu_np.predict_all(**{k: float(vec[i]) for i, k in enumerate(ORDER)})[1]

    base = np.array([ARGS[k] for k in ORDER])
    with pytest.raises(Exception):
        jax.vmap(call)(np.stack([base, base]))


def test_jax_emulator_jit_and_vmap():
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    _, emu_jx = _emus()

    def call(vec):
        return emu_jx.predict_all(*[vec[i] for i in range(14)])[1]

    base = jnp.array([ARGS[k] for k in ORDER])
    batch = jnp.stack([base.at[6].set(v) for v in [0.9, 1.1, 1.3, 1.5]])  # vary mu1
    out_v = jax.vmap(call)(batch)
    out_j = jax.jit(jax.vmap(call))(batch)
    assert np.asarray(out_v).shape == (4, emu_jx.k_plin.shape[0])
    assert np.allclose(np.asarray(out_v), np.asarray(out_j), rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(out_v[0]), np.asarray(call(batch[0])), rtol=1e-12, atol=0)
