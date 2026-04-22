# -*- coding: utf-8 -*-
"""Test Kfuncs_to_tables jit+vmap traceability using realistic isitgr P(k) mocks."""
import sys, importlib.util, types as _types

sys.path.insert(0, '/global/homes/r/rohlf/desilike')
sys.path.insert(0, '/global/homes/r/rohlf/desilike/cosmoprimo')
sys.path.insert(0, '/global/homes/r/rohlf/desilike/fkptjax_muMG/src')

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Generate realistic P(k) from isitgr via cosmoprimo
# ---------------------------------------------------------------------------
print("Generating P(k) from isitgr ...", end='  ', flush=True)
from cosmoprimo.fiducial import DESI
cosmo = DESI(engine='isitgr')
fo = cosmo.get_fourier()
ba = cosmo.get_background()

k = np.logspace(-4, 0, 150)
pk_interp = fo.pk_interpolator(of='delta_cb')

# Three cosmologies: fiducial + two with slightly different Omega_m
from cosmoprimo import Cosmology
cosmos = []
for dOm in [0.0, +0.01, -0.01]:
    c = Cosmology(engine='isitgr',
                  Omega_cdm=0.2589 + dOm, Omega_b=0.0486, h=0.6736,
                  n_s=0.9649, sigma8=0.8101, m_ncdm=[0.06])
    cosmos.append(c)

pks = []
pks_now = []
Oms = []
for c in cosmos:
    f = c.get_fourier()
    pks.append(f.pk_interpolator(of='delta_cb')(k, z=0.51))
    pks_now.append(f.pk_interpolator(of='delta_cb')(k, z=0.0))
    Oms.append(float(c['Omega_m']))

print("OK  (3 cosmologies, Om = %s)" % [round(o,4) for o in Oms])

# Use the fiducial for Om/z passed to the ODE solver (always concrete)
Om_fid = Oms[0]
z_fid  = 0.51

# ---------------------------------------------------------------------------
# Bootstrap Kfuncs_to_tables (skip bao.py import chain)
# ---------------------------------------------------------------------------
for _mod_name in [
    'desilike.theories.galaxy_clustering.bao',
    'desilike.theories.galaxy_clustering.damped_bao',
    'desilike.theories.galaxy_clustering.power_template',
    'desilike.theories.galaxy_clustering.primordial_non_gaussianity',
]:
    sys.modules.setdefault(_mod_name, _types.ModuleType(_mod_name))

_root = '/global/homes/r/rohlf/desilike/desilike/theories/galaxy_clustering'

_gc_pkg = _types.ModuleType('desilike.theories.galaxy_clustering')
_gc_pkg.__path__ = [_root]
_gc_pkg.__package__ = 'desilike.theories.galaxy_clustering'
sys.modules['desilike.theories.galaxy_clustering'] = _gc_pkg

import desilike.theories  # noqa

def _load_module(dotted_name, file_path):
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_module('desilike.theories.galaxy_clustering.base', _root + '/base.py')
try:
    _load_module('desilike.theories.galaxy_clustering.power_template', _root + '/power_template.py')
except Exception:
    pass
_fs_mod = _load_module('desilike.theories.galaxy_clustering.full_shape', _root + '/full_shape.py')
Kfuncs_to_tables = _fs_mod.Kfuncs_to_tables
print("Kfuncs_to_tables loaded OK")

common_kwargs = dict(
    Om=Om_fid, z=z_fid, mu0=0.0,
    model='GR',
    ode_method='RKQS',
    f0_kmax=0.1,
    beyond_eds=False,
    rescale_PS=True,
)

# ---------------------------------------------------------------------------
# Test 1: concrete pk (fiducial cosmology)
# ---------------------------------------------------------------------------
print("\nTest 1: concrete pk (fiducial) ...", end='  ', flush=True)
try:
    out = Kfuncs_to_tables(k, pks[0], pks_now[0], **common_kwargs)
    print("PASS  sigma2w=%.6f" % float(out[0][-2]))
except Exception as e:
    import traceback
    print("FAIL  %s: %s" % (type(e).__name__, e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Test 2: jit over pk
# ---------------------------------------------------------------------------
print("Test 2: jit over pk ...", end='  ', flush=True)
@jax.jit
def _run_jit(pk_arr, pk_now_arr):
    return Kfuncs_to_tables(k, pk_arr, pk_now_arr, **common_kwargs)

try:
    out2 = _run_jit(jnp.asarray(pks[0]), jnp.asarray(pks_now[0]))
    print("PASS  sigma2w=%.6f" % float(out2[0][-2]))
except Exception as e:
    import traceback
    print("FAIL  %s: %s" % (type(e).__name__, e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Test 3: vmap over 3 cosmologies
# ---------------------------------------------------------------------------
print("Test 3: vmap over 3 cosmologies ...", end='  ', flush=True)
@jax.vmap
def _run_vmap(pk_arr, pk_now_arr):
    return Kfuncs_to_tables(k, pk_arr, pk_now_arr, **common_kwargs)

pk_batch     = jnp.stack([jnp.asarray(p) for p in pks])
pk_now_batch = jnp.stack([jnp.asarray(p) for p in pks_now])

try:
    out3 = _run_vmap(pk_batch, pk_now_batch)
    sigma2w_batch = out3[0][-2]
    print("PASS  sigma2w = %s" % [round(float(v), 6) for v in sigma2w_batch])
except Exception as e:
    import traceback
    print("FAIL  %s: %s" % (type(e).__name__, e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Test 4: jit + vmap together
# ---------------------------------------------------------------------------
print("Test 4: jit(vmap) over 3 cosmologies ...", end='  ', flush=True)
_run_jit_vmap = jax.jit(jax.vmap(
    lambda pk_arr, pk_now_arr: Kfuncs_to_tables(k, pk_arr, pk_now_arr, **common_kwargs)
))

try:
    out4 = _run_jit_vmap(pk_batch, pk_now_batch)
    sigma2w_batch4 = out4[0][-2]
    print("PASS  sigma2w = %s" % [round(float(v), 6) for v in sigma2w_batch4])
except Exception as e:
    import traceback
    print("FAIL  %s: %s" % (type(e).__name__, e))
    traceback.print_exc()

print("\nDone.")
