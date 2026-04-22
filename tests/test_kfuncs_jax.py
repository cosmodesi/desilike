# -*- coding: utf-8 -*-
"""Test that Kfuncs_to_tables is jit+vmap traceable over pk."""
import sys, importlib.util, types as _types

sys.path.insert(0, '/global/homes/r/rohlf/desilike')
sys.path.insert(0, '/global/homes/r/rohlf/desilike/cosmoprimo')
sys.path.insert(0, '/global/homes/r/rohlf/desilike/fkptjax_muMG/src')

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Bootstrap the desilike package without loading galaxy_clustering.__init__
# (which would pull in bao.py, which is incompatible with the dr2-dev-fkptjax
# base.py currently installed).
#
# Strategy: load the individual modules we need via importlib and stitch them
# into sys.modules manually so that relative imports inside full_shape.py
# resolve correctly without running __init__.py.
# ---------------------------------------------------------------------------

def _load_module(dotted_name, file_path):
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

_root = '/global/homes/r/rohlf/desilike/desilike/theories/galaxy_clustering'

# Pre-register the galaxy_clustering package as an empty module so that
# Python doesn't try to execute its __init__.py when processing relative
# imports from inside full_shape.py.
_gc_pkg = _types.ModuleType('desilike.theories.galaxy_clustering')
_gc_pkg.__path__ = [_root]
_gc_pkg.__package__ = 'desilike.theories.galaxy_clustering'
sys.modules['desilike.theories.galaxy_clustering'] = _gc_pkg

# Also stub out desilike.theories so it resolves to its package dir.
import desilike.theories as _th_pkg  # noqa: E402 (needs desilike already importable)

# Now load base.py
_base_mod = _load_module('desilike.theories.galaxy_clustering.base',
                         _root + '/base.py')

# Stub power_template with just what full_shape.py imports from it
_pt_stub = _types.ModuleType('desilike.theories.galaxy_clustering.power_template')
sys.modules['desilike.theories.galaxy_clustering.power_template'] = _pt_stub
# Load the real power_template to make sure those classes exist
try:
    _pt_real = _load_module('desilike.theories.galaxy_clustering.power_template',
                            _root + '/power_template.py')
except Exception:
    pass  # if it fails, full_shape will raise a clearer error at import time

# Now load full_shape.py
_fs_mod = _load_module('desilike.theories.galaxy_clustering.full_shape',
                       _root + '/full_shape.py')

Kfuncs_to_tables = _fs_mod.Kfuncs_to_tables
print("Kfuncs_to_tables loaded OK")

# ---------------------------------------------------------------------------
# Build a realistic k / pk grid
# ---------------------------------------------------------------------------
N = 150
k = np.logspace(-4, 0, N)

def _pk_simple(k):
    ns = 0.966
    return k**ns / (1.0 + (k / 0.05)**2) ** 2

pk     = _pk_simple(k) * 1e4
pk_now = pk.copy()

common_kwargs = dict(
    Om=0.31, z=0.5, mu0=0.0,
    fR0_HS=0.0, n_HS=1.0, r_c=0.5,
    model='GR',
    ode_method='RKQS',
    f0_kmax=0.1,
    beyond_eds=False,
    rescale_PS=True,
)

# ---------------------------------------------------------------------------
# Test 1: concrete pk (baseline)
# ---------------------------------------------------------------------------
# Returns (table_w, table_nw, kernel_constants).
# table_w[-2] = sigma2w, table_nw[-4] = sigma2w_NW
def _sigma2w(out):
    return float(out[0][-2])

print("\nTest 1: concrete pk ...", end='  ', flush=True)
try:
    out = Kfuncs_to_tables(k, pk, pk_now, **common_kwargs)
    print("PASS  sigma2w=%.6f" % _sigma2w(out))
except Exception as e:
    import traceback
    print("FAIL  %s: %s" % (type(e).__name__, e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Test 2: jit over pk
# ---------------------------------------------------------------------------
print("Test 2: jit over pk ...", end='  ', flush=True)
@jax.jit
def _run_jit(pk_arr):
    return Kfuncs_to_tables(k, pk_arr, pk_arr, **common_kwargs)

try:
    out2 = _run_jit(jnp.asarray(pk))
    print("PASS  sigma2w=%.6f" % _sigma2w(out2))
except Exception as e:
    import traceback
    print("FAIL  %s: %s" % (type(e).__name__, e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Test 3: vmap over pk (batch of 2)
# ---------------------------------------------------------------------------
print("Test 3: vmap over pk ...", end='  ', flush=True)
@jax.vmap
def _run_vmap(pk_arr):
    return Kfuncs_to_tables(k, pk_arr, pk_arr, **common_kwargs)

pk_batch = jnp.stack([jnp.asarray(pk), jnp.asarray(pk * 1.01)])

try:
    out3 = _run_vmap(pk_batch)
    # out3[0][-2] is sigma2w batched over the 2 spectra
    print("PASS  sigma2w[0]=%.6f  sigma2w[1]=%.6f" % (float(out3[0][-2][0]), float(out3[0][-2][1])))
except Exception as e:
    import traceback
    print("FAIL  %s: %s" % (type(e).__name__, e))
    traceback.print_exc()
