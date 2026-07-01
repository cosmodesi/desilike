"""Tests for DESI DR2 BAO likelihoods.

Synthetic mean/covariance files are written to a temporary directory so these
tests run offline.  The cosmology uses the JAX-native ``'eisenstein_hu'`` engine
so that jit/grad checks do not exercise external-engine finite-difference machinery.

``test_desi_dr2_bao_install`` downloads the real DR2 data files from GitHub
(network required) and verifies end-to-end construction and evaluation.
"""

import numpy as np
import jax
import pytest

from desilike.base import compile, get_params
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from desilike.likelihoods.bao import DESIDR2BAOLikelihood, _TRACER_FILES


# ── fixture helpers ───────────────────────────────────────────────────────────

def _write_mean_file(fn, rows):
    """Write a BAO mean-values file.  rows: list of (z, value, quantity)."""
    with open(fn, 'w') as f:
        f.write('# [z] [value at z] [quantity]\n')
        for z_val, value, quantity in rows:
            f.write(f'{z_val:.8f} {value:.10f} {quantity}\n')


def _write_cov_file(fn, matrix):
    """Write a BAO covariance file as whitespace-separated values."""
    np.savetxt(fn, np.atleast_2d(matrix).ravel()[np.newaxis], fmt='%.15e')


def _make_bao_files(data_dir, tracer_specs):
    """Write mean and covariance files for the given tracer specifications.

    Parameters
    ----------
    data_dir : Path
        Temporary directory.
    tracer_specs : dict
        Mapping tracer_name → (rows, cov_matrix) where rows is a list of
        (z, value, quantity) tuples and cov_matrix is a 2-D array.

    Returns
    -------
    dict mapping tracer_name → (rows, cov_matrix)
    """
    for tracer_name, (rows, cov_matrix) in tracer_specs.items():
        mean_fn, cov_fn = _TRACER_FILES[tracer_name]
        _write_mean_file(data_dir / mean_fn, rows)
        _write_cov_file(data_dir / cov_fn, cov_matrix)
    return tracer_specs


def _all_synthetic_specs():
    """Return synthetic tracer_specs covering all 7 DR2 bins with simple data."""
    return {
        'BGS':       ([(0.295, 7.94, 'DV_over_rs')],                                              np.array([[5.8e-3]])),
        'LRG1':      ([(0.51,  13.59, 'DM_over_rs'), (0.51,  21.86, 'DH_over_rs')],              np.array([[2.8e-2, -3.3e-2], [-3.3e-2, 1.8e-1]])),
        'LRG2':      ([(0.706, 17.35, 'DM_over_rs'), (0.706, 19.46, 'DH_over_rs')],              np.array([[3.2e-2, -2.4e-2], [-2.4e-2, 1.1e-1]])),
        'LRG3+ELG1': ([(0.934, 21.58, 'DM_over_rs'), (0.934, 17.64, 'DH_over_rs')],              np.array([[2.6e-2, -1.1e-2], [-1.1e-2, 4.0e-2]])),
        'ELG2':      ([(1.321, 27.60, 'DM_over_rs'), (1.321, 14.18, 'DH_over_rs')],              np.array([[1.1e-1, -2.9e-2], [-2.9e-2, 5.0e-2]])),
        'QSO':       ([(1.484, 30.51, 'DM_over_rs'), (1.484, 12.82, 'DH_over_rs')],              np.array([[5.8e-1, -2.0e-1], [-2.0e-1, 2.7e-1]])),
        'Lya':       ([(2.33,  8.63,  'DH_over_rs'), (2.33,  38.99, 'DM_over_rs')],              np.array([[1.0e-2, -2.3e-2], [-2.3e-2, 2.8e-1]])),
    }


def _eis_cosmo():
    return CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')


# ── construction ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('zbins', [None, ['BGS', 'LRG1', 'Lya']], ids=['all', 'subset'])
def test_construction(tmp_path, zbins):
    """Correct observable list, flatdata shape, and flatdata values for all-zbins and subset."""
    specs = _all_synthetic_specs()
    _make_bao_files(tmp_path, specs)

    like = DESIDR2BAOLikelihood(zbins=zbins, data_dir=str(tmp_path), cosmo=_eis_cosmo())

    selected = list(specs) if zbins is None else zbins
    assert [obs.name for obs in like.observables] == selected
    expected_flatdata = np.concatenate([np.array([v for _, v, _ in specs[zbin][0]]) for zbin in selected])
    assert like.flatdata.shape == (len(expected_flatdata),)
    np.testing.assert_allclose(np.asarray(like.flatdata), expected_flatdata)


def test_unknown_zbins(tmp_path):
    """Unknown zbin names raise a ValueError at construction."""
    specs = _all_synthetic_specs()
    _make_bao_files(tmp_path, specs)

    with pytest.raises(ValueError, match='Unknown zbins'):
        DESIDR2BAOLikelihood(zbins=['BGS', 'UNKNOWN'], data_dir=str(tmp_path))


# ── jit / grad ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('zbins', [None, ['BGS', 'LRG1', 'Lya']], ids=['all', 'subset'])
def test_jit_and_grad(tmp_path, zbins):
    """logpdf is finite, jit matches eager, gradients are finite and non-zero."""
    specs = _all_synthetic_specs()
    _make_bao_files(tmp_path, specs)

    like = DESIDR2BAOLikelihood(zbins=zbins, data_dir=str(tmp_path), cosmo=_eis_cosmo())
    pipe = compile(like)
    defaults = {p.name: p._value for p in get_params(like)}

    logpdf = pipe(defaults)
    assert np.isfinite(logpdf), f'logpdf not finite: {logpdf}'

    jit_logpdf = jax.jit(pipe)(defaults)
    assert np.isclose(float(logpdf), float(jit_logpdf))

    grad = jax.grad(pipe)(defaults)
    assert all(np.isfinite(v) for v in grad.values()), 'Non-finite gradients'
    # h and omega_cdm both affect distances and should give non-zero gradients.
    assert grad['h'] != 0., 'grad wrt h is zero'
    assert grad['omega_cdm'] != 0., 'grad wrt omega_cdm is zero'


def test_desi_dr2_bao_install(tmp_path, monkeypatch):
    """Install DESIDR2BAOLikelihood to a temporary directory and run it.

    Downloads the 14 DR2 data files from GitHub (network required) and verifies
    the compiled logpdf is finite at fiducial parameters.
    """
    from desilike.install import Installer

    monkeypatch.setenv('DESILIKE_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('DESILIKE_INSTALL_DIR', str(tmp_path))

    DESIDR2BAOLikelihood.install(Installer())

    like = DESIDR2BAOLikelihood(cosmo=_eis_cosmo())
    pipe = compile(like)
    defaults = {p.name: p._value for p in get_params(like)}
    logpdf = pipe(defaults)
    assert np.isfinite(float(logpdf)), f'logpdf not finite: {logpdf}'


if __name__ == '__main__':

    import os
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_construction(Path(tmp), zbins=None)
    with tempfile.TemporaryDirectory() as tmp:
        test_construction(Path(tmp), zbins=['BGS', 'LRG1', 'Lya'])
    with tempfile.TemporaryDirectory() as tmp:
        test_unknown_zbins(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_jit_and_grad(Path(tmp), zbins=None)
    with tempfile.TemporaryDirectory() as tmp:
        test_jit_and_grad(Path(tmp), zbins=['BGS', 'LRG1', 'Lya'])
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['DESILIKE_CONFIG_DIR'] = tmp
        os.environ['DESILIKE_INSTALL_DIR'] = tmp
        test_desi_dr2_bao_install(Path(tmp), type('_MP', (), {'setenv': lambda self, k, v: None})())
    print('All tests passed.')
