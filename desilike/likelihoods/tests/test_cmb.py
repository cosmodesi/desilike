"""Tests for CMB likelihoods: Planck NPIPE (PR4) CamSpec, CamSpec-NPIPE-lite, and ACT DR6 + SPT-3G lensing.

Uses a small synthetic CamSpec-NPIPE-format dataset written to a temporary directory
(via the ``data_dir`` argument) instead of downloading the real (~85 MB compressed)
released data, so these tests run offline and fast. Harmonic :math:`C_\\ell` require a
real Boltzmann engine (the JAX-native ``eisenstein_hu`` engine cannot compute them), so
``cosmo`` is built with ``engine='camb'`` here.

``test_camspec_npipe_lite_install`` downloads the real CamSpec-NPIPE-lite SACC data file
(~300 MB) from GitHub (network required) and verifies end-to-end construction and evaluation.

``test_act_dr6_spt_lensing_install`` downloads the ACT DR6 likelihood data tarball (~220 MB)
from NASA LAMBDA (network required) and verifies end-to-end construction and evaluation
of the ``actplanck_baseline`` variant.

Note
----
``jax.grad`` through the external-engine finite-difference path is not exercised here:
it currently hits an unresolved ``CosmologyInputError`` inside ``cosmoprimo.camb`` when
perturbing parameters at low ``ellmax`` (see the docstring of ``_BasePlanckNPIPECamspecLikelihood``
in ``camspec.py``); only the forward pass and jit-consistency are checked.
"""

import numpy as np
import jax
import pytest

from desilike.base import compile, get_params
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from desilike.likelihoods.cmb import (TTTEEEHighlPlanckNPIPECamspecLikelihood,
                                       TTHighlPlanckNPIPECamspecLikelihood,
                                       CamspecNPIPELiteLikelihood,
                                       ACTDR6SPTLensingLikelihood)


def _write_camspec_fixture(data_dir):
    """Write a small synthetic CamSpec-NPIPE-format dataset: spectra, ell-ranges and a
    diagonal (hence positive-definite) covariance, matching the real data's file layout."""
    all_cls = ['100x100', '143x143', '217x217', '143x217', 'TE', 'EE']
    elllims = {'100x100': (2, 10), '143x143': (2, 12), '217x217': (5, 15),
               '143x217': (5, 15), 'TE': (2, 12), 'EE': (2, 12)}
    nrows = max(hi for _, hi in elllims.values()) + 1

    spectra = np.ones((nrows, len(all_cls)), dtype='f8')
    with open(data_dir / 'like_NPIPE_12.6_unified_data_ranges.txt', 'w') as file:
        for cl in all_cls:
            lo, hi = elllims[cl]
            file.write('{} {} {}\n'.format(cl, lo, hi))

    nx = sum(hi - lo + 1 for lo, hi in elllims.values())
    covariance = (np.eye(nx, dtype='f4') * 1e-2)
    covariance.tofile(data_dir / 'like_NPIPE_12.6_unified_cov.bin')

    np.savetxt(data_dir / 'like_NPIPE_12.6_unified_spectra.txt', spectra)


_LIKELIHOODS = [TTTEEEHighlPlanckNPIPECamspecLikelihood, TTHighlPlanckNPIPECamspecLikelihood]


@pytest.mark.parametrize('Likelihood', _LIKELIHOODS, ids=[cls.__name__ for cls in _LIKELIHOODS])
def test_likelihood(tmp_path, Likelihood):
    """Each CamSpec likelihood constructs, compiles, and evaluates to a finite logpdf,
    matching between direct and jit-compiled evaluation."""
    _write_camspec_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = Likelihood(data_dir=str(tmp_path), cosmo=cosmo)

    assert like.ellmax == 15
    assert like.has_foregrounds

    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}

    logpdf = pipe(defaults)
    assert np.isfinite(logpdf)

    jit_logpdf = jax.jit(pipe)(defaults)
    assert np.isclose(float(logpdf), float(jit_logpdf))


def test_select_cls_excludes_100x100(tmp_path):
    """Both likelihood variants drop '100x100' (per the reference implementation's default)."""
    _write_camspec_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = TTTEEEHighlPlanckNPIPECamspecLikelihood(data_dir=str(tmp_path), cosmo=cosmo)
    assert '100x100' not in like.index_ells
    assert set(like.index_ells) == {'143x143', '217x217', '143x217', 'TE', 'EE'}


def test_tt_only_excludes_polarization(tmp_path):
    """TTHighlPlanckNPIPECamspecLikelihood drops TE/EE spectra and their calibration nuisances."""
    _write_camspec_fixture(tmp_path)
    cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=20, non_linear='mead')))
    like = TTHighlPlanckNPIPECamspecLikelihood(data_dir=str(tmp_path), cosmo=cosmo)
    assert set(like.index_ells) == {'143x143', '217x217', '143x217'}
    assert not hasattr(like, 'calTE')
    assert not hasattr(like, 'calEE')


def test_camspec_npipe_lite_install(tmp_path, monkeypatch):
    """Install CamspecNPIPELiteLikelihood to a temporary directory and run it.

    Downloads the CamSpec-NPIPE-lite SACC FITS file (~300 MB) from GitHub
    (network required) and verifies the compiled logpdf is finite at fiducial parameters.
    """
    from desilike.install import Installer

    monkeypatch.setenv('DESILIKE_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('DESILIKE_INSTALL_DIR', str(tmp_path))

    CamspecNPIPELiteLikelihood.install(Installer())

    like = CamspecNPIPELiteLikelihood()
    pipe = compile(like)
    defaults = {p.name: p._value for p in get_params(like)}
    logpdf = pipe(defaults)
    assert np.isfinite(float(logpdf)), f'logpdf not finite: {logpdf}'


def test_act_dr6_spt_lensing_install(tmp_path, monkeypatch):
    """Install ACTDR6SPTLensingLikelihood and run the actplanck_baseline variant.

    Downloads the ACT DR6 likelihood data tarball (~220 MB) from NASA LAMBDA
    (network required). The ``act_dr6_spt_lenslike`` package must already be importable
    (install with ``pip install .`` from the spt_act_likelihood repository).
    """
    from desilike.install import Installer

    monkeypatch.setenv('DESILIKE_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('DESILIKE_INSTALL_DIR', str(tmp_path))

    ACTDR6SPTLensingLikelihood.install(Installer())

    like = ACTDR6SPTLensingLikelihood(variant='actplanck_baseline')
    pipe = compile(like)
    defaults = {p.name: p._value for p in get_params(like)}
    logpdf = pipe(defaults)
    assert np.isfinite(float(logpdf)), f'logpdf not finite: {logpdf}'


if __name__ == '__main__':

    import os
    import tempfile
    from pathlib import Path

    for Likelihood in _LIKELIHOODS:
        with tempfile.TemporaryDirectory() as tmp:
            test_likelihood(Path(tmp), Likelihood)
    with tempfile.TemporaryDirectory() as tmp:
        test_select_cls_excludes_100x100(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_tt_only_excludes_polarization(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['DESILIKE_CONFIG_DIR'] = tmp
        os.environ['DESILIKE_INSTALL_DIR'] = tmp
        test_camspec_npipe_lite_install(Path(tmp), type('_MP', (), {'setenv': lambda self, k, v: None})())
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['DESILIKE_CONFIG_DIR'] = tmp
        os.environ['DESILIKE_INSTALL_DIR'] = tmp
        test_act_dr6_spt_lensing_install(Path(tmp), type('_MP', (), {'setenv': lambda self, k, v: None})())
