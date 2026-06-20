"""Tests for type Ia supernovae (SN) likelihoods.

Uses small synthetic light-curve/covariance files written to a temporary directory
(via the ``data_dir`` argument) instead of downloading real released data, so these
tests run offline and fast. ``cosmo`` is built with the JAX-native ``eisenstein_hu``
engine so jit/grad checks do not exercise the (unrelated) external-engine
finite-difference machinery.
"""

import numpy as np
import jax
import pytest

from desilike.base import compile, get_params
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from desilike.likelihoods.supernovae import (
    PantheonSNLikelihood, PantheonPlusSNLikelihood, PantheonPlusSHOESSNLikelihood,
    Union3SNLikelihood, Union3p1SNLikelihood, DESY5v1SNLikelihood, DESY5DovekieSNLikelihood,
)


def _write_covariance(fn, n, value=1e-4):
    """Write a CosmoMC-format covariance file: size on the first line, then the flattened matrix."""
    cov = np.eye(n) * value
    with open(fn, 'w') as file:
        file.write(f'{n}\n')
        for row in cov:
            file.write(' '.join(str(v) for v in row) + '\n')


def _make_pantheon(data_dir, z):
    with open(data_dir / 'lcparam_full_long.txt', 'w') as file:
        file.write('# zcmb zhel mb dmb\n')
        for zi in z:
            file.write(f'{zi} {zi} {35. + 5 * np.log10(zi)} 0.1\n')
    _write_covariance(data_dir / 'sys_full_long.txt', len(z))


def _make_pantheonplus(data_dir, z):
    with open(data_dir / 'Pantheon+SH0ES.dat', 'w') as file:
        file.write('zHD zHEL m_b_corr\n')
        for zi in z:
            file.write(f'{zi} {zi} {35. + 5 * np.log10(zi)}\n')
    _write_covariance(data_dir / 'Pantheon+SH0ES_STAT+SYS.cov', len(z))


def _make_pantheonplusshoes(data_dir, z, calibrator_mask=None):
    if calibrator_mask is None:
        calibrator_mask = np.zeros(len(z), dtype=bool)
    with open(data_dir / 'Pantheon+SH0ES.dat', 'w') as file:
        file.write('zHD zHEL m_b_corr IS_CALIBRATOR CEPH_DIST\n')
        for zi, is_calib in zip(z, calibrator_mask):
            file.write(f'{zi} {zi} {35. + 5 * np.log10(zi)} {int(is_calib)} 32.5\n')
    _write_covariance(data_dir / 'Pantheon+SH0ES_STAT+SYS.cov', len(z))


def _write_union3_fits(fn, z):
    """Synthetic fixture matching the union3_release FITS layout: a single (n+1, n+1)
    matrix whose first row (sans corner) is z, first column (sans corner) is mb, and
    remaining (n, n) block is the precision matrix directly (per the data release's README)."""
    import fitsio
    n = len(z)
    data = np.zeros((n + 1, n + 1))
    data[0, 1:] = z
    data[1:, 0] = 35. + 5 * np.log10(z)
    data[1:, 1:] = np.eye(n) * 1e4  # inverse of the 1e-4 covariance used by _write_covariance
    fitsio.write(str(fn), data, clobber=True)


def _make_union3(data_dir, z):
    _write_union3_fits(data_dir / Union3SNLikelihood.data_file, z)


def _make_union3p1(data_dir, z):
    _write_union3_fits(data_dir / Union3p1SNLikelihood.data_file, z)


def _make_des(data_dir, z):
    with open(data_dir / 'DES-SN5YR_HD.csv', 'w') as file:
        file.write('# comment\n')
        file.write('zHD,zHEL,MU,MUERR_FINAL\n')
        for zi in z:
            file.write(f'{zi},{zi},{35. + 5 * np.log10(zi)},0.1\n')
    _write_covariance(data_dir / 'STAT+SYS.txt', len(z))


def _make_des_dovekie(data_dir, z):
    """Synthetic fixture for DESY5DovekieSNLikelihood: 'VARNAMES:'/'SN:'-prefixed
    light-curve file, and a .npz precision matrix stored as an upper-triangle-packed array."""
    with open(data_dir / 'DES-Dovekie_HD.csv', 'w') as file:
        file.write('# comment\n')
        file.write('VARNAMES: CID IDSURVEY zHD zHEL MU MUERR\n')
        for i, zi in enumerate(z):
            file.write(f'SN: SN{i} 10 {zi} {zi} {35. + 5 * np.log10(zi)} 0.1\n')
    n = len(z)
    precision = np.eye(n) * 1e4  # inverse of the 1e-4 covariance used by _write_covariance
    np.savez(data_dir / 'STAT+SYS.npz', nsn=np.array([n]), cov=precision[np.triu_indices(n)], allow_pickle=False)


_CASES = [
    (PantheonSNLikelihood, _make_pantheon),
    (PantheonPlusSNLikelihood, _make_pantheonplus),
    (PantheonPlusSHOESSNLikelihood, _make_pantheonplusshoes),
    (Union3SNLikelihood, _make_union3),
    (Union3p1SNLikelihood, _make_union3p1),
    (DESY5v1SNLikelihood, _make_des),
    (DESY5DovekieSNLikelihood, _make_des_dovekie),
]


@pytest.mark.parametrize('Likelihood,make_fixture', _CASES, ids=[cls.__name__ for cls, _ in _CASES])
def test_likelihood(tmp_path, Likelihood, make_fixture):
    """Each SN likelihood constructs, compiles, evaluates to a finite logpdf,
    and is jit/grad-compatible against synthetic light-curve data."""
    z = np.linspace(0.05, 0.8, 5)
    make_fixture(tmp_path, z)
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
    like = Likelihood(data_dir=str(tmp_path), cosmo=cosmo)

    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}

    logpdf = pipe(defaults)
    assert np.isfinite(logpdf)

    jit_logpdf = jax.jit(pipe)(defaults)
    assert np.isclose(float(logpdf), float(jit_logpdf))

    grad = jax.grad(pipe)(defaults)
    assert all(np.isfinite(v) for v in grad.values())


def test_pantheonplus_zcut(tmp_path):
    """PantheonPlusSNLikelihood drops SNe at z <= 0.01 from both light_curve_params and covariance."""
    z = np.array([0.005, 0.02, 0.1, 0.3, 0.5])
    _make_pantheonplus(tmp_path, z)
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
    like = PantheonPlusSNLikelihood(data_dir=str(tmp_path), cosmo=cosmo)

    assert len(like.light_curve_params['zHD']) == 4  # the z=0.005 entry is cut
    assert np.all(like.light_curve_params['zHD'] > 0.01)
    assert like.covariance.shape == (4, 4)
    assert like.flatdata.shape == (4,)


def test_des_dovekie_precision(tmp_path):
    """DESY5DovekieSNLikelihood unpacks its .npz array as the precision matrix directly
    (no inversion, no extra diagonal term), unlike DESY5v1SNLikelihood's plain covariance."""
    z = np.linspace(0.05, 0.8, 5)
    _make_des_dovekie(tmp_path, z)
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
    like = DESY5DovekieSNLikelihood(data_dir=str(tmp_path), cosmo=cosmo)

    assert len(like.light_curve_params['zHD']) == len(z)
    # The .npz array already *is* the precision matrix (no inversion needed/added).
    np.testing.assert_allclose(like.precision, np.eye(len(z)) * 1e4)


def test_pantheonplusshoes_calibrator(tmp_path):
    """PantheonPlusSHOESSNLikelihood uses the Cepheid host distance (not the cosmological
    prediction) as theory for SNe flagged as calibrators."""
    z = np.array([0.005, 0.1, 0.3, 0.5, 0.7])
    calibrator_mask = np.array([True, False, False, False, False])
    _make_pantheonplusshoes(tmp_path, z, calibrator_mask=calibrator_mask)
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
    like = PantheonPlusSHOESSNLikelihood(data_dir=str(tmp_path), cosmo=cosmo)

    # The z=0.005 calibrator is kept despite z < 0.01, because it is a calibrator.
    assert len(like.light_curve_params['zcmb']) == 5
    compile(like)()
    is_calibrator = like.light_curve_params['is_calibrator']
    assert is_calibrator.sum() == 1
    expected = like.light_curve_params['cepheid_distance'][is_calibrator][0] + like.Mb.value
    assert np.isclose(float(like.flattheory[is_calibrator][0]), expected)


if __name__ == '__main__':

    import tempfile
    from pathlib import Path

    for Likelihood, make_fixture in _CASES:
        with tempfile.TemporaryDirectory() as tmp:
            test_likelihood(Path(tmp), Likelihood, make_fixture)
    with tempfile.TemporaryDirectory() as tmp:
        test_pantheonplus_zcut(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_des_dovekie_precision(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_pantheonplusshoes_calibrator(Path(tmp))
