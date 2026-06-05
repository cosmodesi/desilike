"""Tests for PNG (Primordial Non-Gaussianity) theories."""

import numpy as np
import jax
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _check(result, name=''):
    arr = np.asarray(result)
    assert arr.ndim == 2, f"{name}: expected 2-D result, got shape {arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0, f"{name}: empty result"
    assert np.isfinite(arr).all(), f"{name}: non-finite values"


def _eval(theory, output='poles', **overrides):
    """Compile *theory* and evaluate it at its default parameters (with optional
    *overrides*); return the named output attribute as a numpy array."""
    from desilike.base import compile, params
    pipe = compile(theory)
    values = {par.name: par._value for par in params(theory)}
    values.update(overrides)
    pipe(values)
    return np.asarray(getattr(theory, output))


def _varied(theory):
    """Return the list of non-fixed parameters of *theory*."""
    from desilike.base import params
    return list(params(theory).select(fixed=False))


# ── _png_cosmo (fixed-fiducial alpha + pk) ────────────────────────────────────

class TestPNGCosmo:

    def test_prim(self):
        """method='prim': output shapes, finite values, and k-decreasing alpha."""
        from desilike.theories.galaxy_clustering.png import _png_cosmo
        k = np.geomspace(1e-3, 0.5, 100)
        pk_dd, alpha, f = _png_cosmo('DESI', k, z=1., method='prim', engine='eisenstein_hu')
        assert pk_dd.shape == (len(k),)
        assert alpha.shape == (len(k),)
        assert np.all(pk_dd > 0)
        assert np.all(np.isfinite(alpha))
        assert alpha[0] > alpha[-1], "alpha should decrease with k"
        assert 0. < f < 2.

    def test_transfer(self):
        """method='transfer': output shapes, finite values, and k-decreasing alpha."""
        from desilike.theories.galaxy_clustering.png import _png_cosmo
        k = np.geomspace(1e-3, 0.5, 100)
        pk_dd, alpha, f = _png_cosmo('DESI', k, z=1., method='transfer', engine='eisenstein_hu')
        assert pk_dd.shape == (len(k),)
        assert alpha.shape == (len(k),)
        assert np.all(np.isfinite(alpha))
        assert alpha[0] > alpha[-1], "alpha should decrease with k"

    def test_method_consistency(self):
        """'prim' and 'transfer' alpha should agree in shape to ~5%."""
        from desilike.theories.galaxy_clustering.png import _png_cosmo
        k = np.geomspace(5e-3, 0.2, 50)
        _, alpha_p, _ = _png_cosmo('DESI', k, z=1., method='prim', engine='eisenstein_hu')
        _, alpha_t, _ = _png_cosmo('DESI', k, z=1., method='transfer', engine='eisenstein_hu')
        mid = slice(5, -5)
        ratio = alpha_p[mid] / alpha_t[mid]
        assert np.allclose(ratio, ratio.mean(), rtol=0.05), \
            f"'prim' and 'transfer' alpha shape differ by > 5%: ratio std/mean = {ratio.std() / ratio.mean():.3f}"


# ── PNGTracerSpectrum2Poles ───────────────────────────────────────────────────

class TestPNGTracerSpectrum2Poles:

    def test_basic(self):
        """Basic output shape and finite values."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = PNGTracerSpectrum2Poles(k=k)
        result = _eval(theory, 'poles')
        _check(result, 'PNGTracerSpectrum2Poles')
        assert result.shape == (len(theory.ells), len(k))

    def test_modes(self):
        """All three mode variants run and are sensitive to their fnl parameter."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)

        theory = PNGTracerSpectrum2Poles(k=k, mode='b-p')
        assert not np.allclose(_eval(theory, 'poles', fnl_loc=0.), _eval(theory, 'poles', fnl_loc=100.)), "b-p: fnl_loc had no effect"

        theory = PNGTracerSpectrum2Poles(k=k, mode='bphi')
        assert not np.allclose(_eval(theory, 'poles', fnl_loc=0.), _eval(theory, 'poles', fnl_loc=100.)), "bphi: fnl_loc had no effect"

        theory = PNGTracerSpectrum2Poles(k=k, mode='bfnl')
        assert not np.allclose(_eval(theory, 'poles', bfnl_loc=0.), _eval(theory, 'poles', bfnl_loc=200.)), "bfnl: bfnl_loc had no effect"

    def test_no_png(self):
        """With fnl_loc=0 the monopole is positive (reduces to Kaiser-like bias)."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = PNGTracerSpectrum2Poles(k=k, mode='b-p')
        result = _eval(theory, 'poles', b1=2., fnl_loc=0., p=1., sigmas=0., sn0=0.)
        _check(result, 'PNG fnl=0')
        assert np.all(result[0] > 0), "monopole at fnl=0 should be positive"

    def test_bias_sensitivity(self):
        """b1 variation changes the output."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = PNGTracerSpectrum2Poles(k=k, mode='b-p')
        assert not np.allclose(_eval(theory, 'poles', b1=1.5), _eval(theory, 'poles', b1=3.0)), "b1 variation had no effect"

    def test_methods(self):
        """Both 'prim' and 'transfer' methods produce finite output."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        for method in ['prim', 'transfer']:
            theory = PNGTracerSpectrum2Poles(k=k, method=method)
            _check(_eval(theory, 'poles', fnl_loc=50.), f'PNGTracerSpectrum2Poles method={method}')

    def test_ells(self):
        """Custom ells=(0,) and ells=(0, 2, 4) give correct output shapes."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)

        theory = PNGTracerSpectrum2Poles(k=k, ells=(0,))
        assert _eval(theory, 'poles').shape[0] == 1

        theory = PNGTracerSpectrum2Poles(k=k, ells=(0, 2, 4))
        assert _eval(theory, 'poles').shape[0] == 3

    def test_cross(self):
        """Cross-spectrum (two tracers) runs and namespaces parameters."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        from desilike.base import params
        k = np.linspace(0.02, 0.3, 60)
        theory = PNGTracerSpectrum2Poles(k=k, tracers=('LRG', 'QSO'), mode='b-p')
        names = [p.name for p in params(theory)]
        assert 'LRG.b1' in names and 'QSO.b1' in names
        assert 'LRGxQSO.sn0' in names
        assert 'fnl_loc' in names  # shared, unnamespaced
        _check(_eval(theory, 'poles'), 'PNG cross')

    def test_param_variation(self):
        """Each varied parameter changes the result."""
        from desilike.theories.galaxy_clustering import PNGTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = PNGTracerSpectrum2Poles(k=k)
        for param in _varied(theory):
            lo, hi = np.asarray(param.ref.sample(jax.random.key(0), shape=2))
            r0 = _eval(theory, 'poles', **{param.name: float(lo)})
            r1 = _eval(theory, 'poles', **{param.name: float(hi)})
            _check(r0, f'param={param.name}')
            if not np.isclose(lo, hi):
                assert not np.allclose(r0, r1), f"result invariant to {param.name}"
            break


# ── PNGTracerVelocitySpectrum2Poles ──────────────────────────────────────────

class TestPNGTracerVelocitySpectrum2Poles:

    def test_basic(self):
        """Odd multipoles (ells=1, 3): basic shape and finite values."""
        from desilike.theories.galaxy_clustering import PNGTracerVelocitySpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = PNGTracerVelocitySpectrum2Poles(k=k, ells=(1, 3))
        result = _eval(theory, 'poles')
        _check(result, 'PNGTracerVelocitySpectrum2Poles')
        assert result.shape == (2, len(k))

    def test_fnl_sensitivity(self):
        """fnl_loc changes the velocity cross-spectrum."""
        from desilike.theories.galaxy_clustering import PNGTracerVelocitySpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = PNGTracerVelocitySpectrum2Poles(k=k, mode='b-p')
        assert not np.allclose(_eval(theory, 'poles', fnl_loc=0.), _eval(theory, 'poles', fnl_loc=100.)), \
            "fnl_loc had no effect on velocity spectrum"

    def test_modes(self):
        """All mode variants produce finite output."""
        from desilike.theories.galaxy_clustering import PNGTracerVelocitySpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        for mode in ['b-p', 'bphi', 'bfnl']:
            theory = PNGTracerVelocitySpectrum2Poles(k=k, mode=mode)
            _check(_eval(theory, 'poles'), f'PNGTracerVelocitySpectrum2Poles mode={mode}')

    def test_methods(self):
        """Both 'prim' and 'transfer' methods produce finite output."""
        from desilike.theories.galaxy_clustering import PNGTracerVelocitySpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        for method in ['prim', 'transfer']:
            theory = PNGTracerVelocitySpectrum2Poles(k=k, method=method)
            _check(_eval(theory, 'poles'), f'PNGTracerVelocitySpectrum2Poles method={method}')


if __name__ == '__main__':
    TestPNGCosmo().test_prim()
    TestPNGCosmo().test_transfer()
    TestPNGCosmo().test_method_consistency()
    TestPNGTracerSpectrum2Poles().test_basic()
    TestPNGTracerSpectrum2Poles().test_modes()
    TestPNGTracerSpectrum2Poles().test_no_png()
    TestPNGTracerSpectrum2Poles().test_bias_sensitivity()
    TestPNGTracerSpectrum2Poles().test_methods()
    TestPNGTracerSpectrum2Poles().test_ells()
    TestPNGTracerSpectrum2Poles().test_cross()
    TestPNGTracerSpectrum2Poles().test_param_variation()
    TestPNGTracerVelocitySpectrum2Poles().test_basic()
    TestPNGTracerVelocitySpectrum2Poles().test_fnl_sensitivity()
    TestPNGTracerVelocitySpectrum2Poles().test_modes()
    TestPNGTracerVelocitySpectrum2Poles().test_methods()
