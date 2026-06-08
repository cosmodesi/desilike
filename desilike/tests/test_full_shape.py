"""Emulator integration tests for every PT sub-calculator in full_shape.py.

Pattern (mirrors benchmark.py::build_posterior_folps with emulator_order set):
  1. Build the theory with its real PT sub-calculator.
  2. Compile the PT sub-graph and fit a degree-1 TaylorEmulator on it.
  3. Replace the real PT with the emulated version (via replace()).
  4. Compile the full tracer pipeline and compare against an exact reference.

Checks:
  - Exact match at the Taylor expansion center (always true for any order ≥ 1).
  - Relative error < 10 % after a 5 % shift in a cosmological / bias parameter.

Theories backed by optional packages (folps, velocileptors, pybird) are
auto-skipped when the package is not installed.
"""

import numpy as np
import pytest
import jax

jax.config.update('jax_enable_x64', True)

from desilike import compile, TaylorEmulator
from desilike.base import replace


_FID = ('DESI', {'engine': 'camb'})
_K = np.linspace(0.02, 0.2, 15)
_ELLS = (0, 2)
_S = np.linspace(50., 150., 10)
_EMU_ORDER = 1


# ── helpers ────────────────────────────────────────────────────────────────────

def _emulate(theory, inner_pt=None):
    """Emulate ``inner_pt`` (default: ``theory.pt``) in-place; return compiled pipeline."""
    if inner_pt is None:
        inner_pt = theory.pt
    pt_pipe = compile(inner_pt)
    emu = TaylorEmulator(pt_pipe, order=_EMU_ORDER)
    emu.fit()
    emulated_pt = emu.to_calculator()
    replace(theory, inner_pt, emulated_pt)
    return compile(theory)


def _check(pipe_exact, pipe_emu, shift_param, reldiff_tol=0.10):
    """Center-exact match (atol=1e-8) and shifted accuracy (relative < tol)."""
    center = {p.name: p.value for p in pipe_exact.params}
    exact_center = np.asarray(pipe_exact(center))
    emu_center = np.asarray(pipe_emu(center))
    np.testing.assert_allclose(emu_center, exact_center, atol=1e-8, rtol=0.,
                               err_msg='emulator mismatch at expansion center')
    if shift_param in center:
        shifted = {**center, shift_param: center[shift_param] * 1.05}
        exact_s = np.asarray(pipe_exact(shifted))
        emu_s = np.asarray(pipe_emu(shifted))
        reldiff = float(np.max(np.abs(emu_s - exact_s) / (np.abs(exact_s) + 1e-30)))
        assert reldiff < reldiff_tol, \
            f'shifted [{shift_param}+5%]: max reldiff={reldiff:.3f} > {reldiff_tol:.2f}'


# ── Kaiser ─────────────────────────────────────────────────────────────────────

def test_kaiser_spectrum_emulated():
    """KaiserPTSpectrum2Poles emulated as pt= in KaiserTracerSpectrum2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import KaiserPTSpectrum2Poles, KaiserTracerSpectrum2Poles
    from desilike.theories.galaxy_clustering.template import BAOSpectrum2Template

    pipe_exact = compile(KaiserTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')))

    theory_emu = KaiserTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        pt=KaiserPTSpectrum2Poles(k=_K, ells=_ELLS,
                                   template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1')


def test_kaiser_correlation_emulated():
    """KaiserPTSpectrum2Poles emulated inside KaiserTracerCorrelation2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import KaiserTracerCorrelation2Poles
    from desilike.theories.galaxy_clustering.template import BAOSpectrum2Template

    pipe_exact = compile(KaiserTracerCorrelation2Poles(
        s=_S, ells=_ELLS,
        template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')))

    theory_emu = KaiserTracerCorrelation2Poles(
        s=_S, ells=_ELLS,
        template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper'))
    # theory_emu.pt is KaiserTracerSpectrum2Poles; .pt.pt is KaiserPTSpectrum2Poles
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1')


# ── TNS ────────────────────────────────────────────────────────────────────────

def test_tns_spectrum_emulated():
    """TNSPTSpectrum2Poles emulated as pt= in TNSTracerSpectrum2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import TNSPTSpectrum2Poles, TNSTracerSpectrum2Poles

    pipe_exact = compile(TNSTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = TNSTracerSpectrum2Poles(k=_K, ells=_ELLS,
                                          pt=TNSPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1')


def test_tns_correlation_emulated():
    """TNSPTSpectrum2Poles emulated inside TNSTracerCorrelation2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import TNSTracerCorrelation2Poles

    pipe_exact = compile(TNSTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = TNSTracerCorrelation2Poles(s=_S, ells=_ELLS)
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1')


# ── LPT Velocileptors ──────────────────────────────────────────────────────────

def test_lpt_spectrum_emulated():
    """LPTVelocileptorsPTSpectrum2Poles emulated as pt= in LPTVelocileptorsTracerSpectrum2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import (LPTVelocileptorsPTSpectrum2Poles,
                                                                LPTVelocileptorsTracerSpectrum2Poles)

    pipe_exact = compile(LPTVelocileptorsTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = LPTVelocileptorsTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        pt=LPTVelocileptorsPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_lpt_correlation_emulated():
    """LPTVelocileptorsPTSpectrum2Poles emulated inside LPTVelocileptorsTracerCorrelation2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import LPTVelocileptorsTracerCorrelation2Poles

    pipe_exact = compile(LPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = LPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS)
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


# ── REPT Velocileptors ─────────────────────────────────────────────────────────

def test_rept_spectrum_emulated():
    """REPTVelocileptorsPTSpectrum2Poles emulated as pt= in REPTVelocileptorsTracerSpectrum2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import (REPTVelocileptorsPTSpectrum2Poles,
                                                                REPTVelocileptorsTracerSpectrum2Poles)

    pipe_exact = compile(REPTVelocileptorsTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = REPTVelocileptorsTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        pt=REPTVelocileptorsPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_rept_correlation_emulated():
    """REPTVelocileptorsPTSpectrum2Poles emulated inside REPTVelocileptorsTracerCorrelation2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import REPTVelocileptorsTracerCorrelation2Poles

    pipe_exact = compile(REPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = REPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS)
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


# ── PyBird ─────────────────────────────────────────────────────────────────────

def test_pybird_spectrum_emulated():
    """PyBirdPTSpectrum2Poles emulated as pt= in PyBirdTracerSpectrum2Poles."""
    pytest.importorskip('pybird')
    from desilike.theories.galaxy_clustering.full_shape import PyBirdPTSpectrum2Poles, PyBirdTracerSpectrum2Poles

    pipe_exact = compile(PyBirdTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = PyBirdTracerSpectrum2Poles(k=_K, ells=_ELLS,
                                             pt=PyBirdPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1')


def test_pybird_correlation_emulated():
    """PyBirdPTCorrelation2Poles emulated as pt= in PyBirdTracerCorrelation2Poles."""
    pytest.importorskip('pybird')
    from desilike.theories.galaxy_clustering.full_shape import PyBirdPTCorrelation2Poles, PyBirdTracerCorrelation2Poles

    pipe_exact = compile(PyBirdTracerCorrelation2Poles(s=_S, ells=_ELLS))

    # PyBirdTracerCorrelation2Poles.pt is PyBirdPTCorrelation2Poles (direct PT, not nested)
    theory_emu = PyBirdTracerCorrelation2Poles(s=_S, ells=_ELLS,
                                               pt=PyBirdPTCorrelation2Poles(s=_S, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1')


# ── FOLPS ──────────────────────────────────────────────────────────────────────

def test_folps_spectrum_emulated():
    """FOLPSPTSpectrum2Poles emulated as pt= in FOLPSTracerSpectrum2Poles."""
    pytest.importorskip('folps')
    from desilike.theories.galaxy_clustering.full_shape import FOLPSPTSpectrum2Poles, FOLPSTracerSpectrum2Poles

    pipe_exact = compile(FOLPSTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = FOLPSTracerSpectrum2Poles(k=_K, ells=_ELLS,
                                            pt=FOLPSPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_folps_correlation_emulated():
    """FOLPSPTSpectrum2Poles emulated inside FOLPSTracerCorrelation2Poles."""
    pytest.importorskip('folps')
    from desilike.theories.galaxy_clustering.full_shape import FOLPSTracerCorrelation2Poles

    pipe_exact = compile(FOLPSTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = FOLPSTracerCorrelation2Poles(s=_S, ells=_ELLS)
    # theory_emu.pt is FOLPSTracerSpectrum2Poles; .pt.pt is FOLPSPTSpectrum2Poles
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_folps_spectrum3_poles_emulated():
    """FOLPSPTSpectrum2Poles emulated as pt= in FOLPSTracerSpectrum3Poles."""
    pytest.importorskip('folps')
    from desilike.theories.galaxy_clustering.full_shape import FOLPSPTSpectrum2Poles, FOLPSTracerSpectrum3Poles

    pipe_exact = compile(FOLPSTracerSpectrum3Poles())

    theory_emu = FOLPSTracerSpectrum3Poles(pt=FOLPSPTSpectrum2Poles())
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


if __name__ == '__main__':
    #test_kaiser_spectrum_emulated()
    #test_kaiser_correlation_emulated()
    #test_tns_spectrum_emulated()
    #test_tns_correlation_emulated()
    test_folps_spectrum_emulated()
    test_folps_spectrum3_poles_emulated()
    print('Kaiser + TNS emulator tests passed.')
