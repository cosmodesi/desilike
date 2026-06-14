"""Tests for full-shape theories."""

import numpy as np
import jax
import pytest

jax.config.update('jax_enable_x64', True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _check(result, name=''):
    arr = np.asarray(result)
    assert arr.ndim == 2, f"{name}: expected 2-D result, got shape {arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0, f"{name}: empty result"
    assert np.isfinite(arr).all(), f"{name}: non-finite values"


def _direct_template(**kw):
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template
    return DirectSpectrum2Template(engine='eisenstein_hu', **kw)


def _compile(theory):
    """Compile *theory* **once** and return a reusable runner.

    Calling ``run(**overrides)`` evaluates the pipeline at the theory's default
    parameter values updated with *overrides*, reusing the same compiled
    pipeline -- so a given theory instance is compiled a single time, and
    several features (shape, finiteness, parameter sensitivity) can be checked
    from a minimal number of evaluations. The return value is the pipeline
    output; side-effect attributes (e.g. ``theory.table``) are populated too.
    """
    from desilike.base import compile, params
    pipe = compile(theory)
    defaults = {par.name: par._value for par in params(theory)}

    def run(**overrides):
        return np.asarray(pipe({**defaults, **overrides}))

    return run


def _emulate(theory, inner_pt=None):
    """Fit a degree-1 TaylorEmulator on ``inner_pt`` (default: ``theory.pt``), replace it in-place, return compiled pipeline."""
    from desilike import compile, TaylorEmulator
    from desilike.base import replace
    if inner_pt is None:
        inner_pt = theory.pt
    emu = TaylorEmulator(compile(inner_pt), order=1)
    emu.fit()
    replace(theory, inner_pt, emu.to_calculator())
    return compile(theory)


def _check_emulator(pipe_exact, pipe_emu, shift_param, reldiff_tol=0.10):
    """Center: exact match (atol=1e-8). Shifted by 5 %: relative error < tol."""
    center = {p.name: p.value for p in pipe_exact.params}
    np.testing.assert_allclose(np.asarray(pipe_emu(center)), np.asarray(pipe_exact(center)),
                               atol=1e-8, rtol=0., err_msg='emulator mismatch at expansion center')
    if shift_param in center:
        shifted = {**center, shift_param: center[shift_param] * 1.05}
        exact_s = np.asarray(pipe_exact(shifted))
        emu_s = np.asarray(pipe_emu(shifted))
        reldiff = float(np.max(np.abs(emu_s - exact_s) / (np.abs(exact_s) + 1e-30)))
        assert reldiff < reldiff_tol, f'[{shift_param}+5%] max reldiff={reldiff:.3f} > {reldiff_tol}'


def _check_sensitivity(run, baseline, name, **override):
    """Assert evaluating *run* with *override* moves away from *baseline*."""
    (param_name, value), = override.items()
    assert not np.allclose(baseline, run(**override)), f"{name}: result invariant to {param_name}"


# ── Kaiser ────────────────────────────────────────────────────────────────────

class TestKaiserPoles:

    def test_spectrum_templates(self):
        """KaiserPTSpectrum2Poles: all three templates produce correct shape and finite output."""
        from desilike.theories.galaxy_clustering import (
            KaiserPTSpectrum2Poles, BAOSpectrum2Template, ShapeFitSpectrum2Template,
        )
        k = np.linspace(0.02, 0.3, 60)
        for template in [BAOSpectrum2Template(), ShapeFitSpectrum2Template(), _direct_template()]:
            theory = KaiserPTSpectrum2Poles(k=k, template=template)
            _compile(theory)()  # __call__ returns None; output is in table['pk_dd']
            result = np.asarray(theory.table['pk_dd'])
            _check(result, 'KaiserPTSpectrum2Poles')
            assert result.shape == (len(theory.ells), len(k))

    def test_tracer_spectrum(self):
        """KaiserTracerSpectrum2Poles: shape, b1 sensitivity, and ells=(0,) edge case."""
        from desilike.theories.galaxy_clustering import KaiserTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = KaiserTracerSpectrum2Poles(k=k)
        run = _compile(theory)
        base = run()  # default params (b1 == 1.)
        _check(base, 'KaiserTracerSpectrum2Poles')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'KaiserTracerSpectrum2Poles', b1=2.0)
        # ells=(0,) edge case requires a fresh construction (different output shape)
        assert _compile(KaiserTracerSpectrum2Poles(k=k, ells=(0,)))().shape[0] == 1

    def test_tracer_correlation(self):
        """KaiserTracerCorrelation2Poles: shape and b1 sensitivity."""
        from desilike.theories.galaxy_clustering import KaiserTracerCorrelation2Poles
        s = np.linspace(50., 150., 50)
        theory = KaiserTracerCorrelation2Poles(s=s)
        run = _compile(theory)
        base = run()
        _check(base, 'KaiserTracerCorrelation2Poles')
        assert base.shape == (len(theory.ells), len(s))
        _check_sensitivity(run, base, 'KaiserTracerCorrelation2Poles', b1=2.0)

    def test_emulated(self):
        """KaiserPTSpectrum2Poles emulated as pt= in spectrum and correlation."""
        from desilike import compile
        from desilike.theories.galaxy_clustering import (
            KaiserPTSpectrum2Poles, KaiserTracerSpectrum2Poles, KaiserTracerCorrelation2Poles,
            BAOSpectrum2Template)

        k = np.linspace(0.02, 0.2, 15)
        ells = (0, 2)
        template = BAOSpectrum2Template(z=0.5, fiducial=('DESI', {'engine': 'camb'}), apmode='qparqper')

        pipe_exact = compile(KaiserTracerSpectrum2Poles(k=k, ells=ells, template=template))
        theory_emu = KaiserTracerSpectrum2Poles(k=k, ells=ells,
                                                pt=KaiserPTSpectrum2Poles(k=k, ells=ells, template=template))
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(KaiserTracerCorrelation2Poles(s=s, ells=ells, template=template))
        theory_emu = KaiserTracerCorrelation2Poles(s=s, ells=ells, template=template)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1')


# ── TNS ───────────────────────────────────────────────────────────────────────

class TestTNSPoles:

    def test_spectrum_templates_and_param(self):
        """TNSPTSpectrum2Poles: all three templates give correct shape/finite output,
        and the first varied parameter changes the result."""
        from desilike.theories.galaxy_clustering import (
            TNSPTSpectrum2Poles, BAOSpectrum2Template, ShapeFitSpectrum2Template,
        )
        from desilike.base import params
        k = np.linspace(0.02, 0.3, 60)
        for template in [BAOSpectrum2Template(), ShapeFitSpectrum2Template(), _direct_template()]:
            theory = TNSPTSpectrum2Poles(k=k, template=template)
            _compile(theory)()  # __call__ returns None; output is in table['pk_dd']
            result = np.asarray(theory.table['pk_dd'])
            _check(result, 'TNSPTSpectrum2Poles')
            assert result.shape == (len(theory.ells), len(k))

        # Parameter sensitivity on the (cheap default) template, reusing one compile.
        theory = TNSPTSpectrum2Poles(k=k)
        run = _compile(theory)
        param = list(params(theory).select(fixed=False))[0]
        lo, hi = (float(v) for v in np.asarray(param.ref.sample(jax.random.key(0), shape=2)))
        run(**{param.name: lo})
        r0 = np.asarray(theory.table['pk_dd'])
        _check(r0, 'TNSPTSpectrum2Poles')
        if not np.isclose(lo, hi):
            run(**{param.name: hi})
            r1 = np.asarray(theory.table['pk_dd'])
            assert not np.allclose(r0, r1), f"result invariant to {param.name}"

    def test_tracer_spectrum(self):
        """TNSTracerSpectrum2Poles: shape, b1 sensitivity, and ells=(0, 2) edge case."""
        from desilike.theories.galaxy_clustering import TNSTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = TNSTracerSpectrum2Poles(k=k)
        run = _compile(theory)
        base = run()
        _check(base, 'TNSTracerSpectrum2Poles')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'TNSTracerSpectrum2Poles', b1=2.0)
        assert _compile(TNSTracerSpectrum2Poles(k=k, ells=(0, 2)))().shape[0] == 2

    def test_tracer_correlation(self):
        """TNSTracerCorrelation2Poles: shape and finite values."""
        from desilike.theories.galaxy_clustering import TNSTracerCorrelation2Poles
        s = np.linspace(50., 150., 50)
        theory = TNSTracerCorrelation2Poles(s=s)
        result = _compile(theory)()
        _check(result, 'TNSTracerCorrelation2Poles')
        assert result.shape == (len(theory.ells), len(s))

    def test_emulated(self):
        """TNSPTSpectrum2Poles emulated as pt= in spectrum and correlation."""
        from desilike import compile
        from desilike.theories.galaxy_clustering import (
            TNSPTSpectrum2Poles, TNSTracerSpectrum2Poles, TNSTracerCorrelation2Poles,
        )
        k = np.linspace(0.02, 0.3, 20)
        ells = (0, 2)

        pipe_exact = compile(TNSTracerSpectrum2Poles(k=k, ells=ells))
        theory_emu = TNSTracerSpectrum2Poles(k=k, ells=ells, pt=TNSPTSpectrum2Poles(k=k, ells=ells))
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(TNSTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = TNSTracerCorrelation2Poles(s=s, ells=ells)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1')


# ── LPT velocileptors ─────────────────────────────────────────────────────────

class TestLPTVelocileptors:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        pytest.importorskip('velocileptors')

    def test_matter_spectrum(self):
        """LPTVelocileptorsPTSpectrum2Poles: table shape."""
        from desilike.theories.galaxy_clustering import LPTVelocileptorsPTSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = LPTVelocileptorsPTSpectrum2Poles(k=k)
        _compile(theory)()
        assert np.asarray(theory.table).shape[0] == len(theory.ells)

    def test_tracer_spectrum(self):
        """LPTVelocileptorsTracerSpectrum2Poles: shape, parameter sensitivity and
        ells=(0, 2) edge case in both physical and standard bases (b1p / b1 are the
        respective user-facing parameter names; the attribute is ``self.b1`` in both)."""
        from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)

        theory = LPTVelocileptorsTracerSpectrum2Poles(k=k)  # physical basis
        run = _compile(theory)
        base = run()
        _check(base, 'LPTVelocileptorsTracer (physical)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'LPTVelocileptorsTracer (physical)', b1p=2.0)

        theory_std = LPTVelocileptorsTracerSpectrum2Poles(k=k, prior_basis='standard')
        run_std = _compile(theory_std)
        base_std = run_std()
        _check(base_std, 'LPTVelocileptorsTracer (standard)')
        _check_sensitivity(run_std, base_std, 'LPTVelocileptorsTracer (standard)', b1=2.0)

        assert _compile(LPTVelocileptorsTracerSpectrum2Poles(k=k, ells=(0, 2)))().shape[0] == 2

    def test_tracer_presets(self):
        """LPTVelocileptorsTracerSpectrum2Poles: LRG/ELG/QSO fsat/sigv settings run."""
        from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerSpectrum2Poles
        from desilike.theories.galaxy_clustering.full_shape import get_physical_stochastic_settings
        k = np.linspace(0.02, 0.3, 60)
        for tracer in ['LRG', 'ELG', 'QSO']:
            settings = get_physical_stochastic_settings(tracer)
            theory = LPTVelocileptorsTracerSpectrum2Poles(k=k, **settings)
            _check(_compile(theory)(), f'LPTVelocileptorsTracer tracer={tracer}')

    def test_tracer_correlation(self):
        """LPTVelocileptorsTracerCorrelation2Poles: shape in both bases."""
        from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerCorrelation2Poles
        s = np.linspace(50., 150., 50)
        theory = LPTVelocileptorsTracerCorrelation2Poles(s=s)
        result = _compile(theory)()
        _check(result, 'LPTVelocileptorsTracerCorrelation2Poles')
        assert result.shape == (len(theory.ells), len(s))

        theory_std = LPTVelocileptorsTracerCorrelation2Poles(s=s, prior_basis='standard')
        _check(_compile(theory_std)(), 'LPTVelocileptorsTracerCorrelation2Poles (standard)')

    def test_emulated(self):
        """LPTVelocileptorsPTSpectrum2Poles emulated as pt= in spectrum and correlation."""
        from desilike import compile
        from desilike.theories.galaxy_clustering import (
            LPTVelocileptorsPTSpectrum2Poles,
            LPTVelocileptorsTracerSpectrum2Poles, LPTVelocileptorsTracerCorrelation2Poles,
        )
        k = np.linspace(0.02, 0.3, 20)
        ells = (0, 2)

        pipe_exact = compile(LPTVelocileptorsTracerSpectrum2Poles(k=k, ells=ells))
        theory_emu = LPTVelocileptorsTracerSpectrum2Poles(
            k=k, ells=ells, pt=LPTVelocileptorsPTSpectrum2Poles(k=k, ells=ells))
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1p')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(LPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = LPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1p')


# ── REPT velocileptors ────────────────────────────────────────────────────────

class TestREPTVelocileptors:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        pytest.importorskip('velocileptors')

    def test_matter_spectrum(self):
        """REPTVelocileptorsPTSpectrum2Poles: table shape."""
        from desilike.theories.galaxy_clustering import REPTVelocileptorsPTSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = REPTVelocileptorsPTSpectrum2Poles(k=k)
        _compile(theory)()
        assert np.asarray(theory.table).shape[0] == len(theory.ells)

    def test_tracer_spectrum(self):
        """REPTVelocileptorsTracerSpectrum2Poles: shape and parameter sensitivity
        in both physical and standard bases."""
        from desilike.theories.galaxy_clustering import REPTVelocileptorsTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)

        theory = REPTVelocileptorsTracerSpectrum2Poles(k=k)  # physical basis
        run = _compile(theory)
        base = run()
        _check(base, 'REPTVelocileptorsTracer (physical)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'REPTVelocileptorsTracer (physical)', b1p=2.0)

        theory_std = REPTVelocileptorsTracerSpectrum2Poles(k=k, prior_basis='standard')
        run_std = _compile(theory_std)
        base_std = run_std()
        _check(base_std, 'REPTVelocileptorsTracer (standard)')
        _check_sensitivity(run_std, base_std, 'REPTVelocileptorsTracer (standard)', b1=2.0)

    def test_tracer_correlation(self):
        """REPTVelocileptorsTracerCorrelation2Poles: shape and finite values."""
        from desilike.theories.galaxy_clustering import REPTVelocileptorsTracerCorrelation2Poles
        s = np.linspace(50., 150., 50)
        theory = REPTVelocileptorsTracerCorrelation2Poles(s=s)
        result = _compile(theory)()
        _check(result, 'REPTVelocileptorsTracerCorrelation2Poles')
        assert result.shape == (len(theory.ells), len(s))

    def test_emulated(self):
        """REPTVelocileptorsPTSpectrum2Poles emulated as pt= in spectrum and correlation."""
        from desilike import compile
        from desilike.theories.galaxy_clustering import (
            REPTVelocileptorsPTSpectrum2Poles,
            REPTVelocileptorsTracerSpectrum2Poles, REPTVelocileptorsTracerCorrelation2Poles,
        )
        k = np.linspace(0.02, 0.3, 20)
        ells = (0, 2)

        pipe_exact = compile(REPTVelocileptorsTracerSpectrum2Poles(k=k, ells=ells))
        theory_emu = REPTVelocileptorsTracerSpectrum2Poles(
            k=k, ells=ells, pt=REPTVelocileptorsPTSpectrum2Poles(k=k, ells=ells))
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1p')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(REPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = REPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1p')


# ── PyBird ────────────────────────────────────────────────────────────────────

class TestPyBird:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        pytest.importorskip('pybird')

    def test_matter_spectrum(self):
        """PyBirdPTSpectrum2Poles: runs without error."""
        from desilike.theories.galaxy_clustering import PyBirdPTSpectrum2Poles
        _compile(PyBirdPTSpectrum2Poles(k=np.linspace(0.02, 0.3, 60)))()

    def test_tracer_spectrum(self):
        """PyBirdTracerSpectrum2Poles: shape, b1 sensitivity (eftoflss),
        plus westcoast/eastcoast bases run."""
        from desilike.theories.galaxy_clustering import PyBirdTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = PyBirdTracerSpectrum2Poles(k=k)
        run = _compile(theory)
        base = run()
        _check(base, 'PyBirdTracerSpectrum2Poles')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'PyBirdTracerSpectrum2Poles', b1=2.0)

        for eft_basis in ['westcoast', 'eastcoast']:
            theory = PyBirdTracerSpectrum2Poles(k=k, eft_basis=eft_basis)
            _check(_compile(theory)(), f'PyBirdTracerSpectrum2Poles ({eft_basis})')

    def test_tracer_correlation(self):
        """PyBirdTracerCorrelation2Poles: shape and b1 sensitivity."""
        from desilike.theories.galaxy_clustering import (
            PyBirdPTCorrelation2Poles, PyBirdTracerCorrelation2Poles,
        )
        s = np.linspace(50., 150., 50)
        _compile(PyBirdPTCorrelation2Poles(s=s))()  # smoke-test matter correlation

        theory = PyBirdTracerCorrelation2Poles(s=s)
        run = _compile(theory)
        base = run()
        _check(base, 'PyBirdTracerCorrelation2Poles')
        assert base.shape == (len(theory.ells), len(s))
        _check_sensitivity(run, base, 'PyBirdTracerCorrelation2Poles', b1=2.0)

    def test_emulated(self):
        """PyBirdPTSpectrum2Poles / PyBirdPTCorrelation2Poles emulated in spectrum and correlation."""
        from desilike import compile
        from desilike.theories.galaxy_clustering import (
            PyBirdPTSpectrum2Poles, PyBirdTracerSpectrum2Poles,
            PyBirdPTCorrelation2Poles, PyBirdTracerCorrelation2Poles,
        )
        k = np.linspace(0.02, 0.3, 20)
        ells = (0, 2)

        pipe_exact = compile(PyBirdTracerSpectrum2Poles(k=k, ells=ells))
        theory_emu = PyBirdTracerSpectrum2Poles(k=k, ells=ells, pt=PyBirdPTSpectrum2Poles(k=k, ells=ells))
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(PyBirdTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = PyBirdTracerCorrelation2Poles(s=s, ells=ells, pt=PyBirdPTCorrelation2Poles(s=s, ells=ells))
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1')


# ── FOLPS ─────────────────────────────────────────────────────────────────────

class TestFOLPS:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        pytest.importorskip('folps')

    def test_matter_spectrum(self):
        """FOLPSPTSpectrum2Poles: table shape."""
        from desilike.theories.galaxy_clustering import FOLPSPTSpectrum2Poles
        theory = FOLPSPTSpectrum2Poles(k=np.linspace(0.02, 0.3, 60))
        _compile(theory)()
        assert np.asarray(theory.table[0]).shape[0] > 0

    def test_tracer_spectrum(self):
        """FOLPSTracerSpectrum2Poles: shape, parameter sensitivity and ells=(0, 2)
        edge case across all four prior_basis options."""
        from desilike.theories.galaxy_clustering import FOLPSTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)

        # Default physical_aap basis (p-suffix parameters).
        theory = FOLPSTracerSpectrum2Poles(k=k)
        run = _compile(theory)
        base = run()
        _check(base, 'FOLPSTracerSpectrum2Poles (physical_aap)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'FOLPSTracerSpectrum2Poles (physical_aap)', b1p=2.0)

        # Standard basis (no-suffix parameters).
        theory_std = FOLPSTracerSpectrum2Poles(k=k, prior_basis='standard')
        run_std = _compile(theory_std)
        base_std = run_std()
        _check(base_std, 'FOLPSTracerSpectrum2Poles (standard)')
        _check_sensitivity(run_std, base_std, 'FOLPSTracerSpectrum2Poles (standard)', b1=2.0)

        # Remaining physical variants (all use p-suffix parameters).
        for prior_basis in ['physical', 'tcm_chudaykin_aap']:
            theory_pb = FOLPSTracerSpectrum2Poles(k=k, prior_basis=prior_basis)
            run_pb = _compile(theory_pb)
            base_pb = run_pb()
            _check(base_pb, f'FOLPSTracerSpectrum2Poles ({prior_basis})')
            _check_sensitivity(run_pb, base_pb, f'FOLPSTracerSpectrum2Poles ({prior_basis})', b1p=2.0)

        # fsat / sigv passed directly (e.g. the output of get_physical_stochastic_settings).
        from desilike.theories.galaxy_clustering.full_shape import get_physical_stochastic_settings
        settings = get_physical_stochastic_settings('LRG')
        theory_lrg = FOLPSTracerSpectrum2Poles(k=k, prior_basis='physical_aap', **settings)
        _check(_compile(theory_lrg)(), 'FOLPSTracerSpectrum2Poles (LRG fsat/sigv)')

        assert _compile(FOLPSTracerSpectrum2Poles(k=k, ells=(0, 2)))().shape[0] == 2

    def test_tracer_correlation(self):
        """FOLPSTracerCorrelation2Poles: shape in both bases."""
        from desilike.theories.galaxy_clustering import FOLPSTracerCorrelation2Poles
        s = np.linspace(50., 150., 50)
        theory = FOLPSTracerCorrelation2Poles(s=s)
        result = _compile(theory)()
        _check(result, 'FOLPSTracerCorrelation2Poles')
        assert result.shape == (len(theory.ells), len(s))

        theory_std = FOLPSTracerCorrelation2Poles(s=s, prior_basis='standard')
        _check(_compile(theory_std)(), 'FOLPSTracerCorrelation2Poles (standard)')

    def test_tracer_bispectrum(self):
        """FOLPSTracerSpectrum3Poles: shape, parameter sensitivity and prior_basis variants."""
        from desilike.theories.galaxy_clustering import FOLPSTracerSpectrum3Poles

        k = np.column_stack([np.linspace(0.01, 0.1, 11)] * 2)  # diagonal (k1, k2) pairs

        # Default physical_aap basis.
        theory = FOLPSTracerSpectrum3Poles(k=k)
        run = _compile(theory)
        base = run()
        _check(base, 'FOLPSTracerSpectrum3Poles (physical_aap)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'FOLPSTracerSpectrum3Poles (physical_aap)', b1p=2.0)

        # Standard basis (b1 defaults to 2.0; use 3.0 for sensitivity check).
        theory_std = FOLPSTracerSpectrum3Poles(k=k, prior_basis='standard')
        run_std = _compile(theory_std)
        base_std = run_std()
        _check(base_std, 'FOLPSTracerSpectrum3Poles (standard)')
        _check_sensitivity(run_std, base_std, 'FOLPSTracerSpectrum3Poles (standard)', b1=3.0)

        # Remaining physical variants.
        for prior_basis in ['physical', 'tcm_chudaykin_aap']:
            theory_pb = FOLPSTracerSpectrum3Poles(k=k, prior_basis=prior_basis)
            _check(_compile(theory_pb)(), f'FOLPSTracerSpectrum3Poles ({prior_basis})')

        # Custom ells subset.
        theory_ells = FOLPSTracerSpectrum3Poles(k=k, ells=((0, 0, 0),))
        assert _compile(theory_ells)().shape[0] == 1

    def test_emulated(self):
        """FOLPSPTSpectrum2Poles emulated as pt= in spectrum and correlation."""
        from desilike import compile
        from desilike.theories.galaxy_clustering import (
            FOLPSPTSpectrum2Poles,
            FOLPSTracerSpectrum2Poles, FOLPSTracerCorrelation2Poles,
        )
        k = np.linspace(0.02, 0.3, 20)
        ells = (0, 2)

        pipe_exact = compile(FOLPSTracerSpectrum2Poles(k=k, ells=ells))
        theory_emu = FOLPSTracerSpectrum2Poles(k=k, ells=ells, pt=FOLPSPTSpectrum2Poles(k=k, ells=ells))
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1p')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(FOLPSTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = FOLPSTracerCorrelation2Poles(s=s, ells=ells)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1p')
