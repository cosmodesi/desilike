"""Tests for full-shape theories."""

import numpy as np
import jax
import pytest



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

    Passing ``_jit=True`` runs the same call through a ``jax.jit``-compiled
    version of the pipeline instead (built lazily, once, on first use) -- this
    catches tracer-incompatible code (e.g. stray ``np.*`` calls on traced
    values) that only surfaces under tracing.
    """
    from desilike.base import compile, params
    pipe = compile(theory)
    defaults = {par.name: par._value for par in params(theory)}
    pipe_jit = {}  # lazy cache, built on first _jit=True call

    def run(_jit=False, **overrides):
        if _jit:
            if 'pipe' not in pipe_jit:
                pipe_jit['pipe'] = jax.jit(pipe)
            return np.asarray(pipe_jit['pipe']({**defaults, **overrides}))
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
    """Assert evaluating *run* with *override* moves away from *baseline*.

    Also re-evaluates under ``jax.jit`` (via ``run(_jit=True, **override)``) and checks
    it agrees with the eager result, catching tracer-incompatible code paths.
    """
    (param_name, value), = override.items()
    result = run(**override)
    assert not np.allclose(baseline, result), f"{name}: result invariant to {param_name}"
    jit_result = run(_jit=True, **override)
    np.testing.assert_allclose(jit_result, result, rtol=1e-6, atol=1e-8,
                               err_msg=f"{name}: jit result differs from eager for {param_name}")


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
        _check_sensitivity(run, base, 'KaiserTracerSpectrum2Poles', logA=2.0)
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


# ── JAXEffort ────────────────────────────────────────────────────────────────

class TestJAXEffort:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        pytest.importorskip('jaxeffort')

    def test_tracer_spectrum_standard(self):
        """JAXEffortTracerSpectrum2Poles standard basis: shape, finite output, b1 sensitivity."""
        from desilike.theories.galaxy_clustering.full_shape import JAXEffortTracerSpectrum2Poles
        k = np.linspace(0.01, 0.2, 20)
        theory = JAXEffortTracerSpectrum2Poles(k=k, ells=(0, 2, 4))
        run = _compile(theory)
        base = run()
        _check(base, 'JAXEffortTracerSpectrum2Poles (standard)')
        assert base.shape == (3, len(k))
        _check_sensitivity(run, base, 'JAXEffortTracerSpectrum2Poles (standard)', b1=3.0)
        _check_sensitivity(run, base, 'JAXEffortTracerSpectrum2Poles (standard)', logA=2.5)

    def test_tracer_spectrum_physical(self):
        """JAXEffortTracerSpectrum2Poles physical basis: shape, finite output, b1p sensitivity."""
        from desilike.theories.galaxy_clustering.full_shape import JAXEffortTracerSpectrum2Poles
        k = np.linspace(0.01, 0.2, 20)
        theory = JAXEffortTracerSpectrum2Poles(k=k, ells=(0, 2, 4), prior_basis='physical')
        run = _compile(theory)
        base = run()
        _check(base, 'JAXEffortTracerSpectrum2Poles (physical)')
        assert base.shape == (3, len(k))
        _check_sensitivity(run, base, 'JAXEffortTracerSpectrum2Poles (physical)', b1p=3.0)

    def test_tracer_presets(self):
        """JAXEffortTracerSpectrum2Poles: LRG/ELG/QSO fsat/sigv settings run without error."""
        from desilike.theories.galaxy_clustering.full_shape import (
            JAXEffortTracerSpectrum2Poles, get_physical_stochastic_settings,
        )
        k = np.linspace(0.01, 0.2, 20)
        for tracer in ['LRG', 'ELG', 'QSO']:
            settings = get_physical_stochastic_settings(tracer)
            theory = JAXEffortTracerSpectrum2Poles(k=k, **settings)
            _check(_compile(theory)(), f'JAXEffortTracerSpectrum2Poles tracer={tracer}')

    def test_ells_subset(self):
        """JAXEffortTracerSpectrum2Poles: ells=(0, 2) gives the right output shape."""
        from desilike.theories.galaxy_clustering.full_shape import JAXEffortTracerSpectrum2Poles
        k = np.linspace(0.01, 0.2, 20)
        assert _compile(JAXEffortTracerSpectrum2Poles(k=k, ells=(0, 2)))().shape[0] == 2


# ── COMET ─────────────────────────────────────────────────────────────────────

class TestCOMET:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        pytest.importorskip('comet')

    def test_matter_spectrum(self):
        """COMETPTSpectrum2Poles: table has shape (ndiagrams, nells, nk) with finite values."""
        from desilike.theories.galaxy_clustering.full_shape import COMETPTSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        theory = COMETPTSpectrum2Poles(k=k)
        _compile(theory)()
        table = np.asarray(theory.table)
        assert table.shape == (len(COMETPTSpectrum2Poles._diagrams), len(theory.ells), len(k)), \
            f'table shape mismatch: {table.shape}'
        assert np.isfinite(table).all(), 'COMETPTSpectrum2Poles: non-finite table'

    def test_tracer_spectrum(self):
        """COMETTracerSpectrum2Poles: shape, sensitivity, and all bias/counterterm basis variants."""
        from desilike.theories.galaxy_clustering.full_shape import COMETTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)

        # Default EggScoSmi+Comet basis.
        theory = COMETTracerSpectrum2Poles(k=k)
        run = _compile(theory)
        base = run()
        _check(base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)', logA=2.5)
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)', b1=2.0)

        # Other bias bases (each with Comet counterterms); b1t for AmiGleKok.
        for prior_basis, sensitivity_param in [
            ('AssBauGre+Comet', 'b1'),
            ('AmiGleKok+Comet', 'b1t'),
            ('DESI+Comet',      'b1'),
            ('physical',        'b1p'),
        ]:
            theory_bb = COMETTracerSpectrum2Poles(k=k, prior_basis=prior_basis)
            run_bb = _compile(theory_bb)
            base_bb = run_bb()
            _check(base_bb, f'COMETTracerSpectrum2Poles ({prior_basis})')
            _check_sensitivity(run_bb, base_bb, f'COMETTracerSpectrum2Poles ({prior_basis})', **{sensitivity_param: 2.0})

        # Other counterterm bases (each with EggScoSmi bias).
        for ct_basis in ['ClassPT', 'PBJ', 'DESIct']:
            theory_ct = COMETTracerSpectrum2Poles(k=k, prior_basis=f'EggScoSmi+{ct_basis}')
            _check(_compile(theory_ct)(), f'COMETTracerSpectrum2Poles (EggScoSmi+{ct_basis})')

        # ells subset.
        assert _compile(COMETTracerSpectrum2Poles(k=k, ells=(0, 2)))().shape[0] == 2

    def test_tracer_bispectrum(self):
        """COMETTracerSpectrum3Poles: shape, finite output, b1 sensitivity."""
        from desilike.theories.galaxy_clustering.full_shape import (
            COMETTracerSpectrum3Poles, COMETPTSpectrum3Poles,
        )
        k = np.column_stack([np.linspace(0.02, 0.1, 11)] * 2)  # diagonal (k1, k2) pairs

        theory = COMETTracerSpectrum3Poles(k=k)
        run = _compile(theory)
        base = run()
        _check(base, 'COMETTracerSpectrum3Poles (EggScoSmi+Comet)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles (EggScoSmi+Comet)', logA=2.5)
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles (EggScoSmi+Comet)', b1=2.0)


def test_jit():

    from desilike import compile, get_params
    from desilike.theories import CosmoprimoCosmology
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template, FOLPSTracerSpectrum2Poles
    k = np.linspace(0.02, 0.3, 20)
    ells = (0, 2)
    for engine in ['camb', 'eisenstein_hu']:
        cosmo = CosmoprimoCosmology(engine=engine)
        template = DirectSpectrum2Template(cosmo=cosmo, z=1.)
        pipe = compile(FOLPSTracerSpectrum2Poles(k=k, ells=ells, template=template))

        pipe_jit = jax.jit(pipe)
        for i in range(3):
            params = {param.name: param.prior.sample(key=jax.random.key(42 + i)) for param in get_params(cosmo).select(varied=True)}
            poles = pipe(params)
            poles_jit = pipe_jit(params)
            assert np.allclose(poles_jit, poles)



def test_compile_input():
    """
    Demonstrates ``compile(root, input=fn)``: feed pre-computed cosmological results
    from an external pipeline into a theory pipeline via the ``input`` callable.

    Design
    ------
    ``cosmo_ext`` (CosmoprimoCosmology) runs the real solver and exposes its
    computed arrays as JAX leaves via ``tree_flatten``.

    ``cosmo`` (PrimordialCosmology) is a lightweight proxy in the theory graph: its
    ``__call__`` is a no-op when ``_results`` is already populated, so it acts as a
    pass-through holder for externally provided results.

    ``pipe_ext`` compiles ``cosmo_ext`` and returns its leaves on each call.
    ``pipe`` compiles the theory with an ``input`` callable that injects the
    pre-computed cosmo results into the proxy for side-effects only.
    The parameter dict is passed as the second positional argument::

        cosmo_leaves = pipe_ext(cosmo_params)          # run external cosmo
        poles = pipe(cosmo_leaves, bias_params)        # inject + theory params
    """
    from desilike.base import compile, get_params
    from desilike.theories import PrimordialCosmology, CosmoprimoCosmology
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template, KaiserTracerSpectrum2Poles

    k = np.linspace(0.02, 0.3, 20)

    # cosmo_ext: the real solver (eisenstein_hu is JAX-native, not truly external,
    # but the pattern works identically for camb/class).
    cosmo_ext = CosmoprimoCosmology(engine='eisenstein_hu')

    # cosmo: a lightweight proxy that holds results but does not run a solver.
    # Uses the same Parameter objects as cosmo_ext so naming is consistent.
    cosmo = PrimordialCosmology(params=get_params(cosmo_ext))

    template = DirectSpectrum2Template(cosmo=cosmo, z=1.)
    theory = KaiserTracerSpectrum2Poles(k=k, template=template)

    # Register on cosmo_ext the requirements that the template registered on cosmo.
    # Conversion: cosmo._requirements format  →  add_requirements dict format.
    ext_reqs = {}
    for (method_key, static_items), spec in cosmo._requirements.items():
        if method_key not in ext_reqs:
            ext_reqs[method_key] = []
        for z_val in spec['z']:
            kw = dict(static_items)
            kw['z'] = float(z_val)
            if spec['k'] is not None:
                kw['k'] = spec['k']
            ext_reqs[method_key].append(kw)
    cosmo_ext.add_requirements(ext_reqs)

    # Compile cosmo_ext: output is the flat list of JAX leaves
    # [param_marker, results_0, results_1, ...].
    pipe_ext = compile(cosmo_ext, output=lambda: cosmo_ext.tree_flatten()[0])
    # Capture aux (engine + ordered_specs) after compile so __post_init__ has run.
    _, cosmo_ext_aux = cosmo_ext.tree_flatten()

    # input callable: pure side-effect — injects pre-computed cosmo results into
    # the proxy.  The parameter dict is passed separately as the second arg to pipe.
    def input_fn(cosmo_leaves):
        proxy = PrimordialCosmology.tree_unflatten(cosmo_ext_aux, cosmo_leaves)
        cosmo._results = proxy._results

    pipe = compile(theory, input=input_fn)

    # Separate the compiled params into cosmo vs bias sets.
    ext_param_names = {p.name for p in pipe_ext.params}
    all_defaults = {p.name: p.value for p in pipe.params}
    cosmo_defaults = {name: val for name, val in all_defaults.items() if name in ext_param_names}
    bias_defaults = {name: val for name, val in all_defaults.items() if name not in ext_param_names}

    # Run external cosmo pipeline to get pre-computed leaves.
    cosmo_leaves = pipe_ext(cosmo_defaults)

    # Feed pre-computed cosmo results into the theory pipeline.
    poles = pipe(cosmo_leaves, bias_defaults)
    _check(poles, 'test_compile_input')

    # jax.jit(pipe): input_fn runs as a Python side-effect at trace time; the JAX
    # computation graph then includes the full data-flow from cosmo_leaves to poles.
    poles_jit = jax.jit(pipe)(cosmo_leaves, bias_defaults)
    np.testing.assert_allclose(np.asarray(poles_jit), np.asarray(poles), rtol=1e-5)

    # Cross-check: the same result should come from the monolithic pipeline.
    cosmo_full = CosmoprimoCosmology(engine='eisenstein_hu')
    template_full = DirectSpectrum2Template(cosmo=cosmo_full, z=1.)
    theory_full = KaiserTracerSpectrum2Poles(k=k, template=template_full)
    pipe_full = compile(theory_full)
    ref = np.asarray(pipe_full(all_defaults))
    np.testing.assert_allclose(np.asarray(poles), ref, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(poles_jit), ref, rtol=1e-5)


if __name__ == '__main__':

    test_jit()
    test_compile_input()