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


def _time(fn):
    """Return the wall-clock time (seconds) to call *fn* once."""
    import time
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


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


def _check_sensitivity(run, baseline, name, rtol=1e-6, atol=1e-8, **params_overrides):
    """Assert each param in *params_overrides* individually moves the output away from *baseline*.

    Each param is tested independently: the pipeline is called with only that param shifted,
    so a single failing param is reported cleanly. Also re-evaluates under ``jax.jit`` for each
    and checks agreement with eager, catching tracer-incompatible code paths. *rtol*/*atol*
    default to a tight bit-for-bit-ish bound; loosen for genuinely-compiled (non pure_callback)
    pipelines chaining many GP/spline ops, where jit's XLA fusion can legitimately reorder
    floating-point operations relative to eager dispatch (e.g. COMET, ~1e-6 ULP-level noise).
    """
    for param_name, value in params_overrides.items():
        result = run(**{param_name: value})
        assert not np.allclose(baseline, result), f"{name}: result invariant to {param_name}"
        jit_result = run(_jit=True, **{param_name: value})
        np.testing.assert_allclose(jit_result, result, rtol=rtol, atol=atol,
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
        ells=(0, 2) edge case in both physical and standard bases (b1 is the
        user-facing parameter name in both; physical basis uses narrower priors)."""
        from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)

        theory = LPTVelocileptorsTracerSpectrum2Poles(k=k)  # physical basis
        run = _compile(theory)
        base = run()
        _check(base, 'LPTVelocileptorsTracer (physical)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'LPTVelocileptorsTracer (physical)', b1=2.0)

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
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(LPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = LPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1')


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
        _check_sensitivity(run, base, 'REPTVelocileptorsTracer (physical)', b1=2.0)

        theory_std = REPTVelocileptorsTracerSpectrum2Poles(k=k, prior_basis='standard')
        run_std = _compile(theory_std)
        base_std = run_std()
        _check(base_std, 'REPTVelocileptorsTracer (standard)')
        _check_sensitivity(run_std, base_std, 'REPTVelocileptorsTracer (standard)', b1=2.0)

    def test_fixed_template_recompute_caching(self):
        """REPTVelocileptorsTracerSpectrum2Poles + FixedSpectrum2Template, eager: varying
        only the bias parameter b1 must not re-run the external REPT PT calculator
        (its own params and deps -- FixedSpectrum2Template has none free -- are
        unchanged), only the cheap JAX bias-combination step in the Tracer.

        Note this caching is an eager-only optimization (base.py's _run_graph skips a
        node when its own params are unchanged and no dep was called): jax.jit traces
        and re-embeds every node's pure_callback unconditionally, so a single
        jax.jit(pipe) re-runs REPT on every call regardless of which parameter changed
        -- this test therefore exercises the eager (non-jit) path specifically.
        """
        from desilike.base import compile, params
        from desilike.theories.galaxy_clustering import REPTVelocileptorsTracerSpectrum2Poles, FixedSpectrum2Template

        k = np.linspace(0.02, 0.3, 60)
        theory = REPTVelocileptorsTracerSpectrum2Poles(k=k, template=FixedSpectrum2Template())
        pipe = compile(theory)
        defaults = {p.name: p._value for p in params(theory)}

        base = np.asarray(pipe(defaults))  # first call: also runs REPT
        _check(base, 'REPTVelocileptorsTracer (FixedSpectrum2Template, eager, first call)')
        first_call_time = _time(lambda: pipe(defaults))  # repeat default call: nothing changed at all

        # First time a *different* b1 is seen, the (cheap) bias-combination JAX ops get
        # dispatched for that code path; REPT itself must still be skipped. Treat this as
        # a second warm-up call, then check several more b1 values are all fast.
        _time(lambda: pipe({**defaults, 'b1': 2.0}))

        times, results = [], []
        for b1 in [1.6, 2.4, 3.2]:
            results.append(np.asarray(pipe({**defaults, 'b1': b1})))
            times.append(_time(lambda b1=b1: pipe({**defaults, 'b1': b1})))

        assert not np.allclose(base, results[0]), 'result invariant to b1'
        assert max(times) < 0.3 * first_call_time, (
            f'varying only b1 took {times} (max {max(times):.3f}s), not much faster than '
            f'a repeated default-params call at {first_call_time:.3f}s -- expected the '
            f'external REPT PT calculator (unchanged params/deps) to be skipped, not re-run'
        )

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
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(REPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = REPTVelocileptorsTracerCorrelation2Poles(s=s, ells=ells)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1')


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

        # Default physical_aap basis.
        theory = FOLPSTracerSpectrum2Poles(k=k)
        run = _compile(theory)
        base = run()
        _check(base, 'FOLPSTracerSpectrum2Poles (physical_aap)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'FOLPSTracerSpectrum2Poles (physical_aap)', b1=2.0)

        # Standard basis (different priors, same parameter names).
        theory_std = FOLPSTracerSpectrum2Poles(k=k, prior_basis='standard')
        run_std = _compile(theory_std)
        base_std = run_std()
        _check(base_std, 'FOLPSTracerSpectrum2Poles (standard)')
        _check_sensitivity(run_std, base_std, 'FOLPSTracerSpectrum2Poles (standard)', b1=2.0)

        # Remaining physical variants.
        for prior_basis in ['physical', 'tcm_chudaykin_aap']:
            theory_pb = FOLPSTracerSpectrum2Poles(k=k, prior_basis=prior_basis)
            run_pb = _compile(theory_pb)
            base_pb = run_pb()
            _check(base_pb, f'FOLPSTracerSpectrum2Poles ({prior_basis})')
            _check_sensitivity(run_pb, base_pb, f'FOLPSTracerSpectrum2Poles ({prior_basis})', b1=2.0)

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
        _check_sensitivity(run, base, 'FOLPSTracerSpectrum3Poles (physical_aap)', b1=2.0)

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
        _check_emulator(pipe_exact, _emulate(theory_emu), shift_param='b1')

        s = np.linspace(50., 150., 10)
        pipe_exact = compile(FOLPSTracerCorrelation2Poles(s=s, ells=ells))
        theory_emu = FOLPSTracerCorrelation2Poles(s=s, ells=ells)
        _check_emulator(pipe_exact, _emulate(theory_emu, inner_pt=theory_emu.pt.pt), shift_param='b1')


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
        """JAXEffortTracerSpectrum2Poles physical basis: shape, finite output, b1 sensitivity."""
        from desilike.theories.galaxy_clustering.full_shape import JAXEffortTracerSpectrum2Poles
        k = np.linspace(0.01, 0.2, 20)
        theory = JAXEffortTracerSpectrum2Poles(k=k, ells=(0, 2, 4), prior_basis='physical')
        run = _compile(theory)
        base = run()
        _check(base, 'JAXEffortTracerSpectrum2Poles (physical)')
        assert base.shape == (3, len(k))
        _check_sensitivity(run, base, 'JAXEffortTracerSpectrum2Poles (physical)', b1=3.0)

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

    def test_cosmo_sensitivity(self):
        """COMETTracerSpectrum2Poles/3Poles: jit-vs-eager sensitivity to *cosmological*
        parameters (h, omega_cdm, omega_b, n_s), not just bias -- regression test for
        comet's background/growth machinery (AP via compute_ap_params, the s12/f
        derived via compute_s12_f) under jit. Includes h=0.3, an extreme-but-finite
        value that pushes Om0 = (omega_cdm+omega_b+omega_nu)/h**2 above 1 (so
        Ode0 = 1-Om0 < 0) -- comet/growth.py's growth_factor_lambda used to return NaN
        there (Ode0<=0 breaks its closed form); see comet/ranges.py and
        growth_factor_lambda()'s docstring for the fix."""
        from desilike.theories.galaxy_clustering.full_shape import COMETTracerSpectrum2Poles, COMETTracerSpectrum3Poles
        comet_tol = dict(rtol=2e-3, atol=1e-6)
        cosmo_overrides = [('h', 0.7), ('h', 0.3), ('omega_cdm', 0.13), ('omega_b', 0.0235), ('n_s', 0.98)]

        k = np.linspace(0.02, 0.3, 60)
        theory = COMETTracerSpectrum2Poles(k=k)
        run = _compile(theory)
        base = run()
        for param_name, value in cosmo_overrides:
            result = run(**{param_name: value})
            assert np.isfinite(result).all(), f'COMETTracerSpectrum2Poles ({param_name}={value}): non-finite result'
            _check_sensitivity(run, base, f'COMETTracerSpectrum2Poles ({param_name}={value})',
                               **{param_name: value}, **comet_tol)

        k3 = np.column_stack([np.linspace(0.02, 0.1, 11)] * 2)
        theory3 = COMETTracerSpectrum3Poles(k=k3)
        run3 = _compile(theory3)
        base3 = run3()
        for param_name, value in cosmo_overrides:
            result3 = run3(**{param_name: value})
            assert np.isfinite(result3).all(), f'COMETTracerSpectrum3Poles ({param_name}={value}): non-finite result'
            _check_sensitivity(run3, base3, f'COMETTracerSpectrum3Poles ({param_name}={value})',
                               **{param_name: value}, **comet_tol)

    def test_out_of_training_range(self):
        """Parameters outside comet's training ranges yield NaN poles (both the PT-split and
        pt=False direct paths) instead of narrowed priors: the priors are left untouched, and
        a compile-time warning flags the effective prior truncation."""
        import warnings as _warnings
        from desilike.base import get_params
        from desilike.theories.galaxy_clustering.full_shape import COMETTracerSpectrum2Poles

        k = np.linspace(0.02, 0.3, 20)
        for pt in [None, False]:
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter('always')
                theory = COMETTracerSpectrum2Poles(k=k, pt=pt)
                run = _compile(theory)
                base = run()
            assert any('training range' in str(warning.message) for warning in caught), f'no truncation warning (pt={pt})'
            # priors are no longer narrowed to the emulator training ranges
            prior_limits = get_params(theory)['n_s'].prior.limits
            assert prior_limits[1] > 1.03, prior_limits
            assert np.isfinite(base).all()
            result = run(n_s=1.08)  # outside comet's ns training range (0.9, 1.03)
            assert np.isnan(np.asarray(result)).all(), f'expected all-NaN poles (pt={pt}): {result}'

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
        # COMET is a genuinely-compiled (non pure_callback) GP-emulator pipeline: jit's XLA
        # fusion can reorder floating-point ops relative to eager, giving ~1e-7 relative
        # differences in the GP output (Pk_lin). The bias decomposition then amplifies these
        # by up to ~1000x due to near-cancellation between tree-level and one-loop terms
        # (poles ~5 from terms of order ~400), giving up to ~1e-3 relative error in poles.
        comet_tol = dict(rtol=2e-3, atol=1e-6)
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)', logA=2.5, **comet_tol)
        # All free bias params: b1 (linear), b2/g2/g21 (higher-order). cnlo is fixed=True for VDG_infty.
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)', b1=2.0, b2=0.5, g2=0.5, g21=0.3, **comet_tol)
        # Comet counterterms (c0/c2/c4; cnlo is fixed=True for VDG_infty).
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)', c0=1.0, c2=1.0, c4=1.0, **comet_tol)
        # Stochastic shot-noise params.
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)', NP0=0.1, NP20=0.1, NP22=0.1, **comet_tol)
        # avir (VDG_infty FoG damping): regression test ensuring avir is correctly passed to
        # PX_ell() (which, unlike Pell(), doesn't internally refresh RSD params from its params dict).
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles (EggScoSmi+Comet)', avir=10.0, **comet_tol)

        # Other bias bases (each with Comet counterterms); bias_overrides covers all free bias params,
        # ct_overrides covers all free counterterm + stochastic params.
        # bGam3 (AssBauGre), b3t (AmiGleKok), btd (DESI non-physical) are fixed=True → omitted.
        # cnlo is fixed=True for VDG_infty → omitted. physical uses DESIct (a0/a2/a4) not Comet (c0/c2/c4).
        comet_ct = dict(c0=1.0, c2=1.0, c4=1.0, NP0=0.1, NP20=0.1, NP22=0.1)
        desict_ct = dict(a0=1.0, a2=1.0, a4=1.0, NP0=0.1, NP20=0.1, NP22=0.1)
        for prior_basis, bias_overrides, ct_overrides in [
            ('AssBauGre+Comet', dict(b1=2.0, b2=0.5, bG2=0.5), comet_ct),
            ('AmiGleKok+Comet', dict(b1t=2.0, b2t=0.5, b4t=0.3), comet_ct),
            ('DESI+Comet',      dict(b1=2.0, b2d=0.5, bk2=0.3), comet_ct),
            ('physical',        dict(b1=2.0, b2d=0.5, bk2=0.3, btd=0.2), desict_ct),
        ]:
            theory_bb = COMETTracerSpectrum2Poles(k=k, prior_basis=prior_basis)
            run_bb = _compile(theory_bb)
            base_bb = run_bb()
            _check(base_bb, f'COMETTracerSpectrum2Poles ({prior_basis})')
            _check_sensitivity(run_bb, base_bb, f'COMETTracerSpectrum2Poles ({prior_basis})', **bias_overrides, **comet_tol)
            _check_sensitivity(run_bb, base_bb, f'COMETTracerSpectrum2Poles ({prior_basis})', **ct_overrides, **comet_tol)

        # Other counterterm bases (each with EggScoSmi bias): check all free ct + stochastic params.
        # cnlo/cnlos are fixed=True for VDG_infty → omitted.
        # DESIct non-physical NP20/NP22 omitted: they are not divided by nbar in _get_canonical_params
        # (unlike Comet/ClassPT/PBJ), so NP20=0.1 only produces a ~1e-5 absolute change — below
        # np.allclose's threshold for poles of order 1e4. NP0=0.1 passes because it is h^-3 normalized.
        for ct_basis, ct_overrides in [
            ('ClassPT', dict(b1=2.0, c0s=1.0, c2s=1.0, c4s=1.0, NP0=0.1, NP20s=0.1, NP22s=0.1)),
            ('PBJ',     dict(b1=2.0, c0t=1.0, c2t=1.0, c4t=1.0, NP0=0.1, eps0=0.1, eps2=0.1)),
            ('DESIct',  dict(b1=2.0, a0=1.0, a2=1.0, a4=1.0, NP0=0.1)),
        ]:
            theory_ct = COMETTracerSpectrum2Poles(k=k, prior_basis=f'EggScoSmi+{ct_basis}')
            run_ct = _compile(theory_ct)
            base_ct = run_ct()
            _check(base_ct, f'COMETTracerSpectrum2Poles (EggScoSmi+{ct_basis})')
            _check_sensitivity(run_ct, base_ct, f'COMETTracerSpectrum2Poles (EggScoSmi+{ct_basis})', **ct_overrides, **comet_tol)

        # ells subset.
        assert _compile(COMETTracerSpectrum2Poles(k=k, ells=(0, 2)))().shape[0] == 2

    def test_tracer_spectrum_direct(self):
        """COMETTracerSpectrum2Poles(pt=False): comet's Pell() (monolithic, bias-combined,
        no separate PT calculator) must agree with the default PX_ell()-decomposed path."""
        from desilike.theories.galaxy_clustering.full_shape import COMETTracerSpectrum2Poles
        k = np.linspace(0.02, 0.3, 60)
        comet_tol = dict(rtol=2e-3, atol=1e-6)

        theory = COMETTracerSpectrum2Poles(k=k, pt=False)
        run = _compile(theory)
        base = run()
        _check(base, 'COMETTracerSpectrum2Poles direct (EggScoSmi+Comet)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles direct (EggScoSmi+Comet)', logA=2.5, **comet_tol)
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles direct (EggScoSmi+Comet)', b1=2.0, **comet_tol)
        _check_sensitivity(run, base, 'COMETTracerSpectrum2Poles direct (EggScoSmi+Comet)', avir=10.0, **comet_tol)

        theory_shared = COMETTracerSpectrum2Poles(k=k)
        run_shared = _compile(theory_shared)
        for prior_basis, sensitivity_param, override in [
            ('EggScoSmi+Comet', 'b1', dict(b1=2.0)),
            ('AssBauGre+Comet', 'b1', dict(b1=2.0)),
            ('AmiGleKok+Comet', 'b1t', dict(b1t=2.0)),
            ('DESI+Comet',      'b1', dict(b1=2.0)),
            ('physical_aap',        'b1', dict(b1=2.0)),
            ('EggScoSmi+ClassPT', None, {}),
            ('EggScoSmi+PBJ', None, {}),
            ('EggScoSmi+DESIct', None, {}),
        ]:
            theory_bb = COMETTracerSpectrum2Poles(k=k, prior_basis=prior_basis)
            theory_bb_direct = COMETTracerSpectrum2Poles(k=k, prior_basis=prior_basis, pt=False)
            base_shared = np.asarray(_compile(theory_bb)(**override))
            base_direct = np.asarray(_compile(theory_bb_direct)(**override))
            np.testing.assert_allclose(base_direct, base_shared, rtol=1e-7, atol=1e-8,
                                       err_msg=f'COMETTracerSpectrum2Poles ({prior_basis}): pt=False disagrees with shared PT')

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
        # See test_tracer_spectrum's comment on comet_tol (same XLA-fusion amplification applies).
        comet_tol = dict(rtol=2e-3, atol=1e-6)
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles (EggScoSmi+Comet)', logA=2.5, **comet_tol)
        # All free bias params for EggScoSmi: b1, b2, g2. cnlo is fixed=True for VDG_infty.
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles (EggScoSmi+Comet)', b1=2.0, b2=0.5, g2=0.5, **comet_tol)
        # Bispectrum stochastic params (NP0/NB0/MB0 all enter the coeff vector).
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles (EggScoSmi+Comet)', NP0=0.1, NB0=0.1, MB0=0.1, **comet_tol)
        # avir (VDG_infty FoG damping): regression for BX_ell_Sugi() silently ignoring it.
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles (EggScoSmi+Comet)', avir=5.0, **comet_tol)

    def test_tracer_bispectrum_direct(self):
        """COMETTracerSpectrum3Poles(pt=False): comet's Bell_Sugi() (monolithic,
        bias-combined, no separate PT calculator) must agree with the default
        BX_ell_Sugi()-decomposed path."""
        from desilike.theories.galaxy_clustering.full_shape import COMETTracerSpectrum3Poles
        k = np.column_stack([np.linspace(0.02, 0.1, 11)] * 2)
        comet_tol = dict(rtol=2e-3, atol=1e-6)

        theory = COMETTracerSpectrum3Poles(k=k, pt=False)
        run = _compile(theory)
        base = run()
        _check(base, 'COMETTracerSpectrum3Poles direct (EggScoSmi+Comet)')
        assert base.shape == (len(theory.ells), len(k))
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles direct (EggScoSmi+Comet)', logA=2.5, **comet_tol)
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles direct (EggScoSmi+Comet)', b1=2.0, **comet_tol)
        _check_sensitivity(run, base, 'COMETTracerSpectrum3Poles direct (EggScoSmi+Comet)', avir=5.0, **comet_tol)

        theory_shared = COMETTracerSpectrum3Poles(k=k)
        run_shared = _compile(theory_shared)
        base_shared = np.asarray(run_shared(b1=2.0))
        base_direct = np.asarray(run(b1=2.0))
        np.testing.assert_allclose(base_direct, base_shared, rtol=1e-7, atol=1e-8,
                                   err_msg='COMETTracerSpectrum3Poles: pt=False disagrees with shared PT')

    def test_numpy_backend(self):
        """backend='numpy' produces finite results and agrees with backend='jax' to within ~5%.

        The numpy backend sets _is_external=True so comet is called via jax.pure_callback;
        params are passed as numpy arrays so PTEmu uses the sklearn GP (not the JAX-ported GP),
        guaranteeing a numpy Pk_lin and a proper scipy spline build (with extrapolation_min set).
        The two backends use different GP implementations (sklearn vs JAX port), which naturally
        differ by ~1e-4 in Pk_lin; the bias combination amplifies this by up to ~1000x
        due to near-cancellation, giving up to ~5% relative difference in the final poles.
        """
        from desilike.theories.galaxy_clustering.full_shape import (
            COMETTracerSpectrum2Poles, COMETTracerSpectrum3Poles,
        )
        k2 = np.linspace(0.02, 0.3, 60)
        k3 = np.column_stack([np.linspace(0.02, 0.1, 11)] * 2)

        for Theory, k in [(COMETTracerSpectrum2Poles, k2), (COMETTracerSpectrum3Poles, k3)]:
            for direct in [False, True]:
                pt_arg = False if direct else None  # False = monolithic Pell/Bell_Sugi; None = shared PT table
                name = f'{Theory.__name__}(pt={pt_arg}, backend=...)'
                theory_jax = Theory(k=k, pt=pt_arg, backend='jax')
                theory_np = Theory(k=k, pt=pt_arg, backend='numpy')
                result_jax = np.asarray(_compile(theory_jax)())
                result_np = np.asarray(_compile(theory_np)())
                assert np.isfinite(result_np).all(), f'{name}: non-finite numpy result'
                np.testing.assert_allclose(result_np, result_jax, rtol=0.05, atol=1.0,
                                           err_msg=f'{name}: numpy backend disagrees with jax')


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

    test = TestREPTVelocileptors()
    test.test_fixed_template_jit_timing()
