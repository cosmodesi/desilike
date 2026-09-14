"""Tests for BAO theories."""

import numpy as np
import jax
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_fiducial():
    """Return a cosmoprimo fiducial cosmology that doesn't need CLASS."""
    import cosmoprimo.fiducial as fid
    return fid.DESI(engine='eisenstein_hu')


def _check(result, name=''):
    arr = np.asarray(result)
    assert arr.ndim == 2, f"{name}: expected 2-D result, got shape {arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0, f"{name}: empty result"
    assert np.isfinite(arr).all(), f"{name}: non-finite values"


def _eval(theory, output='poles', **overrides):
    """Compile *theory* and evaluate it at its default parameters (with optional
    *overrides*); return the named output attribute as a numpy array."""
    from desilike.base import build, get_params
    pipe = build(theory)
    values = {par.name: par._value for par in get_params(theory)}
    values.update(overrides)
    pipe(values)
    return np.asarray(getattr(theory, output))


def _varied(theory):
    """Return the list of non-fixed parameters of *theory*."""
    from desilike.base import get_params
    return list(get_params(theory).select(fixed=False))


def _direct_template(**kw):
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template
    return DirectSpectrum2Template(engine='eisenstein_hu', **kw)


# ── BAOSpectrum2Template ──────────────────────────────────────────────────────

class TestBAOSpectrum2Template:

    def test_basic(self):
        """Compile and evaluate; check shapes, positivity, df and AP scaling."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 50)
        tmpl = BAOSpectrum2Template(k=k, z=1., fiducial=fiducial)

        p = get_params(tmpl)
        names = {par.name for par in p}
        assert 'qpar' in names and 'qper' in names and 'df' in names

        pipe = build(tmpl)
        pipe({'qpar': 1., 'qper': 1., 'df': 1.})
        assert tmpl.pk_dd.shape == (50,)
        assert tmpl.pknow_dd.shape == (50,)
        assert np.all(tmpl.pk_dd > 0) and np.all(tmpl.pknow_dd > 0)

        # df scaling
        pipe({'qpar': 1., 'qper': 1., 'df': 2.})
        f2 = float(tmpl.f)
        pipe({'qpar': 1., 'qper': 1., 'df': 1.})
        f1 = float(tmpl.f)
        assert abs(f2 / f1 - 2.) < 1e-10

        # AP scaling of BAO distances
        pipe({'qpar': 1.1, 'qper': 0.9, 'df': 1.})
        assert abs(float(tmpl.DH_over_rd) / tmpl._DH_over_rd_fid - 1.1) < 1e-10
        assert abs(float(tmpl.DM_over_rd) / tmpl._DM_over_rd_fid - 0.9) < 1e-10

    def test_apmodes(self):
        """All AP modes produce no distortion at default parameter values."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        for apmode in ('qparqper', 'qisoqap', 'qiso', 'qap'):
            tmpl = BAOSpectrum2Template(fiducial=fiducial, apmode=apmode)
            p = get_params(tmpl)
            pipe = build(tmpl)
            pipe({par.name: par._value for par in p})
            assert abs(float(tmpl.DH_over_rd) / tmpl._DH_over_rd_fid - 1.) < 1e-10, apmode
            assert abs(float(tmpl.DM_over_rd) / tmpl._DM_over_rd_fid - 1.) < 1e-10, apmode

    def test_only_now(self):
        """only_now replaces pk_dd with pknow_dd."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        tmpl = BAOSpectrum2Template(fiducial=fiducial, only_now=True)
        pipe = build(tmpl)
        pipe({p.name: p._value for p in get_params(tmpl)})
        assert np.allclose(tmpl.pk_dd, tmpl.pknow_dd)


# ── FixedSpectrum2Template ───────────────────────────────────────────────────

class TestFixedSpectrum2Template:

    def test_basic(self):
        """No free parameters; pk_dd/pknow_dd shapes and positivity; qpar=qper=1."""
        from desilike.theories.galaxy_clustering import FixedSpectrum2Template
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 50)
        tmpl = FixedSpectrum2Template(k=k, z=1., fiducial=fiducial)

        assert len(get_params(tmpl)) == 0

        pipe = build(tmpl)
        pipe({})
        assert tmpl.pk_dd.shape == (50,)
        assert tmpl.pknow_dd.shape == (50,)
        assert np.all(tmpl.pk_dd > 0) and np.all(tmpl.pknow_dd > 0)
        assert tmpl.qpar == 1. and tmpl.qper == 1.

        jac, kap, muap = tmpl.ap_k_mu(k, 0.5)
        assert np.allclose(jac, 1.) and np.allclose(kap, k) and np.allclose(muap, 0.5)

    def test_only_now(self):
        """only_now replaces pk_dd with pknow_dd."""
        from desilike.theories.galaxy_clustering import FixedSpectrum2Template
        from desilike.base import build

        fiducial = _make_fiducial()
        tmpl = FixedSpectrum2Template(fiducial=fiducial, only_now=True)
        pipe = build(tmpl)
        pipe({})
        assert np.allclose(tmpl.pk_dd, tmpl.pknow_dd)

    def test_downstream_theory(self):
        """Plugs into a downstream Spectrum2Poles theory like any other template."""
        from desilike.theories.galaxy_clustering import FixedSpectrum2Template, DampedBAOWigglesPTSpectrum2Poles
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 50)
        theory = DampedBAOWigglesPTSpectrum2Poles(k=k, template=FixedSpectrum2Template(fiducial=fiducial), ells=(0, 2))
        pipe = build(theory)
        result = pipe({par.name: par._value for par in get_params(theory)})
        _check(np.asarray(result), 'FixedSpectrum2Template downstream')


# ── DampedBAOWigglesPoles ─────────────────────────────────────────────────────

class TestDampedBAOWigglesPoles:

    def test_spectrum_basic(self):
        """DampedBAOWigglesPTSpectrum2Poles: build and evaluate, check shapes and parameter names."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesPTSpectrum2Poles
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 50)
        theory = DampedBAOWigglesPTSpectrum2Poles(k=k, template=BAOSpectrum2Template(fiducial=fiducial), ells=(0, 2))

        p = get_params(theory)
        names = {par.name for par in p}
        assert {'b1', 'dbeta', 'sigmapar', 'sigmaper', 'qpar'} <= names

        pipe = build(theory)
        pipe({par.name: par._value for par in p})
        assert theory.poles.shape == (2, 50)
        assert np.all(np.isfinite(theory.poles))

    def test_spectrum_models(self):
        """DampedBAOWigglesPTSpectrum2Poles: all model variants run."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesPTSpectrum2Poles
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 30)
        for model in ('standard', 'fix-damping', 'move-all', 'fog-damping'):
            theory = DampedBAOWigglesPTSpectrum2Poles(k=k, template=BAOSpectrum2Template(fiducial=fiducial), model=model)
            pipe = build(theory)
            pipe({par.name: par._value for par in get_params(theory)})
            assert theory.poles.shape == (2, 30), model

    def test_spectrum_reciso(self):
        """DampedBAOWigglesPTSpectrum2Poles: reciso reconstruction runs."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesPTSpectrum2Poles
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        theory = DampedBAOWigglesPTSpectrum2Poles(template=BAOSpectrum2Template(fiducial=fiducial), mode='reciso')
        pipe = build(theory)
        pipe({par.name: par._value for par in get_params(theory)})
        assert np.all(np.isfinite(theory.poles))

    def test_spectrum_templates(self):
        """DampedBAOWigglesPTSpectrum2Poles: BAO, ShapeFit, Direct templates all work."""
        from desilike.theories.galaxy_clustering import (
            DampedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template, ShapeFitSpectrum2Template,
        )
        k = np.linspace(0.02, 0.3, 60)
        for template in [BAOSpectrum2Template(), ShapeFitSpectrum2Template(), _direct_template()]:
            theory = DampedBAOWigglesPTSpectrum2Poles(k=k, template=template)
            result = _eval(theory, 'poles')
            _check(result, 'DampedBAOWigglesPTSpectrum2Poles')
            assert result.shape == (len(theory.ells), len(k))

        # parameter sensitivity
        theory = DampedBAOWigglesPTSpectrum2Poles(k=k)
        for param in _varied(theory):
            lo, hi = np.asarray(param.ref.sample(jax.random.key(0), shape=2))
            r0 = _eval(theory, 'poles', **{param.name: float(lo)})
            r1 = _eval(theory, 'poles', **{param.name: float(hi)})
            _check(r0, 'DampedBAOWigglesPTSpectrum2Poles')
            if not np.isclose(lo, hi):
                assert not np.allclose(r0, r1), f"result invariant to {param.name}"
            break

    def test_tracer_spectrum_basic(self):
        """DampedBAOWigglesTracerSpectrum2Poles: build, evaluate, broadband params present."""
        from desilike.theories.galaxy_clustering import (
            BAOSpectrum2Template, DampedBAOWigglesPTSpectrum2Poles, DampedBAOWigglesTracerSpectrum2Poles,
        )
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 50)
        pt = DampedBAOWigglesPTSpectrum2Poles(k=k, template=BAOSpectrum2Template(fiducial=fiducial))
        tracer = DampedBAOWigglesTracerSpectrum2Poles(k=k, pt=pt, ells=(0, 2))

        p = get_params(tracer)
        names = {par.name for par in p}
        assert 'al0_-3' in names and 'al2_1' in names
        assert 'b1' in names and 'qpar' in names

        pipe = build(tracer)
        param_vals = {par.name: par._value for par in p}
        pipe(param_vals)
        assert tracer.poles.shape == (2, 50)
        assert np.all(np.isfinite(tracer.poles))

        # zero broadband → tracer equals pt
        bb_zero = {n: 0. for n in names if n.startswith('al')}
        pipe({**param_vals, **bb_zero})
        assert np.allclose(tracer.poles, tracer.pt.poles, rtol=1e-12)

    def test_tracer_spectrum_broadband(self):
        """DampedBAOWigglesTracerSpectrum2Poles: custom broadband, template switching, ells, sensitivity."""
        from desilike.theories.galaxy_clustering import (
            DampedBAOWigglesTracerSpectrum2Poles, DampedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template,
        )
        k = np.linspace(0.02, 0.3, 60)

        theory = DampedBAOWigglesTracerSpectrum2Poles(k=k, broadband="power3")
        _check(_eval(theory, 'poles'), 'DampedBAOWigglesTracerSpectrum2Poles broadband')

        for template in [BAOSpectrum2Template()]:
            theory = DampedBAOWigglesTracerSpectrum2Poles(k=k, pt=DampedBAOWigglesPTSpectrum2Poles(k=k, template=template))
            _check(_eval(theory, 'poles'), 'DampedBAOWigglesTracerSpectrum2Poles template')

        theory = DampedBAOWigglesTracerSpectrum2Poles(k=k, ells=(0,))
        assert _eval(theory, 'poles').shape[0] == 1

        theory = DampedBAOWigglesTracerSpectrum2Poles(k=k)
        r0 = _eval(theory, 'poles')
        bb_param = next(p for p in _varied(theory) if p.basename.startswith('al'))
        r1 = _eval(theory, 'poles', **{bb_param.name: 1e-2})
        assert not np.allclose(r0, r1), "broadband param had no effect"

    def test_correlation_basic(self):
        """DampedBAOWigglesPTCorrelation2Poles: build and evaluate, check shapes."""
        from desilike.theories.galaxy_clustering import (
            BAOSpectrum2Template, DampedBAOWigglesPTSpectrum2Poles, DampedBAOWigglesPTCorrelation2Poles,
        )
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        s = np.linspace(50., 150., 51)
        pt = DampedBAOWigglesPTSpectrum2Poles(template=BAOSpectrum2Template(fiducial=fiducial))
        corr = DampedBAOWigglesPTCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

        pipe = build(corr)
        pipe({par.name: par._value for par in get_params(corr)})
        assert corr.poles.shape == (2, 51)
        assert np.all(np.isfinite(corr.poles))

    def test_correlation_templates(self):
        """DampedBAOWigglesPTCorrelation2Poles: BAO and ShapeFit templates work."""
        from desilike.theories.galaxy_clustering import (
            DampedBAOWigglesPTCorrelation2Poles, DampedBAOWigglesPTSpectrum2Poles,
            BAOSpectrum2Template,
        )
        s = np.linspace(50., 150., 50)
        theory = DampedBAOWigglesPTCorrelation2Poles(s=s)
        result = _eval(theory, 'poles')
        _check(result, 'DampedBAOWigglesPTCorrelation2Poles')
        assert result.shape == (len(theory.ells), len(s))
        assert theory.s is not None

        for template in [BAOSpectrum2Template()]:
            theory = DampedBAOWigglesPTCorrelation2Poles(s=s, pt=DampedBAOWigglesPTSpectrum2Poles(template=template))
            _check(_eval(theory, 'poles'), 'DampedBAOWigglesPTCorrelation2Poles template')

        # parameter sensitivity
        theory = DampedBAOWigglesPTCorrelation2Poles(s=s)
        for param in _varied(theory):
            lo, hi = np.asarray(param.ref.sample(jax.random.key(0), shape=2))
            r0 = _eval(theory, 'poles', **{param.name: float(lo)})
            r1 = _eval(theory, 'poles', **{param.name: float(hi)})
            _check(r0, 'DampedBAOWigglesPTCorrelation2Poles')
            if not np.isclose(lo, hi):
                assert not np.allclose(r0, r1), f"result invariant to {param.name}"
            break

    def test_tracer_correlation_basic(self):
        """DampedBAOWigglesTracerCorrelation2Poles: build, evaluate, broadband params, zero-bb identity."""
        from desilike.theories.galaxy_clustering import (
            BAOSpectrum2Template, DampedBAOWigglesPTSpectrum2Poles,
            DampedBAOWigglesPTCorrelation2Poles, DampedBAOWigglesTracerCorrelation2Poles,
        )
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        s = np.linspace(50., 150., 51)
        pt = DampedBAOWigglesPTSpectrum2Poles(template=BAOSpectrum2Template(fiducial=fiducial))
        tracer = DampedBAOWigglesTracerCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

        p = get_params(tracer)
        names = {par.name for par in p}
        assert 'al0_-2' in names and 'al2_1' in names
        assert 'b1' in names and 'qpar' in names

        pipe = build(tracer)
        param_vals = {par.name: par._value for par in p}
        pipe(param_vals)
        assert tracer.poles.shape == (2, 51)
        assert np.all(np.isfinite(tracer.poles))

        # zero broadband → tracer equals bare correlation
        bb_zero = {n: 0. for n in names if n.startswith('al')}
        pipe({**param_vals, **bb_zero})
        # `DampedBAOWigglesPTCorrelation2Poles.__init__` sizes the pt it is handed, which
        # invalidates this graph; we have finished calling it (only `tracer.poles` is read below).
        bare = DampedBAOWigglesPTCorrelation2Poles(s=s, pt=pt, ells=(0, 2))
        bare_pipe = build(bare)
        # the bare correlation has no broadband parameters, and a name a pipeline does not have
        # is an error rather than a silent drop
        bare_pipe({name: value for name, value in param_vals.items() if name in bare_pipe.params})
        assert np.allclose(tracer.poles, bare.poles, rtol=1e-10)

    def test_tracer_correlation_broadband(self):
        """DampedBAOWigglesTracerCorrelation2Poles: custom broadband, ells, sensitivity."""
        from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelation2Poles
        s = np.linspace(50., 150., 50)

        theory = DampedBAOWigglesTracerCorrelation2Poles(s=s, broadband="power3")
        _check(_eval(theory, 'poles'), 'DampedBAOWigglesTracerCorrelation2Poles broadband')

        theory = DampedBAOWigglesTracerCorrelation2Poles(s=s, ells=(0,))
        assert _eval(theory, 'poles').shape[0] == 1

        theory = DampedBAOWigglesTracerCorrelation2Poles(s=s)
        r0 = _eval(theory, 'poles')
        bb_param = next(p for p in _varied(theory) if p.basename.startswith('al'))
        r1 = _eval(theory, 'poles', **{bb_param.name: 1.})
        assert not np.allclose(r0, r1), "broadband param had no effect"


# ── ResummedBAOWigglesPoles ───────────────────────────────────────────────────

class TestResummedBAOWigglesPoles:

    def test_spectrum_basic(self):
        """ResummedBAOWigglesPTSpectrum2Poles: build and evaluate, check shapes and parameter names."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template, ResummedBAOWigglesPTSpectrum2Poles
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 50)
        theory = ResummedBAOWigglesPTSpectrum2Poles(k=k, template=BAOSpectrum2Template(fiducial=fiducial), ells=(0, 2))

        p = get_params(theory)
        names = {par.name for par in p}
        assert {'b1', 'dbeta', 'd', 'qpar'} <= names

        pipe = build(theory)
        pipe({par.name: par._value for par in p})
        assert theory.poles.shape == (2, 50)
        assert np.all(np.isfinite(theory.poles))

    def test_spectrum_modes(self):
        """ResummedBAOWigglesPTSpectrum2Poles: reconstruction modes and model variants run."""
        from desilike.theories.galaxy_clustering import BAOSpectrum2Template, ResummedBAOWigglesPTSpectrum2Poles
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 30)
        for mode in ('', 'recsym', 'reciso'):
            theory = ResummedBAOWigglesPTSpectrum2Poles(k=k, template=BAOSpectrum2Template(fiducial=fiducial), mode=mode)
            pipe = build(theory)
            pipe({par.name: par._value for par in get_params(theory)})
            assert np.all(np.isfinite(theory.poles)), mode

        for model in ('standard', 'move-all', 'fog-damping'):
            theory = ResummedBAOWigglesPTSpectrum2Poles(k=k, template=BAOSpectrum2Template(fiducial=fiducial), model=model)
            pipe = build(theory)
            pipe({par.name: par._value for par in get_params(theory)})
            assert np.all(np.isfinite(theory.poles)), model

    def test_spectrum_templates(self):
        """ResummedBAOWigglesPTSpectrum2Poles: BAO, ShapeFit, Direct templates all work."""
        from desilike.theories.galaxy_clustering import (
            ResummedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template, ShapeFitSpectrum2Template,
        )
        k = np.linspace(0.02, 0.3, 60)
        for template in [BAOSpectrum2Template(), ShapeFitSpectrum2Template(), _direct_template()]:
            theory = ResummedBAOWigglesPTSpectrum2Poles(k=k, template=template)
            result = _eval(theory, 'poles')
            _check(result, 'ResummedBAOWigglesPTSpectrum2Poles')
            assert result.shape == (len(theory.ells), len(k))

        # parameter sensitivity
        theory = ResummedBAOWigglesPTSpectrum2Poles(k=k)
        for param in _varied(theory):
            lo, hi = np.asarray(param.ref.sample(jax.random.key(0), shape=2))
            r0 = _eval(theory, 'poles', **{param.name: float(lo)})
            r1 = _eval(theory, 'poles', **{param.name: float(hi)})
            _check(r0, 'ResummedBAOWigglesPTSpectrum2Poles')
            if not np.isclose(lo, hi):
                assert not np.allclose(r0, r1), f"result invariant to {param.name}"
            break

    def test_tracer_spectrum_basic(self):
        """ResummedBAOWigglesTracerSpectrum2Poles: build and evaluate."""
        from desilike.theories.galaxy_clustering import (
            BAOSpectrum2Template, ResummedBAOWigglesPTSpectrum2Poles, ResummedBAOWigglesTracerSpectrum2Poles,
        )
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        k = np.linspace(0.01, 0.3, 50)
        pt = ResummedBAOWigglesPTSpectrum2Poles(k=k, template=BAOSpectrum2Template(fiducial=fiducial))
        tracer = ResummedBAOWigglesTracerSpectrum2Poles(k=k, pt=pt, ells=(0, 2))

        pipe = build(tracer)
        pipe({par.name: par._value for par in get_params(tracer)})
        assert tracer.poles.shape == (2, 50)
        assert np.all(np.isfinite(tracer.poles))

    def test_tracer_spectrum_broadband(self):
        """ResummedBAOWigglesTracerSpectrum2Poles: custom broadband, template switching, ells, sensitivity."""
        from desilike.theories.galaxy_clustering import (
            ResummedBAOWigglesTracerSpectrum2Poles, ResummedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template,
        )
        k = np.linspace(0.02, 0.3, 60)

        theory = ResummedBAOWigglesTracerSpectrum2Poles(k=k, broadband="power3")
        _check(_eval(theory, 'poles'), 'ResummedBAOWigglesTracerSpectrum2Poles broadband')

        for template in [BAOSpectrum2Template()]:
            theory = ResummedBAOWigglesTracerSpectrum2Poles(k=k, pt=ResummedBAOWigglesPTSpectrum2Poles(k=k, template=template))
            _check(_eval(theory, 'poles'), 'ResummedBAOWigglesTracerSpectrum2Poles template')

        theory = ResummedBAOWigglesTracerSpectrum2Poles(k=k, ells=(0,))
        assert _eval(theory, 'poles').shape[0] == 1

        theory = ResummedBAOWigglesTracerSpectrum2Poles(k=k)
        r0 = _eval(theory, 'poles')
        bb_param = next(p for p in _varied(theory) if p.basename.startswith('al'))
        r1 = _eval(theory, 'poles', **{bb_param.name: 1e-2})
        assert not np.allclose(r0, r1), "broadband param had no effect"

    def test_correlation_basic(self):
        """ResummedBAOWigglesPTCorrelation2Poles: build and evaluate, check shapes."""
        from desilike.theories.galaxy_clustering import (
            BAOSpectrum2Template, ResummedBAOWigglesPTSpectrum2Poles, ResummedBAOWigglesPTCorrelation2Poles,
        )
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        s = np.linspace(50., 150., 51)
        pt = ResummedBAOWigglesPTSpectrum2Poles(template=BAOSpectrum2Template(fiducial=fiducial))
        corr = ResummedBAOWigglesPTCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

        pipe = build(corr)
        pipe({par.name: par._value for par in get_params(corr)})
        assert corr.poles.shape == (2, 51)
        assert np.all(np.isfinite(corr.poles))

    def test_correlation_templates(self):
        """ResummedBAOWigglesPTCorrelation2Poles: BAO template works, parameter sensitivity."""
        from desilike.theories.galaxy_clustering import (
            ResummedBAOWigglesPTCorrelation2Poles, ResummedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template,
        )
        s = np.linspace(50., 150., 50)
        theory = ResummedBAOWigglesPTCorrelation2Poles(s=s)
        result = _eval(theory, 'poles')
        _check(result, 'ResummedBAOWigglesPTCorrelation2Poles')
        assert result.shape == (len(theory.ells), len(s))

        for template in [BAOSpectrum2Template()]:
            theory = ResummedBAOWigglesPTCorrelation2Poles(s=s, pt=ResummedBAOWigglesPTSpectrum2Poles(template=template))
            _check(_eval(theory, 'poles'), 'ResummedBAOWigglesPTCorrelation2Poles template')

        theory = ResummedBAOWigglesPTCorrelation2Poles(s=s)
        for param in _varied(theory):
            lo, hi = np.asarray(param.ref.sample(jax.random.key(0), shape=2))
            r0 = _eval(theory, 'poles', **{param.name: float(lo)})
            r1 = _eval(theory, 'poles', **{param.name: float(hi)})
            _check(r0, 'ResummedBAOWigglesPTCorrelation2Poles')
            if not np.isclose(lo, hi):
                assert not np.allclose(r0, r1), f"result invariant to {param.name}"
            break

    def test_tracer_correlation_basic(self):
        """ResummedBAOWigglesTracerCorrelation2Poles: build and evaluate."""
        from desilike.theories.galaxy_clustering import (
            BAOSpectrum2Template, ResummedBAOWigglesPTSpectrum2Poles, ResummedBAOWigglesTracerCorrelation2Poles,
        )
        from desilike.base import build, get_params

        fiducial = _make_fiducial()
        s = np.linspace(50., 150., 51)
        pt = ResummedBAOWigglesPTSpectrum2Poles(template=BAOSpectrum2Template(fiducial=fiducial))
        tracer = ResummedBAOWigglesTracerCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

        pipe = build(tracer)
        pipe({par.name: par._value for par in get_params(tracer)})
        assert tracer.poles.shape == (2, 51)
        assert np.all(np.isfinite(tracer.poles))

    def test_tracer_correlation_broadband(self):
        """ResummedBAOWigglesTracerCorrelation2Poles: custom broadband, ells, sensitivity."""
        from desilike.theories.galaxy_clustering import ResummedBAOWigglesTracerCorrelation2Poles
        s = np.linspace(50., 150., 50)

        theory = ResummedBAOWigglesTracerCorrelation2Poles(s=s, broadband="power3")
        _check(_eval(theory, 'poles'), 'ResummedBAOWigglesTracerCorrelation2Poles broadband')

        theory = ResummedBAOWigglesTracerCorrelation2Poles(s=s, ells=(0,))
        assert _eval(theory, 'poles').shape[0] == 1

        theory = ResummedBAOWigglesTracerCorrelation2Poles(s=s)
        r0 = _eval(theory, 'poles')
        bb_param = next(p for p in _varied(theory) if p.basename.startswith('al'))
        r1 = _eval(theory, 'poles', **{bb_param.name: 1.})
        assert not np.allclose(r0, r1), "broadband param had no effect"
