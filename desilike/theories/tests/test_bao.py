"""Tests for BAO theories."""

import numpy as np
import pytest


def _make_fiducial():
    """Return a cosmoprimo fiducial cosmology that doesn't need CLASS."""
    import cosmoprimo.fiducial as fid
    return fid.BOSS(engine='eisenstein_hu')


def test_bao_template_basic():
    """BAOSpectrum2Template: compile, evaluate, check shapes and scaling."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 50)
    tmpl = BAOSpectrum2Template(k=k, z=1., fiducial=fiducial)

    # params() without full compile: build_graph runs __post_init__ only.
    p = params(tmpl)
    names = {par.name for par in p}
    assert 'qpar' in names
    assert 'qper' in names
    assert 'df' in names

    pipe = compile(tmpl)
    out = pipe({'qpar': 1., 'qper': 1., 'df': 1.})

    assert tmpl.pk_dd.shape == (50,)
    assert tmpl.pknow_dd.shape == (50,)
    assert np.all(tmpl.pk_dd > 0)
    assert np.all(tmpl.pknow_dd > 0)

    # df scaling
    pipe({'qpar': 1., 'qper': 1., 'df': 2.})
    f_at_2 = float(tmpl.f)
    pipe({'qpar': 1., 'qper': 1., 'df': 1.})
    f_at_1 = float(tmpl.f)
    assert abs(f_at_2 / f_at_1 - 2.) < 1e-10

    # AP scaling of BAO distances
    pipe({'qpar': 1.1, 'qper': 0.9, 'df': 1.})
    assert abs(float(tmpl.DH_over_rd) / tmpl._DH_over_rd_fid - 1.1) < 1e-10
    assert abs(float(tmpl.DM_over_rd) / tmpl._DM_over_rd_fid - 0.9) < 1e-10


def test_bao_template_apmodes():
    """BAOSpectrum2Template: all AP modes produce sensible results."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    for apmode in ('qparqper', 'qisoqap', 'qiso', 'qap'):
        tmpl = BAOSpectrum2Template(fiducial=fiducial, apmode=apmode)
        p = params(tmpl)
        pipe = compile(tmpl)

        # Default parameter values (all ones) -> no distortion.
        param_vals = {par.name: par._value for par in p}
        pipe(param_vals)
        # At default (no distortion), DH/rd should equal fiducial.
        assert abs(float(tmpl.DH_over_rd) / tmpl._DH_over_rd_fid - 1.) < 1e-10, apmode
        assert abs(float(tmpl.DM_over_rd) / tmpl._DM_over_rd_fid - 1.) < 1e-10, apmode


def test_bao_template_only_now():
    """BAOSpectrum2Template: only_now replaces pk_dd with pknow_dd."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    tmpl = BAOSpectrum2Template(fiducial=fiducial, only_now=True)
    pipe = compile(tmpl)
    pipe({p.name: p._value for p in params(tmpl)})
    assert np.allclose(tmpl.pk_dd, tmpl.pknow_dd)


def test_damped_bao_basic():
    """DampedBAOWigglesSpectrum2Poles: compile and evaluate, check shapes."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesSpectrum2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 50)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    theory = DampedBAOWigglesSpectrum2Poles(k=k, template=tmpl, ells=(0, 2))

    p = params(theory)
    param_names = {par.name for par in p}
    assert 'b1' in param_names
    assert 'dbeta' in param_names
    assert 'sigmapar' in param_names
    assert 'sigmaper' in param_names
    assert 'qpar' in param_names  # inherited from template

    pipe = compile(theory)
    param_vals = {par.name: par._value for par in p}
    spectrum = pipe(param_vals)

    assert theory.spectrum.shape == (2, 50)
    assert np.all(np.isfinite(theory.spectrum))


def test_damped_bao_models():
    """DampedBAOWigglesSpectrum2Poles: all model variants run."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesSpectrum2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 30)
    for model in ('standard', 'fix-damping', 'move-all', 'fog-damping'):
        tmpl = BAOSpectrum2Template(fiducial=fiducial)
        theory = DampedBAOWigglesSpectrum2Poles(k=k, template=tmpl, model=model)
        pipe = compile(theory)
        p = params(theory)
        pipe({par.name: par._value for par in p})
        assert theory.spectrum.shape == (2, 30), model


def test_damped_bao_reciso():
    """DampedBAOWigglesSpectrum2Poles: reciso reconstruction runs."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesSpectrum2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 30)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    theory = DampedBAOWigglesSpectrum2Poles(k=k, template=tmpl, mode='reciso')
    pipe = compile(theory)
    p = params(theory)
    pipe({par.name: par._value for par in p})
    assert np.all(np.isfinite(theory.spectrum))


def test_resummed_bao_basic():
    """ResummedBAOWigglesSpectrum2Poles: compile and evaluate, check shapes."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, ResummedBAOWigglesSpectrum2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 50)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    theory = ResummedBAOWigglesSpectrum2Poles(k=k, template=tmpl, ells=(0, 2))

    p = params(theory)
    param_names = {par.name for par in p}
    assert 'b1' in param_names
    assert 'dbeta' in param_names
    assert 'd' in param_names
    assert 'qpar' in param_names  # inherited from template

    pipe = compile(theory)
    param_vals = {par.name: par._value for par in p}
    pipe(param_vals)

    assert theory.spectrum.shape == (2, 50)
    assert np.all(np.isfinite(theory.spectrum))


def test_resummed_bao_modes():
    """ResummedBAOWigglesSpectrum2Poles: reconstruction modes and model variants run."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, ResummedBAOWigglesSpectrum2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 30)
    for mode in ('', 'recsym', 'reciso'):
        tmpl = BAOSpectrum2Template(fiducial=fiducial)
        theory = ResummedBAOWigglesSpectrum2Poles(k=k, template=tmpl, mode=mode)
        pipe = compile(theory)
        p = params(theory)
        pipe({par.name: par._value for par in p})
        assert np.all(np.isfinite(theory.spectrum)), mode

    for model in ('standard', 'move-all', 'fog-damping'):
        tmpl = BAOSpectrum2Template(fiducial=fiducial)
        theory = ResummedBAOWigglesSpectrum2Poles(k=k, template=tmpl, model=model)
        pipe = compile(theory)
        p = params(theory)
        pipe({par.name: par._value for par in p})
        assert np.all(np.isfinite(theory.spectrum)), model


def test_damped_tracer_basic():
    """DampedBAOWigglesTracerSpectrum2Poles: compile, evaluate, check broadband params."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesTracerSpectrum2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 50)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    from desilike.theories.galaxy_clustering import DampedBAOWigglesSpectrum2Poles
    pt = DampedBAOWigglesSpectrum2Poles(k=k, template=tmpl)
    tracer = DampedBAOWigglesTracerSpectrum2Poles(k=k, pt=pt, ells=(0, 2))

    p = params(tracer)
    param_names = {par.name for par in p}
    # Broadband params for ells 0 and 2.
    assert 'al0_-3' in param_names
    assert 'al2_1' in param_names
    # Physics params from pt.
    assert 'b1' in param_names
    assert 'qpar' in param_names

    pipe = compile(tracer)
    param_vals = {par.name: par._value for par in p}
    pipe(param_vals)

    assert tracer.spectrum.shape == (2, 50)
    assert np.all(np.isfinite(tracer.spectrum))

    # At default broadband=0, tracer spectrum equals pt spectrum.
    bb_zero = {n: 0. for n in param_names if n.startswith('al')}
    pipe({**param_vals, **bb_zero})
    assert np.allclose(tracer.spectrum, tracer.pt.spectrum, rtol=1e-12)


def test_resummed_tracer_basic():
    """ResummedBAOWigglesTracerSpectrum2Poles: compile and evaluate."""
    from desilike.theories.galaxy_clustering import (BAOSpectrum2Template, ResummedBAOWigglesSpectrum2Poles,
                                                     ResummedBAOWigglesTracerSpectrum2Poles)
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    k = np.linspace(0.01, 0.3, 50)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    pt = ResummedBAOWigglesSpectrum2Poles(k=k, template=tmpl)
    tracer = ResummedBAOWigglesTracerSpectrum2Poles(k=k, pt=pt, ells=(0, 2))

    pipe = compile(tracer)
    p = params(tracer)
    param_vals = {par.name: par._value for par in p}
    pipe(param_vals)

    assert tracer.spectrum.shape == (2, 50)
    assert np.all(np.isfinite(tracer.spectrum))


def test_damped_corr_basic():
    """DampedBAOWigglesCorrelation2Poles: compile and evaluate, check shapes."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesSpectrum2Poles, DampedBAOWigglesCorrelation2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    s = np.linspace(50., 150., 51)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    pt = DampedBAOWigglesSpectrum2Poles(template=tmpl)
    corr = DampedBAOWigglesCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

    p = params(corr)
    pipe = compile(corr)
    param_vals = {par.name: par._value for par in p}
    pipe(param_vals)

    assert corr.correlation.shape == (2, 51)
    assert np.all(np.isfinite(corr.correlation))


def test_resummed_corr_basic():
    """ResummedBAOWigglesCorrelation2Poles: compile and evaluate, check shapes."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, ResummedBAOWigglesSpectrum2Poles, ResummedBAOWigglesCorrelation2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    s = np.linspace(50., 150., 51)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    pt = ResummedBAOWigglesSpectrum2Poles(template=tmpl)
    corr = ResummedBAOWigglesCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

    p = params(corr)
    pipe = compile(corr)
    param_vals = {par.name: par._value for par in p}
    pipe(param_vals)

    assert corr.correlation.shape == (2, 51)
    assert np.all(np.isfinite(corr.correlation))


def test_damped_tracer_corr_basic():
    """DampedBAOWigglesTracerCorrelation2Poles: compile, evaluate, check broadband params."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, DampedBAOWigglesSpectrum2Poles, DampedBAOWigglesTracerCorrelation2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    s = np.linspace(50., 150., 51)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    pt = DampedBAOWigglesSpectrum2Poles(template=tmpl)
    tracer = DampedBAOWigglesTracerCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

    p = params(tracer)
    param_names = {par.name for par in p}
    assert 'al0_-2' in param_names
    assert 'al2_1' in param_names
    assert 'b1' in param_names
    assert 'qpar' in param_names

    pipe = compile(tracer)
    param_vals = {par.name: par._value for par in p}
    pipe(param_vals)

    assert tracer.correlation.shape == (2, 51)
    assert np.all(np.isfinite(tracer.correlation))

    # At default broadband=0, tracer correlation equals bare correlation.
    bb_zero = {n: 0. for n in param_names if n.startswith('al')}
    pipe({**param_vals, **bb_zero})
    from desilike.theories.galaxy_clustering import DampedBAOWigglesCorrelation2Poles
    bare = DampedBAOWigglesCorrelation2Poles(s=s, pt=pt, ells=(0, 2))
    bare_pipe = compile(bare)
    bare_pipe(param_vals)
    assert np.allclose(tracer.correlation, bare.correlation, rtol=1e-10)


def test_resummed_tracer_corr_basic():
    """ResummedBAOWigglesTracerCorrelation2Poles: compile and evaluate."""
    from desilike.theories.galaxy_clustering import BAOSpectrum2Template, ResummedBAOWigglesSpectrum2Poles, ResummedBAOWigglesTracerCorrelation2Poles
    from desilike.base import compile, params

    fiducial = _make_fiducial()
    s = np.linspace(50., 150., 51)
    tmpl = BAOSpectrum2Template(fiducial=fiducial)
    pt = ResummedBAOWigglesSpectrum2Poles(template=tmpl)
    tracer = ResummedBAOWigglesTracerCorrelation2Poles(s=s, pt=pt, ells=(0, 2))

    pipe = compile(tracer)
    p = params(tracer)
    param_vals = {par.name: par._value for par in p}
    pipe(param_vals)

    assert tracer.correlation.shape == (2, 51)
    assert np.all(np.isfinite(tracer.correlation))


if __name__ == '__main__':

    test_bao_template_basic()
    test_bao_template_apmodes()
    test_bao_template_only_now()
    test_damped_bao_basic()
    test_damped_bao_models()
    test_damped_bao_reciso()
    test_resummed_bao_basic()
    test_resummed_bao_modes()
    test_damped_tracer_basic()
    test_resummed_tracer_basic()
    test_damped_corr_basic()
    test_resummed_corr_basic()
    test_damped_tracer_corr_basic()
    test_resummed_tracer_corr_basic()
