"""Tests for galaxy clustering observables and ObservablesGaussianLikelihood."""

import numpy as np
import pytest
import scipy as sp


def _make_fiducial():
    import cosmoprimo.fiducial as fid
    return fid.BOSS(engine='eisenstein_hu')


def _make_spectrum_theory(k, ells=(0, 2)):
    from desilike.theories.galaxy_clustering import (DampedBAOWigglesPTSpectrum2Poles,
                                                      DampedBAOWigglesTracerSpectrum2Poles,
                                                      BAOSpectrum2Template)
    fiducial = _make_fiducial()
    tmpl = BAOSpectrum2Template(k=k, z=0.5, fiducial=fiducial)
    pt = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=ells)
    return DampedBAOWigglesTracerSpectrum2Poles(k=k, pt=pt, ells=ells)


def _make_correlation_theory(s, ells=(0, 2)):
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelation2Poles, DampedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template
    fiducial = _make_fiducial()
    kin = np.geomspace(1e-4, 0.6, 300)
    tmpl = BAOSpectrum2Template(k=kin, z=0.5, fiducial=fiducial)
    pt = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=ells)
    theory = DampedBAOWigglesTracerCorrelation2Poles(s=s, pt=pt, ells=ells)
    return theory


# ── Spectrum2PolesObservable ────────────────────────────────────────────────

def test_spectrum2poles_no_window():
    """Spectrum2PolesObservable: no window, flattheory matches theory.poles.ravel()."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import compile

    k = np.linspace(0.01, 0.3, 30)
    ells = (0, 2)
    theory = _make_spectrum_theory(k, ells)
    obs = Spectrum2PolesObservable(data=None, theory=theory, k=k, ells=ells)

    pipe = compile(obs)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    assert obs.flattheory.shape == (len(ells) * len(k),)
    assert obs.flatdata.shape == (len(ells) * len(k),)
    np.testing.assert_allclose(np.asarray(obs.flattheory), np.ravel(theory.poles), rtol=1e-10)


def test_spectrum2poles_with_data():
    """Spectrum2PolesObservable: data provided, flatdata set correctly."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import compile

    k = np.linspace(0.01, 0.3, 20)
    ells = (0, 2)
    theory = _make_spectrum_theory(k, ells)
    rng = np.random.default_rng(0)
    data = rng.normal(size=len(ells) * len(k))
    cov = np.diag(np.ones(len(ells) * len(k)) * 0.1)
    obs = Spectrum2PolesObservable(data=data, theory=theory, k=k, ells=ells, covariance=cov)

    pipe = compile(obs)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    np.testing.assert_array_equal(obs.flatdata, data)
    assert obs.covariance.shape == (len(ells) * len(k), len(ells) * len(k))


def test_spectrum2poles_with_window():
    """Spectrum2PolesObservable: window matrix applied correctly."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import compile

    k_data = np.linspace(0.02, 0.25, 15)
    kin = np.linspace(0.01, 0.3, 30)
    ells = (0, 2)
    theory = _make_spectrum_theory(kin, ells)

    n_data = len(ells) * len(k_data)
    n_theory = len(ells) * len(kin)
    rng = np.random.default_rng(1)
    window = rng.normal(size=(n_data, n_theory))

    obs = Spectrum2PolesObservable(data=None, theory=theory, k=k_data, ells=ells,
                                   window=window, kin=kin)

    pipe = compile(obs)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    assert obs.flattheory.shape == (n_data,)
    expected = window @ np.ravel(theory.poles)
    np.testing.assert_allclose(np.asarray(obs.flattheory), expected, rtol=1e-6)


# ── Correlation2PolesObservable ─────────────────────────────────────────────

def test_correlation2poles_no_window():
    """Correlation2PolesObservable: no window, flattheory matches theory.poles.ravel()."""
    from desilike.observables import Correlation2PolesObservable
    from desilike.base import compile

    s = np.linspace(20., 180., 20)
    ells = (0, 2)
    theory = _make_correlation_theory(s, ells)
    obs = Correlation2PolesObservable(data=None, theory=theory, s=s, ells=ells)

    pipe = compile(obs)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    assert obs.flattheory.shape == (len(ells) * len(s),)
    np.testing.assert_allclose(np.asarray(obs.flattheory), np.ravel(theory.poles), rtol=1e-10)


def test_correlation2poles_with_data():
    """Correlation2PolesObservable: data and covariance stored correctly."""
    from desilike.observables import Correlation2PolesObservable
    from desilike.base import compile

    s = np.linspace(20., 180., 15)
    ells = (0, 2)
    theory = _make_correlation_theory(s, ells)
    rng = np.random.default_rng(2)
    n = len(ells) * len(s)
    data = rng.normal(size=n)
    cov_diag = np.ones(n) * 1e-4
    obs = Correlation2PolesObservable(data=data, theory=theory, s=s, ells=ells, covariance=cov_diag)

    pipe = compile(obs)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    np.testing.assert_array_equal(obs.flatdata, data)
    # 1-D covariance should be expanded to diagonal matrix
    assert obs.covariance.shape == (n, n)
    np.testing.assert_allclose(np.diag(obs.covariance), cov_diag)


# ── ObservablesGaussianLikelihood ────────────────────────────────────────────

def test_gaussian_likelihood_basic():
    """ObservablesGaussianLikelihood: logpdf is finite, theory() shape matches flatdata."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile

    k = np.linspace(0.01, 0.3, 20)
    ells = (0, 2)
    theory = _make_spectrum_theory(k, ells)
    n = len(ells) * len(k)
    rng = np.random.default_rng(3)
    data = rng.normal(size=n)
    cov = np.diag(np.ones(n) * 0.1)
    obs = Spectrum2PolesObservable(data=data, theory=theory, k=k, ells=ells)
    like = ObservablesGaussianLikelihood(observables=obs, covariance=cov)

    pipe = compile(like)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    assert np.isfinite(float(like.logpdf))
    assert like.flattheory.shape == (n,)


def test_gaussian_likelihood_obs_covariance():
    """ObservablesGaussianLikelihood: uses observable's covariance when none provided."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile

    k = np.linspace(0.01, 0.3, 20)
    ells = (0, 2)
    theory = _make_spectrum_theory(k, ells)
    n = len(ells) * len(k)
    rng = np.random.default_rng(4)
    data = rng.normal(size=n)
    cov = np.diag(np.ones(n) * 0.05)
    obs = Spectrum2PolesObservable(data=data, theory=theory, k=k, ells=ells, covariance=cov)
    like = ObservablesGaussianLikelihood(observables=obs)

    pipe = compile(like)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    np.testing.assert_allclose(like.precision, np.linalg.inv(cov), rtol=1e-10)
    assert np.isfinite(float(like.logpdf))


def test_gaussian_likelihood_scale_covariance():
    """ObservablesGaussianLikelihood: scale_covariance rescales precision."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile

    k = np.linspace(0.01, 0.3, 20)
    ells = (0, 2)
    theory = _make_spectrum_theory(k, ells)
    n = len(ells) * len(k)
    rng = np.random.default_rng(5)
    data = rng.normal(size=n)
    cov = np.diag(np.ones(n) * 0.1)

    obs1 = Spectrum2PolesObservable(data=data, theory=theory, k=k, ells=ells)
    like1 = ObservablesGaussianLikelihood(observables=obs1, covariance=cov, scale_covariance=1.)

    from desilike.theories.galaxy_clustering import (DampedBAOWigglesPTSpectrum2Poles,
                                                      DampedBAOWigglesTracerSpectrum2Poles,
                                                      BAOSpectrum2Template)
    fiducial = _make_fiducial()
    tmpl2 = BAOSpectrum2Template(k=k, z=0.5, fiducial=fiducial)
    pt2 = DampedBAOWigglesPTSpectrum2Poles(template=tmpl2, ells=ells)
    theory2 = DampedBAOWigglesTracerSpectrum2Poles(k=k, pt=pt2, ells=ells)
    obs2 = Spectrum2PolesObservable(data=data, theory=theory2, k=k, ells=ells)
    like2 = ObservablesGaussianLikelihood(observables=obs2, covariance=cov, scale_covariance=2.)

    pipe1 = compile(like1)
    p1 = {p.name: float(p.value) for p in pipe1.params}
    pipe1(p1)

    pipe2 = compile(like2)
    p2 = {p.name: float(p.value) for p in pipe2.params}
    pipe2(p2)

    np.testing.assert_allclose(like2.precision * 2., like1.precision, rtol=1e-10)


def test_gaussian_likelihood_multi_observable():
    """ObservablesGaussianLikelihood: two observables sharing a template are concatenated correctly."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile
    from desilike.theories.galaxy_clustering import (DampedBAOWigglesPTSpectrum2Poles,
                                                      BAOSpectrum2Template)

    k = np.linspace(0.01, 0.3, 15)
    ells = (0, 2)
    n = len(ells) * len(k)
    fiducial = _make_fiducial()

    # Use the same theory object for both observables so all parameters are shared.
    tmpl = BAOSpectrum2Template(k=k, z=0.5, fiducial=fiducial)
    theory1 = theory2 = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=ells)

    rng = np.random.default_rng(6)
    data1 = rng.normal(size=n)
    data2 = rng.normal(size=n)
    obs1 = Spectrum2PolesObservable(data=data1, theory=theory1, k=k, ells=ells, name='tracer1')
    obs2 = Spectrum2PolesObservable(data=data2, theory=theory2, k=k, ells=ells, name='tracer2')

    cov = np.diag(np.ones(2 * n) * 0.1)
    like = ObservablesGaussianLikelihood(observables=[obs1, obs2], covariance=cov)

    pipe = compile(like)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    assert like.flatdata.shape == (2 * n,)
    assert like.flattheory.shape == (2 * n,)
    assert np.isfinite(float(like.logpdf))
    np.testing.assert_array_equal(like.flatdata[:n], data1)
    np.testing.assert_array_equal(like.flatdata[n:], data2)


def test_gaussian_likelihood_hartlap():
    """ObservablesGaussianLikelihood: Hartlap correction scales precision correctly."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile

    k = np.linspace(0.01, 0.3, 10)
    ells = (0, 2)
    n = len(ells) * len(k)
    theory = _make_spectrum_theory(k, ells)
    rng = np.random.default_rng(7)
    data = rng.normal(size=n)
    cov = np.diag(np.ones(n) * 0.1)
    nobs = 500

    obs = Spectrum2PolesObservable(data=data, theory=theory, k=k, ells=ells)
    like = ObservablesGaussianLikelihood(observables=obs, covariance=cov,
                                         correct_covariance=dict(correction='hartlap', nobs=nobs))

    pipe = compile(like)
    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    hartlap = (nobs - n - 2.) / (nobs - 1.)
    prec_ref = np.linalg.inv(cov) * hartlap
    np.testing.assert_allclose(like.precision, prec_ref, rtol=1e-10)


# ── lsstypes tests ───────────────────────────────────────────────────────────

def _make_spectrum2_lsstypes(size=10):
    import lsstypes as types
    edges = np.linspace(0., 0.2, size + 1)
    edges = np.column_stack([edges[:-1], edges[1:]])
    k = np.mean(edges, axis=-1)
    ells = [0, 2]
    poles = [types.Mesh2SpectrumPole(k=k, num_raw=np.zeros_like(k), k_edges=edges, ell=ell) for ell in ells]
    return types.Mesh2SpectrumPoles(poles)


def _make_spectrum2_window(data, size=20):
    import lsstypes as types
    edges = np.linspace(0., 0.2, size + 1)
    edges = np.column_stack([edges[:-1], edges[1:]])
    k = np.mean(edges, axis=-1)
    ells = data.ells
    theory_poles = [types.Mesh2SpectrumPole(k=k, num_raw=np.zeros_like(k), k_edges=edges, ell=ell) for ell in ells]
    theory = types.ObservableTree(theory_poles, ells=ells, wa_orders=[0] * len(ells))
    window = np.zeros((data.size, theory.size))
    return types.WindowMatrix(observable=data, theory=theory, value=window)


def _make_spectrum3_lsstypes(size=10):
    import lsstypes as types
    edges = np.linspace(0., 0.2, size + 1)
    edges = np.column_stack([edges[:-1], edges[1:]])
    edges = np.concatenate([edges[:, None, :]] * 2, axis=1)
    k = np.mean(edges, axis=-1)
    ells = [(0, 0, 0), (2, 0, 2)]
    poles = [types.Mesh3SpectrumPole(k=k, num_raw=np.zeros_like(k[..., 0]), k_edges=edges, basis='sugiyama-diagonal', ell=ell) for ell in ells]
    return types.Mesh3SpectrumPoles(poles)


def _make_spectrum3_window(data, size=15):
    import lsstypes as types

    def get_grid(*arrays):
        arrays = np.meshgrid(*arrays, indexing='ij')
        return np.column_stack([array.ravel() for array in arrays])

    edges1d = np.linspace(0., 0.2, size + 1)
    edges1d = np.column_stack([edges1d[:-1], edges1d[1:]])
    k1d = np.mean(edges1d, axis=-1)
    edges2d = np.column_stack([get_grid(edges1d[..., axis], edges1d[..., axis])[:, None, :] for axis in range(2)])
    k2d = get_grid(k1d, k1d)
    ells = [(0, 0, 0), (2, 0, 2)]
    theory_poles = [types.Mesh3SpectrumPole(k=k2d, num_raw=np.zeros_like(k2d[..., 0]), k_edges=edges2d, basis='sugiyama', ell=ell) for ell in ells]
    theory = types.Mesh3SpectrumPoles(theory_poles)
    window = np.zeros((data.size, theory.size))
    return types.WindowMatrix(observable=data, theory=theory, value=window)


def _make_correlation2_lsstypes():
    import lsstypes as types

    def get_count(seed=42):
        rng = np.random.RandomState(seed=seed)
        coords = ['s', 'mu']
        edges = [np.linspace(0., 200., 21), np.linspace(-1., 1., 11)]
        edges = [np.column_stack([e[:-1], e[1:]]) for e in edges]
        coords_values = [np.mean(e, axis=-1) for e in edges]
        counts = 1. + rng.uniform(size=tuple(v.size for v in coords_values))
        return types.Count2(counts=counts, norm=np.ones_like(counts),
                            **{coord: val for coord, val in zip(coords, coords_values)},
                            **{f'{coord}_edges': val for coord, val in zip(coords, edges)},
                            coords=coords, attrs=dict(los='x'))

    counts = {label: get_count(seed=i) for i, label in enumerate(['DD', 'DR', 'RD', 'RR'])}
    correlation = types.Count2Correlation(**counts)
    return correlation.project(ells=[0, 2], kw_window=dict(RR=correlation.get('RR')))


def test_spectrum2poles_lsstypes():
    """Spectrum2PolesObservable: lsstypes input gives same flatdata as numpy input."""
    import lsstypes as types
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import compile
    from desilike.theories.galaxy_clustering import DampedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template

    data = _make_spectrum2_lsstypes(size=8)
    window = _make_spectrum2_window(data, size=12)
    covariance = types.CovarianceMatrix(observable=data, value=np.eye(data.size))

    fiducial = _make_fiducial()
    tmpl = BAOSpectrum2Template(k=np.linspace(0.01, 0.2, 12), z=0.5, fiducial=fiducial)
    theory = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=[0, 2])

    # lsstypes path
    obs = Spectrum2PolesObservable(data=data, theory=theory, window=window, covariance=covariance, name='obs1')
    assert obs.flatdata.shape == (data.size,)
    assert obs.covariance.shape == (data.size, data.size)

    # numpy path — must give same flatdata
    obs2 = Spectrum2PolesObservable(
        data=data.value(), theory=theory,
        k=[pole.coords('k') for pole in data], ells=data.ells,
        window=window.value(), kin=next(iter(window.theory)).coords('k'),
        ellsin=window.theory.ells, covariance=covariance.value(), name='obs2')
    np.testing.assert_array_equal(obs2.flatdata, obs.flatdata)

    # both paths compile and produce finite logpdf
    like = Spectrum2PolesObservable(data=data, theory=theory, window=window, covariance=covariance, name='obs1')
    from desilike.likelihoods import ObservablesGaussianLikelihood
    likelihood = ObservablesGaussianLikelihood(observables=[like], covariance=covariance)
    pipe = compile(likelihood)
    pipe({p.name: float(p.value) for p in pipe.params})
    assert np.isfinite(float(likelihood.logpdf))


def test_spectrum3poles_lsstypes():
    """Spectrum3PolesObservable: lsstypes input gives same flatdata as numpy input."""
    import lsstypes as types
    from desilike.observables import Spectrum3PolesObservable
    from desilike.base import compile
    from desilike.theories.galaxy_clustering import DampedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template

    data = _make_spectrum3_lsstypes(size=5)
    window = _make_spectrum3_window(data, size=4)
    covariance = types.CovarianceMatrix(observable=data, value=np.eye(data.size))

    # Use a simple theory that has a 'spectrum' attribute (reuse 2-pole with flat k)
    fiducial = _make_fiducial()
    kin = next(iter(window.theory)).coords('k')
    tmpl = BAOSpectrum2Template(k=kin[..., 0], z=0.5, fiducial=fiducial)
    theory = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=[(0, 0, 0), (2, 0, 2)])

    obs = Spectrum3PolesObservable(data=data, theory=theory, window=window, covariance=covariance, name='obs1')
    assert obs.flatdata.shape == (data.size,)

    obs2 = Spectrum3PolesObservable(
        data=data.value(), theory=theory,
        k=[pole.coords('k') for pole in data],
        ells=data.ells, window=window.value(),
        kin=next(iter(window.theory)).coords('k'), ellsin=window.theory.ells,
        covariance=covariance.value(), name='obs2')
    np.testing.assert_array_equal(obs2.flatdata, obs.flatdata)


def test_correlation2poles_lsstypes():
    """Correlation2PolesObservable: lsstypes input gives same flatdata as numpy input."""
    import lsstypes as types
    from desilike.observables import Correlation2PolesObservable
    from desilike.base import compile
    from desilike.theories.galaxy_clustering import (DampedBAOWigglesTracerCorrelation2Poles,
                                                      DampedBAOWigglesPTSpectrum2Poles,
                                                      BAOSpectrum2Template)

    data, window = _make_correlation2_lsstypes()
    covariance = types.CovarianceMatrix(observable=data, value=np.eye(data.size))

    fiducial = _make_fiducial()
    kin = np.geomspace(1e-4, 0.6, 300)
    sin = next(iter(window.theory)).coords('s')
    tmpl = BAOSpectrum2Template(k=kin, z=0.5, fiducial=fiducial)
    pt = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=window.theory.ells)
    theory = DampedBAOWigglesTracerCorrelation2Poles(s=sin, pt=pt, ells=window.theory.ells)

    obs = Correlation2PolesObservable(data=data, theory=theory, window=window, covariance=covariance, name='obs1')
    assert obs.flatdata.shape == (data.size,)
    assert obs.covariance.shape == (data.size, data.size)

    # numpy path: use obs.window (already matched to data ells) so row count matches
    obs2 = Correlation2PolesObservable(
        data=data.value(), theory=theory,
        s=[pole.coords('s') for pole in data], ells=data.ells,
        window=obs.window.value(), sin=next(iter(obs.window.theory)).coords('s'),
        ellsin=obs.window.theory.ells, covariance=covariance.value(), name='obs2')
    np.testing.assert_array_equal(obs2.flatdata, obs.flatdata)

    like = Correlation2PolesObservable(data=data, theory=theory, window=window, covariance=covariance, name='obs1')
    from desilike.likelihoods import ObservablesGaussianLikelihood
    likelihood = ObservablesGaussianLikelihood(observables=[like], covariance=covariance)
    pipe = compile(likelihood)
    pipe({p.name: float(p.value) for p in pipe.params})
    assert np.isfinite(float(likelihood.logpdf))


def test_gaussian_likelihood_lsstypes_covariance():
    """ObservablesGaussianLikelihood: lsstypes CovarianceMatrix accepted for multi-observable case."""
    import lsstypes as types
    from desilike.observables import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.base import compile
    from desilike.theories.galaxy_clustering import DampedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template

    data = _make_spectrum2_lsstypes(size=8)
    covariance = types.CovarianceMatrix(observable=data, value=np.eye(data.size))

    fiducial = _make_fiducial()
    tmpl = BAOSpectrum2Template(k=np.linspace(0.01, 0.2, 8), z=0.5, fiducial=fiducial)
    theory = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=[0, 2])

    obs1 = Spectrum2PolesObservable(data=data, theory=theory, name='obs1')
    obs2 = Spectrum2PolesObservable(data=data, theory=theory, name='obs2')

    # Single-observable likelihood with lsstypes covariance
    like_single = ObservablesGaussianLikelihood(observables=[obs1], covariance=covariance)
    pipe_single = compile(like_single)
    pipe_single({p.name: float(p.value) for p in pipe_single.params})

    # Multi-observable likelihood with lsstypes block-diagonal covariance
    cov_joint = types.CovarianceMatrix(
        observable=types.ObservableTree([data] * 2, observables=['obs1', 'obs2']),
        value=sp.linalg.block_diag(covariance.value(), covariance.value()))
    like_joint = ObservablesGaussianLikelihood(observables=[obs1, obs2], covariance=cov_joint)
    pipe_joint = compile(like_joint)
    pipe_joint({p.name: float(p.value) for p in pipe_joint.params})

    assert np.isfinite(float(like_single.logpdf))
    assert np.isfinite(float(like_joint.logpdf))
    # Joint logpdf = 2 × single because data=zeros, theory≈signal → residuals same for both obs
    np.testing.assert_allclose(float(like_joint.logpdf), 2. * float(like_single.logpdf), rtol=1e-10)


# ── templates ────────────────────────────────────────────────────────────────

def test_spectrum2poles_templates_scalar():
    """templates: scalar Parameter contribution is added to flattheory and is discoverable in the graph."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import Parameter, compile

    k = np.linspace(0.01, 0.3, 20)
    ells = (0, 2)
    n = len(ells) * len(k)
    theory = _make_spectrum_theory(k, ells)

    rng = np.random.default_rng(20)
    template_array = rng.normal(size=n)
    param = Parameter('my_template', value=2.5, prior={'dist': 'norm', 'loc': 0., 'scale': 1.}, fixed=False)

    obs = Spectrum2PolesObservable(data=None, theory=theory, k=k, ells=ells,
                                   templates=[(param, template_array)])
    assert len(obs.templates) == 1
    assert obs.templates[0][0].name == 'my_template'
    np.testing.assert_array_equal(obs.templates[0][1], template_array)

    pipe = compile(obs)
    assert any(p.name == 'my_template' for p in pipe.params)

    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    expected = obs._window_matrix @ np.ravel(theory.poles) + template_array * pipe_params['my_template']
    np.testing.assert_allclose(np.asarray(obs.flattheory), expected, rtol=1e-6)


def test_spectrum2poles_templates_vector():
    """templates: vector Parameter contribution (shape=(m,)) is added to flattheory."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import Parameter, compile

    k = np.linspace(0.01, 0.3, 20)
    ells = (0, 2)
    n = len(ells) * len(k)
    m = 3
    theory = _make_spectrum_theory(k, ells)

    rng = np.random.default_rng(21)
    template_array = rng.normal(size=(n, m))
    param_value = np.array([1., -0.5, 2.])
    param = Parameter('my_vector_template', value=param_value,
                      prior={'dist': 'norm', 'loc': 0., 'scale': 1.}, shape=(m,), fixed=False)

    obs = Spectrum2PolesObservable(data=None, theory=theory, k=k, ells=ells,
                                   templates=[(param, template_array)])
    assert obs.templates[0][1].shape == (n, m)

    pipe = compile(obs)
    pipe_params = {}
    for p in pipe.params:
        pipe_params[p.name] = np.asarray(p.value) if p.shape else float(p.value)
    pipe(pipe_params)

    expected = obs._window_matrix @ np.ravel(theory.poles) + template_array @ pipe_params['my_vector_template']
    np.testing.assert_allclose(np.asarray(obs.flattheory), expected, rtol=1e-6)


def test_spectrum2poles_templates_dict():
    """templates: dict argument is converted to Parameter, discovered in graph."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import Parameter, compile

    k = np.linspace(0.01, 0.3, 15)
    ells = (0, 2)
    n = len(ells) * len(k)
    theory = _make_spectrum_theory(k, ells)

    rng = np.random.default_rng(22)
    template_array = rng.normal(size=n)
    param_dict = {'name': 'dict_template', 'value': 1.0,
                  'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}, 'fixed': False}

    obs = Spectrum2PolesObservable(data=None, theory=theory, k=k, ells=ells,
                                   templates=[(param_dict, template_array)])
    assert isinstance(obs.templates[0][0], Parameter)
    assert obs.templates[0][0].name == 'dict_template'

    pipe = compile(obs)
    assert any(p.name == 'dict_template' for p in pipe.params)

    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)
    assert obs.flattheory.shape == (n,)


def test_spectrum2poles_templates_shape_error():
    """templates: array with wrong shape raises ValueError at construction."""
    from desilike.observables import Spectrum2PolesObservable
    from desilike.base import Parameter

    k = np.linspace(0.01, 0.3, 10)
    ells = (0, 2)
    n = len(ells) * len(k)
    theory = _make_spectrum_theory(k, ells)
    param = Parameter('bad_template', value=1., fixed=False)

    with pytest.raises(ValueError, match='shape'):
        Spectrum2PolesObservable(data=None, theory=theory, k=k, ells=ells,
                                 templates=[(param, np.ones(n + 1))])


def test_correlation2poles_templates():
    """Correlation2PolesObservable: scalar template contribution is added to flattheory."""
    from desilike.observables import Correlation2PolesObservable
    from desilike.base import Parameter, compile

    s = np.linspace(20., 180., 20)
    ells = (0, 2)
    n = len(ells) * len(s)
    theory = _make_correlation_theory(s, ells)

    rng = np.random.default_rng(23)
    template_array = rng.normal(size=n)
    param = Parameter('xi_template', value=-1.5, prior={'dist': 'norm', 'loc': 0., 'scale': 1.}, fixed=False)

    obs = Correlation2PolesObservable(data=None, theory=theory, s=s, ells=ells,
                                      templates=[(param, template_array)])
    assert len(obs.templates) == 1

    pipe = compile(obs)
    assert any(p.name == 'xi_template' for p in pipe.params)

    pipe_params = {p.name: float(p.value) for p in pipe.params}
    pipe(pipe_params)

    expected = obs._window_matrix @ np.ravel(theory.poles) + template_array * pipe_params['xi_template']
    np.testing.assert_allclose(np.asarray(obs.flattheory), expected, rtol=1e-6)


def test_spectrum3poles_templates():
    """Spectrum3PolesObservable: templates are stored and Parameter is discovered in graph."""
    import lsstypes as types
    from desilike.observables import Spectrum3PolesObservable
    from desilike.base import Parameter
    from desilike.theories.galaxy_clustering import DampedBAOWigglesPTSpectrum2Poles, BAOSpectrum2Template

    data = _make_spectrum3_lsstypes(size=5)
    window = _make_spectrum3_window(data, size=4)

    fiducial = _make_fiducial()
    kin = next(iter(window.theory)).coords('k')
    tmpl = BAOSpectrum2Template(k=kin[..., 0], z=0.5, fiducial=fiducial)
    theory = DampedBAOWigglesPTSpectrum2Poles(template=tmpl, ells=[(0, 0, 0), (2, 0, 2)])

    n = data.size
    rng = np.random.default_rng(24)
    template_array = rng.normal(size=n)
    param = Parameter('b3_template', value=0.3, prior={'dist': 'norm', 'loc': 0., 'scale': 1.}, fixed=False)

    obs = Spectrum3PolesObservable(data=data, theory=theory, window=window,
                                   templates=[(param, template_array)])
    assert len(obs.templates) == 1
    assert obs.templates[0][0].name == 'b3_template'
    np.testing.assert_array_equal(obs.templates[0][1], template_array)
