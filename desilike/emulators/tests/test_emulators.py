"""Tests for TaylorEmulator.

Covers:
- Custom JAX Calculator  (analytic polynomial: exact at expansion order)
- Kaiser galaxy-clustering model (KaiserTracerSpectrum2Poles + BAOSpectrum2Template)
- Save / load round-trip
- to_calculator() drop-in replacement
"""

import os
import tempfile

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from desilike import compile, TaylorEmulator
from desilike.base import Calculator
from desilike.parameter import Parameter


# ── shared fiducial shorthand ─────────────────────────────────────────────────

_FID = ('DESI', {'engine': 'camb'})


# ── custom toy calculators ────────────────────────────────────────────────────

class QuadraticModel(Calculator):
    """f(a, b) = c0 + c1*a + c2*b + c3*a**2 + c4*a*b + c5*b**2.

    Coefficients stored as class attributes so we can compare analytically.
    """
    C = np.array([1.0, 2.0, -1.0, 0.5, 1.5, -0.3])

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        c = self.C
        self.out = (c[0]
                    + c[1] * self.a.value + c[2] * self.b.value
                    + c[3] * self.a.value ** 2 + c[4] * self.a.value * self.b.value
                    + c[5] * self.b.value ** 2)
        return self.out

    def tree_flatten(self):
        return [self.out], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.out = children[0]
        return obj

    @classmethod
    def analytic(cls, a, b):
        c = cls.C
        return (c[0] + c[1] * a + c[2] * b
                + c[3] * a ** 2 + c[4] * a * b + c[5] * b ** 2)


class GrowthCalculator(Calculator):
    """Simulates a non-JAX cosmology code: D(omega_m) = omega_m**0.55."""
    _is_external = True

    def __init__(self, omega_m):
        self.omega_m = omega_m

    def __call__(self):
        self.growth = np.array(float(self.omega_m.value) ** 0.55)
        return self

    def tree_flatten(self):
        return [self.growth], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.growth = children[0]
        return obj


class PowerLaw(Calculator):
    """P(k) = A * k**ns * D**2, a pure JAX node downstream of GrowthCalculator."""

    _K = jnp.linspace(0.01, 0.3, 15)

    def __init__(self, cosmo, A, ns):
        self.cosmo = cosmo
        self.A = A
        self.ns = ns

    def __call__(self):
        self.pk = self.A.value * self._K ** self.ns.value * self.cosmo.growth ** 2
        return self.pk

    def tree_flatten(self):
        return [self.pk], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk = children[0]
        return obj


class MultiOutNoReturn(Calculator):
    """__call__ returns None; outputs (array, scalar, tuple-of-arrays) live in attrs.

    Mirrors the shape of FOLPSPTSpectrum2Poles: a side-effect-only __call__ whose
    parameter-dependent state is exposed entirely via tree_flatten children.
    """

    _K = jnp.linspace(0.1, 1.0, 5)

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        self.arr = self.a.value * self._K + self.b.value          # linear in a, b
        self.scal = self.a.value ** 2                              # quadratic in a
        self.tup = (self.a.value * self._K, self.b.value + self._K)
        # no return → None

    def tree_flatten(self):
        children = [self.arr, self.scal, self.tup[0], self.tup[1]]
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.arr, obj.scal = children[0], children[1]
        obj.tup = (children[2], children[3])
        return obj


class SumMultiOut(Calculator):
    """Downstream node reading the (None-returning) MultiOutNoReturn attributes."""

    def __init__(self, src):
        self.src = src

    def __call__(self):
        self.out = self.src.arr + self.src.scal + self.src.tup[0] + self.src.tup[1]
        return self.out

    def tree_flatten(self):
        return [self.out], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.out = children[0]
        return obj


# ── test helpers ──────────────────────────────────────────────────────────────

def _make_quadratic_graph():
    a = Parameter('a', value=1.0, fixed=False)
    b = Parameter('b', value=0.5, fixed=False)
    return compile(QuadraticModel(a=a, b=b))


def _make_powerlaw_graph():
    omega_m = Parameter('omega_m', value=0.3, fixed=False)
    A = Parameter('A', value=1.0, fixed=False)
    ns = Parameter('ns', value=0.96, fixed=False)
    cosmo = GrowthCalculator(omega_m=omega_m)
    return compile(PowerLaw(cosmo=cosmo, A=A, ns=ns))


def _make_kaiser_graph(k=None, ells=(0, 2)):
    from desilike.theories.galaxy_clustering.full_shape import KaiserTracerSpectrum2Poles
    from desilike.theories.galaxy_clustering.template import BAOSpectrum2Template
    if k is None:
        k = np.linspace(0.02, 0.2, 15)
    template = BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')
    theory = KaiserTracerSpectrum2Poles(k=k, ells=ells, template=template)
    return compile(theory)


# ── custom Calculator tests ───────────────────────────────────────────────────

def test_taylor_custom_center():
    """At-center prediction must match the exact function value."""
    pipe = _make_quadratic_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    center = {p.name: p.value for p in pipe.params}
    rv, _ = emu.predict(center)
    assert abs(float(rv) - float(pipe(center))) < 1e-12


def test_taylor_custom_exact_at_order():
    """Quadratic model is exactly represented at order=2 everywhere."""
    pipe = _make_quadratic_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    for a, b in [(0.5, 1.5), (-0.3, 0.8), (2.0, -1.0), (1.0, 0.0)]:
        rv, _ = emu.predict({'a': a, 'b': b})
        exact = float(QuadraticModel.analytic(a, b))
        assert abs(float(rv) - exact) < 1e-10, \
            f'(a={a}, b={b}): pred={float(rv):.8f}, exact={exact:.8f}'


def test_taylor_custom_grad():
    """jax.grad of the emulator predict matches finite differences."""
    pipe = _make_quadratic_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    params = {'a': 1.2, 'b': 0.7}
    grad = jax.grad(lambda p: emu.predict(p)[0])(params)
    eps = 1e-5
    for name in params:
        fd = (float(emu.predict({**params, name: params[name] + eps})[0])
              - float(emu.predict({**params, name: params[name] - eps})[0])) / (2 * eps)
        assert abs(float(grad[name]) - fd) < 1e-8, \
            f'grad[{name}]: emu={float(grad[name]):.6f}, fd={fd:.6f}'


def test_taylor_custom_to_calculator():
    """to_calculator() gives a drop-in Calculator whose compiled graph agrees with predict()."""
    pipe = _make_quadratic_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    calc = emu.to_calculator()
    pipe2 = compile(calc)
    assert set(pipe2.params.names()) == {'a', 'b'}
    for a, b in [(0.5, 1.2), (1.5, -0.3)]:
        p = {'a': a, 'b': b}
        via_calc = float(pipe2(p))
        via_predict = float(emu.predict(p)[0])
        assert abs(via_calc - via_predict) < 1e-12, \
            f'to_calculator vs predict at (a={a}, b={b}): {via_calc:.8f} vs {via_predict:.8f}'


def test_taylor_custom_external():
    """TaylorEmulator works with external (_is_external=True, FD path) upstream."""
    pipe = _make_powerlaw_graph()
    center = {p.name: p.value for p in pipe.params}
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    rv, _ = emu.predict(center)
    exact = np.asarray(pipe(center))
    assert np.allclose(np.asarray(rv), exact, atol=1e-10), \
        f'center mismatch, max={np.max(np.abs(np.asarray(rv) - exact)):.2e}'

    # Gradient of sum(pk) wrt each input should be finite
    grad = jax.grad(lambda p: jnp.sum(emu.predict(p)[0]))(center)
    for name in center:
        assert np.isfinite(float(grad[name])), f'grad[{name}] is not finite'


def test_taylor_custom_write_read():
    """write / read round-trip; predictions must agree."""
    pipe = _make_quadratic_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    params = {'a': 0.8, 'b': 1.3}
    pred_orig = float(emu.predict(params)[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'quad_emu.h5')
        emu.write(fn)
        emu2 = TaylorEmulator.read(fn)
    pred_loaded = float(emu2.predict(params)[0])
    assert abs(pred_orig - pred_loaded) < 1e-12, \
        f'write/read mismatch: orig={pred_orig:.8f}, loaded={pred_loaded:.8f}'
    # to_calculator() must also work after loading (no graph available)
    calc = emu2.to_calculator()
    pipe2 = compile(calc)
    assert abs(float(pipe2(params)) - pred_orig) < 1e-12


# ── None-returning, multi-child calculator (FOLPS-shaped) ─────────────────────

def _make_multiout_graph():
    a = Parameter('a', value=1.3, fixed=False)
    b = Parameter('b', value=0.4, fixed=False)
    return compile(MultiOutNoReturn(a=a, b=b))


def test_taylor_none_return_predict():
    """A None-returning root emulates its tree children; predict returns (None, {})."""
    pipe = _make_multiout_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    assert emu._return_kind == 'none'
    assert emu._n_children == 4
    rv, derived = emu.predict({'a': 1.3, 'b': 0.4})
    assert rv is None
    assert derived == {}


def test_taylor_none_return_children_exact():
    """Children are Taylor-expanded (exact at order 2 here) and vary with params."""
    pipe = _make_multiout_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    for a, b in [(1.3, 0.4), (2.0, -0.5), (0.7, 1.1)]:
        _, children, _ = emu._predict_children({'a': a, 'b': b})
        K = np.asarray(MultiOutNoReturn._K)
        np.testing.assert_allclose(np.asarray(children[0]), a * K + b, atol=1e-10)
        np.testing.assert_allclose(np.asarray(children[1]), a ** 2, atol=1e-10)
        np.testing.assert_allclose(np.asarray(children[2]), a * K, atol=1e-10)
        np.testing.assert_allclose(np.asarray(children[3]), b + K, atol=1e-10)


def test_taylor_none_return_in_pipeline():
    """to_calculator() drop-in feeds a downstream node that reads the emulated attrs."""
    pipe = _make_multiout_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    emulated = emu.to_calculator()
    pipe2 = compile(SumMultiOut(src=emulated))
    assert set(pipe2.params.names()) == {'a', 'b'}
    for a, b in [(1.3, 0.4), (1.8, -0.2)]:
        K = np.asarray(MultiOutNoReturn._K)
        expected = (a * K + b) + a ** 2 + (a * K) + (b + K)
        np.testing.assert_allclose(np.asarray(pipe2({'a': a, 'b': b})), expected, atol=1e-9)


def test_taylor_none_return_write_read():
    """write/read round-trip preserves the per-child emulation and None return."""
    pipe = _make_multiout_graph()
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'multiout_emu.h5')
        emu.write(fn)
        emu2 = TaylorEmulator.read(fn)
    assert emu2._return_kind == 'none'
    rv, _ = emu2.predict({'a': 1.6, 'b': 0.2})
    assert rv is None
    _, ch1, _ = emu._predict_children({'a': 1.6, 'b': 0.2})
    _, ch2, _ = emu2._predict_children({'a': 1.6, 'b': 0.2})
    for c1, c2 in zip(ch1, ch2):
        np.testing.assert_allclose(np.asarray(c1), np.asarray(c2), atol=1e-12)


# ── Kaiser model tests ────────────────────────────────────────────────────────

class TestTaylorKaiser:
    """TaylorEmulator tests with KaiserTracerSpectrum2Poles + BAOSpectrum2Template.

    The graph is compiled and the emulator is fitted once for the whole class
    (scope='class') to keep the suite fast.
    """

    @pytest.fixture(scope='class')
    def graph(self):
        return _make_kaiser_graph()

    @pytest.fixture(scope='class')
    def emulator(self, graph):
        emu = TaylorEmulator(graph, order=2)
        emu.fit()
        return emu

    def test_center(self, graph, emulator):
        """At-center prediction is numerically identical to the exact graph."""
        center = {p.name: p.value for p in graph.params}
        rv, _ = emulator.predict(center)
        exact = np.asarray(graph(center))
        assert np.allclose(np.asarray(rv), exact, atol=0., rtol=0.), \
            f'center mismatch, max abs diff = {np.max(np.abs(np.asarray(rv) - exact)):.2e}'

    def test_grad(self, graph, emulator):
        """Gradient of sum(spectrum) at center matches the exact JAX gradient."""
        center = {p.name: p.value for p in graph.params}
        grad_emu = jax.grad(lambda p: jnp.sum(emulator.predict(p)[0]))(center)
        grad_exact = jax.grad(lambda p: jnp.sum(graph(p)))(center)
        for name in emulator._input_param_names:
            g_e = float(grad_emu[name])
            g_x = float(grad_exact[name])
            scale = max(abs(g_x), 1e-6)
            assert abs(g_e - g_x) / scale < 1e-5, \
                f'grad[{name}]: emu={g_e:.6f}, exact={g_x:.6f}, reldiff={(abs(g_e-g_x)/scale):.2e}'

    def test_accuracy(self, graph, emulator):
        """A small parameter shift gives a Taylor prediction within 5 % of exact (order-2 accuracy)."""
        center = {p.name: p.value for p in graph.params}
        shifted = {**center, 'b1': center['b1'] * 1.05, 'qpar': center['qpar'] * 1.02}
        rv, _ = emulator.predict(shifted)
        exact = np.asarray(graph(shifted))
        reldiff = np.max(np.abs(np.asarray(rv) - exact) / (np.abs(exact) + 1e-30))
        assert reldiff < 0.05, f'order-2 accuracy check failed: max reldiff = {reldiff:.3f}'

    def test_to_calculator(self, graph, emulator):
        """to_calculator() compiles into a pipeline that agrees with predict()."""
        center = {p.name: p.value for p in graph.params}
        shifted = {**center, 'b1': 2.1, 'qpar': 1.02}
        calc = emulator.to_calculator()
        pipe2 = compile(calc)
        assert set(pipe2.params.names()) == set(graph.params.names())
        via_calc = np.asarray(pipe2(shifted))
        via_predict = np.asarray(emulator.predict(shifted)[0])
        assert np.allclose(via_calc, via_predict, atol=0., rtol=0.), \
            f'to_calculator vs predict mismatch, max diff={np.max(np.abs(via_calc - via_predict)):.2e}'

    def test_emulated_pt_in_tracer(self):
        """Emulate KaiserPTSpectrum2Poles and use it as pt= in KaiserTracerSpectrum2Poles.

        The emulated PT is a drop-in for the exact PT:
        - At the expansion center the tracer output is identical.
        - At a small parameter shift the relative error is < 5 % (order-2 accuracy).
        """
        from desilike.theories.galaxy_clustering.full_shape import KaiserPTSpectrum2Poles, KaiserTracerSpectrum2Poles
        from desilike.theories.galaxy_clustering.template import BAOSpectrum2Template

        k = np.linspace(0.02, 0.2, 15)
        ells = (0, 2)

        # Build and emulate KaiserPTSpectrum2Poles
        template = BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')
        pt = KaiserPTSpectrum2Poles(k=k, ells=ells)
        pt.update(template=template)
        pt_pipe = compile(pt)
        pt_emu = TaylorEmulator(pt_pipe, order=2)
        pt_emu.fit()
        emulated_pt = pt_emu.to_calculator()

        # Tracer with emulated PT — pass k/ells matching the emulator's fixed grid
        pipe_emu = compile(KaiserTracerSpectrum2Poles(k=k, ells=ells, pt=emulated_pt))

        # Exact reference
        template2 = BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')
        pt2 = KaiserPTSpectrum2Poles(k=k, ells=ells)
        pt2.update(template=template2)
        pipe_exact = compile(KaiserTracerSpectrum2Poles(k=k, ells=ells, pt=pt2))

        center = {p.name: p.value for p in pipe_exact.params}
        assert set(pipe_emu.params.names()) == set(pipe_exact.params.names())

        # At the expansion center the emulator is exact
        diff_center = np.max(np.abs(np.asarray(pipe_emu(center)) - np.asarray(pipe_exact(center))))
        assert diff_center == 0., f'center mismatch: {diff_center:.2e}'

        # At a small shift the tracer output is within 5 % of exact
        shifted = {**center, 'b1': 2.0, 'qpar': 1.02}
        exact_shifted = np.asarray(pipe_exact(shifted))
        emu_shifted = np.asarray(pipe_emu(shifted))
        reldiff = np.max(np.abs(emu_shifted - exact_shifted) / (np.abs(exact_shifted) + 1e-30))
        assert reldiff < 0.05, f'order-2 accuracy check failed: max reldiff = {reldiff:.3f}'

    def test_write_read(self, graph, emulator):
        """write / read round-trip; loaded emulator predicts identically."""
        center = {p.name: p.value for p in graph.params}
        shifted = {**center, 'b1': 2.1, 'qpar': 1.02}
        pred_orig = np.asarray(emulator.predict(shifted)[0])
        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, 'kaiser_emu.h5')
            emulator.write(fn)
            emu2 = TaylorEmulator.read(fn)
        pred_loaded = np.asarray(emu2.predict(shifted)[0])
        assert np.allclose(pred_orig, pred_loaded, atol=0., rtol=0.), \
            f'write/read mismatch, max diff={np.max(np.abs(pred_orig - pred_loaded)):.2e}'
        # to_calculator() must also work after loading
        calc2 = emu2.to_calculator()
        pipe2 = compile(calc2)
        assert np.allclose(np.asarray(pipe2(shifted)), pred_orig, atol=0., rtol=0.)


if __name__ == '__main__':
    pipe = _make_kaiser_graph()
    center = {p.name: p.value for p in pipe.params}
    print('params:', list(center.keys()))
    emu = TaylorEmulator(pipe, order=2)
    emu.fit()
    rv, _ = emu.predict(center)
    exact = pipe(center)
    print(f'center diff: {float(jnp.max(jnp.abs(rv - exact))):.2e}')
    grad_emu = jax.grad(lambda p: jnp.sum(emu.predict(p)[0]))(center)
    grad_exact = jax.grad(lambda p: jnp.sum(pipe(p)))(center)
    for name in ['b1', 'qpar', 'df']:
        print(f'  grad[{name}]: emu={float(grad_emu[name]):.4f}  exact={float(grad_exact[name]):.4f}')
