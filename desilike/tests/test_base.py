"""Tests for desilike/base.py — cosmology-inspired pipeline (Cosmology → PowerSpectrum → GaussianChi2)."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from desilike.base import Calculator, ExternalCalculator, Likelihood, GaussianLikelihood, SumLikelihood, Prior, Posterior, CompiledGraph, compile
from desilike.parameter import Parameter


# ── shared static data ────────────────────────────────────────────────────────

K = jnp.linspace(0.01, 0.3, 30)
DATA = jnp.array(np.random.default_rng(0).normal(1.0, 0.1, len(K)))


# ── toy calculators ───────────────────────────────────────────────────────────

class Cosmology(ExternalCalculator):
    """Non-JAX: growth_factor = omega_m^0.55 / (1 + z), growth_rate = omega_m^0.55."""
    _call_count = 0

    def __post_init__(self, omega_m, z):
        self.omega_m = omega_m
        self.z = z

    def __call__(self):
        Cosmology._call_count += 1
        self.growth_factor = np.array(self.omega_m ** 0.55 / (1.0 + self.z))
        self.growth_rate = np.array(self.omega_m ** 0.55)
        return self

    def tree_flatten(self):
        return [self.growth_factor, self.growth_rate], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.growth_factor = children[0]
        obj.growth_rate = children[1]
        return obj


class PowerSpectrum(Calculator):
    """JAX-native: P(k) = A * k^ns * D^2."""

    def __post_init__(self, cosmo, A, ns):
        self.cosmo = cosmo
        self.A = A
        self.ns = ns

    def __call__(self):
        self.pk = self.A * K ** self.ns * self.cosmo.growth_factor ** 2
        return self.pk

    def tree_flatten(self):
        return [self.pk], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk = children[0]
        return obj


class GaussianChi2(Calculator):
    """JAX-native: logL = -0.5 * sum((theory - data)^2 / sigma^2)."""

    def __post_init__(self, spectrum, data, sigma=0.1):
        self.spectrum = spectrum
        self._data = data
        self._sigma = sigma

    def __call__(self):
        self.loglikelihood = -0.5 * jnp.sum(((self.spectrum.pk - self._data) / self._sigma) ** 2)
        return self.loglikelihood

    def tree_flatten(self):
        return [self.loglikelihood], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.loglikelihood = children[0]
        return obj


# ── shared GaussianLikelihood subclass for marginalisation tests ──────────────

class _LinearTheory(GaussianLikelihood):
    """theory = (A + alpha) * K, linear in alpha."""

    def __post_init__(self, A, alpha, data, covariance):
        self.A = A
        self.alpha = alpha
        self.flatdata = jnp.asarray(data)
        self.precision = jnp.linalg.inv(jnp.asarray(covariance))

    def __call__(self):
        self.flattheory = (self.A + self.alpha) * K
        return super().__call__()


# ── analytic reference and node factory ───────────────────────────────────────

def analytic_logL(omega_m, z, A, ns, data=DATA, sigma=0.1):
    D = omega_m ** 0.55 / (1.0 + z)
    theory = A * np.array(K) ** ns * D ** 2
    return -0.5 * np.sum(((theory - np.array(data)) / sigma) ** 2)


def _make_nodes():
    """Return (omega_m, z, A, ns, cosmo, spectrum, likelihood) with default parameter values."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianChi2(spectrum=spectrum, data=DATA)
    return omega_m, z, A, ns, cosmo, spectrum, likelihood


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    return compile(_make_nodes()[-1])


# ── basic correctness ─────────────────────────────────────────────────────────

def test_param_names(pipeline):
    assert set(pipeline.params.names()) == {'omega_m', 'z', 'A', 'ns'}


def test_correctness(pipeline):
    got = float(pipeline(omega_m=0.3, z=0.5, A=1.0, ns=0.96)[0])
    expected = analytic_logL(0.3, 0.5, 1.0, 0.96)
    assert abs(got - expected) < 1e-8, f"got {got}, expected {expected}"


def test_eager_attrs_updated(pipeline):
    """After an eager call, all node attributes reflect the computed values."""
    omega_m, z, A, ns = 0.28, 0.6, 1.1, 0.97
    pipeline(omega_m=omega_m, z=z, A=A, ns=ns)
    cosmo, spectrum, likelihood = pipeline.nodes

    expected_gf = omega_m ** 0.55 / (1.0 + z)
    expected_pk = A * np.array(K) ** ns * expected_gf ** 2

    assert abs(float(cosmo.growth_factor) - expected_gf) < 1e-8
    assert abs(float(cosmo.growth_rate) - omega_m ** 0.55) < 1e-8
    assert jnp.allclose(jnp.array(spectrum.pk), jnp.array(expected_pk), atol=1e-8)
    assert abs(float(likelihood.loglikelihood) - analytic_logL(omega_m, z, A, ns)) < 1e-8


# ── JAX transforms ────────────────────────────────────────────────────────────

def test_jit(pipeline):
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.2, 'ns': 0.98}
    assert abs(float(jax.jit(pipeline)(params)[0]) - analytic_logL(0.3, 0.5, 1.2, 0.98)) < 1e-8


def test_grad(pipeline):
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    grad = jax.grad(pipeline, has_aux=True)(params)[0]
    eps = 1e-5
    for name in pipeline.params.names():
        fd = (float(pipeline({**params, name: params[name] + eps})[0]) - float(pipeline({**params, name: params[name] - eps})[0])) / (2 * eps)
        assert abs(float(grad[name]) - fd) < 1e-4, f"grad[{name}]: got {float(grad[name]):.6f}, fd {fd:.6f}"


def test_vmap(pipeline):
    omega_m_vals = jnp.linspace(0.25, 0.35, 5)
    params_batch = {'omega_m': omega_m_vals, 'z': jnp.full(5, 0.5), 'A': jnp.ones(5), 'ns': jnp.full(5, 0.96)}
    batched = jax.vmap(pipeline)(params_batch)[0]
    looped = jnp.stack([pipeline({'omega_m': float(omega_m_vals[i]), 'z': 0.5, 'A': 1.0, 'ns': 0.96})[0] for i in range(5)])
    assert jnp.allclose(batched, looped, atol=1e-8)


def test_jit_grad(pipeline):
    params = {'omega_m': 0.28, 'z': 0.6, 'A': 1.1, 'ns': 0.97}
    grad_eager = jax.grad(pipeline, has_aux=True)(params)[0]
    grad_jit = jax.jit(lambda p: jax.grad(pipeline, has_aux=True)(p)[0])(params)
    assert all(jnp.allclose(grad_eager[k], grad_jit[k], atol=1e-8) for k in grad_eager)


def test_jacrev_external():
    """jax.jacobian (jacrev) gives correct Jacobian for ExternalCalculator pipelines."""
    _, _, _, _, _, spectrum, likelihood = _make_nodes()
    pipe_pk = compile(likelihood, output=lambda: spectrum.pk)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    jac_rev = jax.jacobian(lambda p: pipe_pk(p)[0])(params)

    for name in params:
        assert jac_rev[name].shape == (len(K),), f"jacrev[{name}].shape = {jac_rev[name].shape}"
    eps = 1e-5
    for name in params:
        fd = (np.asarray(pipe_pk({**params, name: params[name] + eps})[0]) -
              np.asarray(pipe_pk({**params, name: params[name] - eps})[0])) / (2 * eps)
        assert np.allclose(np.asarray(jac_rev[name]), fd, atol=1e-4), \
            f"jacrev vs FD mismatch for {name}: max err = {np.abs(np.asarray(jac_rev[name]) - fd).max():.2e}"


def test_jacfwd_grad_external():
    """jax.jacfwd(jax.grad(pipe)) works for ExternalCalculator pipelines via custom_jvp FD rule."""
    pipe = compile(_make_nodes()[-1])
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    grad_fn = lambda p: jax.grad(pipe, has_aux=True)(p)[0]
    hess_jacfwd = jax.jacfwd(grad_fn)(params)

    eps = 1e-5
    for p in pipe.params:
        ref = (float(grad_fn({**params, p.name: params[p.name] + eps})[p.name]) - float(grad_fn({**params, p.name: params[p.name] - eps})[p.name])) / (2 * eps)
        assert abs(float(hess_jacfwd[p.name][p.name]) - ref) < 1e-3, f"H[{p.name},{p.name}]: jacfwd={float(hess_jacfwd[p.name][p.name]):.6f}, FD={ref:.6f}"

    hess_jax = jax.hessian(lambda p: pipe(p)[0])(params)
    for p in pipe.params:
        assert abs(float(hess_jax[p.name][p.name]) - float(hess_jacfwd[p.name][p.name])) < 1e-10


# ── caching ───────────────────────────────────────────────────────────────────

def test_external_cache():
    """ExternalCalculator() skipped when params and deps unchanged."""
    _call_count = [0]

    class CountedCosmology(ExternalCalculator):
        def __post_init__(self, omega_m, z):
            self.omega_m = omega_m
            self.z = z

        def __call__(self):
            _call_count[0] += 1
            self.growth_factor = np.array(self.omega_m ** 0.55 / (1.0 + self.z))
            self.growth_rate = np.array(self.omega_m ** 0.55)
            return self

        def tree_flatten(self):
            return [self.growth_factor, self.growth_rate], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.growth_factor = children[0]
            obj.growth_rate = children[1]
            return obj

    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = CountedCosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    pipe = compile(GaussianChi2(spectrum=spectrum, data=DATA))
    _call_count[0] = 0

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    pipe(params)
    pipe(params)
    assert _call_count[0] == 1, f"Expected 1 (cache hit on repeat), got {_call_count[0]}"
    pipe({'omega_m': 0.35, 'z': 0.5, 'A': 1.0, 'ns': 0.96})
    assert _call_count[0] == 2, f"Expected 2 (new params trigger rerun), got {_call_count[0]}"


def test_jax_cache():
    """Calculator() skipped in eager mode when params and deps unchanged."""
    _call_count = [0]

    class CountedSpectrum(Calculator):
        def __post_init__(self, cosmo, A, ns):
            self.cosmo = cosmo
            self.A = A
            self.ns = ns

        def __call__(self):
            _call_count[0] += 1
            self.pk = self.A * K ** self.ns * self.cosmo.growth_factor ** 2
            return self.pk

        def tree_flatten(self):
            return [self.pk], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.pk = children[0]
            return obj

    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = CountedSpectrum(cosmo=cosmo, A=A, ns=ns)
    pipe = compile(GaussianChi2(spectrum=spectrum, data=DATA))
    _call_count[0] = 0

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    pipe(params)
    pipe(params)
    assert _call_count[0] == 1, f"Expected 1 (cache hit on repeat), got {_call_count[0]}"
    pipe({'omega_m': 0.3, 'z': 0.5, 'A': 1.2, 'ns': 0.96})
    assert _call_count[0] == 2, f"Expected 2 (own param changed), got {_call_count[0]}"
    pipe({'omega_m': 0.35, 'z': 0.5, 'A': 1.2, 'ns': 0.96})
    assert _call_count[0] == 3, f"Expected 3 (dep rerun), got {_call_count[0]}"


# ── output / pytree ───────────────────────────────────────────────────────────

def test_custom_output():
    """output= lambda reads any pytree of calculator attrs; grad flows through it."""
    _, _, _, _, _, spectrum, likelihood = _make_nodes()
    pipe_pk = compile(likelihood, output=lambda: spectrum.pk)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    D = 0.3 ** 0.55 / 1.5
    expected_pk = 1.0 * np.array(K) ** 0.96 * D ** 2
    assert jnp.allclose(pipe_pk(params)[0], jnp.array(expected_pk), atol=1e-8)

    _, _, _, _, _, spectrum2, likelihood2 = _make_nodes()
    pipe_tuple = compile(likelihood2, output=lambda: (likelihood2.loglikelihood, spectrum2.pk))
    (logL, pk), _ = pipe_tuple(params)
    assert abs(float(logL) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8
    assert jnp.allclose(pk, jnp.array(expected_pk), atol=1e-8)
    grad = jax.grad(lambda p: jnp.sum(pipe_pk(p)[0]))(params)
    assert set(grad.keys()) == {'omega_m', 'z', 'A', 'ns'}


def test_pytree_registration():
    """Calculators are registered JAX pytrees: tree_leaves, tree_map, and jit work natively."""
    _, _, _, _, _, spectrum, _ = _make_nodes()
    pipe = compile(spectrum)
    pipe(omega_m=0.3, z=0.5, A=1.0, ns=0.96)

    leaves = jax.tree_util.tree_leaves(spectrum)
    assert any(isinstance(l, (np.ndarray, jnp.ndarray)) for l in leaves)

    doubled = jax.tree_util.tree_map(lambda x: x * 2, spectrum)
    assert jnp.allclose(doubled.pk, spectrum.pk * 2)

    children, aux = spectrum.tree_flatten()
    reconstructed = PowerSpectrum.tree_unflatten(aux, children)
    assert jnp.allclose(reconstructed.pk, spectrum.pk)


# ── array-valued parameters ───────────────────────────────────────────────────

def test_array_param_jax():
    """Calculator: array-valued weight parameter flows correctly through pipeline."""

    class WeightedLikelihood(Calculator):
        def __post_init__(self, spectrum, data, w):
            self.spectrum = spectrum
            self._data = data
            self.w = w

        def __call__(self):
            self.loglikelihood = -0.5 * jnp.sum(self.w * (self.spectrum.pk - self._data) ** 2)
            return self.loglikelihood

        def tree_flatten(self):
            return [self.loglikelihood], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.loglikelihood = children[0]
            return obj

    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    w_param = Parameter('w', value=np.ones(len(K)))
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    pipe = compile(WeightedLikelihood(spectrum=spectrum, data=DATA, w=w_param))

    w = np.ones(len(K))
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96, 'w': jnp.array(w)}
    got = float(pipe(params)[0])
    D = 0.3 ** 0.55 / 1.5
    expected = -0.5 * np.sum(w * (1.0 * np.array(K) ** 0.96 * D ** 2 - np.array(DATA)) ** 2)
    assert abs(got - expected) < 1e-8

    grad = jax.grad(pipe, has_aux=True)(params)[0]
    assert grad['w'].shape == (len(K),)
    assert 'omega_m' in grad


def test_array_param_external():
    """ExternalCalculator: array-valued parameter (k-weights) FD grad is correct."""

    class WeightedCosmology(ExternalCalculator):
        def __post_init__(self, omega_m, k_weights):
            self.omega_m = omega_m
            self.k_weights = k_weights

        def __call__(self):
            self.weighted_D = np.asarray(self.k_weights) * np.array(self.omega_m ** 0.55)
            return self

        def tree_flatten(self):
            return [self.weighted_D], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.weighted_D = children[0]
            return obj

    class WeightedChi2(Calculator):
        def __post_init__(self, wcos):
            self.wcos = wcos

        def __call__(self):
            self.loglikelihood = -0.5 * jnp.sum(self.wcos.weighted_D ** 2)
            return self.loglikelihood

        def tree_flatten(self):
            return [self.loglikelihood], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.loglikelihood = children[0]
            return obj

    omega_m = Parameter('omega_m', value=0.3)
    k_weights_param = Parameter('k_weights', value=np.ones(len(K)))
    wcos = WeightedCosmology(omega_m=omega_m, k_weights=k_weights_param)
    pipe = compile(WeightedChi2(wcos=wcos))

    k_weights = np.ones(len(K))
    params = {'omega_m': 0.3, 'k_weights': jnp.array(k_weights)}
    got = float(pipe(params)[0])
    D = 0.3 ** 0.55
    assert abs(got - (-0.5 * np.sum((k_weights * D) ** 2))) < 1e-8

    grad = jax.grad(pipe, has_aux=True)(params)[0]
    assert grad['k_weights'].shape == (len(K),)
    assert jnp.allclose(grad['k_weights'], jnp.array(-D ** 2 * k_weights), atol=1e-4)
    fd_om = (float(pipe({**params, 'omega_m': jnp.array(0.3 + 1e-5)})[0]) - float(pipe({**params, 'omega_m': jnp.array(0.3 - 1e-5)})[0])) / 2e-5
    assert abs(float(grad['omega_m']) - fd_om) < 1e-4


# ── compilation edge cases ────────────────────────────────────────────────────

def test_internal_init():
    """Calculator and Parameter objects created inside init() are auto-discovered."""

    class InternalCosmology(ExternalCalculator):
        def __post_init__(self):
            self.omega_m = Parameter('omega_m', value=0.3)
            self.z = Parameter('z', value=0.5)

        def __call__(self):
            self.growth_factor = np.array(self.omega_m ** 0.55 / (1.0 + self.z))
            self.growth_rate = np.array(self.omega_m ** 0.55)
            return self

        def tree_flatten(self):
            return [self.growth_factor, self.growth_rate], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.growth_factor = children[0]
            obj.growth_rate = children[1]
            return obj

    class InternalSpectrum(Calculator):
        def __post_init__(self):
            self.cosmo = InternalCosmology()
            self.A = Parameter('A', value=1.0)
            self.ns = Parameter('ns', value=0.96)

        def __call__(self):
            self.pk = self.A * K ** self.ns * self.cosmo.growth_factor ** 2
            return self.pk

        def tree_flatten(self):
            return [self.pk], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.pk = children[0]
            return obj

    pipe = compile(InternalSpectrum())
    assert set(pipe.params.names()) == {'omega_m', 'z', 'A', 'ns'}
    D = 0.3 ** 0.55 / 1.5
    assert jnp.allclose(pipe(omega_m=0.3, z=0.5, A=1.0, ns=0.96)[0], jnp.array(1.0 * np.array(K) ** 0.96 * D ** 2), atol=1e-8)


def test_duplicate_param_name_raises():
    """Two distinct Parameter objects with the same name raise ValueError at compile time."""

    class DupCosmology(ExternalCalculator):
        def __post_init__(self):
            self.omega_m = Parameter('omega_m', value=0.3)

        def __call__(self):
            self.growth_factor = np.array(self.omega_m ** 0.55)
            return self

        def tree_flatten(self):
            return [self.growth_factor], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.growth_factor = children[0]
            return obj

    class DupSpectrum(Calculator):
        def __post_init__(self, cosmo):
            self.cosmo = cosmo
            self.omega_m = Parameter('omega_m', value=0.3)  # different object, same name

        def __call__(self):
            self.pk = self.omega_m * self.cosmo.growth_factor
            return self.pk

        def tree_flatten(self):
            return [self.pk], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.pk = children[0]
            return obj

    with pytest.raises(ValueError, match='omega_m'):
        compile(DupSpectrum(cosmo=DupCosmology()))


def test_fd_acc():
    """param.fd_acc=4 gives smaller gradient error than fd_acc=2 at the same large step size."""

    class SinCosmology(ExternalCalculator):
        def __post_init__(self, omega_m):
            self.omega_m = omega_m

        def __call__(self):
            x = float(self.omega_m)
            self.val = np.array(np.sin(x) + x ** 3)
            return self

        def tree_flatten(self):
            return [self.val], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.val = children[0]
            return obj

    class TrivialLikelihood(Calculator):
        def __post_init__(self, cosmo):
            self.cosmo = cosmo

        def __call__(self):
            self.out = jnp.sum(self.cosmo.val)
            return self.out

        def tree_flatten(self):
            return [self.out], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.out = children[0]
            return obj

    x0 = 0.3
    analytic_grad = float(np.cos(x0) + 3 * x0 ** 2)
    large_eps = 1e-2

    for acc, tol in [(2, 2e-4), (4, 1e-8)]:
        om = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=acc)
        pipe = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om)))
        g = float(jax.grad(pipe, has_aux=True)({'omega_m': jnp.array(x0)})[0]['omega_m'])
        assert abs(g - analytic_grad) < tol, f'fd_acc={acc}: grad error {abs(g - analytic_grad):.2e} >= tol {tol:.2e}'

    om2 = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=2)
    om4 = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=4)
    pipe2 = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om2)))
    pipe4 = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om4)))
    err2 = abs(float(jax.grad(pipe2, has_aux=True)({'omega_m': jnp.array(x0)})[0]['omega_m']) - analytic_grad)
    err4 = abs(float(jax.grad(pipe4, has_aux=True)({'omega_m': jnp.array(x0)})[0]['omega_m']) - analytic_grad)
    assert err4 < err2, f'fd_acc=4 error {err4:.2e} should be smaller than fd_acc=2 error {err2:.2e}'


# ── tracer safety / parameter mutation ────────────────────────────────────────

def test_no_tracer_leakage_after_jit(pipeline):
    """After jax.jit(pipe)(params), node attrs and param values are concrete, not stale tracers."""
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    jax.jit(pipeline)(params)
    cosmo, spectrum, likelihood = pipeline.nodes

    for p in pipeline.params:
        assert not isinstance(p._value, jax.core.Tracer), f'{p.name}.value is a stale tracer'
        assert isinstance(np.asarray(p.value), np.ndarray)
    assert not isinstance(cosmo.growth_factor, jax.core.Tracer)
    assert not isinstance(spectrum.pk, jax.core.Tracer)
    assert not isinstance(likelihood.loglikelihood, jax.core.Tracer)
    assert np.isfinite(float(pipeline()[0]))


def test_inplace_mutation_after_compile():
    """param.value changed in place after compile: pipeline uses the new value as default."""
    _, _, A, _, _, _, likelihood = _make_nodes()
    pipe = compile(likelihood)

    assert abs(float(pipe()[0]) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8
    A.value = 1.5
    assert abs(float(pipe()[0]) - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8
    jax.jit(pipe)({'omega_m': 0.3, 'z': 0.5, 'A': 1.5, 'ns': 0.96})
    assert abs(float(pipe()[0]) - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8


def test_update_before_compile():
    """Calculator.update() replaces init arguments before compile() is called."""
    _, _, _, _, _, spectrum, likelihood = _make_nodes()
    spectrum.update(A=Parameter('A', value=1.5))
    pipe = compile(likelihood)

    assert pipe.params['A'].value == 1.5
    assert abs(float(pipe(omega_m=0.3, z=0.5, A=1.5, ns=0.96)[0]) - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8

    likelihood.update(sigma=0.2)
    pipe2 = compile(likelihood)
    assert abs(float(pipe2(omega_m=0.3, z=0.5, A=1.5, ns=0.96)[0]) - analytic_logL(0.3, 0.5, 1.5, 0.96, sigma=0.2)) < 1e-8


# ── prior / posterior ─────────────────────────────────────────────────────────

def test_prior_standalone():
    """Prior alone: sums logpdf over non-fixed params, skips fixed ones."""
    omega_m = Parameter('omega_m', value=0.3, prior=dict(dist='norm', loc=0.3, scale=0.01))
    A = Parameter('A', value=1.0, prior=dict(dist='uniform', limits=(0.5, 2.0)))
    ns = Parameter('ns', value=0.96, fixed=True)
    pipe = compile(Prior(omega_m=omega_m, A=A, ns=ns))

    params = {'omega_m': 0.3, 'A': 1.0}
    got = float(pipe(params)[0])
    expected = float(jax.scipy.stats.norm.logpdf(0.3, loc=0.3, scale=0.01) +
                     jax.scipy.stats.uniform.logpdf(1.0, loc=0.5, scale=1.5))
    assert abs(got - expected) < 1e-8
    assert float(pipe({'omega_m': 0.3, 'A': 0.3})[0]) == float(-jnp.inf)

    grad = jax.grad(pipe, has_aux=True)(params)[0]
    assert 'omega_m' in grad and 'A' in grad
    fd_om = (float(pipe({'omega_m': 0.3 + 1e-5, 'A': 1.0})[0]) - float(pipe({'omega_m': 0.3 - 1e-5, 'A': 1.0})[0])) / 2e-5
    assert abs(float(grad['omega_m']) - fd_om) < 1e-6


def test_prior_in_posterior():
    """Prior combined with likelihood via a hand-rolled posterior node."""

    class LogPosterior(Calculator):
        def __post_init__(self, likelihood, prior):
            self.likelihood = likelihood
            self.prior = prior

        def __call__(self):
            self.logposterior = self.likelihood.loglikelihood + self.prior.logpdf
            return self.logposterior

        def tree_flatten(self):
            return [self.logposterior], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.logposterior = children[0]
            return obj

    omega_m = Parameter('omega_m', value=0.3, prior=dict(dist='norm', loc=0.3, scale=0.01))
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)

    prior = Prior(omega_m=omega_m, z=z, A=A, ns=ns)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianChi2(spectrum=spectrum, data=DATA)
    pipe = compile(LogPosterior(likelihood=likelihood, prior=prior))

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    got = float(pipe(params)[0])
    logP = float(jax.scipy.stats.norm.logpdf(0.3, loc=0.3, scale=0.01))
    assert abs(got - (analytic_logL(0.3, 0.5, 1.0, 0.96) + logP)) < 1e-8
    assert set(jax.grad(pipe, has_aux=True)(params)[0].keys()) == {'omega_m', 'z', 'A', 'ns'}
    assert abs(float(jax.jit(pipe)(params)[0]) - got) < 1e-8


def test_posterior_value_and_grad():
    """Posterior = logL + logPrior; grad flows through both."""
    omega_m = Parameter('omega_m', value=0.3, prior=dict(dist='norm', loc=0.3, scale=0.01))
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)

    prior = Prior(omega_m=omega_m, z=z, A=A, ns=ns)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianChi2(spectrum=spectrum, data=DATA)
    pipe = compile(Posterior(likelihood, prior))

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    got = float(pipe(params)[0])
    logP = float(jax.scipy.stats.norm.logpdf(0.3, loc=0.3, scale=0.01))
    assert abs(got - (analytic_logL(0.3, 0.5, 1.0, 0.96) + logP)) < 1e-8
    assert set(jax.grad(pipe, has_aux=True)(params)[0].keys()) == {'omega_m', 'z', 'A', 'ns'}
    assert abs(float(jax.jit(pipe)(params)[0]) - got) < 1e-8


def test_posterior_early_exit():
    """Likelihood is not called when logprior == -inf (eager mode)."""
    _call_count = [0]

    class CountedLikelihood(Calculator):
        def __post_init__(self, spectrum, data, sigma=0.1):
            self.spectrum = spectrum
            self._data = data
            self._sigma = sigma

        def __call__(self):
            _call_count[0] += 1
            self.loglikelihood = -0.5 * jnp.sum(((self.spectrum.pk - self._data) / self._sigma) ** 2)
            return self.loglikelihood

        def tree_flatten(self):
            return [self.loglikelihood], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.loglikelihood = children[0]
            return obj

    omega_m = Parameter('omega_m', value=0.3, prior=dict(dist='uniform', limits=(0.2, 0.5)))
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)

    prior = Prior(omega_m=omega_m, z=z, A=A, ns=ns)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = CountedLikelihood(spectrum=spectrum, data=DATA)
    pipe = compile(Posterior(likelihood, prior))
    _call_count[0] = 0

    pipe({'omega_m': 0.35, 'z': 0.5, 'A': 1.0, 'ns': 0.96})
    assert _call_count[0] == 1, f"Expected 1 call, got {_call_count[0]}"
    assert float(pipe({'omega_m': 0.1, 'z': 0.5, 'A': 1.0, 'ns': 0.96})[0]) == float(-jnp.inf)
    assert _call_count[0] == 1, f"Expected still 1 call (no likelihood), got {_call_count[0]}"
    jax.jit(pipe)({'omega_m': 0.1, 'z': 0.5, 'A': 1.0, 'ns': 0.96})
    assert _call_count[0] == 2, f"Expected 2 calls (jit always traces), got {_call_count[0]}"


# ── GaussianLikelihood / analytic marginalisation ─────────────────────────────

def test_gaussian_likelihood_base():
    """GaussianLikelihood: logpdf, tree_flatten, and grad flow through theory."""

    class SpectrumLikelihood(GaussianLikelihood):
        def __post_init__(self, spectrum, data, covariance):
            self.spectrum = spectrum
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.linalg.inv(jnp.asarray(covariance))

        def __call__(self):
            self.flattheory = self.spectrum.pk
            return super().__call__()

    sigma = 0.1
    _, _, _, _, _, spectrum, _ = _make_nodes()
    lik = SpectrumLikelihood(spectrum=spectrum, data=DATA, covariance=np.eye(len(K)) * sigma ** 2)
    pipe = compile(lik)

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    assert abs(float(pipe(params)[0]) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8

    lik()
    children, aux = lik.tree_flatten()
    assert len(children) == 3
    recon = SpectrumLikelihood.tree_unflatten(aux, children)
    assert jnp.allclose(recon.flattheory, lik.flattheory)
    assert jnp.allclose(recon.precision, lik.precision)
    assert set(jax.grad(pipe, has_aux=True)(params)[0].keys()) == {'omega_m', 'z', 'A', 'ns'}


def test_analytic_marginalization():
    """AnalyticMarginalization: logpdf matches analytical formula; grad flows through theta."""
    sigma_d, sigma_alpha = 0.1, 2.0
    A_val, alpha_0 = 1.0, 0.0

    A = Parameter('A', value=A_val)
    alpha = Parameter('alpha', value=alpha_0, derived='marg', prior=dict(dist='norm', loc=0., scale=sigma_alpha))
    pipe = compile(Posterior(_LinearTheory(A=A, alpha=alpha, data=DATA, covariance=np.eye(len(K)) * sigma_d ** 2), Prior()))

    params = {'A': A_val, 'alpha': alpha_0}
    got = float(pipe(params)[0])

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np
    B = K_np[:, None]
    P = np.eye(len(K)) / sigma_d ** 2
    P_alpha = np.array([[1.0 / sigma_alpha ** 2]])
    F = B.T @ P @ B + P_alpha
    b = B.T @ (P @ r)
    expected = float(-0.5 * r @ P @ r + 0.5 * float(b @ np.linalg.solve(F, b)) + 0.5 * np.log(1.0 / sigma_alpha ** 2) - 0.5 * np.log(float(F.flat[0])))
    assert abs(got - expected) < 1e-6, f'marg logpdf: got {got:.8f}, expected {expected:.8f}'

    grad = jax.grad(pipe, has_aux=True)(params)[0]
    eps = 1e-5
    fd = (float(pipe({'A': A_val + eps, 'alpha': alpha_0})[0]) - float(pipe({'A': A_val - eps, 'alpha': alpha_0})[0])) / (2 * eps)
    assert abs(float(grad['A']) - fd) < 1e-4
    assert abs(float(jax.jit(pipe)(params)[0]) - got) < 1e-8


def test_best_fit_solved():
    """derived='best': profile likelihood — parameter at MLE, no volume factor."""
    sigma_d = 0.1
    A_val, alpha_0 = 1.0, 0.0

    A = Parameter('A', value=A_val)
    alpha = Parameter('alpha', value=alpha_0, derived='best')
    pipe = compile(Posterior(_LinearTheory(A=A, alpha=alpha, data=DATA, covariance=np.eye(len(K)) * sigma_d ** 2), Prior()))

    params = {'A': A_val, 'alpha': alpha_0}
    got = float(pipe(params)[0])

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np
    B = K_np[:, None]
    P = np.eye(len(K)) / sigma_d ** 2
    F = B.T @ P @ B
    b = B.T @ (P @ r)
    expected = float(-0.5 * r @ P @ r + 0.5 * float(b @ np.linalg.solve(F, b)))
    assert abs(got - expected) < 1e-6

    eps = 1e-5
    fd = (float(pipe({'A': A_val + eps, 'alpha': alpha_0})[0]) - float(pipe({'A': A_val - eps, 'alpha': alpha_0})[0])) / (2 * eps)
    assert abs(float(jax.grad(pipe, has_aux=True)(params)[0]['A']) - fd) < 1e-4
    assert abs(float(jax.jit(pipe)(params)[0]) - got) < 1e-8


def test_mixed_marg_best():
    """Mixed derived='marg' and derived='best': volume factor only for 'marg' param."""

    class TwoParamTheory(GaussianLikelihood):
        def __post_init__(self, A, alpha_m, alpha_b, data, covariance):
            self.A = A
            self.alpha_m = alpha_m
            self.alpha_b = alpha_b
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.linalg.inv(jnp.asarray(covariance))

        def __call__(self):
            self.flattheory = (self.A + self.alpha_m + self.alpha_b) * K
            return super().__call__()

    sigma_d, sigma_m = 0.1, 2.0
    A_val = 1.0

    A = Parameter('A', value=A_val)
    alpha_m = Parameter('alpha_m', value=0.0, derived='marg', prior=dict(dist='norm', loc=0., scale=sigma_m))
    alpha_b = Parameter('alpha_b', value=0.0, derived='best')
    lik = TwoParamTheory(A=A, alpha_m=alpha_m, alpha_b=alpha_b, data=DATA, covariance=np.eye(len(K)) * sigma_d ** 2)
    pipe = compile(Posterior(lik, Prior()))

    params = {'A': A_val, 'alpha_m': 0.0, 'alpha_b': 0.0}
    got = float(pipe(params)[0])

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np
    P = np.eye(len(K)) / sigma_d ** 2
    Bmat = np.column_stack([K_np, K_np])
    F_full = Bmat.T @ P @ Bmat + np.diag([1.0 / sigma_m ** 2, 0.0])
    b_vec = Bmat.T @ (P @ r)
    quad = float(b_vec @ np.linalg.solve(F_full, b_vec))
    _, logdet_F = np.linalg.slogdet(F_full)
    expected = float(-0.5 * r @ P @ r + 0.5 * quad + 0.5 * np.log(1.0 / sigma_m ** 2) - 0.5 * float(logdet_F) + 0.5 * np.log(float(F_full[1, 1])))
    assert abs(got - expected) < 1e-6, f'mixed logpdf: got {got:.8f}, expected {expected:.8f}'

    assert 'A' in jax.grad(pipe, has_aux=True)(params)[0]
    assert abs(float(jax.jit(pipe)(params)[0]) - got) < 1e-8


def test_custom_jvp_linear_theory():
    """custom_jvp on the linear-alpha term: jacfwd uses the analytic rule; grad(posterior) works correctly."""

    @jax.custom_jvp
    def linear_term(alpha, template):
        return alpha * template

    @linear_term.defjvp
    def _(primals, tangents):
        alpha, template = primals
        dalpha, _ = tangents
        return alpha * template, dalpha * template

    class MixedTheory(GaussianLikelihood):
        def __post_init__(self, A, ns, alpha, data, cov):
            self.A = A
            self.ns = ns
            self.alpha = alpha
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.linalg.inv(jnp.asarray(cov))

        def __call__(self):
            self.flattheory = self.A * K ** self.ns + linear_term(self.alpha.value, K)
            return super().__call__()

    sigma_d, sigma_alpha = 0.1, 2.0
    A_val, ns_val, alpha_0 = 1.0, 0.96, 0.0

    A = Parameter('A', value=A_val)
    ns = Parameter('ns', value=ns_val)
    alpha = Parameter('alpha', value=alpha_0, derived='marg', prior=dict(dist='norm', loc=0., scale=sigma_alpha))
    lik = MixedTheory(A=A, ns=ns, alpha=alpha, data=DATA, cov=np.eye(len(K)) * sigma_d ** 2)
    pipe = compile(Posterior(lik, Prior()))

    params = {'A': A_val, 'ns': ns_val, 'alpha': alpha_0}
    got = float(pipe(params)[0])

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np ** ns_val
    B = K_np[:, None]
    P = np.eye(len(K)) / sigma_d ** 2
    P_alpha = np.array([[1.0 / sigma_alpha ** 2]])
    F = B.T @ P @ B + P_alpha
    b = B.T @ (P @ r)
    expected = float(-0.5 * r @ P @ r + 0.5 * float(b @ np.linalg.solve(F, b)) + 0.5 * np.log(float(P_alpha.flat[0])) - 0.5 * np.log(float(F.flat[0])))
    assert abs(got - expected) < 1e-6

    grad = jax.grad(pipe, has_aux=True)(params)[0]
    eps = 1e-5
    for name in ('A', 'ns'):
        fd = (float(pipe({**params, name: params[name] + eps})[0]) - float(pipe({**params, name: params[name] - eps})[0])) / (2 * eps)
        assert abs(float(grad[name]) - fd) < 1e-4
    assert abs(float(jax.jit(pipe)(params)[0]) - got) < 1e-8


def test_sum_likelihood():
    """SumLikelihood: logpdf is the sum of components; Posterior marginalizes only components that depend on solved params."""

    K1 = jnp.linspace(0.01, 0.2, 20)
    K2 = jnp.linspace(0.1, 0.3, 15)
    rng = np.random.default_rng(42)
    data1 = jnp.array(rng.normal(1.0, 0.1, len(K1)))
    data2 = jnp.array(rng.normal(0.5, 0.2, len(K2)))
    sigma1, sigma2, sigma_alpha = 0.1, 0.2, 1.5

    class Theory1(GaussianLikelihood):
        def __post_init__(self, A, alpha, data, cov):
            self.A = A
            self.alpha = alpha
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.linalg.inv(jnp.asarray(cov))

        def __call__(self):
            self.flattheory = (self.A + self.alpha) * K1
            return super().__call__()

    class Theory2(GaussianLikelihood):
        def __post_init__(self, B, data, cov):
            self.B = B
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.linalg.inv(jnp.asarray(cov))

        def __call__(self):
            self.flattheory = self.B * K2
            return super().__call__()

    A = Parameter('A', value=1.0)
    B = Parameter('B', value=0.5)
    alpha = Parameter('alpha', value=0.0, derived='marg', prior=dict(dist='norm', loc=0., scale=sigma_alpha))

    lik1 = Theory1(A=A, alpha=alpha, data=data1, cov=np.eye(len(K1)) * sigma1 ** 2)
    lik2 = Theory2(B=B, data=data2, cov=np.eye(len(K2)) * sigma2 ** 2)
    pipe = compile(Posterior(SumLikelihood(lik1, lik2), Prior()))

    A_val, B_val, alpha_val = 1.0, 0.5, 0.0
    params = {'A': A_val, 'B': B_val, 'alpha': alpha_val}
    got = float(pipe(params)[0])

    K1_np, K2_np = np.array(K1), np.array(K2)
    r1 = np.array(data1) - A_val * K1_np
    P1 = np.eye(len(K1)) / sigma1 ** 2
    B_mat = K1_np[:, None]
    P_alpha = np.array([[1.0 / sigma_alpha ** 2]])
    F = B_mat.T @ P1 @ B_mat + P_alpha
    b_vec = B_mat.T @ (P1 @ r1)
    logL1_marg = float(-0.5 * r1 @ P1 @ r1 + 0.5 * float(b_vec @ np.linalg.solve(F, b_vec)) + 0.5 * np.log(float(P_alpha.flat[0])) - 0.5 * np.log(float(F.flat[0])))
    r2 = np.array(data2) - B_val * K2_np
    logL2 = float(-0.5 * r2 @ np.eye(len(K2)) / sigma2 ** 2 @ r2)
    assert abs(got - (logL1_marg + logL2)) < 1e-6

    grad = jax.grad(pipe, has_aux=True)(params)[0]
    assert 'A' in grad and 'B' in grad
    assert abs(float(jax.jit(pipe)(params)[0]) - got) < 1e-8


# ── derived params ────────────────────────────────────────────────────────────

def test_derived_param_export():
    """Derived params (param.derived is True) set by a node's __call__ are readable after eager and JIT pipeline calls."""
    K_loc = jnp.linspace(0.01, 0.3, 20)
    sigma = 0.1
    rng = np.random.default_rng(17)
    data = jnp.array(rng.normal(1.0, sigma, 20))

    class TheoryWithDerived(GaussianLikelihood):
        def __post_init__(self, A, ns, data, sigma=0.1):
            self.A = A
            self.ns = ns
            self._sigma = sigma
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.eye(len(data)) / sigma ** 2
            self.chi2 = Parameter('chi2', value=0.0, derived=True)

        def __call__(self):
            self.flattheory = self.A * K_loc ** self.ns
            result = GaussianLikelihood.__call__(self)
            r = self.flatdata - self.flattheory
            self.chi2.value = jnp.sum(r ** 2) / self._sigma ** 2
            return result

    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    lik = TheoryWithDerived(A=A, ns=ns, data=data, sigma=sigma)
    pipe = compile(lik)

    def expected_chi2(A_val, ns_val):
        r = np.array(data) - A_val * np.array(K_loc) ** ns_val
        return float(np.sum(r ** 2) / sigma ** 2)

    params1 = {'A': 1.2, 'ns': 0.95}
    params2 = {'A': 0.9, 'ns': 0.98}

    _, deriveds = pipe(params1)
    assert abs(float(pipe.params['chi2'].value) - expected_chi2(1.2, 0.95)) < 1e-6
    assert abs(float(deriveds['chi2']) - expected_chi2(1.2, 0.95)) < 1e-6

    _, deriveds = jax.jit(pipe)(params2)
    for p in pipe._derived_params:
        p._value = np.asarray(deriveds[p.name])
    assert abs(float(pipe.params['chi2'].value) - expected_chi2(0.9, 0.98)) < 1e-6

    n = 4
    A_batch = jnp.linspace(0.8, 1.2, n)
    ns_batch = jnp.full(n, 0.96)
    expected_batch = jnp.array([expected_chi2(float(A_batch[i]), 0.96) for i in range(n)])
    _, deriveds = jax.jit(jax.vmap(pipe))({'A': A_batch, 'ns': ns_batch})
    for p in pipe._derived_params:
        p._value = np.asarray(deriveds[p.name])
    assert jnp.allclose(pipe.params['chi2'].value, expected_batch, atol=1e-6)


def test_init_params_immediate():
    """Parameters declared in __init__ are available before compile(), and the pipeline still works."""
    K_loc = jnp.linspace(0.01, 0.3, 20)
    data = jnp.array(np.random.default_rng(7).normal(0., 0.1, len(K_loc)))

    class QuickLikelihood(GaussianLikelihood):
        def __init__(self, A, ns, data, sigma=0.1):
            self.A = A
            self.ns = ns
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.eye(len(data)) / sigma ** 2

        def __call__(self):
            self.flattheory = self.A * K_loc ** self.ns
            return super().__call__()

    A = Parameter('A', value=1.2, prior=dict(dist='norm', loc=1.0, scale=0.5))
    ns = Parameter('ns', value=0.95)
    lik = QuickLikelihood(A=A, ns=ns, data=data)

    assert lik.A is A
    assert lik.ns is ns

    pipe = compile(lik)
    got = float(pipe({'A': 1.2, 'ns': 0.95})[0])
    r = np.array(data) - 1.2 * np.array(K_loc) ** 0.95
    assert abs(got - float(-0.5 * r @ r / 0.1 ** 2)) < 1e-6


def test_derived_expression_param():
    """Parameter(derived='a * b**2') evaluates and sets value via __call__."""
    omega_m = Parameter('omega_m', value=0.3)
    h = Parameter('h', value=0.67)

    # depends stores actual Parameter refs → __call__() reads their .value
    omega_cdm = Parameter('omega_cdm', derived='omega_m * h**2', depends={'omega_m': omega_m, 'h': h})
    assert omega_cdm._derived == 'omega_m * h**2'

    result = omega_cdm()
    assert abs(result - 0.3 * 0.67 ** 2) < 1e-12

    # updating a dep param propagates to the next __call__
    omega_m.value = 0.4
    result2 = omega_cdm()
    assert abs(result2 - 0.4 * 0.67 ** 2) < 1e-12
    assert abs(omega_cdm.value - 0.4 * 0.67 ** 2) < 1e-12

    # non-expression derived (e.g. derived=True) falls through to return value unchanged
    chi2 = Parameter('chi2', value=3.0, derived=True)
    assert chi2() == 3.0


if __name__ == '__main__':
    _, _, _, _, _, _, likelihood = _make_nodes()
    pipe = compile(likelihood)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    print('logL =', float(pipe(params)[0]))
    print('grad =', jax.grad(pipe, has_aux=True)(params)[0])
    batch = {'omega_m': jnp.linspace(0.25, 0.35, 4), 'z': jnp.full(4, 0.5), 'A': jnp.ones(4), 'ns': jnp.full(4, 0.96)}
    print('vmap logL =', jax.vmap(pipe)(batch)[0])
