"""Tests for desilike/base.py — cosmology-inspired pipeline (Cosmology → PowerSpectrum → GaussianChi2)."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from desilike.base import Calculator, Likelihood, GaussianLikelihood, SumLikelihood, Prior, Posterior, CompiledGraph, compile, pmap
from desilike.parameter import Parameter


# ── shared static data ────────────────────────────────────────────────────────

K = jnp.linspace(0.01, 0.3, 30)
DATA = jnp.array(np.random.default_rng(0).normal(1.0, 0.1, len(K)))


# ── toy calculators ───────────────────────────────────────────────────────────

class Cosmology(Calculator):
    """Non-JAX: growth_factor = omega_m^0.55 / (1 + z), growth_rate = omega_m^0.55."""
    _is_external = True
    _call_count = 0

    def __init__(self, omega_m, z):
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

    def __init__(self, cosmo, A, ns):
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

    def __init__(self, spectrum, data, sigma=0.1):
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

    def __init__(self, A, alpha, data, covariance):
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
    got = float(pipeline(omega_m=0.3, z=0.5, A=1.0, ns=0.96))
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
    assert abs(float(jax.jit(pipeline)(params)) - analytic_logL(0.3, 0.5, 1.2, 0.98)) < 1e-8


def test_grad(pipeline):
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    grad = jax.grad(pipeline)(params)
    eps = 1e-5
    for name in pipeline.params.names():
        fd = (float(pipeline({**params, name: params[name] + eps})) - float(pipeline({**params, name: params[name] - eps}))) / (2 * eps)
        assert abs(float(grad[name]) - fd) < 1e-4, f"grad[{name}]: got {float(grad[name]):.6f}, fd {fd:.6f}"


def test_vmap(pipeline):
    omega_m_vals = jnp.linspace(0.25, 0.35, 5)
    params_batch = {'omega_m': omega_m_vals, 'z': jnp.full(5, 0.5), 'A': jnp.ones(5), 'ns': jnp.full(5, 0.96)}
    batched = jax.vmap(pipeline)(params_batch)
    looped = jnp.stack([pipeline({'omega_m': float(omega_m_vals[i]), 'z': 0.5, 'A': 1.0, 'ns': 0.96}) for i in range(5)])
    assert jnp.allclose(batched, looped, atol=1e-8)


def test_jit_grad(pipeline):
    params = {'omega_m': 0.28, 'z': 0.6, 'A': 1.1, 'ns': 0.97}
    grad_eager = jax.grad(pipeline)(params)
    grad_jit = jax.jit(jax.grad(pipeline))(params)
    assert all(jnp.allclose(grad_eager[k], grad_jit[k], atol=1e-8) for k in grad_eager)


def test_jacrev_external():
    """jax.jacobian (jacrev) gives correct Jacobian for external (_is_external=True) pipelines."""
    _, _, _, _, _, spectrum, likelihood = _make_nodes()
    pipe_pk = compile(likelihood, output=lambda: spectrum.pk)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    jac_rev = jax.jacobian(pipe_pk)(params)

    for name in params:
        assert jac_rev[name].shape == (len(K),), f"jacrev[{name}].shape = {jac_rev[name].shape}"
    eps = 1e-5
    for name in params:
        fd = (np.asarray(pipe_pk({**params, name: params[name] + eps})) -
              np.asarray(pipe_pk({**params, name: params[name] - eps}))) / (2 * eps)
        assert np.allclose(np.asarray(jac_rev[name]), fd, atol=1e-4), \
            f"jacrev vs FD mismatch for {name}: max err = {np.abs(np.asarray(jac_rev[name]) - fd).max():.2e}"


def test_jacfwd_grad_external():
    """jax.jacfwd(jax.grad(pipe)) works for external (_is_external=True) pipelines via custom_jvp FD rule."""
    pipe = compile(_make_nodes()[-1])
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    grad_fn = jax.grad(pipe)
    hess_jacfwd = jax.jacfwd(grad_fn)(params)

    eps = 1e-5
    for p in pipe.params:
        ref = (float(grad_fn({**params, p.name: params[p.name] + eps})[p.name]) - float(grad_fn({**params, p.name: params[p.name] - eps})[p.name])) / (2 * eps)
        assert abs(float(hess_jacfwd[p.name][p.name]) - ref) < 1e-3, f"H[{p.name},{p.name}]: jacfwd={float(hess_jacfwd[p.name][p.name]):.6f}, FD={ref:.6f}"

    hess_jax = jax.hessian(pipe)(params)
    for p in pipe.params:
        assert abs(float(hess_jax[p.name][p.name]) - float(hess_jacfwd[p.name][p.name])) < 1e-10


# ── caching ───────────────────────────────────────────────────────────────────

def test_external_cache():
    """external (_is_external=True)() skipped when params and deps unchanged."""
    _call_count = [0]

    class CountedCosmology(Calculator):
        _is_external = True

        def __init__(self, omega_m, z):
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
        def __init__(self, cosmo, A, ns):
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
    assert jnp.allclose(pipe_pk(params), jnp.array(expected_pk), atol=1e-8)

    _, _, _, _, _, spectrum2, likelihood2 = _make_nodes()
    pipe_tuple = compile(likelihood2, output=lambda: (likelihood2.loglikelihood, spectrum2.pk))
    logL, pk = pipe_tuple(params)
    assert abs(float(logL) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8
    assert jnp.allclose(pk, jnp.array(expected_pk), atol=1e-8)
    grad = jax.grad(lambda p: jnp.sum(pipe_pk(p)))(params)
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
        def __init__(self, spectrum, data, w):
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
    got = float(pipe(params))
    D = 0.3 ** 0.55 / 1.5
    expected = -0.5 * np.sum(w * (1.0 * np.array(K) ** 0.96 * D ** 2 - np.array(DATA)) ** 2)
    assert abs(got - expected) < 1e-8

    grad = jax.grad(pipe)(params)
    assert grad['w'].shape == (len(K),)
    assert 'omega_m' in grad


def test_array_param_external():
    """external (_is_external=True): array-valued parameter (k-weights) FD grad is correct."""

    class WeightedCosmology(Calculator):
        _is_external = True

        def __init__(self, omega_m, k_weights):
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
        def __init__(self, wcos):
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
    got = float(pipe(params))
    D = 0.3 ** 0.55
    assert abs(got - (-0.5 * np.sum((k_weights * D) ** 2))) < 1e-8

    grad = jax.grad(pipe)(params)
    assert grad['k_weights'].shape == (len(K),)
    assert jnp.allclose(grad['k_weights'], jnp.array(-D ** 2 * k_weights), atol=1e-4)
    fd_om = (float(pipe({**params, 'omega_m': jnp.array(0.3 + 1e-5)})) - float(pipe({**params, 'omega_m': jnp.array(0.3 - 1e-5)}))) / 2e-5
    assert abs(float(grad['omega_m']) - fd_om) < 1e-4


# ── compilation edge cases ────────────────────────────────────────────────────

def test_internal_init():
    """Calculator and Parameter objects created inside init() are auto-discovered."""

    class InternalCosmology(Calculator):
        _is_external = True

        def __init__(self):
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
        def __init__(self):
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
    assert jnp.allclose(pipe(omega_m=0.3, z=0.5, A=1.0, ns=0.96), jnp.array(1.0 * np.array(K) ** 0.96 * D ** 2), atol=1e-8)


def test_duplicate_param_name_auto_shared():
    """Two distinct Parameter objects with the same name are auto-unified by build_graph.

    build_graph (and compile) no longer raise — same-named Parameters are merged
    automatically (first-seen wins), equivalent to an implicit share_params() call.
    """

    class DupCosmology(Calculator):
        _is_external = True

        def __init__(self):
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
        def __init__(self, cosmo):
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

    # Should NOT raise — auto-shared instead
    pipe = compile(DupSpectrum(cosmo=DupCosmology()))
    # Only one 'omega_m' in the compiled graph
    assert pipe.params.names() == ['omega_m']
    result = float(pipe({'omega_m': 0.3}))
    expected = 0.3 * 0.3 ** 0.55
    assert abs(result - expected) < 1e-8


def test_fd_acc():
    """param.fd_acc=4 gives smaller gradient error than fd_acc=2 at the same large step size."""

    class SinCosmology(Calculator):
        _is_external = True

        def __init__(self, omega_m):
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
        def __init__(self, cosmo):
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
        g = float(jax.grad(pipe)({'omega_m': jnp.array(x0)})['omega_m'])
        assert abs(g - analytic_grad) < tol, f'fd_acc={acc}: grad error {abs(g - analytic_grad):.2e} >= tol {tol:.2e}'

    om2 = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=2)
    om4 = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=4)
    pipe2 = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om2)))
    pipe4 = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om4)))
    err2 = abs(float(jax.grad(pipe2)({'omega_m': jnp.array(x0)})['omega_m']) - analytic_grad)
    err4 = abs(float(jax.grad(pipe4)({'omega_m': jnp.array(x0)})['omega_m']) - analytic_grad)
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
    assert np.isfinite(float(pipeline()))


def test_inplace_mutation_after_compile():
    """param.value changed in place after compile: pipeline uses the new value as default."""
    _, _, A, _, _, _, likelihood = _make_nodes()
    pipe = compile(likelihood)

    assert abs(float(pipe()) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8
    A.value = 1.5
    assert abs(float(pipe()) - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8
    jax.jit(pipe)({'omega_m': 0.3, 'z': 0.5, 'A': 1.5, 'ns': 0.96})
    assert abs(float(pipe()) - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8


# ── prior / posterior ─────────────────────────────────────────────────────────

def test_prior_standalone():
    """Prior alone: sums logpdf over non-fixed params, skips fixed ones."""
    omega_m = Parameter('omega_m', value=0.3, prior=dict(dist='norm', loc=0.3, scale=0.01))
    A = Parameter('A', value=1.0, prior=dict(dist='uniform', limits=(0.5, 2.0)))
    ns = Parameter('ns', value=0.96, fixed=True)
    pipe = compile(Prior(omega_m=omega_m, A=A, ns=ns))

    # ParameterPrior.logpdf is zero-lag: it subtracts the logpdf at the
    # distribution centre, so each term is 0 at the centre and < 0 elsewhere.
    # omega_m is evaluated off-centre (0.32) to exercise a non-trivial value;
    # A sits in the interior of its uniform prior, contributing 0.
    params = {'omega_m': 0.32, 'A': 1.0}
    got = float(pipe(params))
    norm_lp = lambda x: float(jax.scipy.stats.norm.logpdf(x, loc=0.3, scale=0.01))
    expected = norm_lp(0.32) - norm_lp(0.3)
    assert abs(got - expected) < 1e-8
    assert float(pipe({'omega_m': 0.3, 'A': 0.3})) == float(-jnp.inf)

    grad = jax.grad(pipe)(params)
    assert 'omega_m' in grad and 'A' in grad
    fd_om = (float(pipe({'omega_m': 0.32 + 1e-5, 'A': 1.0})) - float(pipe({'omega_m': 0.32 - 1e-5, 'A': 1.0}))) / 2e-5
    assert abs(float(grad['omega_m']) - fd_om) < 1e-6


def test_prior_in_posterior():
    """Prior combined with likelihood via a hand-rolled posterior node."""

    class LogPosterior(Calculator):
        def __init__(self, likelihood, prior):
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
    got = float(pipe(params))
    # Zero-lag prior: omega_m is evaluated at its centre (loc=0.3), so the prior
    # contributes 0; z/A/ns have no (proper) prior and also contribute 0.
    logP = 0.
    assert abs(got - (analytic_logL(0.3, 0.5, 1.0, 0.96) + logP)) < 1e-8
    assert set(jax.grad(pipe)(params).keys()) == {'omega_m', 'z', 'A', 'ns'}
    assert abs(float(jax.jit(pipe)(params)) - got) < 1e-8


def test_custom_prior_extra_condition():
    """CustomPrior adds a hard constraint (A + ns < 2.) on top of standard parameter priors."""

    class CustomPrior(Prior):
        """Standard logpdf, but returns -inf when A + ns >= 2."""

        def __init__(self, *args, A=None, ns=None, **kwargs):
            # *args forwarded to Prior so prior.update(vc) from Posterior.__init__ works.
            super().__init__(*args, A=A, ns=ns, **kwargs)

        def __call__(self):
            logpdf = super().__call__()
            A, ns = self.params['A'], self.params['ns']
            self.logpdf = jnp.where(A.value + ns.value < 2., logpdf, -jnp.inf)
            return self.logpdf

    A = Parameter('A', value=1.0, prior=dict(dist='norm', loc=1.0, scale=0.5))
    ns = Parameter('ns', value=0.96, prior=dict(dist='norm', loc=0.96, scale=0.05))
    # CustomPrior also works inside a full posterior pipeline.
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianChi2(spectrum=spectrum, data=DATA)
    custom_prior = CustomPrior(A=A, ns=ns)
    pipe2 = compile(Posterior(likelihood, custom_prior))
    params2 = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    got2 = float(pipe2(params2))
    assert abs(got2 - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8
    # Condition violated in the posterior → -inf.
    assert float(pipe2({'omega_m': 0.3, 'z': 0.5, 'A': 1.5, 'ns': 0.96})) == float(-jnp.inf)


def test_custom_prior_reparametrized():
    """CustomPrior sets omega_m = A inside __call__; Posterior propagates this to the likelihood."""

    class CustomPrior(Prior):
        """Reparametrizes omega_m := A before the likelihood runs."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.params['omega_m'].update(derived=True)  # note as derived parameter

        def __call__(self):
            logpdf = super().__call__()
            self.params['omega_m'].value = self.params['A'].value
            return logpdf

    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=0.3)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianChi2(spectrum=spectrum, data=DATA)

    pipe = compile(Posterior(likelihood, CustomPrior(omega_m=omega_m, A=A)))

    # omega_m provided as 0.99 but prior reparametrizes it to A=0.3 → likelihood sees omega_m=0.3.
    params = {'z': 0.5, 'A': 0.3, 'ns': 0.96}
    got = float(pipe(params))
    expected = analytic_logL(0.3, 0.5, 0.3, 0.96)
    assert abs(got - expected) < 1e-8, f'got {got}, expected {expected}'
    # Confirm this differs from the naive (non-reparametrized) result.
    assert abs(got - analytic_logL(0.99, 0.5, 0.3, 0.96)) > 1e-3


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
    got = float(pipe(params))
    # Zero-lag prior: omega_m evaluated at its centre (loc=0.3) → prior contributes 0.
    logP = 0.
    assert abs(got - (analytic_logL(0.3, 0.5, 1.0, 0.96) + logP)) < 1e-8
    assert set(jax.grad(pipe)(params).keys()) == {'omega_m', 'z', 'A', 'ns'}
    assert abs(float(jax.jit(pipe)(params)) - got) < 1e-8


def test_posterior_early_exit():
    """Likelihood is not called when logprior == -inf (eager mode)."""
    _call_count = [0]

    class CountedLikelihood(Calculator):
        def __init__(self, spectrum, data, sigma=0.1):
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
    assert float(pipe({'omega_m': 0.1, 'z': 0.5, 'A': 1.0, 'ns': 0.96})) == float(-jnp.inf)
    assert _call_count[0] == 1, f"Expected still 1 call (no likelihood), got {_call_count[0]}"
    jax.jit(pipe)({'omega_m': 0.1, 'z': 0.5, 'A': 1.0, 'ns': 0.96})
    assert _call_count[0] == 2, f"Expected 2 calls (jit always traces), got {_call_count[0]}"


# ── GaussianLikelihood / analytic marginalisation ─────────────────────────────

def test_gaussian_likelihood_base():
    """GaussianLikelihood: logpdf, tree_flatten, and grad flow through theory."""

    class SpectrumLikelihood(GaussianLikelihood):
        def __init__(self, spectrum, data, covariance):
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
    assert abs(float(pipe(params)) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8

    lik()
    children, aux = lik.tree_flatten()
    assert len(children) == 3
    recon = SpectrumLikelihood.tree_unflatten(aux, children)
    assert jnp.allclose(recon.flattheory, lik.flattheory)
    assert jnp.allclose(recon.precision, lik.precision)
    assert set(jax.grad(pipe)(params).keys()) == {'omega_m', 'z', 'A', 'ns'}


def test_analytic_marginalization():
    """AnalyticMarginalization: logpdf matches analytical formula; grad flows through theta."""
    sigma_d, sigma_alpha = 0.1, 2.0
    A_val, alpha_0 = 1.0, 0.0

    A = Parameter('A', value=A_val)
    alpha = Parameter('alpha', value=alpha_0, derived='marg', prior=dict(dist='norm', loc=0., scale=sigma_alpha))
    pipe = compile(Posterior(_LinearTheory(A=A, alpha=alpha, data=DATA, covariance=np.eye(len(K)) * sigma_d ** 2), Prior()))

    params = {'A': A_val, 'alpha': alpha_0}
    got = float(pipe(params))

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np
    B = K_np[:, None]
    P = np.eye(len(K)) / sigma_d ** 2
    P_alpha = np.array([[1.0 / sigma_alpha ** 2]])
    F = B.T @ P @ B + P_alpha
    b = B.T @ (P @ r)
    expected = float(-0.5 * r @ P @ r + 0.5 * float(b @ np.linalg.solve(F, b)) - 0.5 * np.log(float(F.flat[0])))
    assert abs(got - expected) < 1e-6, f'marg logpdf: got {got:.8f}, expected {expected:.8f}'

    grad = jax.grad(pipe)(params)
    eps = 1e-5
    fd = (float(pipe({'A': A_val + eps, 'alpha': alpha_0})) - float(pipe({'A': A_val - eps, 'alpha': alpha_0}))) / (2 * eps)
    assert abs(float(grad['A']) - fd) < 1e-4
    assert abs(float(jax.jit(pipe)(params)) - got) < 1e-8


def test_best_fit_solved():
    """derived='best': profile likelihood — parameter at MLE, no volume factor."""
    sigma_d = 0.1
    A_val, alpha_0 = 1.0, 0.0

    A = Parameter('A', value=A_val)
    alpha = Parameter('alpha', value=alpha_0, derived='best')
    pipe = compile(Posterior(_LinearTheory(A=A, alpha=alpha, data=DATA, covariance=np.eye(len(K)) * sigma_d ** 2), Prior()))

    params = {'A': A_val, 'alpha': alpha_0}
    got = float(pipe(params))

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np
    B = K_np[:, None]
    P = np.eye(len(K)) / sigma_d ** 2
    F = B.T @ P @ B
    b = B.T @ (P @ r)
    expected = float(-0.5 * r @ P @ r + 0.5 * float(b @ np.linalg.solve(F, b)))
    assert abs(got - expected) < 1e-6

    eps = 1e-5
    fd = (float(pipe({'A': A_val + eps, 'alpha': alpha_0})) - float(pipe({'A': A_val - eps, 'alpha': alpha_0}))) / (2 * eps)
    assert abs(float(jax.grad(pipe)(params)['A']) - fd) < 1e-4
    assert abs(float(jax.jit(pipe)(params)) - got) < 1e-8


def test_mixed_marg_best():
    """Mixed derived='marg' and derived='best': volume factor only for 'marg' param."""

    class TwoParamTheory(GaussianLikelihood):
        def __init__(self, A, alpha_m, alpha_b, data, covariance):
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
    got = float(pipe(params))

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np
    P = np.eye(len(K)) / sigma_d ** 2
    Bmat = np.column_stack([K_np, K_np])
    # Prior precision is added for every solved param (here only alpha_m=index 0 has one).
    F_full = Bmat.T @ P @ Bmat + np.diag([1.0 / sigma_m ** 2, 0.0])
    b_vec = Bmat.T @ (P @ r)
    quad = float(b_vec @ np.linalg.solve(F_full, b_vec))
    # Volume factor (Schur complement over 'best'): + ½ log|P_marg| − ½ log|F| + ½ log|F[best, best]|.
    _, logdet_F = np.linalg.slogdet(F_full)
    expected = float(-0.5 * r @ P @ r + 0.5 * quad - 0.5 * float(logdet_F) + 0.5 * np.log(float(F_full[1, 1])))
    assert abs(got - expected) < 1e-6, f'mixed logpdf: got {got:.8f}, expected {expected:.8f}'

    assert 'A' in jax.grad(pipe)(params)
    assert abs(float(jax.jit(pipe)(params)) - got) < 1e-8


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
        def __init__(self, A, ns, alpha, data, cov):
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
    got = float(pipe(params))

    K_np = np.array(K)
    r = np.array(DATA) - A_val * K_np ** ns_val
    B = K_np[:, None]
    P = np.eye(len(K)) / sigma_d ** 2
    P_alpha = np.array([[1.0 / sigma_alpha ** 2]])
    F = B.T @ P @ B + P_alpha
    b = B.T @ (P @ r)
    expected = float(-0.5 * r @ P @ r + 0.5 * float(b @ np.linalg.solve(F, b)) - 0.5 * np.log(float(F.flat[0])))
    assert abs(got - expected) < 1e-6

    grad = jax.grad(pipe)(params)
    eps = 1e-5
    for name in ('A', 'ns'):
        fd = (float(pipe({**params, name: params[name] + eps})) - float(pipe({**params, name: params[name] - eps}))) / (2 * eps)
        assert abs(float(grad[name]) - fd) < 1e-4
    assert abs(float(jax.jit(pipe)(params)) - got) < 1e-8


def test_sum_likelihood():
    """SumLikelihood: logpdf is the sum of components; Posterior marginalizes only components that depend on solved params."""

    K1 = jnp.linspace(0.01, 0.2, 20)
    K2 = jnp.linspace(0.1, 0.3, 15)
    rng = np.random.default_rng(42)
    data1 = jnp.array(rng.normal(1.0, 0.1, len(K1)))
    data2 = jnp.array(rng.normal(0.5, 0.2, len(K2)))
    sigma1, sigma2, sigma_alpha = 0.1, 0.2, 1.5

    class Theory1(GaussianLikelihood):
        def __init__(self, A, alpha, data, cov):
            self.A = A
            self.alpha = alpha
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.linalg.inv(jnp.asarray(cov))

        def __call__(self):
            self.flattheory = (self.A + self.alpha) * K1
            return super().__call__()

    class Theory2(GaussianLikelihood):
        def __init__(self, B, data, cov):
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
    got = float(pipe(params))

    K1_np, K2_np = np.array(K1), np.array(K2)
    r1 = np.array(data1) - A_val * K1_np
    P1 = np.eye(len(K1)) / sigma1 ** 2
    B_mat = K1_np[:, None]
    P_alpha = np.array([[1.0 / sigma_alpha ** 2]])
    F = B_mat.T @ P1 @ B_mat + P_alpha
    b_vec = B_mat.T @ (P1 @ r1)
    logL1_marg = float(-0.5 * r1 @ P1 @ r1 + 0.5 * float(b_vec @ np.linalg.solve(F, b_vec)) - 0.5 * np.log(float(F.flat[0])))
    r2 = np.array(data2) - B_val * K2_np
    logL2 = float(-0.5 * r2 @ np.eye(len(K2)) / sigma2 ** 2 @ r2)
    assert abs(got - (logL1_marg + logL2)) < 1e-6

    grad = jax.grad(pipe)(params)
    assert 'A' in grad and 'B' in grad
    assert abs(float(jax.jit(pipe)(params)) - got) < 1e-8


# ── derived params ────────────────────────────────────────────────────────────

def test_derived_param_export():
    """Derived params (param.derived is True) set by a node's __call__ are readable after eager and JIT pipeline calls."""
    K_loc = jnp.linspace(0.01, 0.3, 20)
    sigma = 0.1
    rng = np.random.default_rng(17)
    data = jnp.array(rng.normal(1.0, sigma, 20))

    class TheoryWithDerived(GaussianLikelihood):
        def __init__(self, A, ns, data, sigma=0.1):
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

    _, deriveds = pipe(params1, return_derived=True)
    assert abs(float(pipe.params['chi2'].value) - expected_chi2(1.2, 0.95)) < 1e-6
    assert abs(float(deriveds['chi2']) - expected_chi2(1.2, 0.95)) < 1e-6

    # jit: wrap in a lambda so that return_derived=True is a Python constant
    # (jit cannot trace through a Python bool kwarg directly).
    pipe_rd = lambda p: pipe(p, return_derived=True)
    _, deriveds = jax.jit(pipe_rd)(params2)
    for p in pipe._derived_params:
        p._value = np.asarray(deriveds[p.name])
    assert abs(float(pipe.params['chi2'].value) - expected_chi2(0.9, 0.98)) < 1e-6

    n = 4
    A_batch = jnp.linspace(0.8, 1.2, n)
    ns_batch = jnp.full(n, 0.96)
    expected_batch = jnp.array([expected_chi2(float(A_batch[i]), 0.96) for i in range(n)])
    _, deriveds = jax.jit(jax.vmap(pipe_rd))({'A': A_batch, 'ns': ns_batch})
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
    got = float(pipe({'A': 1.2, 'ns': 0.95}))
    r = np.array(data) - 1.2 * np.array(K_loc) ** 0.95
    assert abs(got - float(-0.5 * r @ r / 0.1 ** 2)) < 1e-6


# ── Calculator.clone() ────────────────────────────────────────────────────────

def test_clone_same_result():
    """clone() produces a graph that returns the same value as the original."""
    _, _, _, _, _, spectrum, _ = _make_nodes()
    spec2 = spectrum.clone()
    pipe1 = compile(spectrum)
    pipe2 = compile(spec2)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    assert jnp.allclose(pipe1(params), pipe2(params), atol=1e-8)


def test_clone_shares_init_params():
    """clone() reuses the original's constructor-arg objects (shallow): params are shared.

    Independence is obtained by passing freshly constructed nodes via clone(**kwargs),
    not automatically — see test_clone_override_kwarg.
    """
    _, _, _, _, _, spectrum, likelihood = _make_nodes()
    spec2 = spectrum.clone()
    pipe1 = compile(likelihood)
    pipe2 = compile(spec2)
    for name in pipe2.params.names():
        assert pipe1.params[name] is pipe2.params[name], \
            f"param {name!r} should be shared between original and shallow clone"


def test_clone_mutation_independence():
    """Mutating a param value in the clone does not affect the original pipeline."""
    _, _, A, _, _, spectrum, likelihood = _make_nodes()
    spec2 = spectrum.clone()
    pipe1 = compile(likelihood)
    pipe2 = compile(spec2)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    val1_before = float(pipe1(params))

    # Mutate the clone's A param (find it from pipe2.params)
    pipe2.params['A'].value = 3.0
    val1_after = float(pipe1(params))
    assert abs(val1_before - val1_after) < 1e-8, \
        "Mutating the clone's param changed the original pipeline"


def test_clone_override_kwarg():
    """clone(A=...) overrides the init argument and produces the expected result."""
    _, _, _, _, _, spectrum, _ = _make_nodes()
    new_A = Parameter('A', value=2.0)
    spec2 = spectrum.clone(A=new_A)
    pipe2 = compile(spec2)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 2.0, 'ns': 0.96}
    D = 0.3 ** 0.55 / 1.5
    expected_pk = 2.0 * np.array(K) ** 0.96 * D ** 2
    assert jnp.allclose(pipe2(params), jnp.array(expected_pk), atol=1e-8)


def test_clone_with_external_dep():
    """clone() shares dependency objects (shallow); pipelines called with explicit
    params are unaffected by mutations to the shared defaults' stored values."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    spec2 = spectrum.clone()
    pipe1 = compile(spectrum)
    pipe2 = compile(spec2)

    params = {'omega_m': 0.28, 'z': 0.6, 'A': 1.1, 'ns': 0.97}
    assert jnp.allclose(pipe1(params), pipe2(params), atol=1e-8)

    # Mutate the shared omega_m's stored value; pipe1 called with explicit params
    # overrides the stored value and so stays consistent.
    pipe2.params['omega_m'].value = 0.5
    ref = jnp.array(pipe1(params))
    assert jnp.allclose(ref, pipe1(params), atol=1e-8)  # explicit-param call unaffected
    # pipe2() (no explicit params) uses the mutated default; z, A, ns stay at construction defaults (0.5, 1.0, 0.96)
    D2 = 0.5 ** 0.55 / (1.0 + 0.5)
    expected_pk2 = 1.0 * np.array(K) ** 0.96 * D2 ** 2
    assert jnp.allclose(pipe2(), jnp.array(expected_pk2), atol=1e-8)


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

    # dotted parameter name (e.g. namespace separator) must not be mis-parsed as attribute access
    lrg_b1 = Parameter('LRG_ell0.b1', value=1.5)
    lrg_b2 = Parameter('LRG_ell0.b2', value=1.5 * 0.9984876216336505,
                        derived='LRG_ell0.b1 * 0.9984876216336505',
                        depends={'LRG_ell0.b1': lrg_b1})
    assert abs(lrg_b2() - 1.5 * 0.9984876216336505) < 1e-12
    lrg_b1.value = 2.0
    assert abs(lrg_b2() - 2.0 * 0.9984876216336505) < 1e-12

    # list-form depends: keys are inferred from dep.name
    p_a = Parameter('a', value=3.0)
    p_b = Parameter('b', value=4.0, derived='a * 2', depends=[p_a])
    assert abs(p_b() - 6.0) < 1e-12
    p_a.value = 5.0
    assert abs(p_b() - 10.0) < 1e-12


def test_graph_derived_expression_param():
    """Derived-expression params work inside a Posterior pipeline, including dotted names.

    Regression tests:
    - AttributeError when _run_graph set value on a param whose _call_fn was not None.
    - NameError when a dotted param name (e.g. 'LRG_ell0.b1') appears in a derived expression.
    """
    SCALE = 0.9984876216336505
    sigma_lik = 0.5
    sigma_b2 = 0.1  # non-trivial Gaussian prior width on the derived b2

    class BiasTheory(GaussianLikelihood):
        def __init__(self, b1, b2, data, sigma):
            self.b1 = b1
            self.b2 = b2
            self.flatdata = jnp.asarray(data)
            self.precision = jnp.eye(2) / sigma ** 2

        def __call__(self):
            self.flattheory = jnp.array([self.b1, self.b2])
            return super().__call__()

    data_b = jnp.array([1.5, 1.5 * SCALE])

    # ── plain names ──────────────────────────────────────────────────────────
    b1 = Parameter('b1', value=1.5, prior={'dist': 'uniform', 'limits': [0., 3.]})
    b2 = Parameter('b2', value=1.5 * SCALE, derived=f'b1 * {SCALE!r}', depends=[b1],
                   prior={'dist': 'norm', 'loc': 1.5 * SCALE, 'scale': sigma_b2})
    pipe = compile(Posterior(BiasTheory(b1, b2, data_b, sigma_lik), Prior(b1=b1, b2=b2)))

    # at default params: logL = 0, b2 prior at centre (zero-lag) = 0 → total = 0
    assert abs(float(pipe())) < 1e-8
    # shifting b1 recomputes b2 = b1*SCALE, which is penalised by the b2 prior
    assert float(pipe({'b1': 2.0})) < 0.

    # ── dotted names ('LRG_ell0.b1') ─────────────────────────────────────────
    lrg_b1 = Parameter('LRG_ell0.b1', value=1.5, prior={'dist': 'uniform', 'limits': [0., 3.]})
    lrg_b2 = Parameter('LRG_ell0.b2', value=1.5 * SCALE,
                        derived=f'LRG_ell0.b1 * {SCALE!r}', depends=[lrg_b1],
                        prior={'dist': 'norm', 'loc': 1.5 * SCALE, 'scale': sigma_b2})
    pipe_dot = compile(Posterior(BiasTheory(lrg_b1, lrg_b2, data_b, sigma_lik),
                                 Prior(lrg_b1=lrg_b1, lrg_b2=lrg_b2)))

    assert abs(float(pipe_dot())) < 1e-8
    result_shifted = float(pipe_dot({'LRG_ell0.b1': 2.0}))
    assert result_shifted < 0.

    # gradient w.r.t. lrg_b1 must flow through both logL and the b2 prior chain
    grad = jax.grad(pipe_dot)({'LRG_ell0.b1': 1.5})
    assert jnp.isfinite(grad['LRG_ell0.b1']) and float(grad['LRG_ell0.b1']) != 0.


def test_eager_after_trace_no_stale_state():
    """Eager calls after jax.grad must not see stale trace-escaped state.

    Regression test for the shallow-copy save/restore bug: in-place mutations of
    nested mutable objects (e.g. a cache dict updated inside __call__) are not
    undone by dict(n.__dict__). Without last_params invalidation the next eager
    call with unchanged params skips re-running nodes and exposes stale Tracers.
    """
    # External node with a nested mutable cache dict — simulates cosmoprimo
    # _cosmo/_results that get modified in-place by downstream JAX __call__.
    class CachingCosmology(Calculator):
        _is_external = True

        def __init__(self, omega_m):
            self.omega_m = omega_m
            self._cache = {}  # mutable nested object — not captured by shallow copy

        def __call__(self):
            self._cache['growth'] = np.array(float(self.omega_m) ** 0.55)
            self.growth_factor = self._cache['growth']
            return self

        def tree_flatten(self):
            return [self.growth_factor], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.growth_factor = children[0]
            return obj

    # JAX node that reads from the cosmo node and stores a result in its own
    # mutable dict — simulating a theory calculator with an internal cache.
    class TheoryWithCache(Calculator):
        def __init__(self, cosmo, scale):
            self.cosmo = cosmo
            self.scale = scale
            self._theory_cache = {}

        def __call__(self):
            gf = self.cosmo.growth_factor
            pk = self.scale * gf ** 2
            self._theory_cache['pk'] = pk  # mutates nested mutable in-place
            self.pk = pk
            return self.pk

        def tree_flatten(self):
            return [self.pk], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.pk = children[0]
            return obj

    omega_m = Parameter('omega_m', value=0.3, prior={'dist': 'uniform', 'limits': [0.1, 0.9]})
    scale = Parameter('scale', value=2.0, prior={'dist': 'uniform', 'limits': [0.5, 5.0]})
    cosmo = CachingCosmology(omega_m=omega_m)
    theory = TheoryWithCache(cosmo=cosmo, scale=scale)

    pipe = compile(theory)

    params_P1 = {'omega_m': 0.3, 'scale': 2.0}
    params_P2 = {'omega_m': 0.3, 'scale': 3.0}

    # First eager call establishes baseline
    result_P1 = float(pipe(**params_P1))
    expected_P1 = 2.0 * (0.3 ** 0.55) ** 2
    assert abs(result_P1 - expected_P1) < 1e-8

    # Differentiate w.r.t. scale (JAX param — triggers jax.jvp trace through theory,
    # causing its _theory_cache dict to be mutated with a Tracer during the trace)
    jax.grad(pipe)(params_P1)

    # Eager call with P2 (scale changed) — must compute fresh, not serve stale P1
    result_P2 = float(pipe(**params_P2))
    expected_P2 = 3.0 * (0.3 ** 0.55) ** 2
    assert abs(result_P2 - expected_P2) < 1e-8

    # Eager call back to P1 — must also be correct
    result_P1_again = float(pipe(**params_P1))
    assert abs(result_P1_again - expected_P1) < 1e-8

    # _theory_cache['pk'] must be a plain concrete value, not a stale Tracer
    assert not isinstance(theory._theory_cache.get('pk'), jax.core.Tracer)


@pytest.mark.parametrize('backend', ['jax', 'mpi', 'mpi_and_jax'])
def test_pmap_generic(backend):
    """pmap(fn) batches an arbitrary function over pytree args/outputs like jax.vmap."""
    # scalar-leaf input, pytree (dict) output
    fn = lambda x: {'sq': x ** 2, 'neg': -x}
    x = jnp.linspace(0., 1., 13)            # odd size -> exercises device padding
    out = pmap(fn, backend=backend)(x)
    ref = jax.vmap(fn)(x)
    assert set(out) == set(ref)
    for key in ref:
        assert out[key].shape == ref[key].shape
        assert np.allclose(out[key], ref[key])

    # multiple positional args
    g = lambda a, b: a * b + 1.
    a, b = jnp.arange(7.), jnp.arange(7.) * 2.
    assert np.allclose(pmap(g, backend=backend)(a, b), jax.vmap(g)(a, b))

    # vector-leaf input, scalar output; and tuple output
    h = lambda v: (jnp.sum(v ** 2), v[0] - v[1])
    vb = jnp.arange(12.).reshape(6, 2)
    o0, o1 = pmap(h, backend=backend)(vb)
    r0, r1 = jax.vmap(h)(vb)
    assert np.allclose(o0, r0) and np.allclose(o1, r1)
    assert o0.shape == (6,) and o1.shape == (6,)

    # nested pytree input (dict of arrays)
    k = lambda d: d['a'] + 2. * d['b']
    d = {'a': jnp.arange(5.), 'b': jnp.arange(5.) + 10.}
    assert np.allclose(pmap(k, backend=backend)(d), jax.vmap(k)(d))


def test_pmap_mismatched_batch_raises():
    """pmap requires all batched leaves to share the leading axis size."""
    fn = lambda a, b: a + b
    with pytest.raises(ValueError):
        pmap(fn)(jnp.arange(4.), jnp.arange(5.))


def test_pmap_compiled_graph(pipeline):
    """pmap over a compiled pipeline matches jax.vmap of the same graph."""
    n = 9
    batch = {'omega_m': jnp.linspace(0.25, 0.35, n), 'z': jnp.full(n, 0.5),
             'A': jnp.ones(n), 'ns': jnp.full(n, 0.96)}
    out = pmap(pipeline, backend='mpi_and_jax')(batch)
    ref = jax.vmap(pipeline)(batch)
    assert out.shape == (n,)
    assert np.allclose(out, ref)


def test_build_graph_auto_share_params():
    """build_graph unifies same-named Parameters that are distinct objects across nodes.

    Two calculators constructed independently with a Parameter('omega_m', ...) should be
    compiled together without requiring an explicit share_params() call.
    """
    from desilike.base import build_graph, compile

    omega_m_1 = Parameter('omega_m', value=0.3)
    z_1 = Parameter('z', value=0.5)
    omega_m_2 = Parameter('omega_m', value=0.3)  # distinct object, same name
    z_2 = Parameter('z', value=0.5)              # distinct object, same name
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)

    cosmo1 = Cosmology(omega_m=omega_m_1, z=z_1)
    cosmo2 = Cosmology(omega_m=omega_m_2, z=z_2)
    spec1 = PowerSpectrum(cosmo=cosmo1, A=A, ns=ns)
    spec2 = PowerSpectrum(cosmo=cosmo2, A=A, ns=ns)

    class SumSpectrum(Calculator):
        def __init__(self, s1, s2):
            self.s1 = s1
            self.s2 = s2
        def __call__(self):
            self.pk = self.s1.pk + self.s2.pk
            return self.pk
        def tree_flatten(self):
            return [self.pk], None
        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.pk = children[0]
            return obj

    root = SumSpectrum(spec1, spec2)

    # build_graph should auto-share omega_m_1/omega_m_2 and z_1/z_2, not raise
    ctx = build_graph(root)

    # compile should succeed and produce a graph with only 4 unique params
    pipe = compile(root)
    assert set(pipe.params.names()) == {'omega_m', 'z', 'A', 'ns'}
    assert len(pipe.params) == 4

    # Both branches should see the same canonical omega_m after sharing
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    result = pipe(params)
    D = 0.3 ** 0.55 / 1.5
    expected = 2.0 * np.array(K) ** 0.96 * D ** 2
    assert jnp.allclose(result, jnp.array(expected), atol=1e-8)


if __name__ == '__main__':
    _, _, _, _, _, _, likelihood = _make_nodes()
    pipe = compile(likelihood)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    print('logL =', float(pipe(params)))
    print('grad =', jax.grad(pipe)(params))
    batch = {'omega_m': jnp.linspace(0.25, 0.35, 4), 'z': jnp.full(4, 0.5), 'A': jnp.ones(4), 'ns': jnp.full(4, 0.96)}
    print('vmap logL =', jax.vmap(pipe)(batch))
