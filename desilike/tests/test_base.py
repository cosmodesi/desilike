"""
Tests for desilike/base.py

Scenario (cosmology-inspired):
  Cosmology (External) ──┐
                          ├─▶ PowerSpectrum (JAX) ──▶ GaussianLikelihood (JAX)

Cosmology: non-JAX, simulates a Boltzmann-code-like call.
PowerSpectrum: pure JAX linear model.
GaussianLikelihood: pure JAX chi-squared.

All dependencies are declared in init(), not __init__.
CompiledPipeline calls init() lazily during graph construction.

Tests:
  1. Correctness: pipeline output matches analytic computation.
  2. jax.jit: compiles and gives same result.
  3. jax.grad: gradients match centered finite-difference estimates.
  4. jax.vmap: batch evaluation matches looped evaluation.
  5. Caching: framework-level cache skips call() when inputs unchanged.
  6. Auto-deps: calculators with no explicit _dependencies declaration.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from desilike.base import JAXCalculator, ExternalCalculator, CompiledPipeline, compile
from desilike.parameter import Parameter


# ── shared static data ────────────────────────────────────────────────────────

K = jnp.linspace(0.01, 0.3, 30)
DATA = jnp.array(np.random.default_rng(0).normal(1.0, 0.1, len(K)))


# ── toy calculators ───────────────────────────────────────────────────────────

class Cosmology(ExternalCalculator):
    """Non-JAX: growth_factor = omega_m^0.55 / (1 + z), growth_rate = omega_m^0.55."""
    _call_count = 0

    def init(self, omega_m, z):
        self.omega_m = omega_m
        self.z = z

    def call(self):
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


class PowerSpectrum(JAXCalculator):
    """JAX-native: P(k) = A * k^ns * D^2."""

    def init(self, cosmo, A, ns):
        self.cosmo = cosmo
        self.A = A
        self.ns = ns

    def call(self):
        self.pk = self.A * K ** self.ns * self.cosmo.growth_factor ** 2
        return self.pk

    def tree_flatten(self):
        return [self.pk], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk = children[0]
        return obj


class GaussianLikelihood(JAXCalculator):
    """JAX-native: logL = -0.5 * sum((theory - data)^2 / sigma^2)."""

    def init(self, spectrum, data, sigma=0.1):
        self.spectrum = spectrum
        self._data = data
        self._sigma = sigma

    def call(self):
        self.loglikelihood = -0.5 * jnp.sum(((self.spectrum.pk - self._data) / self._sigma) ** 2)
        return self.loglikelihood

    def tree_flatten(self):
        return [self.loglikelihood], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.loglikelihood = children[0]
        return obj


# ── analytic reference ────────────────────────────────────────────────────────

def analytic_logL(omega_m, z, A, ns, data=DATA, sigma=0.1):
    D = omega_m ** 0.55 / (1.0 + z)
    theory = A * np.array(K) ** ns * D ** 2
    return -0.5 * np.sum(((theory - np.array(data)) / sigma) ** 2)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)
    return compile(likelihood)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_param_names(pipeline):
    assert set(pipeline.params.names()) == {'omega_m', 'z', 'A', 'ns'}


def test_correctness(pipeline):
    got = float(pipeline(omega_m=0.3, z=0.5, A=1.0, ns=0.96))
    expected = analytic_logL(0.3, 0.5, 1.0, 0.96)
    assert abs(got - expected) < 1e-8, f"got {got}, expected {expected}"


def test_jit(pipeline):
    jit_pipeline = jax.jit(pipeline)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.2, 'ns': 0.98}
    got = float(jit_pipeline(params))
    expected = analytic_logL(0.3, 0.5, 1.2, 0.98)
    assert abs(got - expected) < 1e-8


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


def test_external_cache():
    """Framework-level cache: ExternalCalculator.call() skipped when params and deps unchanged."""
    _call_count = [0]

    class CountedCosmology(ExternalCalculator):
        def init(self, omega_m, z):
            self.omega_m = omega_m
            self.z = z

        def call(self):
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
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)
    pipe = compile(likelihood)
    _call_count[0] = 0  # reset after dry-run

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    pipe(params)
    pipe(params)
    assert _call_count[0] == 1, f"Expected 1 (cache hit on repeat), got {_call_count[0]}"

    pipe({'omega_m': 0.35, 'z': 0.5, 'A': 1.0, 'ns': 0.96})
    assert _call_count[0] == 2, f"Expected 2 (new params trigger rerun), got {_call_count[0]}"


def test_eager_attrs_updated(pipeline):
    """After an eager pipe(params) call, all node attributes reflect the computed values."""
    omega_m, z, A, ns = 0.28, 0.6, 1.1, 0.97
    pipeline(omega_m=omega_m, z=z, A=A, ns=ns)

    cosmo, spectrum, likelihood = pipeline.nodes

    expected_gf = omega_m ** 0.55 / (1.0 + z)
    expected_gr = omega_m ** 0.55
    expected_pk = A * np.array(K) ** ns * expected_gf ** 2
    expected_logL = analytic_logL(omega_m, z, A, ns)

    assert abs(float(cosmo.growth_factor) - expected_gf) < 1e-8
    assert abs(float(cosmo.growth_rate) - expected_gr) < 1e-8
    assert jnp.allclose(jnp.array(spectrum.pk), jnp.array(expected_pk), atol=1e-8)
    assert abs(float(likelihood.loglikelihood) - expected_logL) < 1e-8


def test_jit_grad(pipeline):
    params = {'omega_m': 0.28, 'z': 0.6, 'A': 1.1, 'ns': 0.97}
    grad_eager = jax.grad(pipeline)(params)
    grad_jit = jax.jit(jax.grad(pipeline))(params)
    assert all(jnp.allclose(grad_eager[k], grad_jit[k], atol=1e-8) for k in grad_eager)


def test_auto_deps():
    """Dependencies auto-detected from attributes set in init() — no _dependencies needed."""

    class AutoSpectrum(JAXCalculator):
        def init(self, cosmo, A, ns):
            self.cosmo = cosmo
            self.A = A
            self.ns = ns

        def call(self):
            self.pk = self.A * K ** self.ns * self.cosmo.growth_factor ** 2
            return self.pk

        def tree_flatten(self):
            return [self.pk], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.pk = children[0]
            return obj

    class AutoLikelihood(JAXCalculator):
        def init(self, spectrum, data, sigma=0.1):
            self.spectrum = spectrum
            self._data = data
            self._sigma = sigma

        def call(self):
            self.loglikelihood = -0.5 * jnp.sum(((self.spectrum.pk - self._data) / self._sigma) ** 2)
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
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = AutoSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = AutoLikelihood(spectrum=spectrum, data=DATA)
    pipe = compile(likelihood)

    assert set(pipe.params.names()) == {'omega_m', 'z', 'A', 'ns'}
    assert abs(float(pipe(omega_m=0.3, z=0.5, A=1.0, ns=0.96)) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8


def test_jax_cache():
    """JAXCalculator.call() skipped in eager mode when params and deps unchanged."""
    _call_count = [0]

    class CountedSpectrum(JAXCalculator):
        def init(self, cosmo, A, ns):
            self.cosmo = cosmo
            self.A = A
            self.ns = ns

        def call(self):
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
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)
    pipe = compile(likelihood)
    _call_count[0] = 0  # reset after dry-run

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    pipe(params)
    pipe(params)
    assert _call_count[0] == 1, f"Expected 1 (cache hit on repeat), got {_call_count[0]}"

    pipe({'omega_m': 0.3, 'z': 0.5, 'A': 1.2, 'ns': 0.96})  # A changed → spectrum reruns
    assert _call_count[0] == 2, f"Expected 2 (own param changed), got {_call_count[0]}"

    pipe({'omega_m': 0.35, 'z': 0.5, 'A': 1.2, 'ns': 0.96})  # omega_m changed → cosmo reruns → spectrum reruns
    assert _call_count[0] == 3, f"Expected 3 (dep rerun), got {_call_count[0]}"


def test_custom_output():
    """output= lambda reads any pytree of calculator attrs; grad flows through it."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)

    # single array attr
    pipe_pk = compile(likelihood, output=lambda: spectrum.pk)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    D = 0.3 ** 0.55 / 1.5
    expected_pk = 1.0 * np.array(K) ** 0.96 * D ** 2
    assert jnp.allclose(pipe_pk(params), jnp.array(expected_pk), atol=1e-8)

    # tuple output; grad via jnp.sum
    omega_m2 = Parameter('omega_m', value=0.3)
    z2 = Parameter('z', value=0.5)
    A2 = Parameter('A', value=1.0)
    ns2 = Parameter('ns', value=0.96)
    cosmo2 = Cosmology(omega_m=omega_m2, z=z2)
    spectrum2 = PowerSpectrum(cosmo=cosmo2, A=A2, ns=ns2)
    likelihood2 = GaussianLikelihood(spectrum=spectrum2, data=DATA)
    pipe_tuple = compile(likelihood2, output=lambda: (likelihood2.loglikelihood, spectrum2.pk))
    logL, pk = pipe_tuple(params)
    assert abs(float(logL) - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8
    assert jnp.allclose(pk, jnp.array(expected_pk), atol=1e-8)
    grad = jax.grad(lambda p: jnp.sum(pipe_pk(p)))(params)
    assert set(grad.keys()) == {'omega_m', 'z', 'A', 'ns'}


def test_pytree_registration():
    """Calculators are registered JAX pytrees: tree_leaves, tree_map, and jit work natively."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    pipe = compile(spectrum)
    pipe(omega_m=0.3, z=0.5, A=1.0, ns=0.96)

    # JAX sees the output arrays as leaves
    leaves = jax.tree_util.tree_leaves(spectrum)
    assert any(isinstance(l, (np.ndarray, jnp.ndarray)) for l in leaves)

    # tree_map over the node works
    doubled = jax.tree_util.tree_map(lambda x: x * 2, spectrum)
    assert jnp.allclose(doubled.pk, spectrum.pk * 2)

    # round-trip flatten → unflatten preserves values
    children, aux = spectrum.tree_flatten()
    reconstructed = PowerSpectrum.tree_unflatten(aux, children)
    assert jnp.allclose(reconstructed.pk, spectrum.pk)


def test_array_param_jax():
    """JAXCalculator: array-valued parameter (weights vector) flows correctly through pipeline."""

    class WeightedLikelihood(JAXCalculator):
        """logL = -0.5 * sum(w * (theory - data)^2) where w is an array param."""

        def init(self, spectrum, data, w):
            self.spectrum = spectrum
            self._data = data
            self.w = w

        def call(self):
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
    wlik = WeightedLikelihood(spectrum=spectrum, data=DATA, w=w_param)
    pipe = compile(wlik)

    w = np.ones(len(K))
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96, 'w': jnp.array(w)}
    got = float(pipe(params))
    D = 0.3 ** 0.55 / 1.5
    theory = 1.0 * np.array(K) ** 0.96 * D ** 2
    expected = -0.5 * np.sum(w * (theory - np.array(DATA)) ** 2)
    assert abs(got - expected) < 1e-8

    grad = jax.grad(pipe)(params)
    assert grad['w'].shape == (len(K),)
    assert 'omega_m' in grad


def test_array_param_external():
    """ExternalCalculator: array-valued parameter (k-weights) FD grad is correct."""

    class WeightedCosmology(ExternalCalculator):
        """Returns growth_factor weighted by k_weights array param."""

        def init(self, omega_m, k_weights):
            self.omega_m = omega_m
            self.k_weights = k_weights

        def call(self):
            D = np.array(self.omega_m ** 0.55)
            self.weighted_D = np.asarray(self.k_weights) * D
            return self

        def tree_flatten(self):
            return [self.weighted_D], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.weighted_D = children[0]
            return obj

    class SumLikelihood(JAXCalculator):
        def init(self, wcos):
            self.wcos = wcos

        def call(self):
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
    lik = SumLikelihood(wcos=wcos)
    pipe = compile(lik)

    k_weights = np.ones(len(K))
    params = {'omega_m': 0.3, 'k_weights': jnp.array(k_weights)}
    got = float(pipe(params))
    D = 0.3 ** 0.55
    expected = -0.5 * np.sum((k_weights * D) ** 2)
    assert abs(got - expected) < 1e-8

    grad = jax.grad(pipe)(params)
    assert grad['k_weights'].shape == (len(K),)
    expected_grad_kw = -D ** 2 * k_weights
    assert jnp.allclose(grad['k_weights'], jnp.array(expected_grad_kw), atol=1e-4)
    fd_omega_m = (float(pipe({**params, 'omega_m': jnp.array(0.3 + 1e-5)})) - float(pipe({**params, 'omega_m': jnp.array(0.3 - 1e-5)}))) / 2e-5
    assert abs(float(grad['omega_m']) - fd_omega_m) < 1e-4


def test_internal_init():
    """Calculator and Parameter objects created inside init() are auto-discovered."""

    class InternalCosmology(ExternalCalculator):
        def init(self):
            self.omega_m = Parameter('omega_m', value=0.3)
            self.z = Parameter('z', value=0.5)

        def call(self):
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

    class InternalSpectrum(JAXCalculator):
        def init(self):
            self.cosmo = InternalCosmology()   # dep created internally
            self.A = Parameter('A', value=1.0)
            self.ns = Parameter('ns', value=0.96)

        def call(self):
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
    expected_pk = 1.0 * np.array(K) ** 0.96 * D ** 2
    assert jnp.allclose(pipe(omega_m=0.3, z=0.5, A=1.0, ns=0.96), jnp.array(expected_pk), atol=1e-8)


def test_duplicate_param_name_raises():
    """Two distinct Parameter objects with the same name raise ValueError at compile time."""

    class DupCosmology(ExternalCalculator):
        def init(self):
            self.omega_m = Parameter('omega_m', value=0.3)

        def call(self):
            self.growth_factor = np.array(self.omega_m ** 0.55)
            return self

        def tree_flatten(self):
            return [self.growth_factor], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.growth_factor = children[0]
            return obj

    class DupSpectrum(JAXCalculator):
        def init(self, cosmo):
            self.cosmo = cosmo
            self.omega_m = Parameter('omega_m', value=0.3)  # different object, same name

        def call(self):
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
        def init(self, omega_m):
            self.omega_m = omega_m

        def call(self):
            x = float(self.omega_m)
            self.val = np.array(np.sin(x) + x ** 3)  # non-polynomial → acc matters
            return self

        def tree_flatten(self):
            return [self.val], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.val = children[0]
            return obj

    class TrivialLikelihood(JAXCalculator):
        def init(self, cosmo):
            self.cosmo = cosmo

        def call(self):
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
    # deliberately large step to make truncation error visible
    large_eps = 1e-2

    for acc, tol in [(2, 2e-4), (4, 1e-8)]:
        om = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=acc)
        pipe = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om)))
        g = float(jax.grad(pipe)({'omega_m': jnp.array(x0)})['omega_m'])
        err = abs(g - analytic_grad)
        assert err < tol, f'fd_acc={acc}: grad error {err:.2e} >= tol {tol:.2e}'

    # acc=4 error must be strictly smaller than acc=2 error at this step size
    om2 = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=2)
    om4 = Parameter('omega_m', value=x0, fd_eps=large_eps, fd_acc=4)
    pipe2 = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om2)))
    pipe4 = compile(TrivialLikelihood(cosmo=SinCosmology(omega_m=om4)))
    err2 = abs(float(jax.grad(pipe2)({'omega_m': jnp.array(x0)})['omega_m']) - analytic_grad)
    err4 = abs(float(jax.grad(pipe4)({'omega_m': jnp.array(x0)})['omega_m']) - analytic_grad)
    assert err4 < err2, f'fd_acc=4 error {err4:.2e} should be smaller than fd_acc=2 error {err2:.2e}'


def test_no_tracer_leakage_after_jit(pipeline):
    """After jax.jit(pipe)(params), node attrs and param values are concrete, not stale tracers."""
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    jax.jit(pipeline)(params)

    cosmo, spectrum, likelihood = pipeline.nodes

    # param values must be concrete (not JAX tracers)
    for p in pipeline.params:
        assert not isinstance(p._value, jax.core.Tracer), f'{p.name}.value is a stale tracer'
        assert isinstance(np.asarray(p.value), np.ndarray)

    # node output attrs must be concrete numpy/jax arrays, not tracers
    assert not isinstance(cosmo.growth_factor, jax.core.Tracer)
    assert not isinstance(spectrum.pk, jax.core.Tracer)
    assert not isinstance(likelihood.loglikelihood, jax.core.Tracer)

    # a subsequent no-arg eager call must work without UnexpectedTracerError
    result = float(pipeline())
    assert np.isfinite(result)


def test_inplace_mutation_after_compile():
    """param.value changed in place after compile: pipeline uses the new value as default."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)
    pipe = compile(likelihood)

    # Eager no-arg call uses initial defaults
    got0 = float(pipe())
    assert abs(got0 - analytic_logL(0.3, 0.5, 1.0, 0.96)) < 1e-8

    # Mutate A in place — next no-arg call must pick it up
    A.value = 1.5
    got1 = float(pipe())
    assert abs(got1 - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8

    # After a jit call, param.value may hold a stale tracer; no-arg eager call must still work
    jax.jit(pipe)({'omega_m': 0.3, 'z': 0.5, 'A': 1.5, 'ns': 0.96})
    got2 = float(pipe())
    assert abs(got2 - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8


def test_update_before_compile():
    """Calculator.update() replaces init arguments before compile() is called."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)

    # replace A with a new Parameter before compiling
    A2 = Parameter('A', value=1.5)
    spectrum.update(A=A2)

    pipe = compile(likelihood)
    assert pipe.params['A'].value == 1.5

    got = float(pipe(omega_m=0.3, z=0.5, A=1.5, ns=0.96))
    assert abs(got - analytic_logL(0.3, 0.5, 1.5, 0.96)) < 1e-8

    # updating a non-Parameter kwarg (sigma) also takes effect
    likelihood.update(sigma=0.2)
    pipe2 = compile(likelihood)
    got2 = float(pipe2(omega_m=0.3, z=0.5, A=1.5, ns=0.96))
    assert abs(got2 - analytic_logL(0.3, 0.5, 1.5, 0.96, sigma=0.2)) < 1e-8


def test_jacrev_external():
    """jax.jacobian (jacrev) gives correct Jacobian for ExternalCalculator pipelines."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)
    pipe_pk = compile(likelihood, output=lambda: spectrum.pk)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}

    jac_rev = jax.jacobian(pipe_pk)(params)

    # shape: each entry is (len(K),) since output is pk and params are scalar
    for name in params:
        assert jac_rev[name].shape == (len(K),), f"jacrev[{name}].shape = {jac_rev[name].shape}"

    # must agree with manual centered FD
    eps = 1e-5
    for name in params:
        fd = (np.asarray(pipe_pk({**params, name: params[name] + eps})) -
              np.asarray(pipe_pk({**params, name: params[name] - eps}))) / (2 * eps)
        assert np.allclose(np.asarray(jac_rev[name]), fd, atol=1e-4), \
            f"jacrev vs manual FD mismatch for {name}: max err = {np.abs(np.asarray(jac_rev[name]) - fd).max():.2e}"


def test_jacfwd_grad_external():
    """jax.jacfwd(jax.grad(pipe)) works for ExternalCalculator pipelines via custom_jvp FD rule."""
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    cosmo = Cosmology(omega_m=omega_m, z=z)
    spectrum = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianLikelihood(spectrum=spectrum, data=DATA)
    pipe = compile(likelihood)
    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}

    hess_jacfwd = jax.jacfwd(jax.grad(pipe))(params)

    # compare diagonal against manual FD of the gradient
    eps = 1e-5
    grad_fn = jax.grad(pipe)
    for p in pipe.params:
        g_plus = float(grad_fn({**params, p.name: params[p.name] + eps})[p.name])
        g_minus = float(grad_fn({**params, p.name: params[p.name] - eps})[p.name])
        ref = (g_plus - g_minus) / (2 * eps)
        got = float(hess_jacfwd[p.name][p.name])
        assert abs(got - ref) < 1e-3, f"H[{p.name},{p.name}]: jacfwd={got:.6f}, manual_fd={ref:.6f}"

    # jax.hessian is equivalent
    hess_jax = jax.hessian(pipe)(params)
    for p in pipe.params:
        assert abs(float(hess_jax[p.name][p.name]) - float(hess_jacfwd[p.name][p.name])) < 1e-10


if __name__ == '__main__':
    omega_m = Parameter('omega_m', value=0.3)
    z = Parameter('z', value=0.5)
    A = Parameter('A', value=1.0)
    ns = Parameter('ns', value=0.96)
    g = Cosmology(omega_m=omega_m, z=z)
    s = PowerSpectrum(cosmo=g, A=A, ns=ns)
    l = GaussianLikelihood(spectrum=s, data=DATA)
    pipe = compile(l)

    params = {'omega_m': 0.3, 'z': 0.5, 'A': 1.0, 'ns': 0.96}
    print('logL =', float(pipe(params)))
    print('grad =', jax.grad(pipe)(params))

    batch = {'omega_m': jnp.linspace(0.25, 0.35, 4), 'z': jnp.full(4, 0.5), 'A': jnp.ones(4), 'ns': jnp.full(4, 0.96)}
    print('vmap logL =', jax.vmap(pipe)(batch))
    print('All smoke tests passed.')
