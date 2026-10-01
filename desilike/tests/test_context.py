"""Constraints declared by calculators, the constraint-settings context, and parameter overrides."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import desilike
from desilike import profilers, samplers
from desilike.base import build, GaussianLikelihood as BaseGaussianLikelihood, Prior, Posterior, get_params
from desilike.context import constraint_logpdf, current_constraints, override, version
from desilike.parameter import Parameter, Constraint

jax.config.update('jax_enable_x64', True)

MU_X, MU_Y = 0.3, -0.7
SX, SY = 0.05, 0.1
CUT = 0.32          # the calculator is "valid" for x < CUT only; the likelihood peak is past it
SCALE = 0.01


def make_posterior():

    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.flatdata = jnp.array([MU_X + 0.05, MU_Y])
            self.precision = jnp.diag(jnp.array([1. / SX**2, 1. / SY**2]))
            self.valid_range = Constraint('valid_range', scale=SCALE, description='x below the cut')

        def __call__(self):
            # finite everywhere; the violation is reported, not masked
            self.flattheory = jnp.array([self.x, self.y])
            self.valid_range.value = jnp.maximum(self.x - CUT, 0.)
            return super().__call__()

    x = Parameter('x', value=MU_X, prior=dict(dist='uniform', limits=[-1, 1]), ref=dict(dist='norm', loc=MU_X - 0.05, scale=0.01))
    y = Parameter('y', value=MU_Y, prior=dict(dist='uniform', limits=[-1, 1]), ref=dict(dist='norm', loc=MU_Y, scale=SY))
    return build(Posterior(Likelihood(x, y), Prior(x, y)))


def test_constraint_node():
    constraint = Constraint('LRG1.box', scale=0.1, description='training box')
    assert constraint.derived and constraint.scale == 0.1 and constraint.namespace == 'LRG1'
    clone = constraint.clone()
    assert isinstance(clone, Constraint) and clone.scale == 0.1 and clone.description == 'training box'
    leaves, treedef = jax.tree_util.tree_flatten(constraint)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, Constraint) and rebuilt.scale == 0.1


def test_constraints_listed_and_valued():
    posterior = make_posterior()
    listed = get_params(posterior, filter='constraint')
    assert listed.names() == ['valid_range']
    _, derived = posterior({'x': CUT + 0.02, 'y': MU_Y}, return_derived=True)
    assert np.allclose(derived['valid_range'], 0.02)
    _, derived = posterior({'x': CUT - 0.02, 'y': MU_Y}, return_derived=True)
    assert np.allclose(derived['valid_range'], 0.)


def test_get_params_filter():
    """``get_params(..., filter=)`` takes a kind, a class or a callable."""
    posterior = make_posterior()
    assert get_params(posterior, filter='constraint').names() == get_params(posterior, filter=Constraint).names() == ['valid_range']
    parameters = get_params(posterior, filter='parameter').names()
    assert {'x', 'y'} <= set(parameters) and 'valid_range' not in parameters
    assert len(get_params(posterior, filter='variable')) == len(get_params(posterior))
    assert get_params(posterior, filter=lambda variable: variable.basename.startswith('valid')).names() == ['valid_range']
    with pytest.raises(ValueError, match='unknown kind'):
        get_params(posterior, filter='constraints')


def test_modes():
    posterior = make_posterior()
    inside, outside = {'x': CUT - 0.01, 'y': MU_Y}, {'x': CUT + 0.02, 'y': MU_Y}
    with override(posterior, constraints=None):
        reference = float(posterior(outside))
        reference_inside = float(posterior(inside))
    assert np.isfinite(reference)
    # default, outside any context: hard
    assert current_constraints() is None
    assert float(posterior(outside)) == -np.inf
    assert np.allclose(posterior(inside), reference_inside)
    with override(posterior, constraints='soft'):
        assert np.allclose(posterior(outside), reference - 0.5 * (0.02 / SCALE) ** 2)
        assert np.allclose(posterior(inside), reference_inside)
    with override(posterior, constraints={'mode': 'soft', 'scale': {'valid_range': 0.04}}):
        assert np.allclose(posterior(outside), reference - 0.5 * (0.02 / 0.04) ** 2)
    # restored on exit
    assert float(posterior(outside)) == -np.inf


def test_jit_cache_follows_settings():
    """A graph's own jitted function retraces under new settings instead of reusing the old mode."""
    posterior = make_posterior()
    outside = {'x': CUT + 0.02, 'y': MU_Y}
    hard_fn = posterior._jit_call_fn
    with override(posterior, constraints='soft'):
        soft_fn = posterior._jit_call_fn
        assert posterior._jit_call_fn is soft_fn   # cached within one settings version
    assert soft_fn is not hard_fn and posterior._jit_call_fn is not soft_fn
    # a jax.jit the caller wraps around the graph is traced once and keeps the settings of that
    # trace (documented in desilike.context): the kernels enter their own settings inside it
    with override(posterior, constraints='soft'):
        assert np.isfinite(float(jax.jit(lambda params: posterior(params))(outside)))
    start = version()
    with override(posterior, constraints='soft'):
        assert version() == start + 1
    assert version() == start + 2


def test_hard_value_soft_gradient():
    constraint = Constraint('box', scale=0.1)
    grad = jax.grad(lambda value: constraint_logpdf([constraint], {'box': value}, settings={'mode': 'hard', 'grad': 'soft'}))
    assert np.allclose(grad(0.05), -0.05 / 0.1 ** 2)
    value = constraint_logpdf([constraint], {'box': 0.05}, settings={'mode': 'hard', 'grad': 'soft'})
    assert float(value) == -np.inf


def test_profiler_soft_sampler_hard():
    pytest.importorskip('iminuit')
    posterior = make_posterior()
    profiler = profilers.Profiler(posterior, kernel=profilers.Minuit(), rng=42)
    assert profiler.constraints['mode'] == 'soft'
    profiles = profiler.maximize(niterations=2).choice(index='argmax', squeeze=True)
    # the likelihood peaks at 0.35, past the cut: the soft wall holds the best fit just above CUT
    expected = (CUT / SCALE**2 + (MU_X + 0.05) / SX**2) / (1. / SCALE**2 + 1. / SX**2)
    assert np.allclose(profiles.best['x'], expected, atol=2e-4)
    with override(posterior, constraints='hard'):
        sampler = samplers.Sampler(posterior, kernel=samplers.Emcee(nwalkers=8), rng=42)
    assert sampler.constraints['mode'] == 'hard'
    assert samplers.Sampler(posterior, kernel=samplers.Emcee(nwalkers=8), rng=42).constraints['mode'] == 'hard'


@pytest.mark.mpi_skip
def test_sampler_hard_wall():
    pytest.importorskip('emcee')
    posterior = make_posterior()
    sampler = samplers.Sampler(posterior, kernel=samplers.Emcee(nwalkers=8), rng=42)
    chain = sampler.run(max_steps=600, check_every=600)
    if sampler.mpicomm.rank == 0:
        assert np.all(np.asarray(chain['x']) <= CUT)
        assert np.all(np.asarray(chain['valid_range']) == 0.)   # constraint values are stored with the chain


def test_override():
    posterior = make_posterior()
    params = posterior.params
    x, y = params['x'], params['y']
    point = {'x': 0.1, 'y': MU_Y}
    base = float(posterior(point))   # an eager call writes its inputs into the parameters
    old_prior, old_ref, old_value = x.prior, y.ref, x.value
    with override(posterior, value={'x': 0.2}, prior={'x': {'limits': [0., 0.15]}},
                  ref={'y': {'dist': 'norm', 'loc': -0.5, 'scale': 0.2}}, constraints='soft'):
        assert x.value == 0.2
        assert x.prior.limits == (0., 0.15) and x.prior.dist == 'uniform'
        assert y.ref.dist == 'norm' and np.allclose(y.ref.attrs['loc'], -0.5)
        assert current_constraints()['mode'] == 'soft'
        # the narrower prior excludes x = 0.2; inside, the value is unchanged (uniform priors are unnormalised)
        assert np.allclose(float(posterior(point)), base)
        assert float(posterior({'x': 0.2, 'y': MU_Y})) == -np.inf
    assert x.prior is old_prior and y.ref is old_ref and x.value == old_value
    assert np.isfinite(float(posterior({'x': 0.2, 'y': MU_Y})))
    assert current_constraints() is None
    with pytest.raises(ValueError, match='unknown parameters'):
        with override(posterior, value={'nope': 1.}):
            pass
    # restored on error too
    with pytest.raises(RuntimeError):
        with override(posterior, prior={'x': {'limits': [0., 0.15]}}):
            raise RuntimeError
    assert x.prior is old_prior


W0, WA = -0.6, 0.5          # the likelihood peaks at w0 + wa = -0.1, a sigma inside the w0 + wa < 0 cut
SW0, SWA = 0.1, 0.3
W0WA_SCALE = 0.02


def make_w0wa_posterior(peak=(W0, WA)):
    """A Gaussian likelihood in (w0_fld, wa_fld) peaked at *peak*, with a prior declaring w0 + wa < 0
    as a Constraint, the pattern of ``full_shape.tools.get_prior``."""

    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, w0, wa):
            self.w0 = w0
            self.wa = wa
            self.flatdata = jnp.array(peak)
            self.precision = jnp.diag(jnp.array([1. / SW0**2, 1. / SWA**2]))

        def __call__(self):
            self.flattheory = jnp.array([self.w0, self.wa])
            return super().__call__()

    class W0WaPrior(Prior):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.w0_plus_wa = Constraint('w0_plus_wa', scale=W0WA_SCALE)

        def __call__(self):
            self.logpdf = super().__call__()
            self.w0_plus_wa.value = jnp.maximum(self.params['w0_fld'].value + self.params['wa_fld'].value, 0.)
            return self.logpdf

    w0 = Parameter('w0_fld', value=-1., prior=dict(limits=[-3., 1.]), ref=dict(dist='norm', loc=-0.8, scale=0.05))
    wa = Parameter('wa_fld', value=0., prior=dict(limits=[-3., 2.]), ref=dict(dist='norm', loc=0.3, scale=0.05))
    likelihood = Likelihood(w0, wa)
    return build(Posterior(likelihood, prior=W0WaPrior(w0, wa)))


def test_prior_declared_constraint():
    """A constraint declared by the prior (w0 + wa < 0) acts like one declared by the likelihood."""
    posterior = make_w0wa_posterior()
    assert get_params(posterior, filter='constraint').names() == ['w0_plus_wa']
    inside, outside = {'w0_fld': -0.6, 'wa_fld': 0.5}, {'w0_fld': -0.4, 'wa_fld': 0.5}
    with override(posterior, constraints=None):
        free = float(posterior(outside))
    assert float(posterior(outside)) == -np.inf                         # hard by default
    assert np.isfinite(float(posterior(inside)))
    with override(posterior, constraints='soft'):
        assert np.allclose(float(posterior(outside)), free - 0.5 * (0.1 / W0WA_SCALE) ** 2)
    _, derived = posterior(outside, return_derived=True)                 # recorded with the derived outputs
    assert np.allclose(derived['w0_plus_wa'], 0.1)


def test_prior_declared_constraint_kernels():
    pytest.importorskip('iminuit')
    posterior = make_w0wa_posterior()
    # with the cut inactive the best fit is the likelihood peak, inside the region
    profiles = profilers.Profiler(posterior, kernel=profilers.Minuit(), rng=42).maximize(niterations=2).choice(index='argmax', squeeze=True)
    assert np.allclose([profiles.best['w0_fld'], profiles.best['wa_fld']], [W0, WA], atol=1e-3)
    # likelihood peak past the cut (w0 + wa = +0.1): the soft wall holds the best fit at the minimum of
    # chi2 + (w0 + wa)^2 / scale^2, a 2x2 linear problem
    peak = np.array([-0.4, 0.5])
    precision = np.diag([1. / SW0**2, 1. / SWA**2]) + np.ones((2, 2)) / W0WA_SCALE**2
    expected = np.linalg.solve(precision, np.diag([1. / SW0**2, 1. / SWA**2]) @ peak)
    assert expected.sum() > 0.
    profiles = profilers.Profiler(make_w0wa_posterior(peak), kernel=profilers.Minuit(), rng=42).maximize(niterations=2).choice(index='argmax', squeeze=True)
    assert np.allclose([profiles.best['w0_fld'], profiles.best['wa_fld']], expected, atol=1e-3)
    pytest.importorskip('emcee')
    sampler = samplers.Sampler(posterior, kernel=samplers.Emcee(nwalkers=8), rng=42)
    chain = sampler.run(max_steps=1000, check_every=1000)
    if sampler.mpicomm.rank == 0:
        total = np.asarray(chain['w0_fld']) + np.asarray(chain['wa_fld'])
        assert np.all(total <= 0.)
        assert np.all(np.asarray(chain['w0_plus_wa']) == 0.)


def test_exports():
    assert desilike.override is override and desilike.Constraint is Constraint and not hasattr(desilike, 'constraints')
