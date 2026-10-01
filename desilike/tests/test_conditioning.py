"""Conditioner transforms: bijections, Jacobians, and their use by profilers and samplers."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy import stats

from desilike import profilers, samplers
from desilike.base import build, GaussianLikelihood as BaseGaussianLikelihood, Prior, Posterior
from desilike.conditioning import AffineConditioner, Conditioner, Logit, Log, Reparameterization
from desilike.parameter import Parameter, VariableCollection

jax.config.update('jax_enable_x64', True)

MU_X, MU_Y = 0.3, -0.7
SX, SY = 0.05, 0.1


def make_params(x_limits=(-1., 1.), y_limits=(-1., 1.)):
    x = Parameter('x', value=MU_X, prior=dict(dist='uniform', limits=list(x_limits)),
                  ref=dict(dist='norm', loc=MU_X, scale=SX))
    y = Parameter('y', value=MU_Y, prior=dict(dist='uniform', limits=list(y_limits)),
                  ref=dict(dist='norm', loc=MU_Y, scale=SY))
    return x, y


def make_posterior(x_limits=(-1., 1.), y_limits=(-1., 1.)):

    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.flatdata = jnp.array([MU_X, MU_Y])
            self.precision = jnp.diag(jnp.array([1. / SX**2, 1. / SY**2]))

        def __call__(self):
            self.flattheory = jnp.array([self.x, self.y])
            return super().__call__()

    x, y = make_params(x_limits, y_limits)
    return build(Posterior(Likelihood(x, y), Prior(x, y)))


def init_conditioner(conditioner, params=None):
    params = params if params is not None else VariableCollection(list(make_params()))
    conditioner.init(params)
    return conditioner


def test_alias():
    assert Conditioner is AffineConditioner


@pytest.mark.parametrize('rescale', [False, True])
def test_round_trip_and_log_det(rescale):
    params = VariableCollection(list(make_params((-1., 1.), (-2., np.inf))))
    forward = lambda y: jnp.stack([y[..., 0] + 0.1 * y[..., 1] ** 3 + y[..., 1], y[..., 1]], axis=-1)
    conditioners = [
        AffineConditioner(rescale=rescale, transforms=[Logit('x'), Log('y', side='low')]),
        AffineConditioner(rescale=rescale, bounded='logit'),
    ]
    for conditioner in conditioners:
        init_conditioner(conditioner, params)
        z = np.array([[0.3, -0.2], [-1.5, 2.]])
        x = np.asarray(conditioner.forward(z))
        assert np.all(x[:, 0] > -1.) and np.all(x[:, 0] < 1.) and np.all(x[:, 1] > -2.)
        assert np.allclose(conditioner.inverse(x), z)
        for point in z:
            expected = np.linalg.slogdet(jax.jacfwd(conditioner.forward)(jnp.asarray(point)))[1]
            if rescale:  # the affine constant is left out on purpose
                expected -= np.sum(np.log(conditioner._scale))
            assert np.allclose(conditioner.log_abs_det_jacobian(point), expected)
        # bounds of the transformed coordinates are the real line
        assert np.all(np.isinf(conditioner.prior_bounds()))
        # dict form
        sample = conditioner.forward({'x': z[0, 0], 'y': z[0, 1]})
        assert np.allclose([sample['x'], sample['y']], x[0])
    # the automatic log-determinant of a group reparameterization
    transform = Reparameterization(['x', 'y'], forward=forward, inverse=lambda x: x)
    y = jnp.array([[0.2, 0.5], [1., -1.]])
    for point, value in zip(y, transform.log_abs_det_jacobian(y)):
        assert np.allclose(value, np.linalg.slogdet(jax.jacfwd(forward)(point))[1])


def test_coordinate_maps():
    conditioner = init_conditioner(AffineConditioner(rescale=True, bounded='logit'))
    values = np.array([-0.9, 0., 0.4, 0.95])
    assert np.allclose(conditioner.forward_coordinate(conditioner.inverse_coordinate(values, 0), 0), values)


def test_duplicate_and_unbounded():
    with pytest.raises(ValueError, match='more than one transform'):
        init_conditioner(AffineConditioner(transforms=[Logit('x'), Log('x')]))
    x = Parameter('x', value=0., prior=dict(dist='norm', loc=0., scale=1.))
    with pytest.raises(ValueError, match='finite limits'):
        AffineConditioner(transforms=[Logit('x')]).init(VariableCollection([x]))


def test_no_transform_is_unchanged():
    """Without transforms the conditioner is the affine map it always was."""
    conditioner = init_conditioner(AffineConditioner(rescale=True))
    assert conditioner.is_linear
    z = np.array([0.5, -1.])
    assert np.allclose(conditioner.forward(z), z * np.array([SX, SY]) + np.array([MU_X, MU_Y]))
    assert np.allclose(conditioner.log_abs_det_jacobian(z), 0.)


@pytest.mark.parametrize('kernel', ['minuit', 'scipy'])
def test_profiler_logit_matches_affine(kernel):
    """A logit on the prior box moves neither the best fit nor the (Gaussian) errors."""
    if kernel == 'minuit':
        pytest.importorskip('iminuit')
    make_kernel = {'minuit': profilers.Minuit, 'scipy': profilers.Scipy}[kernel]
    results = {}
    for label, conditioner in [('affine', AffineConditioner()), ('logit', AffineConditioner(bounded='logit'))]:
        profiler = profilers.Profiler(make_posterior(), kernel=make_kernel(), conditioner=conditioner, rng=42)
        profiles = profiler.maximize(niterations=2)
        results[label] = profiles.choice(index='argmax', squeeze=True)
    for name, mu in [('x', MU_X), ('y', MU_Y)]:
        assert np.allclose(results['logit'].best[name], mu, atol=1e-3)
    if kernel == 'minuit':
        for name, sigma in [('x', SX), ('y', SY)]:
            assert np.allclose(results['logit'].error[name], sigma, rtol=0.05)


def test_profiler_covariance_with_transform():
    profiler = profilers.Profiler(make_posterior(), kernel=profilers.Scipy(), conditioner=AffineConditioner(bounded='logit'), rng=42)
    profiler.maximize()
    profiles = profiler.covariance()
    cov = np.asarray(profiles.covariance.value)
    assert np.allclose(np.sqrt(np.diag(cov)), [SX, SY], rtol=1e-3)


@pytest.mark.mpi_skip
def test_emcee_truncated_gaussian_with_logit():
    """Sampling in logit space recovers the truncated posterior: the Jacobian is accounted for."""
    pytest.importorskip('emcee')
    upper = MU_X + 0.5 * SX   # cut the x Gaussian half a sigma above its mean
    posterior = make_posterior(x_limits=(-1., upper))
    sampler = samplers.Sampler(posterior, kernel=samplers.Emcee(nwalkers=16), rng=42,
                               conditioner=AffineConditioner(bounded='logit'))
    chain = sampler.run(max_steps=4000, check_every=4000)
    if sampler.mpicomm.rank == 0:
        chain = chain.remove_burnin(0.3) if hasattr(chain, 'remove_burnin') else chain
        x = np.ravel(np.asarray(chain['x']))
        truth = stats.truncnorm((-1. - MU_X) / SX, (upper - MU_X) / SX, loc=MU_X, scale=SX)
        assert np.all(x < upper)
        assert abs(x.mean() - truth.mean()) < 0.1 * truth.std()
        assert abs(x.std() - truth.std()) < 0.1 * truth.std()
        # chains store the posterior in the original parameters, without the Jacobian
        logpost = np.ravel(np.asarray(chain.logposterior))
        y = np.ravel(np.asarray(chain['y']))
        expected = [float(posterior({'x': xx, 'y': yy})) for xx, yy in zip(x[:5], y[:5])]
        assert np.allclose(logpost[:5], expected)
