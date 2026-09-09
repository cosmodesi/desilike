"""The data vector as a node: a likelihood registers it as a `Variable`, so it is read and
overridden through the ordinary parameter path rather than by assigning an attribute.

The motivating case is a synthetic (Asimov) data vector -- evaluate the pipeline at a fiducial
point, feed the prediction back in as the data -- and mock realisations for coverage tests.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from desilike.base import GaussianLikelihood, Posterior, Prior, build, get_params
from desilike.parameter import Parameter, Variable


K = np.linspace(0.01, 0.3, 8)


class _Line(GaussianLikelihood):
    """logpdf of a one-parameter line against a registered data vector."""

    def __init__(self, amplitude=None, data=None, name=None):
        self.name = name or type(self).__name__
        self.amplitude = amplitude if amplitude is not None else Parameter(
            'amplitude', value=1., prior={'limits': [0., 3.]})
        self.flatdata = Variable(f'{self.name}.flatdata', value=jnp.asarray(data))
        self.precision = jnp.eye(len(data))

    def __call__(self):
        self.flattheory = self.amplitude * K
        return super().__call__()


def test_the_data_is_a_node_the_graph_fills_and_a_caller_can_override():
    """It is a Variable on the graph, named for its owner, and invisible to every sampler."""
    data = 1. * K
    like = _Line(data=data)
    assert like.flatdata.name == '_Line.flatdata' and like.flatdata.shape == (len(K),)
    assert like.ndata == len(K)

    graph = build(like)
    assert graph.params.names() == ['amplitude', '_Line.flatdata']
    # `input`, `varied` and `solved` are Parameter properties, so a plain Variable is skipped
    # with no `fixed` flag and no special-casing -- which is why it is not a Parameter.
    assert graph.params.select(varied=True).names() == ['amplitude']
    assert graph.params.select(input=True).names() == ['amplitude']
    assert graph.params.select(solved=True).names() == []

    # The default is the released data, so an ordinary call is unchanged...
    assert float(graph(amplitude=1.)) == pytest.approx(0.)
    # ... and the override reaches the logpdf: a unit shift on every one of len(K) bins.
    shifted = float(graph(**{'amplitude': 1., '_Line.flatdata': np.asarray(data) + 1.}))
    assert shifted == pytest.approx(-0.5 * len(K))
    # And it sticks: an eager call leaves the tree at the values it used, so the vector stays
    # conditioned until something sets it back -- which is what makes a synthetic data vector a
    # state of the pipeline rather than an argument repeated at every call.
    assert float(graph(amplitude=1.)) == pytest.approx(-0.5 * len(K))
    assert float(graph(**{'amplitude': 1., '_Line.flatdata': data})) == pytest.approx(0.)


def test_an_asimov_vector_round_trips_to_zero_chi2():
    """Evaluate at a point, feed the prediction back as the data: chi2 is zero there, and the
    minimum has moved to that point rather than to the one the real data preferred."""
    graph = build(_Line(data=1. * K))
    graph(amplitude=2.)
    asimov = np.asarray(graph.root.flattheory)

    assert float(graph(**{'amplitude': 2., '_Line.flatdata': asimov})) == pytest.approx(0.)
    off = float(graph(**{'amplitude': 2.1, '_Line.flatdata': asimov}))
    assert off < -1e-6
    # the real data still prefers 1., so the two vectors are genuinely different (the graph is
    # still conditioned on `asimov` here, so the real one is passed back explicitly)
    assert float(graph(**{'amplitude': 2., '_Line.flatdata': 1. * K})) < -1e-6


def test_vmap_and_grad_over_the_data():
    """What an attribute cannot do: N realisations under one build, and a gradient of the
    logpdf with respect to the data itself."""
    data = 1. * K
    graph = build(_Line(data=data))

    def logpdf(vector):
        return graph({'amplitude': 1., '_Line.flatdata': vector})

    batch = jnp.asarray(np.array([data, data + 1., data + 2.]))
    values = jax.jit(jax.vmap(logpdf))(batch)
    np.testing.assert_allclose(np.asarray(values), -0.5 * len(K) * np.array([0., 1., 4.]), atol=1e-6)

    gradient = jax.grad(logpdf)(jnp.asarray(data) + 1.)
    # d/dd of -0.5 |d - t|^2 is -(d - t), which is -1 in every bin here
    np.testing.assert_allclose(np.asarray(gradient), -np.ones(len(K)), atol=1e-6)


def test_the_prior_ignores_the_data_variable():
    """`Prior(get_params(likelihood))` is on the critical path of every run and is handed every
    Variable the graph has.  A plain Variable has neither a prior nor a `fixed` flag; it
    contributes zero to the log-prior, so it is dropped rather than read."""
    like = _Line(data=1. * K)
    prior = Prior(get_params(like))
    assert prior.params.names() == ['amplitude']

    # the default prior is built from `get_params(likelihood)`, the path that hands it the data
    graph = build(Posterior(like))
    assert np.isfinite(float(graph(amplitude=1.)))
    # and the data is still overridable through the posterior
    assert float(graph(**{'amplitude': 1., '_Line.flatdata': np.asarray(1. * K) + 1.})) < -1.


def test_two_arms_in_one_likelihood_keep_distinct_names():
    """The namespace is the owner's `name` when it has one, so the same class twice in one
    graph does not collide -- `CompiledGraph` refuses two distinct Variables of one name."""
    from desilike.base import SumLikelihood

    first, second = _Line(data=1. * K, name='north'), _Line(data=1.1 * K, name='south')
    # the amplitude is declared by both and unified by name, as usual
    graph = build(SumLikelihood(first, second))
    assert 'north.flatdata' in graph.params.names() and 'south.flatdata' in graph.params.names()
    at_default = float(graph(amplitude=1.))
    # conditioning one arm on its own prediction leaves only the other arm's mismatch
    graph(amplitude=1.)
    asimov = np.asarray(second.flattheory)
    conditioned = float(graph(**{'amplitude': 1., 'south.flatdata': asimov}))
    assert conditioned > at_default


def test_the_clustering_observables_own_their_data():
    """The vector is registered on the observable, not on the likelihood: the observable is what
    carries a `name`, and `observable.flatdata` stays what the likelihood actually used."""
    import cosmoprimo.fiducial as fid
    from desilike.observables.galaxy_clustering import BAOCompressionObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology

    fiducial = fid.DESI(engine='eisenstein_hu')
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial=fiducial)
    observable = BAOCompressionObservable(data=[1., 1.], covariance=np.diag([1e-4, 1e-4]),
                                          parameters=['qpar', 'qper'], cosmo=cosmo, z=0.5,
                                          fiducial=fiducial)
    likelihood = ObservablesGaussianLikelihood(observables=observable)
    graph = build(likelihood)

    assert isinstance(observable.flatdata, Variable)
    assert observable.flatdata.name == 'bao.flatdata' and 'bao.flatdata' in graph.params.names()
    # the covariance correction counts varied *Parameters*; the data must not inflate nparams
    assert 'bao.flatdata' not in graph.params.select(input=True, varied=True).names()

    at_data = float(graph())
    assert at_data == pytest.approx(0., abs=1e-6)     # data sits at the fiducial prediction
    moved = float(graph(**{'bao.flatdata': np.array([1.01, 1.])}))
    assert moved == pytest.approx(-0.5 * 0.01 ** 2 / 1e-4, rel=1e-3)


def test_marginalisation_uses_the_data_the_graph_was_built_with():
    """Known limitation, asserted so it cannot change silently: analytic marginalisation reads
    each Gaussian component's data once, when `Posterior` is constructed.  Overriding a data
    Variable afterwards moves the plain chi-squared but not the marginalised solve, so a
    synthetic vector wants a fresh `Posterior`."""
    class _Marg(GaussianLikelihood):

        def __init__(self, offset=None, data=None):
            self.offset = offset if offset is not None else Parameter(
                'offset', value=0., derived='marg', prior=dict(dist='norm', loc=0., scale=1.))
            self.flatdata = Variable(f'{type(self).__name__}.flatdata', value=jnp.asarray(data))
            self.precision = jnp.eye(len(data))

        def __call__(self):
            self.flattheory = self.offset * jnp.ones(len(K))
            return super().__call__()

    data = np.zeros(len(K))
    graph = build(Posterior(_Marg(data=data), Prior()))
    baseline = float(graph())
    assert float(graph(**{'_Marg.flatdata': data + 1.})) == pytest.approx(baseline)


def test_neither_the_data_nor_the_theory_can_reach_a_chain():
    """A sampler writes `select(varied=True, derived=False)` and `select(derived=True) +
    select(solved=True)` per sample (`samplers/base.py`).  The data vector is `derived=False`
    and is not a Parameter, so it is in neither set -- which is what keeps a 3013-element array
    out of every row.  `flattheory` is not registered as a node at all, so the same holds.
    """
    like = _Line(data=1. * K)
    graph = build(like)

    assert like.flatdata.derived is False
    sampled = graph.params.select(varied=True, derived=False)
    saved = graph.params.select(derived=True) + graph.params.select(solved=True)
    assert '_Line.flatdata' not in sampled.names() and '_Line.flatdata' not in saved.names()

    graph(amplitude=1.)
    assert not isinstance(like.flattheory, Variable)
    assert not any(name.endswith('flattheory') for name in graph.params.names())
