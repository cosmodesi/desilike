import sys
import numpy as np
import pytest
from jax import numpy as jnp

import desilike.samplers as samplers
from desilike.samples import MCSamples
from desilike.base import compile, GaussianLikelihood as BaseGaussianLikelihood, Prior, Posterior
from desilike.parameter import Parameter
from desilike.distributed import get_mpicomm


SAMPLER_CLS = dict(
    dynesty=samplers.DynestySampler,
    emcee=samplers.EmceeSampler,
    grid=samplers.GridSampler,
    hmc=samplers.HMCSampler,
    importance=samplers.ImportanceSampler,
    mclmc=samplers.MCLMCSampler,
    mhmcmc=samplers.MetropolisHastingsSampler,
    nautilus=samplers.NautilusSampler,
    nuts=samplers.NoUTurnSampler,
    pocomc=samplers.PocoMCSampler,
    qmc=samplers.QMCSampler,
    zeus=samplers.ZeusSampler)
KWARGS_INIT = dict(
    dynesty=dict(dynamic=True, nlive=100),
    nautilus=dict(n_networks=1, n_live=300),
    pocomc=dict(n_effective=200, n_active=100))
KWARGS_INIT_FAST = dict(
    emcee=dict(nwalkers=5),
    dynesty=dict(dynamic=True, nlive=30),
    nautilus=dict(n_networks=1, n_live=100),
    pocomc=dict(n_effective=10, n_active=5))

# Per-parameter grids: a ~ N(0.4, 0.071) covered by [0.01, 0.99]; b ~ N(0.6, 0.316)
# needs a wider range to avoid truncation bias (b grid covers ±4σ around 0.6).
_a_grid = np.linspace(0.01, 0.99, 99)
_b_grid = np.linspace(-0.7, 1.9, 99)
# Iterative MCMC samplers: with ref distributions close to the posterior the default
# convergence criterion (single chain) can stop after very few steps, leaving the mean
# under-sampled.  A min_steps floor guarantees enough effective samples for the accuracy
# assertions, independent of how quickly the chain "looks" converged.
_MCMC_MIN_STEPS = dict(min_steps=3000)
_BLACKJAX_ADAPTATION = dict(adaptation=dict(steps=500))
KWARGS_RUN = dict(
    dynesty=dict(n_effective=0),
    emcee=_MCMC_MIN_STEPS,
    grid=dict(grid=dict(a=_a_grid, b=_b_grid)),
    hmc=dict(min_steps=10000, **_BLACKJAX_ADAPTATION),
    importance=dict(samples=MCSamples(dict(
        a=np.repeat(_a_grid, len(_b_grid)),
        b=np.tile(_b_grid, len(_a_grid))))),
    mclmc=_MCMC_MIN_STEPS,
    mhmcmc=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=sys.maxsize)),
    nautilus=dict(n_eff=100),
    nuts=dict(**_MCMC_MIN_STEPS, **_BLACKJAX_ADAPTATION),
    pocomc=dict(n_total=100, n_evidence=100),
    qmc=dict(size=10000),
    zeus=_MCMC_MIN_STEPS)
KWARGS_RUN_FAST = dict(
    dynesty=dict(maxiter=10),
    importance=dict(samples=MCSamples(dict(
        a=np.repeat(np.linspace(0.05, 0.95, 11), 11),
        b=np.tile(np.linspace(0.05, 0.95, 11), 11)))),
    emcee=dict(max_steps=10),
    grid=dict(grid=np.linspace(0.05, 0.95, 11)),
    hmc=dict(max_steps=10, **_BLACKJAX_ADAPTATION),
    mclmc=dict(max_steps=10),
    mhmcmc=dict(max_steps=10, adaptation=dict(steps=sys.maxsize)),
    nautilus=dict(n_eff=0, n_like_max=100),
    nuts=dict(max_steps=10, **_BLACKJAX_ADAPTATION),
    pocomc=dict(n_total=10, n_evidence=0),
    qmc=dict(size=100),
    zeus=dict(max_steps=10))


@pytest.fixture
def likelihood():

    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.flatdata = jnp.array([0.4, 0.6])
            self.precision = jnp.diag(jnp.array([100., 10.]))
            self.c = Parameter('c', value=0., derived=True)
            self.d = Parameter('d', value=jnp.zeros(3), shape=(3,), derived=True)

        def __call__(self):
            self.flattheory = jnp.array([self.a, self.b])
            self.c.value = self.a + self.b
            self.d.value = jnp.arange(3) * (self.a + self.b)
            return super().__call__()

    # ref distributions approximate the posterior (a ~ N(0.4, 1/sqrt(200)), b ~ N(0.6, 1/sqrt(10))),
    # so rescale=True (which uses ref.std()) whitens the problem to ~unit isotropic.
    a = Parameter('a', prior=dict(dist='norm', limits=[-10, 10.], loc=0.4, scale=0.1),
                  ref=dict(dist='norm', loc=0.4, scale=1. / np.sqrt(200.)))
    b = Parameter('b', prior=dict(dist='uniform', limits=[-10, 10.]),
                  ref=dict(dist='norm', loc=0.6, scale=1. / np.sqrt(10.)))
    like = Likelihood(a, b)
    graph = compile(Posterior(like, Prior(a, b)))
    graph.flatdata = like.flatdata.copy()
    graph.precision = like.precision.copy()
    graph.mpicomm = get_mpicomm()
    return graph


@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_accuracy(likelihood, key):
    # Test that all samplers work with a simple two-dimensional likelihood and
    # produce acceptable results.
    optional_deps = dict(dynesty='dynesty', emcee='emcee', hmc='blackjax', mclmc='blackjax',
                         nautilus='nautilus', nuts='blackjax', pocomc='pocomc', zeus='zeus')
    if key in optional_deps:
        pytest.importorskip(optional_deps[key])

    sampler = SAMPLER_CLS[key](likelihood, rng=42, **KWARGS_INIT.get(key, {}))
    results = sampler.run(**KWARGS_RUN.get(key, {}))

    if sampler.mpicomm.rank == 0:
        # The mean should match.
        mean_samples = results.mean(['a', 'b'])
        assert np.allclose(mean_samples,
                           likelihood.flatdata, atol=0.05, rtol=0)
        # The covariance should match.
        cov_samples = results.covariance(['a', 'b'])
        cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
        cov_err = np.sqrt(
            (cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
        assert np.allclose(cov_samples, cov,
                           atol=3 * cov_err)


@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_rescale(likelihood, key):
    # Same as test_accuracy but exploring the rescaled parameter space (rescale=True):
    # the sampler works in rescaled coordinates while the posterior is evaluated in
    # original space, so the recovered mean/covariance must be unchanged.
    optional_deps = dict(dynesty='dynesty', emcee='emcee', hmc='blackjax', mclmc='blackjax',
                         nautilus='nautilus', nuts='blackjax', pocomc='pocomc', zeus='zeus')
    if key in optional_deps:
        pytest.importorskip(optional_deps[key])

    sampler = SAMPLER_CLS[key](likelihood, rng=42, rescale=True, **KWARGS_INIT.get(key, {}))
    results = sampler.run(**KWARGS_RUN.get(key, {}))

    if sampler.mpicomm.rank == 0:
        mean_samples = results.mean(['a', 'b'])
        assert np.allclose(mean_samples, likelihood.flatdata, atol=0.05, rtol=0)
        cov_samples = results.covariance(['a', 'b'])
        cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
        cov_err = np.sqrt((cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
        assert np.allclose(cov_samples, cov, atol=3 * cov_err)


@pytest.mark.mpi
def test_importance_combine(likelihood):
    # Test that importance sampling can combine two likelihoods without
    # double counting the prior.

    sampler = samplers.GridSampler(likelihood)
    results = sampler.run(grid=dict(a=_a_grid, b=_b_grid))

    sampler = samplers.ImportanceSampler(likelihood)
    results = sampler.run(samples=results, resample=False)

    if sampler.mpicomm.rank == 0:
        cov = np.linalg.inv(2 * likelihood.precision +
                            np.array([[100, 0], [0, 0]]))
        assert np.allclose(results.mean(likelihood.params.select(varied=True)),
                           likelihood.flatdata, atol=1e-3, rtol=0)
        assert np.allclose(results.covariance(likelihood.params.select(varied=True)), cov,
                           atol=1e-3)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee'])
def test_derived(likelihood, key):
    # Test that derived parameters are correctly tracked.

    sampler = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    results = sampler.run(**KWARGS_RUN_FAST.get(key, {}))
    if sampler.mpicomm.rank == 0:
        a, b, c, d = (np.asarray(results[n]) for n in ('a', 'b', 'c', 'd'))
        assert np.allclose(a + b, c)
        for i in range(3):
            assert np.allclose((a + b) * i, d[..., i])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee'])
def test_solved(likelihood, key):
    # Test that solved parameters are correctly tracked.

    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.flatdata = jnp.array([0.4, 0.6])
            self.precision = jnp.diag(jnp.array([100., 10.]))

        def __call__(self):
            self.flattheory = jnp.array([self.a, self.b])
            return super().__call__()

    a = Parameter('a', prior=dict(dist='norm', limits=[0, 1], loc=0.4, scale=0.1))
    b = Parameter('b', derived='best')
    solved_likelihood = compile(Posterior(Likelihood(a, b), Prior(a)))

    def best_fit_b_given_a(likelihood, a):
        data = likelihood.flatdata
        precision = likelihood.precision
        # Here theory = [a, b]
        # d chi2 / db = 2 * P[1] dot ([a, b] - data) = 0
        p10 = precision[1, 0]
        p11 = precision[1, 1]
        b = data[1] - (p10 / p11) * (a - data[0])
        return b

    sampler = SAMPLER_CLS[key](
        solved_likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    results = sampler.run(**KWARGS_RUN_FAST.get(key, {}))
    if sampler.mpicomm.rank == 0:
        a_arr = np.asarray(results['a']).ravel()
        b_arr = np.asarray(results['b']).ravel()
        for i in range(3):
            assert np.allclose(b_arr[i], best_fit_b_given_a(likelihood, a_arr[i]))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_write(likelihood, key, tmp_path):
    # Check that the sampler correctly saves results and state, if applicable.
    optional_deps = dict(dynesty='dynesty', emcee='emcee', hmc='blackjax', mclmc='blackjax',
                         nautilus='nautilus', nuts='blackjax', pocomc='pocomc', zeus='zeus')
    if key in optional_deps:
        pytest.importorskip(optional_deps[key])

    sampler_1 = SAMPLER_CLS[key](
        likelihood, rng=42, directory=tmp_path,
        **KWARGS_INIT_FAST.get(key, {}))
    results_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    # The second sampler should not create any new samples if old results
    # are read correctly.
    sampler_2 = SAMPLER_CLS[key](
        likelihood, rng=43, directory=tmp_path,
        **KWARGS_INIT_FAST.get(key, {}))
    results_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    if sampler_1.mpicomm.rank == 0:
        assert len(results_1) == len(results_2)
        assert np.allclose(results_1.logposterior,
                           results_2.logposterior, atol=1e-6)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER_CLS.keys())
def test_rng(likelihood, key):
    # Test that specifying the random seed leads to reproducible results.
    optional_deps = dict(dynesty='dynesty', emcee='emcee', hmc='blackjax', mclmc='blackjax',
                         nautilus='nautilus', nuts='blackjax', pocomc='pocomc', zeus='zeus')
    if key in optional_deps:
        pytest.importorskip(optional_deps[key])

    if key == 'zeus':
        pytest.skip("Zeus does not support specifying a random seed.")

    sampler_1 = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    results_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = SAMPLER_CLS[key](
        likelihood, rng=42, **KWARGS_INIT_FAST.get(key, {}))
    results_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    if sampler_1.mpicomm.rank == 0:
        assert len(results_1) == len(results_2)
        assert np.allclose(results_1.logposterior,
                           results_2.logposterior, atol=1e-6)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee', 'hmc', 'mhmcmc', 'zeus'])
def test_continue_chain(likelihood, key):
    # Test that we can continue a chain.
    optional_deps = dict(hmc='blackjax', zeus='zeus')
    if key in optional_deps:
        pytest.importorskip(optional_deps[key])

    sampler = SAMPLER_CLS[key](likelihood, rng=42)
    chains_10 = sampler.run(
        burn_in=0, min_steps=10, max_steps=10, concatenate=False)
    sampler = SAMPLER_CLS[key](
        likelihood, rng=43, chains=[c.copy() for c in chains_10] if sampler.mpicomm.rank == 0 else None)
    chains_20 = sampler.run(
        burn_in=0, min_steps=20, max_steps=20, concatenate=False)

    if sampler.mpicomm.rank == 0:
        for chain_10, chain_20 in zip(chains_10, chains_20, strict=True):
            assert len(chain_10) == 10
            assert len(chain_20) == 20
            assert np.allclose(np.asarray(chain_10['a']), np.asarray(chain_20['a'])[:10])


@pytest.mark.mpi
@pytest.mark.parametrize('key', ['emcee', 'hmc', 'mhmcmc', 'zeus'])
def test_multiple_chains(likelihood, key):
    # Test that we can run multiple chains in parallel.
    optional_deps = dict(hmc='blackjax', zeus='zeus')
    if key in optional_deps:
        pytest.importorskip(optional_deps[key])

    nchains = likelihood.mpicomm.size
    sampler = SAMPLER_CLS[key](likelihood, nchains=nchains, rng=42)
    chains_10 = sampler.run(
        burn_in=0, min_steps=10, max_steps=10, concatenate=False)
    if sampler.mpicomm.rank == 0:
        assert len(chains_10) == nchains
    sampler = SAMPLER_CLS[key](
        likelihood, rng=43, chains=[c.copy() for c in chains_10] if sampler.mpicomm.rank == 0 else None)
    chains_20 = sampler.run(
        burn_in=0, min_steps=20, max_steps=20, concatenate=False)

    if sampler.mpicomm.rank == 0:
        for chain_10, chain_20 in zip(chains_10, chains_20, strict=True):
            assert len(chain_10) == 10
            assert len(chain_20) == 20
            assert np.allclose(np.asarray(chain_10['a']), np.asarray(chain_20['a'])[:10])


@pytest.mark.mpi_skip
def test_metropolis_hastings_fast(likelihood):
    # Test we can pass fast parameters to the Metropolis-Hastings sampler.

    sampler = samplers.MetropolisHastingsSampler(
        likelihood, rng=42, fast=['a'], f_fast=1)
    sampler.run(max_steps=100)


# ── New kernel-based API tests ────────────────────────────────────────────────

KERNEL_SAMPLER = dict(
    emcee=lambda: samplers.Emcee(nwalkers=8),
    zeus=lambda: samplers.Zeus(nwalkers=8),
    mhmcmc=lambda: samplers.MetropolisHastings(),
    hmc=lambda: samplers.HMC(),
    nuts=lambda: samplers.NUTS(),
    mclmc=lambda: samplers.MCLMC(),
    numpyro_nuts=lambda: samplers.NumpyroNUTS(),
    numpyro_hmc=lambda: samplers.NumpyroHMC(),
    numpyro_barker=lambda: samplers.NumpyroBarkerMH(),
)
# SA is a gradient-free adaptive sampler; its covariance is inaccurate at low sample counts
# (same behavior as the legacy NumpyroSASampler, which is not in test_accuracy).
KERNEL_SAMPLER_RUNS = dict(numpyro_sa=lambda: samplers.NumpyroSA())
KERNEL_OPTIONAL_DEPS = dict(
    emcee='emcee', zeus='zeus', hmc='blackjax', nuts='blackjax', mclmc='blackjax',
    numpyro_nuts='numpyro', numpyro_hmc='numpyro', numpyro_barker='numpyro',
    numpyro_sa='numpyro',
)
KERNEL_KWARGS_RUN = dict(
    emcee=_MCMC_MIN_STEPS,
    zeus=_MCMC_MIN_STEPS,
    mhmcmc=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=sys.maxsize)),
    hmc=dict(min_steps=10000, **_BLACKJAX_ADAPTATION),
    nuts=dict(**_MCMC_MIN_STEPS, **_BLACKJAX_ADAPTATION),
    mclmc=_MCMC_MIN_STEPS,
    numpyro_nuts=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=500)),
    numpyro_hmc=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=500)),
    numpyro_barker=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=500)),
)
KERNEL_KWARGS_RUN_FAST = dict(
    emcee=dict(max_steps=10),
    zeus=dict(max_steps=10),
    mhmcmc=dict(max_steps=10, adaptation=dict(steps=sys.maxsize)),
    hmc=dict(max_steps=10, **_BLACKJAX_ADAPTATION),
    nuts=dict(max_steps=10, **_BLACKJAX_ADAPTATION),
    mclmc=dict(max_steps=10),
    numpyro_nuts=dict(max_steps=10, adaptation=dict(steps=100)),
    numpyro_hmc=dict(max_steps=10, adaptation=dict(steps=100)),
    numpyro_barker=dict(max_steps=10, adaptation=dict(steps=100)),
    numpyro_sa=dict(max_steps=10, adaptation=dict(steps=100)),
)


@pytest.mark.mpi
@pytest.mark.parametrize('key', KERNEL_SAMPLER.keys())
def test_kernel_accuracy(likelihood, key):
    """Kernel-based Sampler factory produces accurate results."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])

    sampler = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](), rng=42)
    results = sampler.run(**KERNEL_KWARGS_RUN.get(key, {}))

    if sampler.mpicomm.rank == 0:
        mean_samples = results.mean(['a', 'b'])
        assert np.allclose(mean_samples, likelihood.flatdata, atol=0.05, rtol=0)
        cov_samples = results.covariance(['a', 'b'])
        cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
        cov_err = np.sqrt(
            (cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
        assert np.allclose(cov_samples, cov, atol=3 * cov_err)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', KERNEL_SAMPLER_RUNS.keys())
def test_kernel_runs(likelihood, key):
    """Kernel-based Sampler factory runs without error (no accuracy check)."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])
    sampler = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER_RUNS[key](), rng=42)
    sampler.run(**KERNEL_KWARGS_RUN_FAST.get(key, {}))


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee'])
def test_kernel_derived(likelihood, key):
    """Kernel-based Sampler correctly computes derived parameters."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])

    sampler = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](), rng=42)
    results = sampler.run(**KERNEL_KWARGS_RUN_FAST.get(key, {}))
    if sampler.mpicomm.rank == 0:
        a, b, c, d = (np.asarray(results[n]) for n in ('a', 'b', 'c', 'd'))
        assert np.allclose(a + b, c)
        for i in range(3):
            assert np.allclose((a + b) * i, d[..., i])


if __name__ == '__main__':

    from desilike import setup_logging

    setup_logging()

    def likelihood():

        class Likelihood(BaseGaussianLikelihood):

            def __init__(self, a, b):
                self.a = a
                self.b = b
                self.flatdata = jnp.array([0.4, 0.6])
                self.precision = jnp.diag(jnp.array([100., 10.]))
                self.c = Parameter('c', value=0., derived=True)
                self.d = Parameter('d', value=jnp.zeros(3), shape=(3,), derived=True)

            def __call__(self):
                self.flattheory = jnp.array([self.a, self.b])
                self.c.value = self.a + self.b
                self.d.value = jnp.arange(3) * (self.a + self.b)
                return super().__call__()

        # ref distributions approximate the posterior (a ~ N(0.4, 1/sqrt(200)), b ~ N(0.6, 1/sqrt(10))),
        # so rescale=True (which uses ref.std()) whitens the problem to ~unit isotropic.
        a = Parameter('a', prior=dict(dist='norm', limits=[-10, 10.], loc=0.4, scale=0.1),
                    ref=dict(dist='norm', loc=0.4, scale=1. / np.sqrt(200.)))
        b = Parameter('b', prior=dict(dist='uniform', limits=[-10, 10.]),
                    ref=dict(dist='norm', loc=0.6, scale=1. / np.sqrt(10.)))
        like = Likelihood(a, b)
        graph = compile(Posterior(like, Prior(a, b)))
        graph.flatdata = like.flatdata.copy()
        graph.precision = like.precision.copy()
        graph.mpicomm = get_mpicomm()
        return graph

    posterior = likelihood()
    test_accuracy(posterior, 'pocomc')