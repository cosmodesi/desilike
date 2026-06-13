import sys
import numpy as np
import pytest
from jax import numpy as jnp

import desilike.samplers as samplers
from desilike.samples import MCSamples
from desilike.base import compile, GaussianLikelihood as BaseGaussianLikelihood, Prior, Posterior
from desilike.parameter import Parameter
from desilike.distributed import get_mpicomm


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

KERNEL_SAMPLER = dict(
    emcee=lambda: samplers.Emcee(nwalkers=8),
    zeus=lambda: samplers.Zeus(nwalkers=8),
    mhmcmc=lambda: samplers.MH(),
    hmc=lambda: samplers.BlackjaxHMC(),
    nuts=lambda: samplers.BlackjaxNUTS(),
    mclmc=lambda: samplers.BlackjaxMCLMC(),
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

NESTED_KERNEL_SAMPLER = dict(
    dynesty=lambda: samplers.Dynesty(dynamic=True),
    nautilus=lambda: samplers.Nautilus(),
    pocomc=lambda: samplers.PocoMC(),
)
NESTED_KERNEL_OPTIONAL_DEPS = dict(
    dynesty='dynesty', nautilus='nautilus', pocomc='pocomc',
)
NESTED_KERNEL_KWARGS_RUN = dict(
    dynesty=dict(maxiter=10),
    nautilus=dict(n_eff=0, n_like_max=100),
    pocomc=dict(n_total=10, n_evidence=0),
)


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


# ── Accuracy ──────────────────────────────────────────────────────────────────

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


@pytest.mark.mpi
@pytest.mark.parametrize('key', KERNEL_SAMPLER.keys())
def test_kernel_rescale(likelihood, key):
    """Rescaling the parameter space does not change the recovered posterior."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])

    sampler = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](), rng=42, rescale=True)
    results = sampler.run(**KERNEL_KWARGS_RUN.get(key, {}))

    if sampler.mpicomm.rank == 0:
        mean_samples = results.mean(['a', 'b'])
        assert np.allclose(mean_samples, likelihood.flatdata, atol=0.05, rtol=0)
        cov_samples = results.covariance(['a', 'b'])
        cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
        cov_err = np.sqrt((cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
        assert np.allclose(cov_samples, cov, atol=3 * cov_err)


# ── Derived / solved parameters ───────────────────────────────────────────────

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


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee'])
def test_kernel_solved(likelihood, key):
    """Kernel-based Sampler correctly computes analytically-solved parameters."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])

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

    def best_fit_b_given_a(like, a):
        data = like.flatdata
        precision = like.precision
        p10 = precision[1, 0]
        p11 = precision[1, 1]
        return data[1] - (p10 / p11) * (a - data[0])

    sampler = samplers.Sampler(solved_likelihood, kernel=KERNEL_SAMPLER[key](), rng=42)
    results = sampler.run(**KERNEL_KWARGS_RUN_FAST.get(key, {}))
    if sampler.mpicomm.rank == 0:
        a_arr = np.asarray(results['a']).ravel()
        b_arr = np.asarray(results['b']).ravel()
        for idx in range(3):
            assert np.allclose(b_arr[idx], best_fit_b_given_a(likelihood, a_arr[idx]))


# ── Checkpointing / determinism / chain continuation ─────────────────────────

@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', KERNEL_SAMPLER.keys())
def test_kernel_write(likelihood, key, tmp_path):
    """Second sampler reading a saved checkpoint returns identical results."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])

    sampler_1 = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](),
                                  rng=42, output_dir=tmp_path)
    results_1 = sampler_1.run(**KERNEL_KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](),
                                  rng=43, output_dir=tmp_path)
    results_2 = sampler_2.run(**KERNEL_KWARGS_RUN_FAST.get(key, {}))

    if sampler_1.mpicomm.rank == 0:
        assert len(results_1) == len(results_2)
        assert np.allclose(results_1.logposterior, results_2.logposterior, atol=1e-6)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', KERNEL_SAMPLER.keys())
def test_kernel_rng(likelihood, key):
    """Fixing the random seed leads to reproducible results."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])
    if key == 'zeus':
        pytest.skip('Zeus does not support specifying a random seed.')

    sampler_1 = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](), rng=42)
    results_1 = sampler_1.run(**KERNEL_KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](), rng=42)
    results_2 = sampler_2.run(**KERNEL_KWARGS_RUN_FAST.get(key, {}))

    if sampler_1.mpicomm.rank == 0:
        assert len(results_1) == len(results_2)
        assert np.allclose(results_1.logposterior, results_2.logposterior, atol=1e-6)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee', 'hmc', 'mhmcmc', 'zeus'])
def test_kernel_continue_chain(likelihood, key):
    """A chain can be continued from a checkpoint."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])

    sampler = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](), rng=42)
    chains_10 = sampler.run(
        burn_in=0, min_steps=10, max_steps=10, concatenate=False)
    sampler = samplers.Sampler(
        likelihood, kernel=KERNEL_SAMPLER[key](), rng=43,
        chains=[c.copy() for c in chains_10] if sampler.mpicomm.rank == 0 else None)
    chains_20 = sampler.run(
        burn_in=0, min_steps=20, max_steps=20, concatenate=False)

    if sampler.mpicomm.rank == 0:
        for chain_10, chain_20 in zip(chains_10, chains_20, strict=True):
            assert len(chain_10) == 10
            assert len(chain_20) == 20
            assert np.allclose(np.asarray(chain_10['a']), np.asarray(chain_20['a'])[:10])


@pytest.mark.mpi
@pytest.mark.parametrize('key', ['emcee', 'hmc', 'mhmcmc', 'zeus'])
def test_kernel_multiple_chains(likelihood, key):
    """Multiple chains can be run in parallel across MPI ranks."""
    if key in KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(KERNEL_OPTIONAL_DEPS[key])

    nchains = likelihood.mpicomm.size
    sampler = samplers.Sampler(likelihood, kernel=KERNEL_SAMPLER[key](),
                                nparallel=nchains, rng=42)
    chains_10 = sampler.run(
        burn_in=0, min_steps=10, max_steps=10, concatenate=False)
    if sampler.mpicomm.rank == 0:
        assert len(chains_10) == nchains
    sampler = samplers.Sampler(
        likelihood, kernel=KERNEL_SAMPLER[key](), rng=43,
        chains=[c.copy() for c in chains_10] if sampler.mpicomm.rank == 0 else None)
    chains_20 = sampler.run(
        burn_in=0, min_steps=20, max_steps=20, concatenate=False)

    if sampler.mpicomm.rank == 0:
        for chain_10, chain_20 in zip(chains_10, chains_20, strict=True):
            assert len(chain_10) == 10
            assert len(chain_20) == 20
            assert np.allclose(np.asarray(chain_10['a']), np.asarray(chain_20['a'])[:10])


# ── MH fast-slow decomposition ────────────────────────────────

@pytest.mark.mpi_skip
def test_mh_fast_slow(likelihood):
    """MH kernel accepts fast parameters."""
    sampler = samplers.Sampler(
        likelihood, kernel=samplers.MH(fast=['a'], f_fast=1), rng=42)
    sampler.run(max_steps=100)


# ── Nested kernels ─────────────────────────────────────────────────────────────

@pytest.mark.mpi
@pytest.mark.parametrize('key', NESTED_KERNEL_SAMPLER.keys())
def test_nested_kernel_runs(likelihood, key):
    """Nested-kernel Sampler factory runs without error (smoke test)."""
    if key in NESTED_KERNEL_OPTIONAL_DEPS:
        pytest.importorskip(NESTED_KERNEL_OPTIONAL_DEPS[key])
    sampler = samplers.Sampler(likelihood, kernel=NESTED_KERNEL_SAMPLER[key](), rng=42)
    sampler.run(**NESTED_KERNEL_KWARGS_RUN.get(key, {}))


# ── Static kernels ─────────────────────────────────────────────────────────────

@pytest.mark.mpi
def test_static_kernel_grid(likelihood):
    """Grid kernel via Sampler factory produces accurate results."""
    sampler = samplers.Sampler(likelihood, kernel=samplers.Grid(), rng=42)
    results = sampler.run(grid=dict(a=_a_grid, b=_b_grid))

    if sampler.mpicomm.rank == 0:
        mean_samples = results.mean(['a', 'b'])
        assert np.allclose(mean_samples, likelihood.flatdata, atol=0.05, rtol=0)
        cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
        cov_err = np.sqrt((cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
        assert np.allclose(results.covariance(['a', 'b']), cov, atol=3 * cov_err)


@pytest.mark.mpi
def test_static_kernel_qmc(likelihood):
    """QMC kernel via Sampler factory produces accurate results."""
    sampler = samplers.Sampler(likelihood, kernel=samplers.QMC(), rng=42)
    results = sampler.run(size=10000)

    if sampler.mpicomm.rank == 0:
        mean_samples = results.mean(['a', 'b'])
        assert np.allclose(mean_samples, likelihood.flatdata, atol=0.05, rtol=0)
        cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
        cov_err = np.sqrt((cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
        assert np.allclose(results.covariance(['a', 'b']), cov, atol=3 * cov_err)


@pytest.mark.mpi
def test_static_kernel_importance(likelihood):
    """Importance kernel correctly reweights a grid sample (combine likelihoods)."""
    grid_sampler = samplers.Sampler(likelihood, kernel=samplers.Grid(), rng=42)
    grid_results = grid_sampler.run(grid=dict(a=_a_grid, b=_b_grid))

    imp_sampler = samplers.Sampler(likelihood, kernel=samplers.Importance(), rng=42)
    results = imp_sampler.run(samples=grid_results, resample=False)

    if imp_sampler.mpicomm.rank == 0:
        cov = np.linalg.inv(2 * likelihood.precision + np.array([[100, 0], [0, 0]]))
        assert np.allclose(results.mean(likelihood.params.select(varied=True)),
                           likelihood.flatdata, atol=1e-3, rtol=0)
        assert np.allclose(results.covariance(likelihood.params.select(varied=True)),
                           cov, atol=1e-3)


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

    likelihood = likelihood()
    sampler = samplers.Sampler(likelihood, kernel=samplers.Emcee(nwalkers=8), rng=42)
    results = sampler.run(**_MCMC_MIN_STEPS)
