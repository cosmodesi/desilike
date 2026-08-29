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

SAMPLER = dict(
    emcee=lambda: samplers.Emcee(nwalkers=8),
    zeus=lambda: samplers.Zeus(nwalkers=8),
    mhmcmc=lambda: samplers.MH(),
    # num_integration_steps must keep the trajectory away from a full oscillation period:
    # after mass-matrix adaptation the space is ~unit Gaussian (period 2*pi) and the default
    # 60 steps x adapted step size ~0.85 lands near an integer number of periods, so the
    # trajectory nearly returns to its start and one direction stops mixing (tau ~ 2400).
    hmc=lambda: samplers.BlackjaxHMC(num_integration_steps=10),
    nuts=lambda: samplers.BlackjaxNUTS(),
    mclmc=lambda: samplers.BlackjaxMCLMC(),
    numpyro_nuts=lambda: samplers.NumpyroNUTS(),
    numpyro_hmc=lambda: samplers.NumpyroHMC(),
    numpyro_barker=lambda: samplers.NumpyroBarkerMH(),
    numpyro_aies=lambda: samplers.NumpyroAIES(nwalkers=8),
    numpyro_ess=lambda: samplers.NumpyroESS(nwalkers=8),
    dynesty=lambda: samplers.Dynesty(dynamic=True, nlive=100),
    nautilus=lambda: samplers.Nautilus(n_networks=1, n_live=300),
    pocomc=lambda: samplers.PocoMC(n_effective=200, n_active=100),
)
# SA is a gradient-free adaptive sampler; its covariance is inaccurate at low sample counts
# (same behavior as the legacy NumpyroSASampler, which is not in test_accuracy).
SAMPLER_RUNS = dict(numpyro_sa=lambda: samplers.NumpyroSA())
OPTIONAL_DEPS = dict(
    emcee='emcee', zeus='zeus', hmc='blackjax', nuts='blackjax', mclmc='blackjax',
    numpyro_nuts='numpyro', numpyro_hmc='numpyro', numpyro_barker='numpyro',
    numpyro_sa='numpyro', numpyro_aies='numpyro', numpyro_ess='numpyro',
    dynesty='dynesty', nautilus='nautilus', pocomc='pocomc',
)
KWARGS_RUN = dict(
    emcee=_MCMC_MIN_STEPS,
    zeus=_MCMC_MIN_STEPS,
    mhmcmc=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=sys.maxsize)),
    hmc=dict(min_steps=10000, **_BLACKJAX_ADAPTATION),
    nuts=dict(**_MCMC_MIN_STEPS, **_BLACKJAX_ADAPTATION),
    mclmc=_MCMC_MIN_STEPS,
    numpyro_nuts=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=500)),
    numpyro_hmc=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=500)),
    numpyro_barker=dict(**_MCMC_MIN_STEPS, adaptation=dict(steps=500)),
    numpyro_aies=_MCMC_MIN_STEPS,
    numpyro_ess=_MCMC_MIN_STEPS,
    dynesty=dict(n_effective=0),
    nautilus=dict(n_eff=100),
    pocomc=dict(n_total=100, n_evidence=100),
)
KWARGS_RUN_FAST = dict(
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
    numpyro_aies=dict(max_steps=10),
    numpyro_ess=dict(max_steps=10),
    dynesty=dict(maxiter=10),
    nautilus=dict(n_eff=0, n_like_max=100),
    pocomc=dict(n_total=10, n_evidence=0),
)


def make_likelihood(flatdata=(0.4, 0.6)):
    """Compile the two-parameter Gaussian posterior used throughout these tests.

    *flatdata* shifts the data vector, which shifts the posterior: passing a perturbed
    value gives a second, slightly wrong posterior -- a stand-in for an emulated one.
    """

    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.flatdata = jnp.asarray(flatdata, dtype=float)
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


@pytest.fixture
def likelihood():
    return make_likelihood()


# ── Accuracy ──────────────────────────────────────────────────────────────────

@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER.keys())
def test_kernel_accuracy(likelihood, key):
    """Kernel-based Sampler factory produces accurate results."""
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])

    sampler = samplers.Sampler(likelihood, kernel=SAMPLER[key](), rng=42)
    results = sampler.run(**KWARGS_RUN.get(key, {}))

    if sampler.mpicomm.rank == 0:
        mean_samples = results.mean(['a', 'b'])
        assert np.allclose(mean_samples, likelihood.flatdata, atol=0.05, rtol=0)
        cov_samples = results.covariance(['a', 'b'])
        cov = np.linalg.inv(likelihood.precision + np.array([[100, 0], [0, 0]]))
        cov_err = np.sqrt(
            (cov**2 + np.outer(np.diag(cov), np.diag(cov))) / 100)
        assert np.allclose(cov_samples, cov, atol=3 * cov_err)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER_RUNS.keys())
def test_kernel_runs(likelihood, key):
    """Kernel-based Sampler factory runs without error (no accuracy check)."""
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])
    sampler = samplers.Sampler(likelihood, kernel=SAMPLER_RUNS[key](), rng=42)
    sampler.run(**KWARGS_RUN_FAST.get(key, {}))


@pytest.mark.mpi
@pytest.mark.parametrize('key', SAMPLER.keys())
def test_kernel_rescale(likelihood, key):
    """Rescaling the parameter space does not change the recovered posterior."""
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])

    sampler = samplers.Sampler(likelihood, kernel=SAMPLER[key](), rng=42,
                               conditioner=samplers.AffineConditioner(rescale=True))
    results = sampler.run(**KWARGS_RUN.get(key, {}))

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
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])

    sampler = samplers.Sampler(likelihood, kernel=SAMPLER[key](), rng=42)
    results = sampler.run(**KWARGS_RUN_FAST.get(key, {}))
    if sampler.mpicomm.rank == 0:
        a, b, c, d = (np.asarray(results[n]) for n in ('a', 'b', 'c', 'd'))
        assert np.allclose(a + b, c)
        for i in range(3):
            assert np.allclose((a + b) * i, d[..., i])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['emcee'])
def test_kernel_solved(likelihood, key):
    """Kernel-based Sampler correctly computes analytically-solved parameters."""
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])

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

    sampler = samplers.Sampler(solved_likelihood, kernel=SAMPLER[key](), rng=42)
    results = sampler.run(**KWARGS_RUN_FAST.get(key, {}))
    if sampler.mpicomm.rank == 0:
        a_arr = np.asarray(results['a']).ravel()
        b_arr = np.asarray(results['b']).ravel()
        for idx in range(3):
            assert np.allclose(b_arr[idx], best_fit_b_given_a(likelihood, a_arr[idx]))


# ── Checkpointing / determinism / chain continuation ─────────────────────────

@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER.keys())
def test_kernel_write(likelihood, key, tmp_path):
    """Second sampler reading a saved checkpoint returns identical results."""
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])

    sampler_1 = samplers.Sampler(likelihood, kernel=SAMPLER[key](),
                                  rng=42, output_dir=tmp_path)
    results_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = samplers.Sampler(likelihood, kernel=SAMPLER[key](),
                                  rng=43, output_dir=tmp_path)
    results_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    if sampler_1.mpicomm.rank == 0:
        assert len(results_1) == len(results_2)
        assert np.allclose(results_1.logposterior, results_2.logposterior, atol=1e-6)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', SAMPLER.keys())
def test_kernel_rng(likelihood, key):
    """Fixing the random seed leads to reproducible results."""
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])
    if key == 'zeus':
        pytest.skip('Zeus does not support specifying a random seed.')
    if key == 'pocomc':
        pytest.skip('pocoMC adaptive beta schedule is not deterministic across runs.')

    sampler_1 = samplers.Sampler(likelihood, kernel=SAMPLER[key](), rng=42)
    results_1 = sampler_1.run(**KWARGS_RUN_FAST.get(key, {}))

    sampler_2 = samplers.Sampler(likelihood, kernel=SAMPLER[key](), rng=42)
    results_2 = sampler_2.run(**KWARGS_RUN_FAST.get(key, {}))

    if sampler_1.mpicomm.rank == 0:
        assert len(results_1) == len(results_2)
        assert np.allclose(results_1.logposterior, results_2.logposterior, atol=1e-6)


@pytest.mark.mpi
@pytest.mark.parametrize('key', ['emcee', 'hmc', 'mhmcmc', 'zeus'])
def test_kernel_multiple_chains(likelihood, key):
    """Multiple chains can be run in parallel across MPI ranks."""
    if key in OPTIONAL_DEPS:
        pytest.importorskip(OPTIONAL_DEPS[key])

    nchains = likelihood.mpicomm.size
    sampler = samplers.Sampler(likelihood, kernel=SAMPLER[key](),
                                nparallel=nchains, rng=42)
    chains_10 = sampler.run(
        burn_in=0, min_steps=10, max_steps=10, concatenate=False)
    if sampler.mpicomm.rank == 0:
        assert len(chains_10) == nchains


# ── PocoMC Gaussian proposal ──────────────────────────────────────────────────

@pytest.mark.mpi_skip
@pytest.mark.parametrize('rescale,use_proposal', [
    (False, False),
    (True,  False),
    ('diag', False),
    ('full', False),
    (False, True),
    (True,  True),
    ('diag', True),
    ('full', True),
])
def test_pocomc_gaussian_proposal(rescale, use_proposal):
    """PocoMC runs without error under all AffineConditioner × proposal combinations.

    Parameters have Gaussian priors with hard bounds to exercise both the
    Gaussian-proposal branch and the per-parameter bound clipping.
    """
    pytest.importorskip('pocomc')
    from desilike.samples import Covariance

    # Two parameters: both Gaussian with hard bounds (exercises clipping in the PPF).
    a = Parameter('a', prior=dict(dist='norm', limits=[-1., 2.], loc=0.4, scale=0.1),
                  ref=dict(dist='norm', loc=0.4, scale=0.05), value=0.4)
    b = Parameter('b', prior=dict(dist='norm', loc=0.6, scale=0.4),
                  ref=dict(dist='norm', loc=0.6, scale=0.1), value=0.6)

    class Likelihood(BaseGaussianLikelihood):
        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.flatdata = jnp.array([0.4, 0.6])
            self.precision = jnp.diag(jnp.array([1., 1.]))

        def __call__(self):
            self.flattheory = jnp.array([self.a, self.b])
            return super().__call__()

    graph = compile(Posterior(Likelihood(a, b), Prior(a, b)))

    proposal_cov = None
    if use_proposal:
        proposal_cov = Covariance(np.diag([0.15**2, 0.5**2]), params=[a, b])

    covariance = None
    if rescale in ('diag', 'full'):
        # Off-diagonal entry triggers the Cholesky path for 'full'.
        cov_arr = np.array([[0.08**2, 0.5 * 0.08 * 0.35],
                            [0.5 * 0.08 * 0.35, 0.35**2]])
        covariance = Covariance(cov_arr, params=[a, b])

    conditioner = samplers.AffineConditioner(covariance=covariance, rescale=rescale)
    sampler = samplers.Sampler(
        graph, kernel=samplers.PocoMC(n_effective=100, n_active=50),
        rng=42, conditioner=conditioner, proposal=proposal_cov)
    sampler.run(n_total=50, n_evidence=0)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('kind', ['covariance', 'generic'])
def test_pocomc_proposal_accuracy(likelihood, kind):
    """A shifted, over-dispersed proposal must leave the posterior unbiased.

    The tempered target is ``proposal * (posterior / proposal)^beta``, so at beta = 1 the
    samples follow the exact posterior for any covering proposal; the proposal only shapes
    the annealing path.  Analytic posterior of the `likelihood` fixture:
    a ~ N(0.4, 1/sqrt(200)), b ~ N(0.6, 1/sqrt(10)).  The proposal is shifted by 1 sigma
    and 1.5x wider — the supported (over-covering) regime.  A proposal 30% *narrower* was
    measured to leave 5-17% residual under-dispersion (PocoMC's rejuvenation does not fully
    re-expand an under-covered start), which is why inflation is required, and why this
    test does not exercise the under-dispersed direction.

    'generic' passes the same Gaussian through the object interface (logpdf/ppf) instead
    of a Covariance, covering the custom-proposal code path.
    """
    pytest.importorskip('pocomc')
    from desilike.samples import Covariance

    analytic_mean = {'a': 0.4, 'b': 0.6}
    analytic_std = {'a': 1. / np.sqrt(200.), 'b': 1. / np.sqrt(10.)}
    mu = np.array([analytic_mean['a'] + analytic_std['a'], analytic_mean['b'] - analytic_std['b']])
    sigma = 1.5 * np.array([analytic_std['a'], analytic_std['b']])

    if kind == 'covariance':
        params = {param.name: param for param in likelihood.params if not (param.derived or param.fixed)}
        proposal = Covariance(np.diag(sigma**2), params=[params['a'].clone(value=mu[0]),
                                                         params['b'].clone(value=mu[1])])
    else:
        from jax.scipy import stats as jstats
        from jax.scipy.special import ndtri

        class GaussianProposal:
            """Diagonal Gaussian through the generic logpdf/ppf interface (JAX-traceable)."""

            def logpdf(self, x):
                return jnp.sum(jstats.norm.logpdf(x, loc=mu, scale=sigma))

            def ppf(self, u):
                return mu + sigma * jnp.clip(ndtri(u), -1e38, 1e38)

        proposal = GaussianProposal()

    sampler = samplers.Sampler(likelihood, kernel=samplers.PocoMC(n_effective=256, n_active=128),
                               rng=42, proposal=proposal)
    samples = sampler.run(n_total=1000, n_evidence=0)
    if samples is None:  # non-main ranks
        return
    weights = np.asarray(samples.aweight)
    weights = weights / weights.sum()
    for name in ('a', 'b'):
        values = np.ravel(samples[name].value)
        mean = np.sum(weights * values)
        std = np.sqrt(np.sum(weights * (values - mean)**2))
        assert abs(mean - analytic_mean[name]) < 0.15 * analytic_std[name], \
            f'{name}: mean {mean} vs analytic {analytic_mean[name]} +- {analytic_std[name]}'
        assert abs(std / analytic_std[name] - 1.) < 0.15, \
            f'{name}: std {std} vs analytic {analytic_std[name]}'


# ── MH fast-slow decomposition ────────────────────────────────

@pytest.mark.mpi_skip
def test_mh_fast_slow(likelihood):
    """MH kernel accepts fast parameters."""
    sampler = samplers.Sampler(
        likelihood, kernel=samplers.MH(fast=['a'], f_fast=1), rng=42)
    sampler.run(max_steps=100)


def _skip_if_mpi():
    """Skip a single-process test under MPI.

    ``pytest.mark.mpi_skip`` is only enforced when pytest-mpi is installed; without it a
    test that drives the pool from the main rank alone deadlocks the workers rather than
    being skipped.
    """
    if get_mpicomm().size > 1:
        pytest.skip('single-process test')


# ── SMC bridge ────────────────────────────────────────────────────────────────

# Analytic posterior of the `likelihood` fixture: a ~ N(0.4, 1/sqrt(200)), b ~ N(0.6, 1/sqrt(10)).
_ANALYTIC_MEAN = np.array([0.4, 0.6])
_ANALYTIC_STD = np.array([1. / np.sqrt(200.), 1. / np.sqrt(10.)])


def _analytic_logevidence(graph, mean=_ANALYTIC_MEAN, std=_ANALYTIC_STD):
    """Return log Z of *graph*, whose posterior is exactly Gaussian.

    The graph returns an unnormalized log-posterior, but a Gaussian is fixed by its shape,
    so evaluating it at the mean and dividing by the normalized Gaussian there gives the
    normalization exactly (the prior box at +-10 sigma cuts nothing).
    """
    log_post = float(graph(dict(zip(('a', 'b'), mean)), return_derived=False))
    log_gauss = float(-0.5 * len(mean) * np.log(2. * np.pi) - np.sum(np.log(std)))
    return log_post - log_gauss


def _shifted_proposal(likelihood, shift=1., scale=1.5):
    """A Gaussian proposal shifted by *shift* sigma and *scale* times wider."""
    from desilike.samples import Covariance
    params = {param.name: param for param in likelihood.params if not (param.derived or param.fixed)}
    center = _ANALYTIC_MEAN + shift * _ANALYTIC_STD * np.array([1., -1.])
    covariance = Covariance(np.diag((scale * _ANALYTIC_STD)**2),
                            params=[params['a'].clone(value=center[0]),
                                    params['b'].clone(value=center[1])])
    return samplers.GaussianProposal(covariance)


def _check_posterior(samples, atol_mean=0.15, rtol_std=0.1):
    """Check equal-weight *samples* against the analytic posterior, in units of sigma."""
    weights = np.asarray(samples.weight, dtype='f8')
    weights = weights / weights.sum()
    for i, name in enumerate(('a', 'b')):
        values = np.ravel(samples[name].value)
        mean = np.sum(weights * values)
        std = np.sqrt(np.sum(weights * (values - mean)**2))
        assert abs(mean - _ANALYTIC_MEAN[i]) < atol_mean * _ANALYTIC_STD[i], \
            f'{name}: mean {mean} vs analytic {_ANALYTIC_MEAN[i]} +- {_ANALYTIC_STD[i]}'
        assert abs(std / _ANALYTIC_STD[i] - 1.) < rtol_std, \
            f'{name}: std {std} vs analytic {_ANALYTIC_STD[i]}'


@pytest.mark.mpi_skip
def test_smc_gaussian_proposal(likelihood):
    """Bridging from a shifted, over-dispersed Gaussian recovers the posterior and its evidence."""
    _skip_if_mpi()
    sampler = samplers.Sampler(likelihood, kernel=samplers.SMC(nparticles=512), rng=42,
                               proposal=_shifted_proposal(likelihood))
    samples = sampler.run()

    _check_posterior(samples)
    assert samples.attrs['beta'] == 1.
    # The proposal is normalized, so the bridge returns the actual log-evidence.
    assert abs(samples.attrs['logevidence'] - _analytic_logevidence(likelihood)) < 0.2
    # Derived parameters are carried through the whole schedule, not recomputed at the end.
    assert np.allclose(np.ravel(samples['c'].value),
                       np.ravel(samples['a'].value) + np.ravel(samples['b'].value))


@pytest.mark.mpi
def test_smc_prior_annealing(likelihood):
    """Without a proposal the bridge is the usual prior-to-posterior annealing.

    Marked ``mpi`` rather than ``mpi_skip`` so the padded, pooled evaluation path is
    exercised with several ranks: the batch a rank is handed must stay one fixed shape
    however many particles are live.
    """
    sampler = samplers.Sampler(likelihood, kernel=samplers.SMC(nparticles=512), rng=42)
    samples = sampler.run()

    if sampler.mpicomm.rank != 0:
        return
    _check_posterior(samples)
    # No proposal, so delayed acceptance is off: screening on the prior would screen nothing.
    assert samples.attrs['beta'] == 1.
    # Annealing all the way from the prior needs many more temperatures than a bridge does.
    assert len(samples.attrs['history']) > 3


@pytest.mark.mpi_skip
def test_smc_delayed_acceptance(likelihood):
    """Delayed acceptance changes the cost, not the posterior."""
    _skip_if_mpi()
    results = {}
    for delayed in (False, True):
        sampler = samplers.Sampler(
            likelihood, kernel=samplers.SMC(nparticles=512, delayed=delayed), rng=42,
            proposal=_shifted_proposal(likelihood))
        results[delayed] = sampler.run()
        _check_posterior(results[delayed])

    # The screening stage rejects before paying for the posterior, so it must cost less.
    assert results[True].attrs['nevaluations'] < results[False].attrs['nevaluations']
    # And both must land on the same evidence.
    assert abs(results[True].attrs['logevidence'] - results[False].attrs['logevidence']) < 0.2


@pytest.mark.mpi_skip
def test_smc_seeds(likelihood):
    """Two seeds scatter around the same posterior."""
    _skip_if_mpi()
    means = []
    for seed in (42, 7):
        sampler = samplers.Sampler(likelihood, kernel=samplers.SMC(nparticles=512), rng=seed,
                                   proposal=_shifted_proposal(likelihood))
        samples = sampler.run()
        _check_posterior(samples)
        means.append(samples.mean(['a', 'b']))
    assert np.allclose(means[0], means[1], atol=0.15 * _ANALYTIC_STD, rtol=0)


@pytest.mark.mpi_skip
def test_smc_samples_proposal():
    """The emulated-to-exact bridge: an existing chain plus the density it was sampled under.

    The 'emulated' posterior is the same Gaussian with a perturbed data vector, so it sits
    about 1 sigma off in b -- the emulator error the bridge has to remove. Reweighting the
    chain would work here too; the point is that the machinery does, and that it removes the
    shift rather than inheriting it.
    """
    _skip_if_mpi()
    exact = make_likelihood()
    emulated_data = np.array([0.4 + 0.5 * _ANALYTIC_STD[0], 0.6 + _ANALYTIC_STD[1]])
    emulated = make_likelihood(flatdata=emulated_data)
    # a keeps its N(0.4, 0.1) prior, so shifting the data moves the mean only half as far;
    # b's prior is uniform, so its mean follows the data exactly.
    emulated_mean = np.array([0.5 * (0.4 + emulated_data[0]), emulated_data[1]])

    # A grid under the emulated posterior stands in for a converged chain: its weights are
    # the emulated posterior, so SamplesProposal draws from exactly the density it is given.
    grid_sampler = samplers.Sampler(emulated, kernel=samplers.Grid(), rng=42)
    chain = grid_sampler.run(grid=dict(a=_a_grid, b=_b_grid))

    proposal = samplers.SamplesProposal(emulated, chain)
    sampler = samplers.Sampler(exact, kernel=samplers.SMC(nparticles=512), rng=42, proposal=proposal)
    samples = sampler.run()

    _check_posterior(samples)
    # log(Z_exact / Z_emulated), both graphs sharing the same unknown constant.
    expected = _analytic_logevidence(exact) - _analytic_logevidence(emulated, mean=emulated_mean)
    assert abs(samples.attrs['logevidence'] - expected) < 0.2


@pytest.mark.mpi_skip
def test_smc_product_proposal(likelihood):
    """A product of independent factors over disjoint blocks is a valid proposal."""
    _skip_if_mpi()
    from desilike.samples import Covariance
    params = {param.name: param for param in likelihood.params if not (param.derived or param.fixed)}
    factors = [samplers.GaussianProposal(
        Covariance(np.diag([(1.5 * _ANALYTIC_STD[i])**2]),
                   params=[params[name].clone(value=_ANALYTIC_MEAN[i] + _ANALYTIC_STD[i])]))
        for i, name in enumerate(('a', 'b'))]

    sampler = samplers.Sampler(likelihood, kernel=samplers.SMC(nparticles=512), rng=42,
                               proposal=samplers.ProductProposal(*factors))
    samples = sampler.run()

    _check_posterior(samples)
    # The factorized proposal is normalized too, so the evidence is still the true one.
    assert abs(samples.attrs['logevidence'] - _analytic_logevidence(likelihood)) < 0.2


@pytest.mark.mpi_skip
@pytest.mark.parametrize('names', [('a', 'b'), ('a',)])
def test_smc_covariance_proposal(likelihood, names):
    """A Covariance is still accepted as a proposal, covering all parameters or only some.

    It is wrapped into a GaussianProposal inside a ProductProposal; the parameters the
    covariance leaves out keep their own prior.
    """
    _skip_if_mpi()
    from desilike.samples import Covariance
    params = {param.name: param for param in likelihood.params if not (param.derived or param.fixed)}
    index = [('a', 'b').index(name) for name in names]
    center = _ANALYTIC_MEAN[index] + _ANALYTIC_STD[index]
    covariance = Covariance(np.diag((1.5 * _ANALYTIC_STD[index])**2),
                            params=[params[name].clone(value=value)
                                    for name, value in zip(names, center)])

    sampler = samplers.Sampler(likelihood, kernel=samplers.SMC(nparticles=512), rng=42,
                               proposal=covariance)
    _check_posterior(sampler.run())


@pytest.mark.mpi_skip
def test_proposal_errors(likelihood):
    """Malformed proposals are rejected where they are built, not deep inside a run."""
    from desilike.samples import Covariance
    params = {param.name: param for param in likelihood.params if not (param.derived or param.fixed)}

    def gaussian(name):
        return samplers.GaussianProposal(
            Covariance(np.diag([0.1**2]), params=[params[name].clone(value=0.5)]))

    with pytest.raises(ValueError, match='disjoint'):
        samplers.Sampler(likelihood, kernel=samplers.SMC(), rng=42,
                         proposal=samplers.ProductProposal(gaussian('a'), gaussian('a')))

    # Parameters no factor covers are not an error: they keep their own prior. Cover b, and
    # shifting a must move the proposal density exactly as a's (normal) prior does.
    sampler = samplers.Sampler(likelihood, kernel=samplers.SMC(), rng=42,
                               proposal=samplers.ProductProposal(gaussian('b')))
    lo, hi = np.zeros((1, 2)), np.array([[0.2, 0.]])
    shift = (float(np.asarray(sampler.prior_logpdf(hi))[0])
             - float(np.asarray(sampler.prior_logpdf(lo))[0]))
    a0 = float(np.asarray(sampler.conditioner.forward(np.zeros(2)))[0])
    prior_a = params['a'].prior
    assert np.isclose(shift, float(prior_a.logpdf(jnp.array(a0 + 0.2)) - prior_a.logpdf(jnp.array(a0))))

    # A factor covering only part of the space is not usable on its own: the sampler would
    # feed it vectors of the wrong width. It has to say so, not narrow itself silently.
    with pytest.raises(ValueError, match='ProductProposal'):
        samplers.Sampler(likelihood, kernel=samplers.SMC(), rng=42, proposal=gaussian('a'))

    # A product handed as a factor is flattened, not nested.
    product = samplers.ProductProposal(gaussian('a'), samplers.ProductProposal(gaussian('b')))
    assert len(product.factors) == 2
    samplers.Sampler(likelihood, kernel=samplers.SMC(), rng=42, proposal=product)

    class RvsOnly:
        """A proposal that can be drawn from but not inverted from the unit cube."""

        def logpdf(self, x):
            return jnp.zeros(())

        def rvs(self, size, rng):
            return rng.random((size, 2))

    # SMC draws through rvs, so it accepts this; kernels that need a ppf must say so clearly.
    sampler = samplers.Sampler(likelihood, kernel=samplers.SMC(), rng=42, proposal=RvsOnly())
    with pytest.raises(NotImplementedError, match='ppf'):
        sampler.prior_ppf(np.full((1, 2), 0.5))

    with pytest.raises(TypeError, match='logpdf'):
        samplers.Sampler(likelihood, kernel=samplers.SMC(), rng=42, proposal=object())


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
    results = imp_sampler.run(samples=grid_results, combine=True)

    if imp_sampler.mpicomm.rank == 0:
        cov = np.linalg.inv(2 * likelihood.precision + np.array([[100, 0], [0, 0]]))
        assert np.allclose(results.mean(likelihood.params.select(varied=True)),
                           likelihood.flatdata, atol=1e-3, rtol=0)
        assert np.allclose(results.covariance(likelihood.params.select(varied=True)),
                           cov, atol=1e-3)


@pytest.mark.mpi
def test_static_kernel_importance_reweight(likelihood):
    """Importance reweighting an already-weighted grid to the same posterior is the identity."""
    grid_sampler = samplers.Sampler(likelihood, kernel=samplers.Grid(), rng=42)
    grid_results = grid_sampler.run(grid=dict(a=_a_grid, b=_b_grid))

    imp_sampler = samplers.Sampler(likelihood, kernel=samplers.Importance(), rng=42)
    results = imp_sampler.run(samples=grid_results)
    # Every rank must take part in the second run, so it cannot sit inside the rank-0 branch.
    equal = imp_sampler.run(samples=grid_results, resample=True, reuse=False)

    if imp_sampler.mpicomm.rank == 0:
        # The weighted grid already represents the posterior, so reweighting it to the
        # very same posterior must leave the weights (and hence the moments) unchanged.
        params = likelihood.params.select(varied=True)
        assert np.allclose(results.aweight, grid_results.aweight, atol=0, rtol=1e-8)
        assert np.allclose(results.mean(params), grid_results.mean(params), atol=0, rtol=1e-6)
        assert results.attrs['ess_correction'] > 0.999 * len(results.aweight)
        assert abs(results.attrs['logevidence']) < 1e-8

        # Resampling to equal weight preserves the moments to Monte-Carlo accuracy.
        assert np.allclose(equal.aweight, 1.)
        assert np.allclose(equal.mean(params), grid_results.mean(params), atol=3e-2, rtol=0)


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
    sampler = samplers.Sampler(likelihood, kernel=samplers.PocoMC(n_effective=200, n_active=100), rng=42)
    results = sampler.run(n_total=100, n_evidence=100)


@pytest.mark.mpi_skip
@pytest.mark.parametrize('key', ['numpyro_nuts', 'numpyro_hmc', 'numpyro_barker', 'numpyro_sa'])
def test_kernel_nparallel_batches(likelihood, key):
    """Vectorized chains keep sampling across several kernel calls.

    NumPyro caches its compiled functions on the kernel instance, so a multi-chain run that
    wrapped that kernel in a fresh MCMC for the second batch used to raise
    "vmap ... rank should be at least 1"; and since the first batch is short by the samples
    already present, the compiled sample count changed on the second batch too.  ``check_every``
    here is small enough that ``Sampler.run`` issues several calls: a short first one, then
    full-length ones.
    """
    pytest.importorskip('numpyro')
    nparallel, check_every, max_steps = 4, 10, 30
    kernel = (SAMPLER.get(key) or SAMPLER_RUNS[key])()
    sampler = samplers.Sampler(likelihood, kernel=kernel, nparallel=nparallel, rng=42)
    results = sampler.run(min_steps=max_steps, max_steps=max_steps, check_every=check_every,
                          adaptation=dict(steps=50))

    if sampler.mpicomm.rank == 0:
        for name in ['a', 'b']:
            values = np.asarray(results[name])
            assert values.size, f'{key}: no samples returned'
            assert np.all(np.isfinite(values)), f'{key}: non-finite samples'


def test_sampler_accepts_an_uncompiled_calculator():
    """`Sampler(posterior)` compiles a Calculator rather than failing on `.params` -- but only the
    no-argument form; `compile(root, output=...)` is a choice a sampler cannot make.  An already
    compiled graph is stored as-is: compiling runs the whole pipeline, so a needless recompile is
    a real cost, not a formality.
    """
    from desilike.base import CompiledGraph

    class Toy(BaseGaussianLikelihood):

        def __init__(self, a):
            self.a = a
            self.flatdata = jnp.array([0.4])
            self.precision = jnp.diag(jnp.array([100.]))

        def __call__(self):
            self.flattheory = jnp.array([self.a])
            return super().__call__()

    def posterior():
        a = Parameter('a', prior=dict(dist='uniform', limits=[-10., 10.]),
                      ref=dict(dist='norm', loc=0.4, scale=0.1))
        return Posterior(Toy(a), Prior(a))

    sampler = samplers.Sampler(posterior(), samplers.Grid())
    assert [param.name for param in sampler.varied_params] == ['a']
    assert isinstance(sampler.posterior, CompiledGraph)

    already = compile(posterior())
    assert samplers.Sampler(already, samplers.Grid()).posterior is already
