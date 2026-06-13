"""Benchmark samplers on a multi-tracer BAO correlation-function posterior.

Multi-tracer posterior
----------------------
Two independent BAO likelihoods (LRG at z=0.5, ELG at z=0.8) are combined
via SumLikelihood.  The BAO stretch parameters (``qpar``, ``qper``) are shared
across tracers (they have the same names in both templates and are merged by
SumLikelihood).  Tracer-specific nuisance parameters (``b1``, ``sigmas``,
broadband polynomials) are namespaced under the tracer label.

Samplers benchmarked
--------------------
- emcee        (ensemble MCMC)
- nautilus     (learned importance)
- pocomc       (preconditioned Monte Carlo)
- hmc          (Hamiltonian Monte Carlo via BlackJAX)
- nuts         (No-U-Turn Sampler via BlackJAX)
- numpyro_nuts (No-U-Turn Sampler via NumPyro)
- numpyro_hmc  (HMC via NumPyro)
- numpyro_barker (Barker MH via NumPyro)
- numpyro_sa   (Sample Adaptive MCMC via NumPyro)

Run directly::

    python -m desilike.samplers.tests.benchmark_samplers

or with specific samplers::

    python -m desilike.samplers.tests.benchmark_samplers --samplers emcee nuts numpyro_nuts
"""

import time
import sys
import tempfile
from pathlib import Path

import numpy as np

from desilike import setup_logging
import jax.numpy as jnp
from desilike.base import get_params, Posterior, SumLikelihood, compile, GaussianLikelihood as BaseGaussianLikelihood, Prior
from desilike.parameter import Parameter
from desilike.theories.galaxy_clustering import (BAOSpectrum2Template,
                                                  DampedBAOWigglesPTSpectrum2Poles,
                                                  DampedBAOWigglesTracerCorrelation2Poles)
from desilike.observables import Correlation2PolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.samples import diagnostics
import desilike.samplers as samplers
import desilike.profilers as profilers


# ── multi-tracer BAO posterior ───────────────────────────────────────────────

ELLS = (0, 2)
S = np.arange(20., 180., 4.)
BROADBAND = 'pcs2'

TRACERS = {
    'LRG': dict(z=0.5),
    'ELG': dict(z=0.8),
}

TRACERS = {
    'LRG': dict(z=0.5),
}


def build_posterior_bao_multi(s=S, ells=ELLS, tracers=None, marginalize=True):
    """Multi-tracer BAO correlation posterior.

    Two independent likelihoods (one per tracer) are combined via SumLikelihood.
    BAO stretch parameters (``qpar``, ``qper``) are shared; nuisance parameters
    (``b1``, ``sigmas``, broadband) are per-tracer.

    Parameters
    ----------
    s : array
        Separation bins (Mpc/h).
    ells : tuple of int
        Multipole orders.
    tracers : dict or None
        Mapping from tracer label to ``dict(z=...)``.  Defaults to :data:`TRACERS`.
    marginalize : bool
        If True, broadband parameters are analytically marginalized.

    Returns
    -------
    Posterior
    """
    if tracers is None:
        tracers = TRACERS
    n = len(ells) * len(s)
    rng = np.random.default_rng(42)

    from cosmoprimo.fiducial import BOSS
    fiducial = BOSS(engine='eisenstein_hu')

    likelihoods = []
    for tracer_label, tracer_kwargs in tracers.items():
        z = tracer_kwargs.get('z', 0.5)
        template = BAOSpectrum2Template(z=z, fiducial=fiducial)
        pt = DampedBAOWigglesPTSpectrum2Poles(template=template, tracers=tracer_label)
        theory = DampedBAOWigglesTracerCorrelation2Poles(
            s=s, pt=pt, ells=ells, broadband=BROADBAND, tracers=tracer_label)

        data = rng.normal(scale=1e-3, size=n)
        covariance = np.diag(np.full(n, 1e-6))
        window = np.eye(n)

        observable = Correlation2PolesObservable(
            data=data, theory=theory, s=s, ells=ells,
            window=window, sin=s, ellsin=ells, covariance=covariance)
        data = compile(observable, output=lambda: observable.flattheory)()
        observable.update(data=data)
        like = ObservablesGaussianLikelihood(observables=observable)

        if marginalize:
            for param in get_params(like).select(basename='*l*', fixed=False):
                param.update(derived='best')
        likelihoods.append(like)

    likelihood = SumLikelihood(likelihoods)
    return Posterior(likelihood)


# ── ill-conditioned Gaussian posterior ──────────────────────────────────────

def build_posterior_gaussian(ndim=20, condition_number=1e4, seed=42):
    """Multivariate Gaussian posterior with a high-condition-number precision matrix.

    The covariance is ``Q @ diag(eigenvalues) @ Q.T`` where ``Q`` is a random
    orthogonal matrix and the ``ndim`` eigenvalues are geometrically spaced from
    ``1`` to ``condition_number``.  The mean is the origin.

    No ``ref`` distributions are provided, so samplers ca
    nnot use the posterior
    geometry for preconditioning — they must adapt from the broad uniform priors.

    Parameters
    ----------
    ndim : int
        Number of parameters.
    condition_number : float
        Ratio of largest to smallest eigenvalue of the covariance matrix.
    seed : int
        RNG seed for the random rotation matrix.

    Returns
    -------
    Posterior
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((ndim, ndim)))
    eigenvalues = np.geomspace(1., condition_number, ndim)
    precision = (Q * (1. / eigenvalues)) @ Q.T   # Q @ diag(1/λ) @ Q.T
    #precision = np.diag(1. / eigenvalues)

    class _IllConditionedGaussian(BaseGaussianLikelihood):

        def __init__(self, params, precision_matrix):
            self.ndim = len(params)
            for param in params:
                setattr(self, param.name, param)
            self.flatdata = jnp.zeros(self.ndim)
            self.precision = jnp.array(precision_matrix)

        def __call__(self):
            self.flattheory = jnp.array([getattr(self, f'x{param_idx}')
                                         for param_idx in range(self.ndim)])
            return super().__call__()

    prior_limit = 10. * condition_number ** 0.5
    params = [Parameter(f'x{param_idx}',
                        value=0.,
                        prior=dict(dist='uniform', limits=[-prior_limit, prior_limit]),
                        ref=dict(dist='uniform', limits=[-prior_limit / 10., prior_limit / 10.]))
              for param_idx in range(ndim)]
    like = _IllConditionedGaussian(params, precision)
    return Posterior(like, Prior(*params))


# ── sampler configuration ────────────────────────────────────────────────────

def _sampler_config(ndim=1):
    """Return (kernel, sampler_kwargs, run_kwargs) per sampler name."""
    return {
        'emcee': (
            samplers.Emcee(nwalkers=4 * ndim),
            dict(rng=42),
            dict(gelman_rubin=1.1, min_steps=50, max_steps=2000),
        ),
        'nautilus': (
            samplers.Nautilus(n_networks=2, n_live=300),
            dict(rng=42),
            dict(n_eff=200),
        ),
        'pocomc': (
            samplers.PocoMC(n_effective=500, n_active=200),
            dict(),
            dict(),
        ),
        'hmc': (
            samplers.BlackjaxHMC(),
            dict(rng=42),
            dict(adaptation=dict(steps=500), gelman_rubin=1.1, min_steps=50, max_steps=2000),
        ),
        'mclmc': (
            samplers.BlackjaxMCLMC(),
            dict(rescale=True, rng=42),
            dict(adaptation=dict(steps=1000, diagonal_preconditioning=False), gelman_rubin=1.1, min_steps=50, max_steps=2000),
        ),
        'nuts': (
            samplers.BlackjaxNUTS(step_size=0.05),
            dict(rescale=True, rng=42),
            dict(adaptation=dict(initial_step_size=0.1, target_acceptance_rate=0.8, steps=500, is_mass_matrix_diagonal=False), gelman_rubin=1.1, min_steps=200),
        ),
        'numpyro_nuts': (
            samplers.NumpyroNUTS(),
            dict(rescale=True, rng=42),
            dict(adaptation=dict(steps=500, dense_mass=True), gelman_rubin=1.1, min_steps=50, max_steps=2000),
        ),
        'numpyro_hmc': (
            samplers.NumpyroHMC(),
            dict(rng=42),
            dict(adaptation=dict(steps=500), gelman_rubin=1.1, min_steps=50, max_steps=2000),
        ),
        'numpyro_barker': (
            samplers.NumpyroBarkerMH(),
            dict(rescale=True, rng=42),
            dict(adaptation=dict(steps=500, dense_mass=True), gelman_rubin=1.1, min_steps=50, max_steps=5000),
        ),
        'numpyro_sa': (
            samplers.NumpyroSA(),
            dict(rng=42),
            dict(adaptation=dict(steps=500, dense_mass=True), gelman_rubin=1.1, min_steps=50, max_steps=2000),
        ),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _ess_from_samples(samples):
    """Estimate ESS from a single MCSamples object; returns None on failure."""
    try:
        iact = diagnostics.integrated_autocorrelation_time(
            [samples], check_valid='ignore')
        return float(samples.size / np.max(iact))
    except Exception:
        return None


def _gr_from_samples(samples):
    """Estimate max Gelman-Rubin from a single split MCSamples; returns None on failure."""
    try:
        value = float(np.max(diagnostics.gelman_rubin(
            [samples], method='diag', nsplits=4)))
        return value
    except Exception:
        return None


# ── benchmark harness ────────────────────────────────────────────────────────

POSTERIORS = {
    'bao': build_posterior_bao_multi,
    'gaussian': build_posterior_gaussian,
}

PROFILER_KERNELS = {
    'minuit': lambda: profilers.Minuit(),
    'scipy':  lambda: profilers.Scipy(),
    'bobyqa': lambda: profilers.BOBYQA(),
    'optax':  lambda: profilers.Optax(),
}


def _build_posterior(posterior, marginalize=True):
    if callable(posterior):
        build_fn = posterior
        kwargs = {}
    else:
        if posterior not in POSTERIORS:
            raise ValueError(f'Unknown posterior {posterior!r}; choose from {list(POSTERIORS)}')
        build_fn = POSTERIORS[posterior]
        kwargs = dict(marginalize=marginalize) if posterior == 'bao' else {}
    return build_fn(**kwargs)


def print_priors(params):
    print(f"{'param':20} {'prior':50} {'reference':50} derived")
    for p in params:
        print(f"{p.name:20} {str(p.prior):50} {str(p.ref):50} {p.derived}")


def run_benchmark(sampler_names=None, profiler_names=None, posterior='bao', directory=None):
    """Build the posterior and run each requested sampler and profiler.

    Parameters
    ----------
    sampler_names : list of str or None
        Samplers to run.  ``None`` runs all five; ``[]`` skips sampling.
    profiler_names : list of str or None
        Profilers to run.  ``None`` skips profiling; pass a list to enable.
    posterior : str or callable
        ``'bao'`` for the multi-tracer BAO pipeline, ``'gaussian'`` for the
        ill-conditioned Gaussian, or a callable returning a ``Posterior``.
    marginalize : bool
        Analytically marginalize broadband parameters (BAO only).
    directory : str or None
        Root directory for sampler checkpoints.  ``None`` disables checkpointing.
    """

    profiler_posterior = compile(_build_posterior(posterior, marginalize=True))
    # ── profilers ─────────────────────────────────────────────────────────────
    profiler_results = {}
    if profiler_names:
        for name in profiler_names:
            if name not in PROFILER_KERNELS:
                print(f'[{name}] unknown profiler — skipping')
                continue
            print(f'─── {name} {"─" * (50 - len(name))}')
            try:
                profiler = profilers.Profiler(profiler_posterior, kernel=PROFILER_KERNELS[name](), rng=42)
                t_start = time.perf_counter()
                profiler.maximize()
                elapsed = time.perf_counter() - t_start
                print(f'  time    : {elapsed:.1f} s')
                print(profiler.profiles.to_stats(tablefmt='pretty'))
                profiler_results[name] = dict(time=elapsed)
                #profiler.covariance()
            except Exception as exc:
                import traceback
                print(f'  ERROR: {exc}')
                traceback.print_exc()
                profiler_results[name] = dict(error=str(exc))
            print()

        col_w = 12
        header = f'{"profiler":<{col_w}}  {"time (s)":>10}'
        sep = '─' * len(header)
        print(sep)
        print(header)
        print(sep)
        for name, res in profiler_results.items():
            if 'error' in res:
                print(f'{name:<{col_w}}  {"ERROR":>10}')
            else:
                print(f'{name:<{col_w}}  {res["time"]:>10.1f}')
        print(sep)
        print()

    # ── samplers ──────────────────────────────────────────────────────────────
    sampler_results = {}
    if sampler_names is None:
        sampler_names = list(_sampler_config())
    if sampler_names:
        config = _sampler_config()
        for name in sampler_names:
            if name not in config:
                print(f'[{name}] unknown sampler — skipping')
                continue

            if name in ['hmc', 'nuts', 'mclmc'][:0]:
                sampler_posterior = _build_posterior(posterior, marginalize=False)
            else:
                sampler_posterior = _build_posterior(posterior)

            config = _sampler_config(ndim=len(get_params(sampler_posterior).names(varied=True, derived=False)))
            kernel, sampler_kwargs, run_kwargs = config[name]

            sampler_dir = None
            if directory is not None:
                sampler_dir = Path(directory) / name

            print(f'─── {name} {"─" * (50 - len(name))}')
            try:
                sampler_kwargs = dict(sampler_kwargs)
                if sampler_kwargs.get('rescale', False):
                    profiles = profiler.profiles.choice(index='argmax', squeeze=True)
                    best, error, covariance = profiles.best, profiles.error, profiles.covariance
                    sampler_kwargs['covariance'] = covariance
                    print(covariance._value)
                    error = {param: covariance.std(param) for param in covariance.names()}
                    for param in get_params(sampler_posterior):
                        if param.name in profiles.covariance.names():
                            param.update(ref=dict(dist='norm', loc=best[param.name], scale=error[param.name]))
                            print(param, param.ref)

                sampler = samplers.Sampler(compile(sampler_posterior), kernel=kernel, directory=sampler_dir, **sampler_kwargs)
                t_start = time.perf_counter()
                chain = sampler.run(**run_kwargs)
                elapsed = time.perf_counter() - t_start

                nsamples = chain.size if chain is not None else 0
                ess = _ess_from_samples(chain) if chain is not None and nsamples > 0 else None
                gr = _gr_from_samples(chain) if chain is not None and nsamples > 0 else None

                print(f'  time    : {elapsed:.1f} s')
                print(f'  samples : {nsamples}')
                if ess is not None:
                    print(f'  ESS     : {ess:.1f}')
                if gr is not None:
                    print(f'  max GR  : {gr:.4f}')
                sampler_results[name] = dict(time=elapsed, nsamples=nsamples, ess=ess, gr=gr)

            except Exception as exc:
                import traceback
                print(f'  ERROR: {exc}')
                traceback.print_exc()
                sampler_results[name] = dict(error=str(exc))
            print()

        col_w = 12
        header = (f'{"sampler":<{col_w}}  {"time (s)":>10}  {"nsamples":>10}'
                  f'  {"ESS":>8}  {"max GR":>8}')
        sep = '─' * len(header)
        print(sep)
        print(header)
        print(sep)
        for name, res in sampler_results.items():
            if 'error' in res:
                print(f'{name:<{col_w}}  {"ERROR":>10}')
            else:
                ess_str = f'{res["ess"]:.1f}' if res['ess'] is not None else '–'
                gr_str = f'{res["gr"]:.4f}' if res['gr'] is not None else '–'
                print(f'{name:<{col_w}}  {res["time"]:>10.1f}  {res["nsamples"]:>10}'
                      f'  {ess_str:>8}  {gr_str:>8}')
        print(sep)
        print()

    return sampler_results, profiler_results


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--posterior', default='bao', choices=list(POSTERIORS),
                        help='Posterior to benchmark (default: bao)')
    parser.add_argument('--samplers', nargs='*', metavar='SAMPLER', default=[],
                        help='Samplers to run (default: none)')
    parser.add_argument('--profilers', nargs='*', metavar='PROFILER',
                        choices=list(PROFILER_KERNELS), default=[],
                        help='Profilers to run (default: none)')
    args = parser.parse_args()
    setup_logging()
    with tempfile.TemporaryDirectory() as tmpdir:
        run_benchmark(sampler_names=args.samplers,
                      profiler_names=args.profilers,
                      posterior=args.posterior,
                      directory=tmpdir)
