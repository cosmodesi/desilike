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
- emcee   (ensemble MCMC)
- nautilus (learned importance)
- pocomc  (preconditioned Monte Carlo)
- hmc     (Hamiltonian Monte Carlo via BlackJAX)
- nuts    (No-U-Turn Sampler via BlackJAX)

Run directly::

    python -m desilike.samplers.tests.benchmark_samplers

or with specific samplers::

    python -m desilike.samplers.tests.benchmark_samplers emcee nuts
"""

import time
import sys
import tempfile

import numpy as np

from desilike.base import get_params, Posterior, SumLikelihood
from desilike.theories.galaxy_clustering import (BAOSpectrum2Template,
                                                  DampedBAOWigglesPTSpectrum2Poles,
                                                  DampedBAOWigglesTracerCorrelation2Poles)
from desilike.observables import Correlation2PolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.samples import diagnostics
import desilike.samplers as samplers


# ── multi-tracer BAO posterior ───────────────────────────────────────────────

ELLS = (0, 2)
S = np.arange(20., 180., 4.)
BROADBAND = 'pcs2'

TRACERS = {
    'LRG': dict(z=0.5),
    'ELG': dict(z=0.8),
}


def build_posterior_bao_multi(s=S, ells=ELLS, tracers=None, marginalize=False):
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
        like = ObservablesGaussianLikelihood(observables=observable)

        if marginalize:
            for param in get_params(like).select(basename='*l*', fixed=False):
                param.update(derived='best')
        likelihoods.append(like)

    likelihood = SumLikelihood(likelihoods)
    return Posterior(likelihood)


# ── sampler configuration ────────────────────────────────────────────────────

def _sampler_config(ndim):
    """Return (SamplerClass, init_kwargs, run_kwargs) per sampler name."""
    nwalkers = max(4 * ndim, 20)
    return {
        'emcee': (
            samplers.EmceeSampler,
            dict(nwalkers=nwalkers),
            dict(gelman_rubin=1.1, min_steps=500, max_steps=2000),
        ),
        'nautilus': (
            samplers.NautilusSampler,
            dict(n_networks=2, n_live=300),
            dict(n_eff=200),
        ),
        'pocomc': (
            samplers.PocoMCSampler,
            dict(n_effective=500, n_active=200),
            dict(),
        ),
        'hmc': (
            samplers.HMCSampler,
            dict(),
            dict(gelman_rubin=1.1, min_steps=500, max_steps=2000),
        ),
        'nuts': (
            samplers.NoUTurnSampler,
            dict(),
            dict(gelman_rubin=1.1, min_steps=500, max_steps=2000),
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

def run_benchmark(sampler_names=None, marginalize=False, directory=None):
    """Build the posterior and run each requested sampler, printing a summary.

    Parameters
    ----------
    sampler_names : list of str or None
        Samplers to run.  Defaults to all five.
    marginalize : bool
        Whether to analytically marginalize broadband parameters.
    directory : str or None
        Root directory for sampler checkpoints; a sub-directory per sampler is
        created.  ``None`` disables checkpointing.
    """
    if sampler_names is None:
        sampler_names = ['emcee', 'nautilus', 'pocomc', 'hmc', 'nuts']

    print('\nBuilding multi-tracer BAO posterior …', end=' ', flush=True)
    t0 = time.perf_counter()
    posterior = build_posterior_bao_multi(marginalize=marginalize)
    build_time = time.perf_counter() - t0
    print(f'done ({build_time * 1e3:.0f} ms)')

    from desilike.base import get_params
    varied_params = get_params(posterior).select(fixed=False, derived=False)
    ndim = len(varied_params)
    tracer_labels = list(TRACERS)
    print(f'Tracers : {" + ".join(tracer_labels)}')
    print(f'ndim    : {ndim}  ({", ".join(p.name for p in varied_params)})')
    print(f'marg.   : {marginalize}')
    print()

    config = _sampler_config(ndim)

    results = {}
    for name in sampler_names:
        if name not in config:
            print(f'[{name}] unknown sampler — skipping')
            continue
        cls, init_kwargs, run_kwargs = config[name]
        sampler_dir = None
        if directory is not None:
            import pathlib
            sampler_dir = pathlib.Path(directory) / name

        print(f'─── {name} {"─" * (50 - len(name))}')
        try:
            sampler_obj = cls(posterior, directory=sampler_dir, **init_kwargs)
            t_start = time.perf_counter()
            chain = sampler_obj.run(**run_kwargs)
            elapsed = time.perf_counter() - t_start

            nsamples = len(chain) if chain is not None else 0
            ess = _ess_from_samples(chain) if chain is not None and nsamples > 0 else None
            gr = _gr_from_samples(chain) if chain is not None and nsamples > 0 else None

            print(f'  time    : {elapsed:.1f} s')
            print(f'  samples : {nsamples}')
            if ess is not None:
                print(f'  ESS     : {ess:.1f}')
            if gr is not None:
                print(f'  max GR  : {gr:.4f}')
            results[name] = dict(time=elapsed, nsamples=nsamples, ess=ess, gr=gr)

        except Exception as exc:
            import traceback
            print(f'  ERROR: {exc}')
            traceback.print_exc()
            results[name] = dict(error=str(exc))
        print()

    # Summary table
    col_w = 12
    header = (f'{"sampler":<{col_w}}  {"time (s)":>10}  {"nsamples":>10}'
              f'  {"ESS":>8}  {"max GR":>8}')
    sep = '─' * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, res in results.items():
        if 'error' in res:
            print(f'{name:<{col_w}}  {"ERROR":>10}')
        else:
            ess_str = f'{res["ess"]:.1f}' if res['ess'] is not None else '–'
            gr_str = f'{res["gr"]:.4f}' if res['gr'] is not None else '–'
            print(f'{name:<{col_w}}  {res["time"]:>10.1f}  {res["nsamples"]:>10}'
                  f'  {ess_str:>8}  {gr_str:>8}')
    print(sep)
    return results


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    requested = sys.argv[1:] or None
    with tempfile.TemporaryDirectory() as tmpdir:
        run_benchmark(sampler_names=requested, directory=tmpdir)
