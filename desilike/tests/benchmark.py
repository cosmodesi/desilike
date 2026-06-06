"""Benchmark full galaxy-clustering pipelines.

Pipelines
---------
BAO correlation function:
    DampedBAOWigglesTracerCorrelation2Poles → Correlation2PolesObservable
    → ObservablesGaussianLikelihood → Posterior.

Full-shape (velocileptors REPT):
    CosmoprimoCosmology + DirectSpectrum2Template
    → REPTVelocileptorsTracerSpectrum2Poles → Spectrum2PolesObservable
    → ObservablesGaussianLikelihood → Posterior.

Run directly::

    python -m desilike.tests.benchmark
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from desilike.base import compile, params as get_params, Posterior
from desilike.theories.galaxy_clustering import (BAOSpectrum2Template,
                                                 DampedBAOWigglesPTSpectrum2Poles,
                                                 DampedBAOWigglesTracerCorrelation2Poles,
                                                 CosmoprimoCosmology, DirectSpectrum2Template,
                                                 FOLPSTracerSpectrum2Poles)
from desilike.observables import Correlation2PolesObservable, Spectrum2PolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood


# ── BAO correlation-function pipeline ────────────────────────────────────────

ELLS_BAO = (0, 2)
S = np.arange(20., 180., 4.)
BROADBAND = 'pcs2'


def build_posterior_bao(s=S, ells=ELLS_BAO, marginalize=False):
    """BAO correlation pipeline with diagonal window.

    With ``marginalize=True`` the broadband parameters (bl*) are solved
    analytically (``derived='best'``) instead of being sampled.
    """
    n = len(ells) * len(s)

    from cosmoprimo.fiducial import BOSS
    fiducial = BOSS(engine='eisenstein_hu')
    template = BAOSpectrum2Template(z=0.5, fiducial=fiducial)
    pt = DampedBAOWigglesPTSpectrum2Poles(template=template, ells=ells)
    theory = DampedBAOWigglesTracerCorrelation2Poles(s=s, pt=pt, ells=ells, broadband=BROADBAND)

    window = np.eye(n)
    rng = np.random.default_rng(42)
    data = rng.normal(scale=1e-3, size=n)
    covariance = np.diag(np.full(n, 1e-6))

    observable = Correlation2PolesObservable(data=data, theory=theory, s=s, ells=ells,
                                             window=window, sin=s, ellsin=ells,
                                             covariance=covariance)
    likelihood = ObservablesGaussianLikelihood(observables=observable)
    if marginalize:
        for param in get_params(likelihood).select(basename='bl*'):
            param.update(derived='best')
    return Posterior(likelihood)


# ── FOLPS full-shape pipeline ────────────────────────────────────────────────

ELLS_FOLPS = (0, 2, 4)
K = np.linspace(0.02, 0.2, 101)


def build_posterior_folps(k=K, ells=ELLS_FOLPS, marginalize=False):
    """FOLPS full-shape pipeline (no window matrix).

    With ``marginalize=True`` the counter-term (alpha*) and stochastic (sn*)
    parameters are solved analytically (``derived='best'``).
    """
    n = len(ells) * len(k)

    cosmo = CosmoprimoCosmology(engine='eisenstein_hu')
    template = DirectSpectrum2Template(z=0.8, cosmo=cosmo, engine='eisenstein_hu')
    theory = FOLPSTracerSpectrum2Poles(k=k, template=template, ells=ells)

    rng = np.random.default_rng(42)
    data = rng.normal(scale=1e2, size=n)
    covariance = np.diag(np.full(n, 1e4))

    observable = Spectrum2PolesObservable(data=data, theory=theory, k=k, ells=ells,
                                          covariance=covariance)
    likelihood = ObservablesGaussianLikelihood(observables=observable)
    if marginalize:
        for param in get_params(likelihood).select(basename='alpha*'):
            param.update(derived='best')
        for param in get_params(likelihood).select(basename='sn*'):
            param.update(derived='best')
    return Posterior(likelihood)


# ── timing harness ───────────────────────────────────────────────────────────

def _bench(label, fn, number=5, warmup=3):
    """Time *fn(i)* over *number* calls; return ms/call.

    *fn* receives the iteration index so callers can vary inputs and avoid the
    compiled graph's identical-params result cache.
    """
    for bench_idx in range(warmup):
        jax.block_until_ready(fn(bench_idx))
    start = time.perf_counter()
    for bench_idx in range(number):
        jax.block_until_ready(fn(bench_idx))
    dt = (time.perf_counter() - start) / number * 1e3
    print(f'  {label:<26s} {dt:9.4f} ms/call   ({number} calls)')
    return dt


def run(label, build_fn, vary_param=None, batch_size=8, run=('eager', 'jit', 'grad', 'vmap'), **kwargs):
    """Compile and benchmark one pipeline variant."""
    print(f'\n=== {label} ===')
    pipe = compile(build_fn())

    params = {p.name: float(p.value) for p in pipe.params if not p.derived}
    print(f'  sampled parameters ({len(params)}): {", ".join(params)}')
    print(f'  logpdf at center: {float(pipe(params)):.4f}\n')

    if vary_param is None or vary_param not in params:
        vary_param = next(iter(params))

    def perturbed(bench_idx):
        return {**params, vary_param: params[vary_param] + 1e-4 * bench_idx}

    def batch(bench_idx):
        return {name: jnp.full(batch_size, value) for name, value in perturbed(bench_idx).items()}

    if 'eager' in run:
        _bench('eager call', lambda bench_idx: pipe(perturbed(bench_idx)), **kwargs)
    if 'jit' in run:
        jit_pipe = jax.jit(pipe)
        _bench('jit call', lambda bench_idx: jit_pipe(perturbed(bench_idx)), **kwargs)
    if 'grad' in run:
        grad = jax.jit(jax.grad(pipe))
        _bench('jit grad call', lambda bench_idx: grad(perturbed(bench_idx)), **kwargs)
    if 'vmap' in run:
        vpipe = jax.jit(jax.vmap(pipe))
        dt = _bench(f'jit vmap call (n={batch_size})', lambda bench_idx: vpipe(batch(bench_idx)), **kwargs)
        print(f'  {"→ per-sample":<26s} {dt / batch_size:9.4f} ms/sample')


def main(test=('folps',)):
    if 'bao' in test:
        print(f'\n{"─" * 60}')
        print(f'BAO correlation: ells={ELLS_BAO}, s=arange(20, 180, 4) ({len(S)} points), '
            f'data size={len(ELLS_BAO) * len(S)}, broadband={BROADBAND!r}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_bao(marginalize=False), vary_param='b1')
        run('with analytic marg. (bl* → best)', lambda: build_posterior_bao(marginalize=True), vary_param='b1')

    if 'folps' in test:
        print(f'\n{"─" * 60}')
        print(f'FOLPS: ells={ELLS_FOLPS}, k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps(marginalize=False), vary_param='logA', warmup=2, number=2, run=('eager', 'jit'))
        run('with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps(marginalize=True), vary_param='logA', warmup=2, number=2, run=('eager', 'jit'))


if __name__ == '__main__':
    main()
