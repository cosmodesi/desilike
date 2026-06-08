"""Benchmark full galaxy-clustering pipelines.

Pipelines
---------
BAO correlation function:
    DampedBAOWigglesTracerCorrelation2Poles → Correlation2PolesObservable
    → ObservablesGaussianLikelihood → Posterior.

FOLPS full-shape:
    CosmoprimoCosmology + DirectSpectrum2Template
    → FOLPSTracerSpectrum2Poles → Spectrum2PolesObservable
    → ObservablesGaussianLikelihood → Posterior.

FOLPS full-shape with TaylorEmulator on FOLPSPTSpectrum2Poles:
    Same as above but the PT sub-graph is replaced by a degree-3 Taylor emulator
    fitted once before timing.

Run directly::

    python -m desilike.tests.benchmark
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from desilike.base import compile, get_params, Posterior, replace, SumLikelihood
from desilike import TaylorEmulator
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
    pt = DampedBAOWigglesPTSpectrum2Poles(template=template)
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
        for param in get_params(likelihood).select(basename='*l*', fixed=False):
            param.update(derived='best')
    return Posterior(likelihood)


# ── FOLPS full-shape pipeline ────────────────────────────────────────────────

ELLS_FOLPS = (0, 2, 4)
K = np.linspace(0.02, 0.2, 101)


def build_posterior_folps(k=K, ells=ELLS_FOLPS, marginalize=False, emulator_order=None):
    """FOLPS full-shape pipeline (no window matrix).

    With ``marginalize=True`` the counter-term (alpha*) and stochastic (sn*)
    parameters are solved analytically (``derived='best'``).
    With ``emulator_order`` not None, the PT sub-graph is replaced by a Taylor
    emulator of that polynomial order before building the observable.
    """
    n = len(ells) * len(k)

    cosmo = CosmoprimoCosmology(engine='camb')
    template = DirectSpectrum2Template(z=0.8, cosmo=cosmo)
    theory = FOLPSTracerSpectrum2Poles(k=k, template=template, ells=ells)

    if emulator_order is not None:
        print(f'  fitting TaylorEmulator (order={emulator_order}) on PT sub-graph …', end=' ', flush=True)
        t0 = time.perf_counter()
        pt_emulator = TaylorEmulator(compile(theory.pt), order=emulator_order)
        pt_emulator.fit()
        print(f'done ({(time.perf_counter() - t0) * 1e3:.0f} ms)')
        replace(theory, theory.pt, pt_emulator.to_calculator())

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


# ── FOLPS multi-tracer pipeline ──────────────────────────────────────────────

TRACERS_FOLPS = ['LRG', 'ELG']
Z_TRACERS = {'LRG': 0.8, 'ELG': 1.1}


def build_posterior_folps_multitracer(k=K, ells=ELLS_FOLPS, tracers=TRACERS_FOLPS, marginalize=False, emulator_order=None):
    """FOLPS SumLikelihood over independent per-tracer pipelines.

    Each tracer gets its own CosmoprimoCosmology + DirectSpectrum2Template +
    FOLPSTracerSpectrum2Poles.  All parameters (including cosmological) are
    namespaced under the tracer name so they are sampled independently.

    With ``marginalize=True`` the alpha* and sn* parameters of every tracer
    are solved analytically.
    With ``emulator_order`` not None, each tracer's PT sub-graph is replaced by
    a Taylor emulator of that polynomial order.
    """
    n = len(ells) * len(k)
    rng = np.random.default_rng(42)

    likelihoods = []
    for tracer in tracers:
        cosmo = CosmoprimoCosmology(engine='camb')
        template = DirectSpectrum2Template(z=Z_TRACERS.get(tracer, 0.8), cosmo=cosmo)
        theory = FOLPSTracerSpectrum2Poles(k=k, template=template, ells=ells, tracers=(tracer,))
        params = get_params(theory, level=1)
        theory.update(params=params)

        if emulator_order is not None:
            pt_emulator = TaylorEmulator(compile(theory.pt), order=emulator_order)
            pt_emulator.fit()
            replace(theory, theory.pt, pt_emulator.to_calculator())

        data = rng.normal(scale=1e2, size=n)
        covariance = np.diag(np.full(n, 1e4))
        observable = Spectrum2PolesObservable(data=data, theory=theory, k=k, ells=ells,
                                              covariance=covariance)

        lik = ObservablesGaussianLikelihood(observables=observable)
        if marginalize:
            for param in get_params(lik).select(basename='alpha*'):
                param.update(derived='best')
            for param in get_params(lik).select(basename='sn*'):
                param.update(derived='best')
        likelihoods.append(lik)

    likelihood = SumLikelihood(likelihoods)
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


def run(label, build_fn, vary_param=None, batch_size=8, run=('eager', 'jit', 'grad', 'vmap', 'profile'), **kwargs):
    """Compile and benchmark one pipeline variant."""
    print(f'\n=== {label} ===')
    pipe = compile(build_fn())

    params = {p.name: float(p.value) for p in pipe.params.select(fixed=False, derived=False)}
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
    if 'profile' in run:
        from desilike.profilers import MinuitProfiler
        profiler = MinuitProfiler(pipe, seed=42)
        profiler.maximize()
        profiles = profiler.profiles
        print(profiles.to_stats(tablefmt='pretty'))


def main(test=('folps_multi', 'folps_multi_emu')):
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

    if 'folps_multi' in test:
        tracers = TRACERS_FOLPS
        print(f'\n{"─" * 60}')
        print(f'FOLPS multi-tracer ({"+".join(tracers)}): ells={ELLS_FOLPS}, '
            f'k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size per tracer={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps_multitracer(marginalize=False),
            vary_param=f'{tracers[0]}::logA', warmup=2, number=2, run=('eager', 'jit'))
        run('with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps_multitracer(marginalize=True),
            vary_param=f'{tracers[0]}::logA', warmup=2, number=2, run=('eager', 'jit'))

    if 'folps_emu' in test:
        print(f'\n{"─" * 60}')
        print(f'FOLPS + TaylorEmulator(order=1) on PT: ells={ELLS_FOLPS}, '
            f'k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps(marginalize=False, emulator_order=1), vary_param='logA', warmup=2, number=10)
        run('with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps(marginalize=True, emulator_order=1), vary_param='logA', warmup=2, number=10)

    if 'folps_multi_emu' in test:
        tracers = TRACERS_FOLPS
        print(f'\n{"─" * 60}')
        print(f'FOLPS multi-tracer ({"+".join(tracers)}) + TaylorEmulator(order=1) on PT: '
            f'ells={ELLS_FOLPS}, k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size per tracer={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps_multitracer(marginalize=False, emulator_order=1),
            vary_param=f'{tracers[0]}::logA', warmup=2, number=10, run=('eager', 'jit', 'grad', 'vmap'))
        run('with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps_multitracer(marginalize=True, emulator_order=1),
            vary_param=f'{tracers[0]}::logA', warmup=2, number=10, run=('eager', 'jit', 'grad'))


if __name__ == '__main__':
    main(test=('bao',))