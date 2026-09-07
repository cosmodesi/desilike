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

FOLPS eisenstein_hu vs TaylorEmulator (``folps_vs_emu``):
    Head-to-head: ``engine='eisenstein_hu'`` (JAX-native, no emulator) vs
    ``engine='camb'`` + ``TaylorEmulator(order=1)`` on the PT sub-graph.

COMET full-shape (GP-emulator-based):
    COMETTracerSpectrum2Poles (+ COMETTracerSpectrum3Poles, optionally) →
    Spectrum2(3)PolesObservable → ObservablesGaussianLikelihood → Posterior.
    ``direct=True`` uses the ``pt=False`` path (comet's monolithic Pell()/
    Bell_Sugi(), no separate PT calculator) instead of the default, shared-PT
    PX_ell()/BX_ell_Sugi()-decomposed path -- see build_posterior_comet().

Run directly::

    python -m desilike.tests.benchmark
"""

import functools
import time

import numpy as np
import jax
import jax.numpy as jnp

from desilike.base import compile, get_params, Posterior, Prior, replace, SumLikelihood
from desilike import TaylorEmulator
from desilike.theories import ACECosmology
from desilike.theories.galaxy_clustering import (BAOSpectrum2Template,
                                                 DampedBAOWigglesPTSpectrum2Poles,
                                                 DampedBAOWigglesTracerCorrelation2Poles,
                                                 CosmoprimoCosmology, DirectSpectrum2Template,
                                                 FOLPSTracerSpectrum2Poles, FOLPSTracerSpectrum3Poles,
                                                 COMETPTSpectrum2Poles, COMETPTSpectrum3Poles,
                                                 COMETTracerSpectrum2Poles, COMETTracerSpectrum3Poles)
from desilike.observables.galaxy_clustering import Correlation2PolesObservable, Spectrum2PolesObservable, Spectrum3PolesObservable
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
    data = compile(theory)()
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
TRACERS_FOLPS = ['LRG', 'ELG']
Z_TRACERS = {'LRG1': 0.4, 'LRG2': 0.6, 'LRG3': 0.8, 'LRG': 0.8, 'ELG': 1.1, 'QSO': 2.1}
K3 = np.column_stack([np.linspace(0.01, 0.1, 11)] * 2)
ELLS3_FOLPS = ((0, 0, 0), (2, 0, 2))

# Packaged / local ACE emulator set (pure-JAX Boltzmann replacement): background + linear pk
# serve DirectSpectrum2Template, the Capse Cl networks are unused here but kept so the very
# same cosmo can also feed a CMB likelihood.
ACE_ENGINE = {'background': 'ACE_mnuw0wacdm_ln10As_basis',
              'fourier': 'mnuw0wacdm_class',
              'harmonic': 'capse_mnuw0wacdm_250001'}


def _ace_base_dir(engine):
    """Directory holding the non-packaged emulators of *engine* (e.g. the local Capse set).

    ``None`` (the ACECosmology default, ``Installer().install_dir / 'ace-emulators'``) unless
    a directory-based emulator is requested and only found next to the desilike checkout.
    """
    from pathlib import Path
    from desilike.install import Installer
    from desilike.theories.primordial_cosmology import _PACKAGED_EMULATORS

    names = list(engine.values()) if isinstance(engine, dict) else [engine]
    dir_names = [name for name in names if name not in _PACKAGED_EMULATORS and name != 'ace']
    if not dir_names:
        return None
    candidates = [Path(Installer().install_dir) / 'ace-emulators',
                  Path(__file__).parent.parent.parent.parent]  # desilike checkout's parent
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in dir_names):
            return candidate
    raise FileNotFoundError(f'emulator directories {dir_names} not found under any of {candidates}')


def build_posterior_folps(k=K, ells=ELLS_FOLPS, tracers=None, marginalize=False, emulator_order=None,
                          include_3poles=False, k3=K3, ells3=ELLS3_FOLPS, engine='camb',
                          prior_basis='physical_aap'):
    """FOLPS full-shape pipeline, optionally multi-tracer and/or with bispectrum.

    When ``tracers`` is ``None`` a single un-namespaced pipeline is built.
    When ``tracers`` is a list (e.g. ``['LRG', 'ELG']``) each tracer gets its
    own CosmoprimoCosmology + DirectSpectrum2Template + FOLPSTracerSpectrum2Poles
    with parameters namespaced under the tracer name; the per-tracer likelihoods
    are combined via SumLikelihood.

    With ``marginalize=True`` the counter-term (alpha*) and stochastic (sn*)
    parameters are solved analytically (``derived='best'``).
    With ``emulator_order`` not None, each PT sub-graph is replaced by a Taylor
    emulator of that polynomial order.
    With ``include_3poles=True`` each tracer also gets a FOLPSTracerSpectrum3Poles
    (bispectrum) sharing the same PT sub-graph as the power spectrum.
    ``k3`` (shape (N, 2)) and ``ells3`` control the bispectrum k-grid and multipoles.
    ``engine`` selects the Boltzmann solver: a cosmoprimo engine name (``'camb'``,
    ``'eisenstein_hu'``, …) for :class:`CosmoprimoCosmology`, or ``'ace'`` / an
    emulator-set dict (e.g. :data:`ACE_ENGINE`) for the pure-JAX :class:`ACECosmology`.
    ``prior_basis`` is the FOLPS bias/counterterm/stochastic parameterization.
    """
    n2 = len(ells) * len(k)
    n3 = len(ells3) * len(k3)
    rng = np.random.default_rng(42)
    tracer_list = [None] if tracers is None else list(tracers)

    is_ace = isinstance(engine, dict) or engine == 'ace'
    if is_ace:
        base_dir = _ace_base_dir(engine)
        cosmo = ACECosmology(engine=engine, base_dir=base_dir, fiducial='DESI')
    else:
        cosmo = CosmoprimoCosmology(engine=engine, fiducial='DESI')
    params = get_params(cosmo)
    for param in params.select(basename=['w0_fld', 'wa_fld']):
        param.update(fixed=False)
    if is_ace:
        # The emulators are only trained over a finite hypercube and return NaN outside it;
        # clip the (much wider) default priors to the training ranges.
        params = ACECosmology.truncate_priors(params, engine=engine, base_dir=base_dir)
    cosmo.update(params=params)
    likelihoods = []
    for tracer in tracer_list:
        z = Z_TRACERS.get(tracer, 0.8)
        template = DirectSpectrum2Template(z=z, cosmo=cosmo)
        tracer_arg = (tracer,) if tracer is not None else None
        theory = FOLPSTracerSpectrum2Poles(k=k, template=template, ells=ells, tracers=tracer_arg, prior_basis=prior_basis)
        params = get_params(theory, level=1)
        if marginalize:
            for param in params.select(basename=['alpha*', 'sn*']):
                param.update(derived='best')
        theory.update(params=params)

        if emulator_order is not None:
            print(f'  fitting TaylorEmulator (order={emulator_order}) on PT sub-graph …', end=' ', flush=True)
            t0 = time.perf_counter()
            pt_emulator = TaylorEmulator(compile(theory.pt), order=emulator_order)
            pt_emulator.fit()
            print(f'done ({(time.perf_counter() - t0) * 1e3:.0f} ms)')
            replace(theory, theory.pt, pt_emulator.to_calculator())

        data2 = rng.normal(scale=1e2, size=n2)
        observable2 = Spectrum2PolesObservable(data=data2, theory=theory, k=k, ells=ells)
        observables = [observable2]
        covariances = [np.diag(np.full(n2, 1e4))]

        if include_3poles:
            theory3 = FOLPSTracerSpectrum3Poles(k=k3, pt=theory.pt, ells=ells3, tracers=tracer_arg, prior_basis=prior_basis)
            data3 = rng.normal(scale=1e8, size=n3)
            observable3 = Spectrum3PolesObservable(data=data3, theory=theory3, k=k3, ells=ells3)
            observables.append(observable3)
            covariances.append(np.diag(np.full(n3, 1e16)))

        from scipy.linalg import block_diag
        like = ObservablesGaussianLikelihood(observables=observables, covariance=block_diag(*covariances))
        if marginalize:
            for param in get_params(like).select(basename='alpha*'):
                param.update(derived='best')
            for param in get_params(like).select(basename='sn*'):
                param.update(derived='best')
        likelihoods.append(like)

    likelihood = SumLikelihood(likelihoods)

    def CustomPrior(Prior):

        def __call__(self):
            self.logpdf = super().__call__()
            self.logpdf = jnp.where(self.params['w0_fld'].value + self.params['wa_fld'].value < 0, self.logpdf, -jnp.inf)
            return self.logpdf

    return Posterior(likelihood, prior=CustomPrior(get_params(likelihood)))


# ── COMET full-shape pipeline ────────────────────────────────────────────────

ELLS_COMET = (0, 2, 4)
Z_COMET = 1.0
K_COMET = np.linspace(0.02, 0.3, 60)
K3_COMET = np.column_stack([np.linspace(0.02, 0.1, 11)] * 2)
ELLS3_COMET = ((0, 0, 0), (2, 0, 2))


def build_posterior_comet(k=K_COMET, ells=ELLS_COMET, z=Z_COMET, prior_basis='physical_aap',
                          direct=False, include_3poles=False, k3=K3_COMET, ells3=ELLS3_COMET, marginalize=False):
    """COMET full-shape pipeline.

    With ``direct=True``, both COMETTracerSpectrum2Poles and (if requested)
    COMETTracerSpectrum3Poles use ``pt=False`` (comet's monolithic, bias-combined
    Pell()/Bell_Sugi(), no separate PT calculator/diagram decomposition) instead
    of the default shared-PT path -- compare the two to see whether skipping the
    PX_ell()/BX_ell_Sugi() decomposition is worth it for a single (non-shared)
    tracer. With ``include_3poles=True``, COMETTracerSpectrum3Poles gets its own
    COMETPTSpectrum3Poles (unlike FOLPS, COMET's power-spectrum and bispectrum PTs
    are different classes -- they cannot share one PT sub-graph), but both PTs
    share the same ``cosmo`` (so e.g. the narrowed ``h`` prior below applies to
    both).
    """
    n2 = len(ells) * len(k)

    # COMET's GP emulator is only trained/valid over a finite cosmological hypercube;
    # out-of-range samples yield NaN outputs (see _comet_params_validity), which a profiler
    # exploring the full prior box (e.g. Minuit) can and does hit. Narrow omega_b here for
    # benchmark stability, on a cosmo built upfront and threaded through (a post-hoc
    # get_params(theory).select('h').update(...) does *not* propagate -- the
    # Posterior/likelihood machinery ends up reading a different copy of the parameter).
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
    params = get_params(cosmo)
    params['omega_b'].update(prior=dict(dist='norm', loc=params['omega_b'].value, scale=0.003))
    cosmo.update(params=params)

    if direct:
        theory = COMETTracerSpectrum2Poles(k=k, z=z, ells=ells, prior_basis=prior_basis, pt=False, cosmo=cosmo)
    else:
        pt = COMETPTSpectrum2Poles(k=k, z=z, ells=ells, cosmo=cosmo)
        theory = COMETTracerSpectrum2Poles(k=k, z=z, ells=ells, prior_basis=prior_basis, pt=pt)
    # Anchor mock data to the theory prediction at default parameters.  Using a fixed
    # absolute noise scale (1e2) with COMET amplitudes (~1e3-1e4) makes chi2 ride the
    # model amplitude to its prior boundary rather than recovering a meaningful best fit.
    data2 = np.asarray(compile(theory)()).ravel()
    observable2 = Spectrum2PolesObservable(data=data2, theory=theory, k=k, ells=ells)
    observables = [observable2]
    covariances = [np.diag(800**2 * np.ones_like(data2))]

    if include_3poles:
        if direct:
            theory3 = COMETTracerSpectrum3Poles(k=k3, z=z, ells=ells3, prior_basis=prior_basis, pt=False, cosmo=cosmo)
        else:
            pt3 = COMETPTSpectrum3Poles(k=k3, z=z, ells=ells3, cosmo=cosmo)
            theory3 = COMETTracerSpectrum3Poles(k=k3, z=z, ells=ells3, prior_basis=prior_basis, pt=pt3)
        data3 = np.asarray(compile(theory3)()).ravel()
        observable3 = Spectrum3PolesObservable(data=data3, theory=theory3, k=k3, ells=ells3)
        observables.append(observable3)
        covariances.append(1000**2 * np.ones_like(data2))

    from scipy.linalg import block_diag
    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=block_diag(*covariances))
    if marginalize:
        for param in get_params(likelihood).select(basename=['a0', 'a2', 'a4', 'NP0', 'NP20', 'NP22']):
            param.update(derived='best')
    return Posterior(likelihood)


# ── timing harness ───────────────────────────────────────────────────────────

_COSMO_BASENAMES = frozenset({'omega_b', 'omega_cdm', 'h', 'n_s', 'logA', 'w0_fld', 'wa_fld', 'm_ncdm'})


def _bench_grad_subsets(pipe, params, number=5, warmup=3):
    """Compare jit(grad) timing for bias-only vs cosmo-only vs all-params differentiation.

    Closing over one group as concrete Python floats makes JAX constant-fold that
    sub-graph: grad w.r.t. bias only never differentiates through the Boltzmann/PT
    emulator; grad w.r.t. cosmo only never differentiates through the bias combination.
    The difference reveals which part of the pipeline dominates the gradient cost.
    """
    cosmo_names = sorted(name for name in params if name.split('.')[-1] in _COSMO_BASENAMES)
    bias_names = sorted(name for name in params if name not in cosmo_names)

    cosmo_vals = {name: params[name] for name in cosmo_names}
    bias_vals = {name: params[name] for name in bias_names}

    print(f'  cosmo params ({len(cosmo_names)}): {", ".join(cosmo_names)}')
    print(f'  bias  params ({len(bias_names)}): {", ".join(bias_names)}')
    print()

    def _time_grad(label, grad_fn, base_dict, vary_name):
        try:
            for bench_idx in range(warmup):
                g = grad_fn({**base_dict, vary_name: base_dict[vary_name] + 1e-4 * bench_idx})
            leaves = jax.tree_util.tree_leaves(g)
            finite = all(np.isfinite(float(v)) for v in leaves)
            start = time.perf_counter()
            for bench_idx in range(number):
                jax.block_until_ready(grad_fn({**base_dict, vary_name: base_dict[vary_name] + 1e-4 * bench_idx}))
            dt = (time.perf_counter() - start) / number * 1e3
            suffix = '' if finite else '  [NaN gradients]'
            print(f'  {label:<32s} {dt:9.4f} ms/call   ({number} calls){suffix}')
        except (MemoryError, Exception) as exc:
            short = str(exc).split('\n')[0][:60]
            print(f'  {label:<32s} {"FAILED":>9s}          [{type(exc).__name__}: {short}]')
            dt = float('nan')

    vary_bias = bias_names[0]
    vary_cosmo = cosmo_names[0]

    def pipe_bias_only(bias_dict):
        return pipe({**cosmo_vals, **bias_dict})

    def pipe_cosmo_only(cosmo_dict):
        return pipe({**bias_vals, **cosmo_dict})

    grad_bias = jax.jit(jax.grad(pipe_bias_only))
    grad_cosmo = jax.jit(jax.grad(pipe_cosmo_only))
    grad_full = jax.jit(jax.grad(pipe))

    _time_grad('jit grad (bias only)', grad_bias, bias_vals, vary_bias)
    _time_grad('jit grad (cosmo only)', grad_cosmo, cosmo_vals, vary_cosmo)
    _time_grad('jit grad (all params)', grad_full, params, vary_bias)


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


def run(label, build_fn, vary_param=None, batch_size=8, run=('eager', 'jit', 'grad', 'vmap', 'profile'),
       profile_kwargs=None, **kwargs):
    """Compile and benchmark one pipeline variant."""
    print(f'\n=== {label} ===')
    pipe = compile(build_fn())

    params = {p.name: float(p.value) for p in pipe.params.select(fixed=False, derived=False)}
    solved_params = pipe.params.select(solved=True).names()
    print(f'  sampled parameters ({len(params)}): {", ".join(params)}')
    print(f'  solved parameters ({len(solved_params)}): {", ".join(solved_params)}')
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
        from desilike import profilers
        profiler = profilers.Profiler(pipe, kernel=profilers.Minuit(), rng=69)
        t0 = time.perf_counter()
        profiler.maximize(**(profile_kwargs or {}))
        print(f'  {"profile (maximize)":<26s} {(time.perf_counter() - t0) * 1e3:9.1f} ms total')
        profiles = profiler.profiles
        print(profiles.to_stats(tablefmt='pretty'))


def main(test=('folps_multi', 'folps_multi_emu', 'folps_vs_emu')):
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
        run('without analytic marg.', lambda: build_posterior_folps(tracers=tracers, marginalize=False),
            vary_param=f'logA', warmup=2, number=2, run=('eager', 'jit'))
        run('with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps(tracers=tracers, marginalize=True),
            vary_param=f'logA', warmup=2, number=2, run=('eager', 'jit'))

    if 'folps_emu' in test:
        print(f'\n{"─" * 60}')
        print(f'FOLPS + TaylorEmulator(order=1) on PT: ells={ELLS_FOLPS}, '
            f'k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps(marginalize=False, emulator_order=1), vary_param='logA', warmup=2, number=10)
        #run('with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps(marginalize=True, emulator_order=1), vary_param='logA', warmup=2, number=10)

    if 'folps_noemu' in test:
        print(f'\n{"─" * 60}')
        print(f'FOLPS: eisenstein_hu (no emulator) vs camb + TaylorEmulator(order=1)')
        print(f'ells={ELLS_FOLPS}, k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('EH (no emulator), analytic marg.',
            lambda: build_posterior_folps(tracers=['LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'], marginalize=True, engine='camb'),
            vary_param='logA', warmup=2, number=10, run=('jit',))

    if 'folps_3poles' in test:
        print(f'\n{"─" * 60}')
        print(f'FOLPS: ells={ELLS_FOLPS}, k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps(marginalize=False, include_3poles=True), vary_param='logA', warmup=2, number=2, run=('eager', 'jit'))

    if 'folps_emu_3poles' in test:
        print(f'\n{"─" * 60}')
        print(f'FOLPS + TaylorEmulator(order=1) on PT: ells={ELLS_FOLPS}, '
            f'k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps(marginalize=False, emulator_order=1, include_3poles=True), vary_param='logA', warmup=2, number=10, run=('eager', 'jit'))

    if 'folps_multi_emu' in test:
        tracers = TRACERS_FOLPS
        print(f'\n{"─" * 60}')
        print(f'FOLPS multi-tracer ({"+".join(tracers)}) + TaylorEmulator(order=1) on PT: '
            f'ells={ELLS_FOLPS}, k=linspace(0.02, 0.2, {len(K)}) ({len(K)} points), '
            f'data size per tracer={len(ELLS_FOLPS) * len(K)}')
        print(f'{"─" * 60}')
        run('without analytic marg.', lambda: build_posterior_folps(tracers=tracers, marginalize=False, emulator_order=1),
            vary_param=f'logA', warmup=2, number=10, run=('eager', 'jit', 'grad', 'vmap'))
        run('with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps(tracers=tracers, marginalize=True, emulator_order=1),
            vary_param=f'logA', warmup=2, number=10, run=('eager', 'jit', 'grad'))

    if 'folps_multi_3poles' in test:
        tracers = TRACERS_FOLPS
        print(f'\n{"─" * 60}')
        print(f'FOLPS multi-tracer ({"+".join(tracers)}) 2pt+3pt: ells2={ELLS_FOLPS}, ells3={ELLS3_FOLPS}, '
            f'k2=linspace(0.02, 0.2, {len(K)}) ({len(K)} pts), '
            f'k3 diagonal ({len(K3)} pts), '
            f'data size per tracer={len(ELLS_FOLPS) * len(K) + len(ELLS3_FOLPS) * len(K3)}')
        print(f'{"─" * 60}')
        run('2pt+3pt without analytic marg.', lambda: build_posterior_folps(tracers=tracers, marginalize=False, include_3poles=True),
            vary_param=f'logA', warmup=2, number=2, run=('eager', 'jit'))
        run('2pt+3pt with analytic marg. (alpha*+sn* → best)', lambda: build_posterior_folps(tracers=tracers, marginalize=True, include_3poles=True),
            vary_param=f'logA', warmup=2, number=2, run=('eager', 'jit'))

    if 'folps_ace_3poles' in test:
        print(f'\n{"─" * 60}')
        print(f'FOLPS 2pt+3pt with ACECosmology ({", ".join(f"{k}={v}" for k, v in ACE_ENGINE.items())}): '
            f'ells2={ELLS_FOLPS}, ells3={ELLS3_FOLPS}, '
            f'k2=linspace(0.02, 0.2, {len(K)}) ({len(K)} pts), k3 diagonal ({len(K3)} pts), '
            f"prior_basis='physical_aap', data size={len(ELLS_FOLPS) * len(K) + len(ELLS3_FOLPS) * len(K3)}")
        print(f'{"─" * 60}')
        run('2pt+3pt without analytic marg.',
            lambda: build_posterior_folps(marginalize=False, include_3poles=True, engine=ACE_ENGINE, prior_basis='physical_aap'),
            vary_param='logA', warmup=2, number=2, run=('eager', 'jit'))
        run('2pt+3pt with analytic marg. (alpha*+sn* → best)',
            lambda: build_posterior_folps(marginalize=True, include_3poles=True, engine=ACE_ENGINE, prior_basis='physical_aap'),
            vary_param='logA', warmup=2, number=2, run=('eager', 'jit'))

    if 'comet' in test:
        print(f'\n{"─" * 60}')
        print(f'COMET: ells={ELLS_COMET}, k=linspace(0.02, 0.3, {len(K_COMET)}) ({len(K_COMET)} points), '
            f'data size={len(ELLS_COMET) * len(K_COMET)}')
        print(f'{"─" * 60}')
        # Mock data is theory-anchored (see build_posterior_comet's comment), so the
        # chi2 surface has genuine structure for Minuit to climb rather than riding a
        # parameter to its prior boundary.
        run('shared PT (PX_ell, marg)', lambda: build_posterior_comet(direct=False, marginalize=False),
            vary_param='n_s', warmup=2, number=5, run=('jit', 'grad'))
        run('shared PT (PX_ell, marg)', lambda: build_posterior_comet(direct=False, marginalize=True),
            vary_param='n_s', warmup=2, number=5, run=('jit', 'grad'))
        run('direct (Pell, pt=False)', lambda: build_posterior_comet(direct=True),
            vary_param='b1', warmup=2, number=5, run=('jit', 'grad', 'profile'))

    if 'comet_grad_split' in test:
        print(f'\n{"─" * 60}')
        print(f'COMET grad split (bias-only vs cosmo-only vs full): ells={ELLS_COMET}, '
              f'k=linspace(0.02, 0.3, {len(K_COMET)}) ({len(K_COMET)} points), '
              f'data size={len(ELLS_COMET) * len(K_COMET)}')
        print(f'{"─" * 60}')
        for label, build_kw in [
            ('shared PT (PX_ell)', dict(direct=False)),
            ('direct (Pell, pt=False)', dict(direct=True)),
        ]:
            print(f'\n=== {label} ===')
            pipe = compile(build_posterior_comet(**build_kw))
            params_dict = {p.name: float(p.value) for p in pipe.params.select(fixed=False, derived=False)}
            print(f'  sampled parameters ({len(params_dict)}): {", ".join(params_dict)}')
            print(f'  logpdf at center: {float(pipe(params_dict)):.4f}\n')
            _bench_grad_subsets(pipe, params_dict)

    if 'comet_grad_marg' in test:
        print(f'\n{"─" * 60}')
        print(f'COMET grad: effect of analytic marginalization of a0/a2/a4/NP0/NP20/NP22')
        print(f'ells={ELLS_COMET}, k=linspace(0.02, 0.3, {len(K_COMET)}) ({len(K_COMET)} points), '
              f'data size={len(ELLS_COMET) * len(K_COMET)}')
        print(f'{"─" * 60}')
        for label, build_kw in [
            ('shared PT, no marg (16 params)', dict(direct=False, marginalize=False)),
            ('shared PT, marg a0/a2/a4/NP0/NP20/NP22 (10 params)', dict(direct=False, marginalize=True)),
        ]:
            print(f'\n=== {label} ===')
            pipe = compile(build_posterior_comet(**build_kw))
            params_dict = {p.name: float(p.value) for p in pipe.params.select(fixed=False, derived=False)}
            print(f'  sampled parameters ({len(params_dict)}): {", ".join(params_dict)}')
            print(f'  logpdf at center: {float(pipe(params_dict)):.4f}\n')
            vary_name = next(iter(params_dict))
            jit_pipe = jax.jit(pipe)
            _bench('jit call', lambda bench_idx: jit_pipe({**params_dict, vary_name: params_dict[vary_name] + 1e-4 * bench_idx}))

    if 'comet_marg_breakdown' in test:
        print(f'\n{"─" * 60}')
        print(f'COMET marg breakdown: primal vs JVP vs extra likelihood pass')
        print(f'ells={ELLS_COMET}, k=linspace(0.02, 0.3, {len(K_COMET)}) ({len(K_COMET)} points)')
        print(f'{"─" * 60}')
        posterior = compile(build_posterior_comet(direct=False, marginalize=True))
        # Extract the group_fn from the first (and only) group
        for (group_alpha_names, group_alpha_sizes, group_alpha_shapes,
             group_theory_pipe, comp_meta, marg_local, best_local,
             prior_prec, prior_center, stage_i_pipe, stage_i_ids) in posterior.root._groups:
            n_g = sum(group_alpha_sizes)
            group_params = {p.name: jnp.asarray(float(p.value)) for p in group_theory_pipe.params}
            alpha_vec = jnp.concatenate([jnp.ravel(jnp.asarray(group_params[name])) for name in group_alpha_names])

            def group_fn(alpha_vec, _pipe=group_theory_pipe, _params=group_params,
                         _names=group_alpha_names, _sizes=group_alpha_sizes, _shapes=group_alpha_shapes):
                p = dict(_params)
                offset = 0
                for name, size, shape in zip(_names, _sizes, _shapes):
                    p[name] = alpha_vec[offset:offset + size].reshape(shape if shape else ())
                    offset += size
                return _pipe(p)

            # JIT: primal only (jax.linearize but only use theories_concat, not jvp_fn)
            @jax.jit
            def jit_primal(alpha_vec):
                theories, _ = jax.linearize(group_fn, alpha_vec)
                return theories

            # JIT: primal + JVP for 1, 2, 4, 6 tangent directions (linearize approach)
            def make_jit_jvp(n_tangents, _n_g=n_g, _group_fn=group_fn):
                @jax.jit
                def fn(alpha_vec):
                    theories, jvp_fn = jax.linearize(_group_fn, alpha_vec)
                    B_rows = jax.vmap(jvp_fn)(jnp.eye(_n_g)[:n_tangents])
                    return theories, B_rows
                return fn

            jit_jvp_fns = {n: make_jit_jvp(n) for n in (1, 2, 4, n_g)}

            # JIT: value_and_jacfwd approach — vmap over jax.jvp directly,
            # with out_axes=(None, 1) hinting that the primal is batch-invariant.
            def make_jit_jacfwd(n_tangents, _n_g=n_g, _group_fn=group_fn):
                @jax.jit
                def fn(alpha_vec):
                    pushfwd = functools.partial(jax.jvp, _group_fn, (alpha_vec,))
                    basis = jnp.eye(_n_g, dtype=alpha_vec.dtype)[:n_tangents]
                    theories, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
                    return theories, jac.T
                return fn

            jit_jacfwd_fns = {n: make_jit_jacfwd(n) for n in (1, 2, 4, n_g)}

            # JIT: direct group_fn call (plain primal, no linearize overhead)
            @jax.jit
            def jit_group_fn(alpha_vec):
                return group_fn(alpha_vec)

            # Warmup all
            for bench_idx in range(3):
                av = alpha_vec + 1e-4 * bench_idx
                jax.block_until_ready(jit_primal(av))
                jax.block_until_ready(jit_group_fn(av))
                for fn in jit_jvp_fns.values():
                    jax.block_until_ready(fn(av))
                for fn in jit_jacfwd_fns.values():
                    jax.block_until_ready(fn(av))

            number = 5

            def _time(fn):
                start = time.perf_counter()
                for bench_idx in range(number):
                    jax.block_until_ready(fn(alpha_vec + 1e-4 * bench_idx))
                return (time.perf_counter() - start) / number * 1e3

            dt_group = _time(jit_group_fn)
            dt_primal = _time(jit_primal)

            print(f'  group: {group_alpha_names}')
            print(f'  jit(group_fn direct):             {dt_group:9.4f} ms  (plain forward, no linearize)')
            print(f'  jit(linearize primal only):       {dt_primal:9.4f} ms')
            print(f'  --- linearize + vmap(jvp_fn) ---')
            for n_tangents, fn in sorted(jit_jvp_fns.items()):
                dt = _time(fn)
                print(f'  jit(linearize + vmap n={n_tangents}):    {dt:9.4f} ms   (+{dt - dt_primal:6.2f} ms JVP)')
            print(f'  --- vmap(jax.jvp) jacfwd ---')
            for n_tangents, fn in sorted(jit_jacfwd_fns.items()):
                dt = _time(fn)
                print(f'  jit(jacfwd n={n_tangents}):             {dt:9.4f} ms   (+{dt - dt_primal:6.2f} ms JVP)')

            # Two-stage: run Stage i (cosmo+PT) once outside linearize; JVP only through Stage ii (Tracer+obs).
            if stage_i_pipe is not None and stage_i_ids:
                stage_i_params = {p.name: group_params[p.name] for p in stage_i_pipe.params}

                def make_jit_two_stage(n_tangents, _n_g=n_g, _pipe=group_theory_pipe, _params=group_params,
                                       _names=group_alpha_names, _s1_ids=stage_i_ids,
                                       _s1_pipe=stage_i_pipe, _s1_params=stage_i_params):
                    @jax.jit
                    def fn(alpha_vec):
                        s1_flat = _s1_pipe(_s1_params)

                        def thin_fn(av, _s1_flat=s1_flat):
                            p = {**_params, **{name: av[alpha_idx] for alpha_idx, name in enumerate(_names)}}
                            return_val, _, _ = _pipe._run_graph_fn(p, stage_i_ids=_s1_ids, stage_i_flat=_s1_flat)
                            return return_val

                        theories, jvp_fn = jax.linearize(thin_fn, alpha_vec)
                        B_rows = jax.vmap(jvp_fn)(jnp.eye(_n_g)[:n_tangents])
                        return theories, B_rows
                    return fn

                jit_two_stage_fns = {n: make_jit_two_stage(n) for n in (1, 2, 4, n_g)}

                for bench_idx in range(3):
                    av = alpha_vec + 1e-4 * bench_idx
                    for fn in jit_two_stage_fns.values():
                        jax.block_until_ready(fn(av))

                print(f'  --- two-stage: stage_i (cosmo+PT) once, linearize(stage_ii only) ---')
                for n_tangents, fn in sorted(jit_two_stage_fns.items()):
                    dt = _time(fn)
                    print(f'  jit(two-stage n={n_tangents}):           {dt:9.4f} ms')
            break  # only first group

    if 'comet_3poles' in test:
        print(f'\n{"─" * 60}')
        print(f'COMET 2pt+3pt: ells2={ELLS_COMET}, ells3={ELLS3_COMET}, '
            f'k2=linspace(0.02, 0.3, {len(K_COMET)}) ({len(K_COMET)} pts), '
            f'k3 diagonal ({len(K3_COMET)} pts), '
            f'data size={len(ELLS_COMET) * len(K_COMET) + len(ELLS3_COMET) * len(K3_COMET)}')
        print(f'{"─" * 60}')
        # 19 free params (incl. cnloB/NB0/MB0) and an expensive bispectrum evaluation per
        # call means Minuit's default max_iterations=1e5 can take a very long time without
        # converging; cap it here purely for benchmark wall-clock purposes (this is timing
        # the cost-per-iteration machinery, not chasing a converged fit).
        profile_kwargs = dict(max_iterations=500)
        run('shared PT (PX_ell/BX_ell_Sugi)', lambda: build_posterior_comet(direct=False, include_3poles=True),
            vary_param='b1', warmup=2, number=5, run=('jit', 'grad', 'profile'), profile_kwargs=profile_kwargs)
        run('direct (Pell/Bell_Sugi, pt=False)', lambda: build_posterior_comet(direct=True, include_3poles=True),
            vary_param='b1', warmup=2, number=5, run=('jit', 'grad', 'profile'), profile_kwargs=profile_kwargs)


if __name__ == '__main__':

    main(test=('folps_noemu',))