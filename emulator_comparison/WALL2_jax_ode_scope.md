# Wall 2 scope — JAX ODE solver for `Kfuncs_to_tables` (full jit/vmap likelihood)

## Goal & context
Make the `fkpt_pkemu` likelihood fully **`jax.jit`- and `jax.vmap`-able**, so the
MCMC can compile it once and evaluate all walkers/chains in one device call.
Profiling showed the eager pipeline tops out at ~2× (numba ODE + numpy simpson,
done). The big regime (10×+, esp. ensemble samplers) needs the whole pipeline
JAX-traceable.

**Wall 1 (the numpy emulator) is cleared** — `MgEmulatorCosmology(use_jax_emulator=True)`
is jit/vmap-able and bit-for-bit identical (done). **Wall 2 is the ODE machinery
inside `Kfuncs_to_tables`**: the RKQS Python solver + ~39 `float()`/`np.asarray`
concretizations + the numba RHS (opaque to JAX).

Tooling available: **diffrax 0.7.2**, equinox, optimistix, jax 0.9.1 (x64).

## Components to convert (opt-in `backend='jax'`, alongside the existing numba path)

### 1. ODE RHS — binning, in JAX  (~0.5–1 day, easy)
Port `binning_numba.py` (already pure arithmetic) to `binning_jax.py` with `jnp`
(same `_mu`, `_f1`, `_kpp`, `_S2*`, `_S3*`, `nb_firstOrder/secondOrder/thirdOrder`).
The math is identical → validate bit-for-bit vs the numpy `ModelDerivatives`.

### 2. ODE solver — diffrax  (~3–5 days, the bulk / main risk)
Replace `ODESolver`/`odeint.py` (Python RKQS `while` loop, 9 data-dependent
branches) with `diffrax.diffeqsolve` (e.g. `Tsit5`/`Dopri5` + `PIDController`):
- **`DP`** (growth): batched 1st-order ODE over `k_ext` → one `diffeqsolve` with a
  vector state (diffrax handles batched `y`).
- **`kernel_constants`**: the scalar 10-dim 3rd-order ODE.
- Risk: diffrax's adaptive solver ≠ RKQS exactly → results differ at the ODE
  tolerance (~1e-4). The ODE is only ~1e-4 accurate anyway; tune `rtol/atol` so
  the **final multipoles** match the numpy path to ≲1e-4 (validate).

### 3. `Kfuncs_to_tables` de-concretize  (~2 days)
Remove the ~39 `float()`/`np.asarray` casts; keep everything `jnp`:
- `f0`, the σ² integrals, `kernel_constants` outputs (A, Ap, CFD3, CFD3p) → `jnp`.
- **Tension with the eager optimization**: we moved `simpson` to numpy (host) to
  kill device round-trips in the *eager* path. Under jit that's the wrong choice —
  jit fuses the `jnp` simpson into one kernel (no device_put). Resolution: branch on
  `backend` — numpy simpson for eager, `jnp` simpson under jit. (Or just always
  `jnp` when `backend='jax'`.)
- `extrapolate_pklin`, `interp` (interpax), `JaxCalculator.evaluate` are already JAX.

### 4. Provider — DONE (`use_jax_emulator`). ✓

### 5. Wiring  (~1 day)
A `backend` / `use_jax_ode` flag through `Kfuncs_to_tables` and
`fkpt_pkemu_PowerSpectrumMultipoles` (default `'numba'`/eager; opt-in `'jax'`).
`MgEmulatorCosmology(use_jax_emulator=True)` + `backend='jax'` ⇒ the whole
`fkpt_pkemu` theory traces ⇒ MCMCSampler's `vmap`+`jit` succeed.

## Validation plan
- RHS: `binning_jax` vs numpy `ModelDerivatives`, bit-for-bit (atol≈1e-11).
- ODE: D(k), A/Ap/CFD3 (diffrax vs RKQS) within tuned tol.
- End-to-end: multipoles jax-ODE vs numpy-ODE ≲1e-4 (GR + MG cases via
  `emulator_comparison`); `compare_multipoles.py` still sub-percent vs ISiTGR.
- Tracing: `jax.jit(theory)` and `jax.vmap(theory)` succeed (the MCMCSampler
  "Successfully vmap/jit input likelihood" path).
- Benchmark: jit per-eval (expect ~10–15 ms, fused) and vmap over N walkers
  (one batched call) vs the current per-walker loop.

## Effort & payoff
- ~1.5–2 weeks (item 2 dominates).
- Per-eval: ~44 ms → ~10–15 ms (jit-fused, JAX ODE).
- **Ensemble samplers (emcee/zeus, ~46 walkers): one vmapped device call instead
  of 46 sequential → ~10–40×.** Single-chain MCMC: jit gives a few ×.

## Risks / notes
- diffrax-vs-RKQS numerics (mitigated by ODE tolerance + validation).
- Maintain both backends: numba/numpy (eager, no-trace) and JAX (jit/vmap). The
  eager simpson/numpy optimizations are for the non-jit path only.
- Must run with `jax_enable_x64=True` (folps/fkptjax are float64).
- vmap memory: N_walkers × loop arrays on device — check footprint for large N.
- The numba ODE (Path A) and JAX ODE (Path B) are alternative backends for the
  same physics; Wall-1 (emulator) is shared and already done.
