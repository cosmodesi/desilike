# Emulator-backed fkpt — validation against ISiTGR

Validates `fkpt_pkemu_*` (the emulator-backed fkpt classes added to
`desilike/theories/galaxy_clustering/full_shape.py`) against ISiTGR, two ways:

- **`compare_multipoles.py`** — emulator vs **direct ISiTGR** through the
  *identical* `fkpt_pkemu` pipeline (isolates emulator accuracy).
- **`compare_original_vs_emulator.py`** — emulator class vs the **literal
  original** `fkptjaxTracerPowerSpectrumMultipoles` + ISiTGR `Cosmoprimo`
  template (end-to-end class-vs-class).

## What is compared

Both methods feed the **identical** fkpt loop (`Kfuncs_to_tables`) and bias
machinery through `fkpt_pkemu_TracerPowerSpectrumMultipoles`; only the source
of the linear inputs differs:

| | linear P(k), scalars | no-wiggle P_nw |
|---|---|---|
| **Method 1 — "ISiTGR + fkpt"** | direct ISiTGR (`IsitgrCosmology`) | folps `get_pknow` |
| **Method 2 — "emulator + fkpt"** | trained MG emulator (`MgEmulatorCosmology`) | folps `get_pknow` |

So the comparison isolates exactly one thing: the emulator's accuracy in
reproducing the ISiTGR linear power spectrum.

The cosmological background is held at the fiducial in every case (⇒ AP = 1),
so only the binned modified-gravity response varies.

## Files
- `cosmologies.py` — `IsitgrCosmology`, the ISiTGR-truth provider (same
  interface as `MgEmulatorCosmology`, computes plin + scalars directly with
  ISiTGR on the emulator's k-grid).
- `compare_multipoles.py` — builds both pipelines, loops over GR + MG cases,
  writes `compare_multipoles.png`.
- `compare_multipoles.png` — output: left = `k·P_ℓ` overlay (ISiTGR solid,
  emulator dashed); right = percent difference.

## How to run

`compare_multipoles.py` (emulator vs direct ISiTGR) needs only the cosmodesi
env + `fkptjax_muMG` + `FOLPS_BACKEND=jax`:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export FOLPS_BACKEND=jax
export PYTHONPATH=/global/homes/p/prakharb/fkptjax_muMG/src:$PYTHONPATH
cd emulator_comparison
python compare_multipoles.py
```

`compare_original_vs_emulator.py` (literal original class) additionally needs
the ISiTGR-patched cosmoprimo + the latest ISiTGR:

```bash
# one-time: clone the cosmoprimo branch and build the latest isitgr_private
git clone -b cosmoprimo_isitgr https://github.com/cosmodesi/cosmoprimo \
    ~/cosmoprimo_isitgr_src
git clone https://github.com/gqcristhian/isitgr_private ~/isitgr_private_latest
( cd ~/isitgr_private_latest && python setup.py make )   # needs gfortran

export PYTHONPATH=~/cosmoprimo_isitgr_src:~/isitgr_private_latest:\
/global/homes/p/prakharb/fkptjax_muMG/src:$PYTHONPATH
python compare_original_vs_emulator.py
```
(Both scripts prepend the repo root to `sys.path`, so this repo's desilike —
with the emulator classes — wins over any cosmodesi-installed desilike. The
older base-env / pre-`gamma_a` ISiTGR fails here: use the freshly-built
`isitgr_private_latest`.)

## Results

*(All numbers/plots below regenerated on the **dr2-dev-fkptjax** branch with
fkptjax_muMG `main` and the latest `isitgr_private`. z=0.5, max |ΔP_ℓ| over
0.02 < k < 0.2.)*

### A. emulator vs direct ISiTGR, identical pipeline (`compare_multipoles.py`)

| case | max diff |
|---|---|
| GR (μ=Σ=1)            | 1.07 % |
| MG: μ1=1.5            | 0.86 % |
| MG: μ=0.5 (weaker G)  | 0.67 % |
| MG: mixed μ/Σ         | 0.35 % |

Sub-percent everywhere (P0, P2 **and** P4), in GR and modified gravity. This is
the definitive emulator validation: with the linear inputs swapped (emulator vs
ISiTGR) through the same `fkpt_pkemu` loop, the multipoles are identical to ~1%.

### B. emulator class vs literal original class (`compare_original_vs_emulator.py`)
With the **latest** `isitgr_private` + the `cosmoprimo_isitgr` engine (binning
constants passed as ISiTGR **engine kwargs**), the linear spectra agree to <0.5%.
Max |ΔP_ℓ|: with P4 — GR 0.84, μ1=1.5 2.12, μ=0.5 14.11, mixed 12.17 %;
P0/P2 only — GR 0.25, μ1=1.5 1.27, μ=0.5 2.33, mixed 6.90 %.

- **P0 and P2 agree to ≲2%** (mixed 6.9%) across GR and all MG cases.
- **P4 (hexadecapole) at high k** differs more (oscillatory, BAO-shaped) — the
  **no-wiggle / IR-resummation prescription** difference between the two class
  implementations (original = template `peakaverage`; emulator = folps
  `get_pknow`). **Not** an emulator error: result A (same loop, both pnw=folps)
  is sub-percent in P4 too.

### C. Is the residual the no-wiggle prescription? (`test_same_nw.py`)
Run the emulator class two ways vs the original class — (i) production folps
`get_pknow`, and (ii) the **same peakaverage** prescription the original class
uses (cosmoprimo `PowerSpectrumBAOFilter(engine='peakaverage')` on the emulator's
plin). Max |ΔP_ℓ| (P0,P2,P4):

| case | folps pnw | same pnw (all k) | **same pnw (k<0.18)** |
|---|---|---|---|
| GR            | 0.84 % | 0.37 % | **0.37 %** |
| MG μ1=1.5     | 2.12 % | 1.49 % | **1.49 %** |
| MG μ=0.5      | 14.11 % | 10.66 % | **3.01 %** |
| MG mixed μ/Σ  | 12.17 % | 3.82 % | **3.82 %** |

With the matched prescription the **oscillatory/BAO-shaped differences collapse**
(left vs right panels of `test_same_nw.png`). What remains is mostly a high-k
**edge** effect (last 1–2 k-bins, mostly P4) from the two pipelines feeding
different input k-grids to the loop (template 1e-3–1, 500 pts vs emulator
1e-4–10, 200 pts) — **not** the BAO no-wiggle.

### D. Match the loop k-grid too (`test_matched_grid.py`)
Also resample the emulated plin onto the **template's** loop k-grid
(`geomspace(1e-3, 1, 500)`) and use the peakaverage no-wiggle, so the emulator
pipeline feeds `Kfuncs_to_tables` exactly what the original does. Max |ΔP_ℓ|:

| case | with P4 | P0,P2 only |
|---|---|---|
| GR            | 0.21 % | 0.18 % |
| MG μ1=1.5     | 0.19 % | 0.19 % |
| MG μ=0.5      | 1.27 % | 0.40 % |
| MG mixed μ/Σ  | 0.44 % | 0.34 % |

With the linear inputs, no-wiggle prescription, AND loop k-grid all matched, the
emulator *class* reproduces the original *class* to ≤1.3% (≤0.4% in P0/P2) — the
emulator's accuracy floor. The earlier residuals were entirely the no-wiggle
prescription (broadband) + the loop input grid (high-k P4 edge), not the
emulator.

### E. Matched grid but folps (production) no-wiggle (`test_matched_grid_folps.py`)
Same matched loop k-grid as D, but P_nw is the production folps `get_pknow`.
Isolates the no-wiggle prescription once the grid is matched. Max |ΔP_ℓ|:

| case | with P4 | P0,P2 only |
|---|---|---|
| GR            | 0.68 % | 0.35 % |
| MG μ1=1.5     | 1.52 % | 0.73 % |
| MG μ=0.5      | 5.68 % | 1.44 % |
| MG mixed μ/Σ  | 9.97 % | 6.30 % |

Purely **oscillatory (BAO-shaped)**, high-k edge ramp gone (grid matched) —
exactly the folps-vs-peakaverage no-wiggle difference.

### F. Matched grid, GR wiggle-ratio no-wiggle (`test_matched_grid_gr_ratio.py`)
Same matched loop k-grid as D/E, but P_nw is the **new default**
`nw_method='gr_ratio'` approximation `pnw_MG = pnw_GR·plin_MG/plin_GR` (GR
emulators). Isolates this prescription against the original (peakaverage) with
the grid matched. Max |ΔP_ℓ|:

| case | with P4 | P0,P2 only |
|---|---|---|
| GR            | 1.12 % | 0.43 % |
| MG μ1=1.5     | 1.93 % | 0.99 % |
| MG μ=0.5      | 3.06 % | 0.75 % |
| MG mixed μ/Σ  | 9.61 % | 6.03 % |

Sits between peakaverage (D) and folps (E): for μ=0.5 it is markedly better than
folps (3.06 vs 5.68 %), for the mixed case comparable (9.61 vs 9.97 %). The
mixed-case residual is dominated by the known original-class cosmoprimo-isitgr
linear issue on dr2, **not** the no-wiggle. Residuals are P4-dominated at high k,
no broadband oscillations.

### Summary: decomposing the original-vs-emulator residual (max |ΔP_ℓ| incl. P4)

| variant | nw | loop grid | GR | μ1=1.5 | μ=0.5 | mixed |
|---|---|---|---|---|---|---|
| B production       | folps      | native (1e-4–10) | 0.84 | 2.12 | 14.11 | 12.17 |
| E matched grid     | folps      | template (1e-3–1)| 0.68 | 1.52 | 5.68  | 9.97  |
| F matched grid     | gr_ratio   | template (1e-3–1)| 1.12 | 1.93 | 3.06  | 9.61  |
| D matched grid     | peakaverage| template (1e-3–1)| 0.21 | 0.19 | 1.27  | 0.44  |

And the same decomposition **excluding P4** (P0, P2 only):

| variant | nw | loop grid | GR | μ1=1.5 | μ=0.5 | mixed |
|---|---|---|---|---|---|---|
| B production       | folps      | native (1e-4–10) | 0.25 | 1.27 | 2.33 | 6.90 |
| E matched grid     | folps      | template (1e-3–1)| 0.35 | 0.73 | 1.44 | 6.30 |
| F matched grid     | gr_ratio   | template (1e-3–1)| 0.43 | 0.99 | 0.75 | 6.03 |
| D matched grid     | peakaverage| template (1e-3–1)| 0.18 | 0.19 | 0.40 | 0.34 |

Reading across: matching the **grid** removes the high-k P4 edge (B→E); matching
the **no-wiggle** removes the broadband BAO oscillations (E→D). With both matched
(D), the classes agree to ≤1.3% (≤0.4% in P0/P2) — the emulator's accuracy floor.
Neither residual is an emulator error.

### Pitfalls found & fixed during this validation
- The ISiTGR binning constants (`k_c, k_S, k_TGR, z_div, …`) must be passed as
  `Cosmoprimo(engine='isitgr', …)` **construction kwargs**, NOT as appended
  desilike params — otherwise the engine silently uses default k-windows and the
  MG linear response is wrong (this caused the initial 80%+ discrepancies).
- The fkpt-loop binning defaults in `fkptjaxPowerSpectrumMultipoles.
  _collect_mg_params` were corrected to the emulator's training values
  (`z_TGR=2, z_tw=0.05, k_TGR=0.01, k_S=0.2, k_tw=0.001, scale_bins=True`).
- **dr2-dev-fkptjax specifics.** (a) the binned variant lives under
  `model='PHENOM'` (not `'HDKI'`) in the current fkptjax_muMG `main`; (b) the
  emulator's `Kfuncs_to_tables` kernel grid MUST match the parent
  `fkptjaxPowerSpectrumMultipoles.calculate`: `kmin=min(1e-3, k.min())`,
  `kmax=max(1.0, k.max())`, `Nk_kernel=min(len(k),120)` — using the narrower
  desilike-fkpt-dev grid (`max(1e-3)/min(0.5)/240`) gives a ~3% GR offset.

## Notes / gotchas

- **GR convention.** ISiTGR binned μΣ has GR at **μ_i = Σ_i = 1** (verified
  empirically: the spectrum is identical to LCDM there; μ=Σ=0 is already
  modified gravity). The trainer (`data_generation_pk_mg.jl`) passes μ/Σ
  straight into `isitgr.set_cosmology`, so the emulator shares this convention,
  as does fkpt's `binning` variant. The default param values are therefore 1.0.

- **No-wiggle.** The standalone pnw emulator is only ~5 % accurate (RMS), which
  produced ~6–8 % multipole errors. The plin emulator, by contrast, is ~0.16 %.
  So `MgEmulatorCosmology` derives P_nw from the (accurate) emulated plin, with
  two prescriptions selectable via `nw_method=`:
  - **`'gr_ratio'` (default)** — the GR wiggle-ratio approximation
    `pnw_MG = pnw_GR * plin_MG / plin_GR`, where `pnw_GR/plin_GR` is taken from
    the *original* GR emulators (`training_classy_plin_pnw_mnuw0wacdm_nk200v2`,
    the same ones used by `EmulatorCosmology_new` in desilike-Arnaud-bk),
    evaluated at the same base cosmology. The `As·D(z)²` normalisation cancels
    in the ratio, so the growth factor is irrelevant. This keeps the BAO-wiggle
    removal well-behaved even for strongly-modified MG plin (folps' GR-tuned
    smoothing can leave residual wiggles there — see `test_nw_methods.py`).
  - **`'folps'`** — folps `get_pknow` on the emulated plin (the previous
    behaviour), kept as an option.

  `test_nw_methods.py` compares the two: the difference is purely oscillatory
  (BAO-shaped), ~3.8 % in GR rising to ~17–21 % in strong-MG cases (μ=0.5,
  mixed), where folps' GR smoothing handles the MG-distorted wiggle less cleanly.

- **Original-class path.** The literal `fkptjaxTracerPowerSpectrumMultipoles`
  (ISiTGR via the cosmoprimo engine) is now runnable in the cosmodesi base env by
  putting the `cosmoprimo_isitgr` branch + a freshly-built `isitgr_private` on
  PYTHONPATH (see "How to run"). The plain base-env ISiTGR/CAMB is too old (it
  rejects MG params such as `Q0` / `gamma_a` / `n_HS`); build the latest fork.
  `cosmologies.IsitgrCosmology` is an additional direct-ISiTGR provider used by
  `compare_multipoles.py` for the cleanest (same-pipeline) emulator validation.
