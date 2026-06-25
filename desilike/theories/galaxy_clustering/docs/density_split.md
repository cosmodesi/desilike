# Density-Split Pqg Modeling Pipeline

`DensitySplitTracerPowerSpectrumMultipoles` predicts density-split
quantile-galaxy cross-power spectrum multipoles, \(P_{qg}\). The returned
`power` array is ordered as `(quantile, ell, k)` with shape
`(n_quantiles, n_ells, n_k)`.

The production pipeline is intentionally narrow:

- two theory branches: `tree` and `1-loop`;
- one theory file for the model and JAX composite loop backend;
- one observable layer for data loading, covariance, flattening, and plotting;
- fitting scripts that apply priors, analytic marginalization, emulator reuse,
  and optional on-the-fly mock measurements.

Legacy operator and diagnostic branches are not part of the supported fitting
surface.

## Theory Models

| Model | Meaning | Density-split parameters in the theory calculator |
| --- | --- | --- |
| `tree` | Strict Kaiser composite tree model, \(P_{qg,a}=c_{1,a} W_R(k) (b_1 + f\mu^2)^2 P_L(k) + P^{\mathrm{stoch}}_{qg,a}\). | `c1q*`, `s0qg*` |
| `1-loop` | Composite one-loop model, \(c_{1,a} W_R(k) P_{gg,\mathrm{det}}^{1\mathrm{-loop}} + c_{2,a}P_{2,g} - 2k^2 W_R(k)E_a(\mu)P_{gg,\mathrm{lin}} + P^{\mathrm{stoch}}_{qg,a}\), with \(E_a=e_{0,a}+e_{2,a}\mu^2+e_{4,a}\mu^4\). | `c1q*`, `c2q*`, `e0q*`, `e2q*`, `e4q*`, `s0qg*`, `s2qg*`, `s2muqg*`, plus fixed `c3q*` |

`--qg-anisotropic-stochastic` optionally adds `s0muqg*`, an anisotropic
white stochastic response proportional to `shotnoise * mu^2`. This extension is
off by default because it is empirical rather than part of the baseline
Eq. 57/77 stochastic basis.

Both branches define all five equal-probability quantile fields internally.
Quantile 3 coefficients are not independent: they are derived from the
five-quantile partition rule using the independent coefficients for quantiles
1, 2, 4, and 5.

The `1-loop` branch uses the deterministic FOLPS galaxy spectrum for the `c1`
propagation term and an explicit JAX implementation of the composite `c2 P2,g`
loop. This branch requires `backend='jax'`.

For a line-by-line map from the theory implementation to the analytic model
equations, see [density_split_equation_map.md](density_split_equation_map.md).

## Fitting Behavior

The fitting scripts, [run_density_split_minuit.py](../../../../scripts/run_density_split_minuit.py)
and [run_density_split_emcee.py](../../../../scripts/run_density_split_emcee.py),
default to `--theory-model 1-loop`.

By default, the scripts analytically marginalize the density-split linear block
with `derived='.auto'`:

| Model | Analytically marginalized by default |
| --- | --- |
| `tree` | `c1q*`, `s0qg*` for `q = 1, 2, 4, 5` |
| `1-loop` | `c1q*`, `c2q*`, `e0q*`, `e2q*`, `e4q*`, `s0qg*`, `s2qg*`, `s2muqg*` for `q = 1, 2, 4, 5` |

When `--qg-anisotropic-stochastic` is enabled, `s0muqg*` is added to the same
analytic marginalization block unless `--sample-ds-linear` is also passed.

Use `--sample-ds-linear` to sample these amplitudes explicitly instead.
Analytic marginalization is valid with fixed or free cosmology: for each
nonlinear point proposed by Minuit or emcee, desilike recomputes the model
basis and solves/marginalizes the linear block at that point.

Galaxy and cosmology parameters remain sampled. In particular, density-split
fits do not analytically marginalize galaxy `alpha*` counterterms, because the
current model contains products like `c1q * Pgg_det(alpha*)`; solving both
blocks together would not be exact.

## Parameters And Priors

The default prior basis is `physical_aap`.

| Branch | Sampled galaxy sector in `physical_aap` |
| --- | --- |
| `tree` | `b1p` |
| `1-loop` | `b1p`, `b2p`, `bsp`, `alpha0p`, `alpha2p`, `alpha4p`, `X_FoG_pp` |

`b3p` and `ctp` are fixed by default. `c3q*` is present in the one-loop theory
configuration but fixed. Query-galaxy stochastic terms are exposed as
shotnoise-normalized amplitudes `s0qg*`, `s2qg*`, and `s2muqg*`. The optional
anisotropic white extension exposes `s0muqg*`. Density-split auto-spectrum
stochastic terms and old operator amplitudes are not exposed.

Gaussian quantile coefficients initialize the `c1/c2/c3` priors. Derivative
and stochastic terms are centered at zero. These priors regularize the linear
solve or sampled amplitudes; they are not hard constraints.

## Data And Observable Layer

`DensitySplitPowerSpectrumMultipolesObservable` handles:

- raw density-split HDF5 loading;
- quantile and multipole selection;
- `rebin`, `kmin`, and `kmax` cuts;
- covariance loading from raw mocks;
- flattening in quantile-major, ell-major order;
- plotting data and theory.

The observable layer does not contain theory code. It only passes the selected
`k`, quantiles, and ells to the theory calculator.

## Common Commands

Free-cosmology `1-loop` Minuit fit with default analytic marginalization:

```bash
python scripts/run_density_split_minuit.py \
  --theory-model 1-loop \
  --quantiles 1 2 4 5 \
  --ells 0 2 \
  --template direct \
  --template-params h omega_cdm logA omega_b \
  --emulator ./emulator_pqg.npy \
  --kmax 0.2 \
  --composite-loop-resolution default \
  --plot-output pqg_fit.png
```

Sample the density-split amplitudes explicitly:

```bash
python scripts/run_density_split_minuit.py \
  --theory-model 1-loop \
  --sample-ds-linear \
  --quantiles 1 2 4 5 \
  --template direct \
  --template-params h omega_cdm logA omega_b \
  --emulator ./emulator_pqg_sampled.npy \
  --kmax 0.2
```

Tree-level diagnostic fit:

```bash
python scripts/run_density_split_minuit.py \
  --theory-model tree \
  --quantiles 1 2 4 5 \
  --ells 0 2 \
  --template direct \
  --template-params h omega_cdm logA omega_b \
  --kmax 0.2
```

On-the-fly measurement fit:

```bash
python scripts/run_density_split_minuit.py \
  --theory-model 1-loop \
  --quantiles 1 2 4 5 \
  --template direct \
  --template-params h omega_cdm logA omega_b \
  --emulator ./emulator_pqg.npy \
  --kmax 0.2 \
  --on-the-fly-measurements \
  --measurement-overwrite \
  --plot-output test.png
```

On-the-fly measurement from an AbacusSummit truth cosmology:

```bash
python scripts/run_density_split_minuit.py \
  --theory-model 1-loop \
  --quantiles 1 2 4 5 \
  --ells 0 2 \
  --template direct \
  --template-params h omega_cdm logA omega_b \
  --emulator ./emulator_pqg_abacus_c001.npy \
  --kmax 0.2 \
  --on-the-fly-measurements \
  --fiducial-cosmology abacus-c001 \
  --measurement-save-dir scripts/mockfactory_acm_measurements_abacus_c001 \
  --plot-output pqg_abacus_c001.png
```

For emcee, use the same modeling flags with `scripts/run_density_split_emcee.py`.

## Emulators And Loop Resolution

`--emulator` refers to a Taylor emulator for the FOLPS PT calculator
(`theory.pt`), not for the full density-split likelihood. The Minuit script
validates the emulator class and varied PT parameters, then validates the final
observable theory-vector length so stale k-grid or ell-basis emulators fail
fast.

The `1-loop` composite c2 loop supports three quadrature presets:

| Preset | Purpose |
| --- | --- |
| `smoke` | Fast tests and script checks |
| `default` | Production default |
| `high` | Higher-resolution diagnostics |

Changing the loop resolution changes JAX static arguments and can trigger
recompilation.

## On-The-Fly Measurements

`--on-the-fly-measurements` generates and measures a mockfactory/ACM mock before
fitting. The density-split script exposes controls for bias, redshift, box
size, number density, RSD, line of sight, mesh sizes, smoothing radius, and
truth cosmology.

The mock truth cosmology is set with `--fiducial-cosmology`. The default is
`desi`; AbacusSummit cosmologies are available as `abacus-cNNN`, for example
`abacus-c001`, which maps to `cosmoprimo.fiducial.AbacusSummit(1)`. This option
controls only the synthetic measurement generation. It is separate from
`--fiducial`, which is the fiducial cosmology passed to the power-spectrum
template, and from `--template-params`, which controls sampled cosmological
parameters.

The script checks that `kmax` is below half the limiting Nyquist frequency for
the relevant mock and measurement meshes. Existing measurement products are
reused unless `--measurement-overwrite` is passed. DESI measurements keep the
original cache filenames. Non-DESI measurements receive a cosmology suffix such
as `_abacus-c001` to avoid reusing a measurement generated from a different
truth cosmology.

## Removed Interfaces

The supported fitting interface no longer includes:

- `folps_ops_qct` or `folps_composite_qg_1loop` model aliases;
- `opXXqQ` operator amplitudes;
- `bq*`, `beta*`, `bqnabla*`, old `c0/c2/c4` density-split counterterms;
- `--ds-linear-basis`, `--composite-pieces`, `--fix-composite-stochastic`, or
  `--operator-prior-scale`.

Use `--theory-model tree` or `--theory-model 1-loop`.

## Validation Status

Current focused checks cover:

- tree model equality to the strict Kaiser expression;
- no FOLPS one-loop or c2 calls in the tree branch;
- finite one-loop c2 outputs for `kmax = 0.12, 0.15, 0.2`;
- five-quantile partition sum rule and selected-quantile subsetting;
- JAX c2 loop `jit` and `vmap` smoke tests;
- script help and backend defaults;
- default analytic marginalization and `--sample-ds-linear` opt-out;
- likelihood-level finite `.auto` marginalization for both branches.

An explicit Gaussian design-matrix check also confirmed that desilike's
analytic marginalization matches the expected solve for the density-split
linear block at fixed nonlinear parameters, including the q3 partition rule.
