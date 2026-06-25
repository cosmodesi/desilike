# Density-Split Pqg Shot-Noise Study: 2 Gpc/h Multi-Seed

This run repeats the number-density response study with a larger volume and
seed averaging. All mocks use the same cosmology, galaxy bias, RSD, smoothing,
and mesh settings, varying only number density and random seed.

Settings:

- `boxsize=2000 Mpc/h`
- `nmesh=256`, `meshsize=256`, `cellsize=-1`
- `bias=2`, `z=0.5`, `fiducial_cosmology=desi`
- `smoothing_radius=10 Mpc/h`
- `ells=(0, 2)`, `rebin=13`, `0.01 <= k <= 0.2 h/Mpc`
- `seeds = 42, 43, 44`

Galaxy counts:

| seed | nbar `1e-3` | nbar `5e-4` | nbar `2e-4` | nbar `1e-4` |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 7,999,005 | 4,000,445 | 1,599,908 | 801,053 |
| 43 | 7,995,700 | 3,996,416 | 1,597,259 | 800,259 |
| 44 | 7,998,443 | 4,000,025 | 1,599,164 | 799,890 |

## Artifacts

- Measurements: `measurements/seed_*/nbar_*/density_split_power/dsc_pkqg_poles_mockfactory_seed*.h5`
- Raw and averaged arrays: `pqg_shotnoise_measurements.npz`
- Machine-readable summary: `summary.json`
- Multipole comparison: `plots/pqg_multipoles_by_nbar.png`
- Baseline stochastic residuals: `plots/pqg_delta_and_stochastic_fit.png`
- Optional `mu^2` stochastic residuals: `plots/pqg_delta_and_mu0_stochastic_fit.png`
- Partition amplitude plus stochastic residuals: `plots/pqg_delta_and_partition_amplitude_stochastic_fit.png`
- Partition amplitude plus optional `mu^2` stochastic residuals: `plots/pqg_delta_and_partition_amplitude_mu0_stochastic_fit.png`
- Fit summary: `plots/stochastic_fit_summary.png`

## Main Findings

Using `nbar=1e-3` as the reference, the baseline Eq. 57/77 stochastic block
explains only a modest fraction of the seed-averaged number-density response:

| basis | nbar | RMS after / before | variance explained |
| --- | ---: | ---: | ---: |
| partition stochastic | `1e-4` | 0.893 | 0.202 |
| partition stochastic | `2e-4` | 0.952 | 0.093 |
| partition stochastic | `5e-4` | 0.952 | 0.094 |

Adding an anisotropic white stochastic term,
`shotnoise * s0muqg_a * mu^2`, substantially improves the additive stochastic
description:

| basis | nbar | RMS after / before | variance explained |
| --- | ---: | ---: | ---: |
| partition `mu^2` + stochastic | `1e-4` | 0.674 | 0.546 |
| partition `mu^2` + stochastic | `2e-4` | 0.702 | 0.508 |
| partition `mu^2` + stochastic | `5e-4` | 0.791 | 0.375 |

The largest coherent response is still amplitude-like and is already spanned by
the existing density-split `c1q*` block. A partition-constrained amplitude-like
term plus the baseline stochastic block explains most of the averaged response:

| basis | nbar | RMS after / before | variance explained |
| --- | ---: | ---: | ---: |
| partition amplitude + stochastic | `1e-4` | 0.248 | 0.938 |
| partition amplitude + stochastic | `2e-4` | 0.363 | 0.868 |
| partition amplitude + stochastic | `5e-4` | 0.551 | 0.696 |

Including the optional `mu^2` stochastic term on top of that gives a smaller but
measurable additional improvement:

| basis | nbar | RMS after / before | variance explained |
| --- | ---: | ---: | ---: |
| partition amplitude + `mu^2` + stochastic | `1e-4` | 0.201 | 0.960 |
| partition amplitude + `mu^2` + stochastic | `2e-4` | 0.304 | 0.908 |
| partition amplitude + `mu^2` + stochastic | `5e-4` | 0.531 | 0.718 |

Interpretation: the baseline `s0qg/s2qg/s2muqg` terms are not enough by
themselves. Most of the response is a change in quantile selection/amplitude
that the normal fitted `c1q*` freedom should absorb. The remaining additive
anisotropic component is well motivated by this diagnostic, so desilike now has
an optional `--qg-anisotropic-stochastic` flag that enables `s0muqg*`.
