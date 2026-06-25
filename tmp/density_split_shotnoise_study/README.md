# Density-Split Pqg Shot-Noise Study

This study generated matched on-the-fly mockfactory/ACM measurements with the
same cosmology, bias, seed, RSD, smoothing, and mesh settings, varying only the
number density:

- `nbar = 1e-3`: 1,000,069 galaxies
- `nbar = 5e-4`: 500,068 galaxies
- `nbar = 2e-4`: 200,248 galaxies
- `nbar = 1e-4`: 100,047 galaxies

Settings: `boxsize=1000 Mpc/h`, `nmesh=256`, `meshsize=256`,
`cellsize=-1`, `bias=2`, `z=0.5`, `smoothing_radius=10 Mpc/h`,
`ells=(0, 2)`, `seed=42`.

## Artifacts

- Measurements: `measurements/*/density_split_power/dsc_pkqg_poles_mockfactory_seed42.h5`
- Raw arrays: `pqg_shotnoise_measurements.npz`
- Machine-readable summary: `summary.json`
- Multipole comparison plot: `plots/pqg_multipoles_by_nbar.png`
- Stochastic-only residual fit plot: `plots/pqg_delta_and_stochastic_fit.png`
- Amplitude-plus-stochastic residual fit plot: `plots/pqg_delta_and_amplitude_stochastic_fit.png`
- Fit summary plot: `plots/stochastic_fit_summary.png`

## Main Findings

The lower-density catalogs shift the quantile-galaxy multipoles in a coherent,
quantile-dependent way, especially for the extreme quantiles. This is visible
even though the measured power spectrum itself is shot-noise subtracted, so it
is consistent with shot noise entering through the density-split query
selection.

Using `nbar=1e-3` as the reference, the partition-constrained Eq. 77 stochastic
basis alone captures only a small fraction of the measured nbar-dependent
differences:

| nbar | RMS after / before | variance explained |
| --- | ---: | ---: |
| `1e-4` | 0.962 | 0.074 |
| `2e-4` | 0.979 | 0.042 |
| `5e-4` | 0.995 | 0.010 |

An empirical amplitude-plus-stochastic proxy captures more of the smooth
response:

| nbar | RMS after / before | variance explained |
| --- | ---: | ---: |
| `1e-4` | 0.597 | 0.609 |
| `2e-4` | 0.788 | 0.335 |
| `5e-4` | 0.572 | 0.649 |

Interpretation: the new `s0qg/s2qg/s2muqg` terms are useful additive smooth
freedom, but in this single-seed experiment they do not by themselves capture
most of the density-dependent query-selection response. Much of the smooth
effect looks more like a change in quantile response/amplitude, with additional
jagged realization noise left over. A more robust production conclusion should
average this study over several matched seeds and use the sample covariance in
the final fits.
