# Density-Split Equation-To-Code Map

This note maps the current
[`density_split.py`](../density_split.py) implementation to equations in
`Analytic_model_for_density_split_clustering.pdf`. It documents the implemented
quantile-galaxy cross-spectrum, with extra detail on the one-loop pieces.

The current theory calculator returns only `P_{q_a g}` multipoles. It does not
implement density-split auto-spectra.

## Implemented Prediction

`DensitySplitTracerPowerSpectrumMultipoles` exposes two branches
([`density_split.py:14-18`](../density_split.py#L14-L18),
[`density_split.py:236-257`](../density_split.py#L236-L257)):

| Branch | Implemented `P_{q_a g}(k, mu)` | Main equations |
| --- | --- | --- |
| `tree` | `c1_a W_R P_{gg,lin} + shotnoise s0qg_a` | Eqs. 45, 46, 53, 55, 57 |
| `1-loop` | `c1_a W_R P_{gg,FOLPS}^{det} + c2_a P_{2,g}^{impl} - 2 k^2 W_R E_a(mu) P_{gg,lin} + shotnoise S_a(k, mu)` | Eqs. 61, 66, 68, 72, 77 |

The implemented one-loop formula is

```text
P_{q_a g}^{impl}(k, mu)
  = c1_a W_R(k) P_{gg,FOLPS}^{det}(k, mu)
  + c2_a P_{2,g}^{impl}(k, mu)
  - 2 k^2 W_R(k) [e0_a + e2_a mu^2 + e4_a mu^4] P_{gg,lin}(k, mu)
  + P_{q_a g}^{stoch}(k, mu),

P_{q_a g}^{stoch}(k, mu)
  = shotnoise [s0qg_a + k^2 (s2qg_a + s2muqg_a mu^2)].
```

The optional `qg_anisotropic_stochastic=True` extension adds
`shotnoise s0muqg_a mu^2`. This term is empirical, off by default, and is not
part of the baseline Eq. 57/77 mapping.

This is assembled in `calculate`
([`density_split.py:451-481`](../density_split.py#L451-L481)). The three terms
are the `c1` propagation term
([`density_split.py:471-472`](../density_split.py#L471-L472)), the explicit
`c2 P2,g` loop
([`density_split.py:474-478`](../density_split.py#L474-L478)), and the
density-split derivative counterterm
([`density_split.py:475-479`](../density_split.py#L475-L479)), plus the
shotnoise-normalized `P_qg` stochastic term.

## Quantiles And Parameters

The paper's partition identities are Eqs. 6, 33, and 34. The implementation
uses five equal-probability bins and takes quantiles 1, 2, 4, and 5 as
independent:

- Allowed quantiles and independent composite parameters are declared at
  [`density_split.py:14-26`](../density_split.py#L14-L26).
- `_normalize_quantiles` validates the selected bins at
  [`density_split.py:29-38`](../density_split.py#L29-L38).
- Parameter names such as `c1q1`, `c2q5`, and `e4q2` are built by the
  composite helper, while names such as `s0qg1` and `s2muqg5` are built by the
  stochastic helper.
- Quantile 3 is reconstructed by the partition sum rule at
  [`density_split.py:49-52`](../density_split.py#L49-L52). Because all bins have
  equal probability, this implements `q3 = -(q1 + q2 + q4 + q5)` for every
  composite and stochastic coefficient prefix.

The selected parameter block depends on the branch:
[`density_split.py:322-324`](../density_split.py#L322-L324) keeps the FOLPS
galaxy parameters plus the branch's composite and stochastic parameters, while
[`density_split.py:351-365`](../density_split.py#L351-L365) selects only `c1`
and `s0qg` for `tree` and the full `c1/c2/c3/e0/e2/e4/s0qg/s2qg/s2muqg`
block for `1-loop`.

## Gaussian Coefficient Initialization

The Gaussian prior initialization follows the Hermite-coefficient formulas in
Eqs. 35-38 and Appendix A, Eqs. 104-106:

- `_gaussian_quantile_coefficients` constructs equal-probability normal-bin
  edges and `p_a = 1 / N_q` at
  [`density_split.py:55-65`](../density_split.py#L55-L65).
- The normal density `phi` is Eq. 35's `varphi` factor
  ([`density_split.py:67-70`](../density_split.py#L67-L70)).
- The implemented `c1`, `c2`, and `c3` expressions are the explicit Eq. 36,
  Eq. 37, and Eq. 38 numerators divided by the bin probability
  ([`density_split.py:72-82`](../density_split.py#L72-L82)).
- `_configure_composite_parameters` uses these values as prior/reference
  centers for `c1/c2/c3`, while derivative terms are centered at zero
  ([`density_split.py:326-349`](../density_split.py#L326-L349)).

In the current model `c3q*` parameters are configured but fixed
([`density_split.py:328-341`](../density_split.py#L328-L341)). They are not used
in `calculate`.

## Smoothing, AP Remapping, And Multipoles

The paper defines smoothed galaxy kernels as
`Gamma_n = W_R Z^g_n` in Eq. 39. The implementation applies this smoothing in
two places:

- `_smoothing_window` implements the Gaussian and tophat `W_R(k)` options
  ([`density_split.py:85-93`](../density_split.py#L85-L93)).
- `_smoothing_k` chooses whether smoothing is evaluated using observed `k` or
  AP-remapped physical `k`
  ([`density_split.py:96-106`](../density_split.py#L96-L106)).

AP remapping itself is inherited from the FOLPS/PT base machinery and is
available through `self.pt.pt.jac`, `self.pt.pt.kap`, and `self.pt.pt.muap`.
`calculate` reads those AP-remapped arrays at
[`density_split.py:454-456`](../density_split.py#L454-L456), matching the
observed-to-true mapping and Jacobian of Eqs. 85-88. The final observed
multipoles are built by multiplying by the AP Jacobian and projecting over
`mu`:
[`density_split.py:480-481`](../density_split.py#L480-L481), corresponding to
Eq. 8 after Eq. 88.

## Tree Branch

The tree branch is the strict composite large-scale limit of Eqs. 45, 46, 53,
and 55:

- `pgg_lin = (b1 + f mu^2)^2 P_L` is built at
  [`density_split.py:459-460`](../density_split.py#L459-L460), using the galaxy
  linear kernel of Eq. 13 / Eq. 110.
- In the `tree` branch, `pgg_base = pgg_lin` and no composite loop is evaluated
  ([`density_split.py:462-464`](../density_split.py#L462-L464)).
- Each quantile multiplies this by `c1_a W_R`
  ([`density_split.py:470-472`](../density_split.py#L470-L472)).

This implements the deterministic cross-spectrum in Eq. 55 and the white
query-galaxy stochastic completion of Eq. 57. The tree branch does not include
the effective relaxation of Eqs. 51, 52, 59, and 60.

## One-Loop Construction

The one-loop branch is a reduced implementation of the cross-spectrum structure
in Eq. 66. It keeps `c1 P1,g`, an explicit `c2 P2,g`, and the derivative
counterterm from Eq. 72.

### `c1` Propagation Through FOLPS

The paper's one-loop decomposition is Eqs. 20-26 for the galaxy sector and
Eqs. 61, 66, and 67 for the density-split cross-spectrum.

- `_folps_pars` converts the sampled bias basis into FOLPS parameters at
  [`density_split.py:367-401`](../density_split.py#L367-L401).
- `_folps_pkmu` calls FOLPS for the deterministic redshift-space galaxy
  one-loop spectrum at
  [`density_split.py:403-424`](../density_split.py#L403-L424). In `calculate`,
  this is requested with `shotnoise=0.`
  ([`density_split.py:465-466`](../density_split.py#L465-L466)), so it supplies
  the deterministic part used by `c1_a W_R P_{gg,FOLPS}^{det}`.
- `_linear_matter_pk` obtains `P_L(k)` from the FOLPS table at
  [`density_split.py:426-436`](../density_split.py#L426-L436), which is also
  used for `P_{gg,lin}` in the derivative counterterm.

FOLPS is therefore responsible for the galaxy one-loop, galaxy counterterm, IR
resummation, and damping details entering the `c1` propagation term. The
density-split file does not reimplement the full Eq. 112 third-order galaxy
kernel.

### Explicit `c2 P2,g` Loop

The explicit JAX loop implements the `c2_a P2,g` contribution named in Eq. 68.
It corresponds to the part of Eq. 42 proportional to
`(c2_a / 2) Gamma_1 Gamma_1`, crossed with the second-order galaxy kernel
inside the Eq. 63 one-loop integral.

Quadrature setup:

- `_composite_loop_quadrature` builds the 3D loop integral grid:
  logarithmic radial `q`, cosine angle `x`, and azimuth `phi`
  ([`density_split.py:128-138`](../density_split.py#L128-L138)).
- `composite_p2_moments` receives AP-remapped `k, mu`, the linear table,
  growth rate, smoothing radius, and loop-resolution settings
  ([`density_split.py:141-160`](../density_split.py#L141-L160)).
- The two smoothed linear legs `W_R(q) P_L(q)` and `W_R(|k-q|) P_L(|k-q|)` are
  built at [`density_split.py:161-190`](../density_split.py#L161-L190).
- The loop measure `d^3q / (2 pi)^3` is represented by
  [`density_split.py:173-176`](../density_split.py#L173-L176).

Kernel content:

- The code constructs the triangle geometry for `q` and `p = |k-q|`, including
  the line-of-sight cosines for the two internal legs
  ([`density_split.py:179-186`](../density_split.py#L179-L186)).
- `F2`, `G2`, and `S2` are implemented at
  [`density_split.py:192-196`](../density_split.py#L192-L196), matching
  Eqs. 113-115.
- `z_moments` expands the two linear redshift-space factors
  `(b1 + f mu_1^2)(b1 + f mu_2^2)` into coefficients of `b1^2`, `b1`, and
  `1` at [`density_split.py:198-204`](../density_split.py#L198-L204).
- `rb_kernel` and `rf_kernel` are the redshift-space mapping terms in the
  second-order galaxy kernel of Eq. 111
  ([`density_split.py:205-206`](../density_split.py#L205-L206)).
- The six moment families `one`, `F2`, `G2`, `S2`, `Rb`, and `Rf` are stacked at
  [`density_split.py:207-215`](../density_split.py#L207-L215) and named by
  [`density_split.py:25-26`](../density_split.py#L25-L26).

Contraction:

- `contract_p2_moments` contracts those bias-independent moments into
  `P_{2,g}^{impl}(k, mu)`
  ([`density_split.py:223-233`](../density_split.py#L223-L233)).
- The returned combination mirrors Eq. 111:
  `b1 F2 + b2 / 2 + bs S2 + f mu^2 G2 + Rb + Rf`, with the two `Z1`
  factors already expanded by `zcontract`.
- `_composite_p2_moments` JIT-wraps and calls the explicit loop at
  [`density_split.py:438-449`](../density_split.py#L438-L449).
- `calculate` adds `c2_a P_{2,g}^{impl}` at
  [`density_split.py:466-478`](../density_split.py#L466-L478).

This implementation is intentionally a basis calculation: the expensive loop is
independent of the quantile amplitudes, and each quantile receives its own
linear `c2_a` multiplier.

### Derivative Counterterm

Eq. 28 introduces the derivative coefficients, Eq. 29 identifies the
`k^2`, `k^2 mu^2`, and `k^2 mu^4` structures, and Eq. 72 gives the cross-spectrum
counterterm. The code implements the second term in Eq. 72 as

```text
-2 k^2 W_R(k) [e0_a + e2_a mu^2 + e4_a mu^4] P_{gg,lin}(k, mu).
```

The pieces are:

- `e0q`, `e2q`, and `e4q` are fetched with the same quantile sum rule as `c1`
  and `c2`
  ([`density_split.py:474-477`](../density_split.py#L474-L477)).
- `P_{gg,lin}` is built at
  [`density_split.py:459-460`](../density_split.py#L459-L460).
- The counterterm is added at
  [`density_split.py:479`](../density_split.py#L479).

The first term in Eq. 72, `c1_a W_R P_{gg}^{ctr}`, is not separate in this file;
it is included in the FOLPS deterministic galaxy spectrum used by the `c1`
propagation term.

### Query-Galaxy Stochastic Terms

The `P_qg` stochastic completion follows Eq. 57 in the tree branch and Eq. 77
in the one-loop branch:

```text
P_{q_a g}^{stoch} = shotnoise s0qg_a
```

for `tree`, and

```text
P_{q_a g}^{stoch} = shotnoise [s0qg_a + k^2 (s2qg_a + s2muqg_a mu^2)]
```

for `1-loop`. The coefficients use the same equal-bin partition sum rule as the
composite coefficients, so quantile 3 is dependent. These parameters are linear
in the theory vector and are included in the default analytic marginalization
block used by the fitting scripts. The known query partition shot-noise matrix
`N_qry,0_ab` is not included here because the current calculator does not
predict `P_qaqb`.

## Implemented Vs Not Implemented

Implemented:

- Quantile-galaxy cross-spectrum multipoles `P_{q_a g, ell}(k)`.
- The strict-composite tree branch, Eq. 55.
- The reduced one-loop cross-spectrum formula shown above.
- Gaussian coefficient prior centers from Eqs. 35-38 and 104-106.
- Equal-probability partition sum rule for the dependent quantile.
- Explicit `c2 P2,g` loop based on Eqs. 42, 63, 68, and 110-115.
- Density-split derivative counterterms from Eqs. 28, 29, and 72.
- Query-galaxy stochastic terms from Eqs. 57 and 77.

Not implemented in the current file:

- Density-split auto-spectra `P_{q_a q_b}` from Eqs. 47, 54, 56, 70, and 73-79.
- Active `c3_a P3,g` from Eq. 69. `c3q*` parameters exist, are fixed, and are
  not read in `calculate`.
- Density-split auto-spectrum stochastic terms from Eqs. 58, 78, and 79,
  including the partition shot-noise matrix `N_qry,0_ab`.
- The optional effective linear relaxation of Eqs. 51, 52, 59, 60, 94, and 95.
- A separate in-file implementation of the full third-order galaxy kernel in
  Eq. 112; FOLPS supplies the galaxy one-loop sector used by the `c1` term.
