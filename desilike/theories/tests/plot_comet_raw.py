"""
Comparison between Kaiser, desilike-wrapped COMET, and raw PTEmu.Pell call.

The raw PTEmu call uses the DESI fiducial cosmology directly (same as desilike).
Diagnostic values (qpar, qper, AsD, f, b1_canonical) are printed to stdout.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from desilike.base import compile
from desilike.theories import CosmoprimoCosmology
from desilike.theories.galaxy_clustering import (
    KaiserTracerSpectrum2Poles,
    COMETTracerSpectrum2Poles,
    COMETTracerSpectrum3Poles,
)
from desilike.theories.galaxy_clustering.template import DirectSpectrum2Template

Z = 0.706
PRIOR_BASIS = 'physical_aap'
NBAR = 3e-4

k = np.linspace(0.02, 0.30, 80)
ells = (0, 2, 4)
ell_labels = {0: r'$P_0$', 2: r'$P_2$', 4: r'$P_4$'}

B1P = 1.

# DESI fiducial cosmology (cosmoprimo DESI preset)
DESI_PARAMS = {'h': 0.6736, 'wc': 0.12, 'wb': 0.02237, 'ns': 0.9649,
               'As': 2.083, 'z': Z, 'Mnu': 0.06}  # As in 1e-9 units (comet convention); Mnu=0.06 matches cosmoprimo DESI preset

cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial='DESI')
template = DirectSpectrum2Template(cosmo=cosmo, z=Z)

# --- Kaiser ---
theory_kaiser = KaiserTracerSpectrum2Poles(k=k, ells=ells, template=template)
pipe_kaiser = compile(theory_kaiser)
pk_kaiser = np.asarray(pipe_kaiser({'b1': B1P}))

# --- Raw PTEmu.Pell call (no desilike machinery) ---
from comet.PTEmu import PTEmu
emu = PTEmu(model='VDG_infty', use_Mpc=False)
emu.define_nbar(nbar=np.array([NBAR]))
emu.define_fiducial_cosmology(params_fid=DESI_PARAMS, de_model='lambda')


# --- COMET via desilike (pt=False: calls PTEmu.Pell directly) ---
theory_comet = COMETTracerSpectrum2Poles(cosmo=cosmo, pt=False, z=Z, k=k, ells=ells,
                                          prior_basis=PRIOR_BASIS, nbar=NBAR)
pipe_comet = compile(theory_comet)
pk_comet = np.asarray(pipe_comet({'b1p': B1P}))

# Diagnostic: internal COMET AP/growth params and b1_canonical after the call
qpar = float(theory_comet.qpar)
qper = float(theory_comet.qper)
AsD  = float(theory_comet.AsD)
f    = float(theory_comet.f)
A_AP = 1.0 / (qper**2 * qpar)
b1_canonical = B1P / (AsD * A_AP**0.5)
print('--- desilike COMET internals ---')
print(f'  qpar       = {qpar:.6f}')
print(f'  qper       = {qper:.6f}')
print(f'  AsD        = {AsD:.6f}')
print(f'  f          = {f:.6f}')
print(f'  A_AP       = {A_AP:.6f}')
print(f'  b1_canonical (EggScoSmi) = {b1_canonical:.6f}')
# Print cosmo params as seen by desilike
from desilike.theories.galaxy_clustering.full_shape import _cosmo_to_comet
cp = {k: float(v) for k, v in _cosmo_to_comet(theory_comet.cosmo).items()}
print(f'  cosmo params: h={cp["h"]:.6f}  wc={cp["wc"]:.6f}  wb={cp["wb"]:.6f}  As={cp["As"]:.6f}')

# --- desilike COMET with numpy backend (isolates JAX vs numpy) ---
theory_comet_np = COMETTracerSpectrum2Poles(cosmo=cosmo, pt=False, z=Z, k=k, ells=ells,
                                             prior_basis=PRIOR_BASIS, nbar=NBAR, backend='numpy')
pipe_comet_np = compile(theory_comet_np)
pk_comet_np = np.asarray(pipe_comet_np({'b1p': B1P}))
print(f'  P0(k=0.1) desilike-jax  = {np.interp(0.1, k, pk_comet[0]):.4f}')
print(f'  P0(k=0.1) desilike-numpy= {np.interp(0.1, k, pk_comet_np[0]):.4f}')

# Bias params in the same EggScoSmi basis that PTEmu uses by default.
# b1=1 matches b1_canonical from desilike when at fiducial (qpar=qper=AsD=1).
# All nuisance params are zero, matching the desilike call.
raw_params = {**DESI_PARAMS,
              'b1': b1_canonical, 'b2': 0., 'g2': 0., 'g21': 0.,
              'c0': 0., 'c2': 0., 'c4': 0., 'cnlo': 0.,
              'NP0': 0., 'NP20': 0., 'NP22': 0.,
              'cnloB': 0., 'NB0': 0., 'MB0': 0., 'cB1': 0., 'cB2': 0.,
              'avir': 0., 'avirB': 0.}
# q_tr_lo=None → PTEmu computes AP params from cosmology vs fiducial (should give (1,1))
print('RAW params', raw_params)
raw_pell = emu.Pell(k, raw_params, [0, 2, 4], de_model='lambda', ell_for_recon=[0, 2, 4, 6])
pk_raw = np.stack([raw_pell[f'ell{m}'] for m in ells], axis=0)
print('--- raw PTEmu.Pell (no desilike) ---')
print(f'  P0(k=0.1) = {np.interp(0.1, k, pk_raw[0]):.4f}  (desilike: {np.interp(0.1, k, pk_comet[0]):.4f})')
print(f'  P0(k=0.1) Kaiser = {np.interp(0.1, k, pk_kaiser[0]):.4f}')

# --- Manual reproduction of _call_direct with numpy backend ---
# This isolates whether the issue is in the param conversion or in the JAX backend.
from desilike.theories.galaxy_clustering.full_shape import (
    _cosmo_to_comet, _comet_params_to_cosmology,
    _comet_ap_params, _comet_growth_amplitude,
    _load_comet_model,
)

params_manual = {k: float(v) for k, v in _cosmo_to_comet(theory_comet.cosmo).items()}
params_manual['z'] = float(Z)
md_manual = _load_comet_model('VDG_infty', use_mpc=False)

cosmo_base_manual = _comet_params_to_cosmology(params_manual, Z, theory_comet._de_model, backend='numpy')
cosmo_fid_manual  = _comet_params_to_cosmology(
    {k: float(v) for k, v in theory_comet._fid_comet.items()}, Z, theory_comet._de_model, backend='numpy')

qpar_m, qper_m = _comet_ap_params(cosmo_base_manual, cosmo_fid_manual, Z, use_mpc=False)
AsD_m, f_m     = _comet_growth_amplitude(cosmo_base_manual, cosmo_fid_manual, Z)

A_AP_m = 1.0 / (float(qper_m)**2 * float(qpar_m))
b1_m   = B1P / (float(AsD_m) * A_AP_m**0.5)

print('--- manual _call_direct (numpy backend) ---')
print(f'  qpar={float(qpar_m):.6f}  qper={float(qper_m):.6f}  AsD={float(AsD_m):.6f}  f={float(f_m):.6f}')
print(f'  b1_canonical = {b1_m:.6f}')

manual_params = {**params_manual,
                 'b1': b1_m, 'b2': 0., 'g2': 0., 'g21': 0.,
                 'c0': 0., 'c2': 0., 'c4': 0., 'cnlo': 0.,
                 'NP0': 0., 'NP20': 0., 'NP22': 0.,
                 'cnloB': 0., 'NB0': 0., 'MB0': 0., 'cB1': 0., 'cB2': 0.,
                 'avir': 0., 'avirB': 0.}
manual_pell = md_manual.Pell(k, manual_params, [0, 2, 4],
                              de_model=theory_comet._de_model,
                              q_tr_lo=(float(qper_m), float(qpar_m)),
                              ell_for_recon=[0, 2, 4, 6])
pk_manual = np.stack([manual_pell[f'ell{m}'] for m in ells], axis=0)
print(f'  P0(k=0.1) manual = {np.interp(0.1, k, pk_manual[0]):.4f}')

models = [('Kaiser',           pk_kaiser,    'C2', '-'),
          ('COMET-desilike-jax', pk_comet,    'C1', '--'),
          ('COMET-desilike-np',  pk_comet_np, 'C0', '-.'),
          ('COMET-raw',          pk_raw,       'C3', ':'),
          ('COMET-manual',       pk_manual,    'C4', ':')]

fig, axes = plt.subplots(2, len(ells), figsize=(14, 7), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))

for col, ell in enumerate(ells):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    label = ell_labels[ell]

    for name, pk, color, ls in models:
        ax_top.plot(k, k * pk[col], color=color, label=name, lw=1.8, ls=ls)

    ax_top.set_ylabel(rf'$k \, {label[1:-1]}$ [$h^2$ Mpc$^{{-2}}$]')
    ax_top.set_title(label)
    if col == 0:
        ax_top.legend(frameon=False, fontsize=8)

    ref = pk_kaiser[col]
    ax_bot.axhline(1.0, color='k', lw=0.8, ls=':')
    for name, pk, color, ls in models:
        ratio = pk[col] / np.where(np.abs(ref) > 1e-10, ref, np.nan)
        ax_bot.plot(k, ratio, color=color, lw=1.5, ls=ls)
    ax_bot.set_ylabel('X / Kaiser')
    ax_bot.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
    ax_bot.set_ylim(0.5, 1.5)

info = (rf'$z={Z}$, prior_basis={PRIOR_BASIS!r}, $b_{{1p}}={B1P}$  '
        rf'(qpar={qpar:.3f}, qper={qper:.3f}, AsD={AsD:.3f}, f={f:.3f})')
fig.suptitle('Kaiser / COMET-desilike / COMET-raw — ' + info, y=1.01, fontsize=10)
fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), 'plot_comet_raw.png')
fig.savefig(outpath, bbox_inches='tight')
print(f'Saved {outpath}')

# ---------------------------------------------------------------------------
# Bispectrum: compare desilike (jax + numpy) vs raw PTEmu.Bell_Sugi
# Uses EggScoSmi+Comet basis for both desilike backends so that b1 maps
# directly to PTEmu's EggScoSmi b1 without any basis conversion at fiducial.
# ---------------------------------------------------------------------------
k_bi = np.column_stack([np.linspace(0.01, 0.1, 11)] * 2)
bi_ells = ((0, 0, 0), (2, 0, 2))
bi_ell_labels = {(0, 0, 0): r'$B_{000}$', (2, 0, 2): r'$B_{202}$'}

theory_comet_bi_jax = COMETTracerSpectrum3Poles(cosmo=cosmo, pt=False, z=Z, k=k_bi,
                                                 ells=bi_ells, prior_basis='EggScoSmi+Comet',
                                                 nbar=NBAR)
pipe_comet_bi_jax = compile(theory_comet_bi_jax)
bk_comet_jax = np.asarray(pipe_comet_bi_jax({'b1': b1_canonical}))

theory_comet_bi_np = COMETTracerSpectrum3Poles(cosmo=cosmo, pt=False, z=Z, k=k_bi,
                                                ells=bi_ells, prior_basis='EggScoSmi+Comet',
                                                nbar=NBAR, backend='numpy')
pipe_comet_bi_np = compile(theory_comet_bi_np)
bk_comet_np = np.asarray(pipe_comet_bi_np({'b1': b1_canonical}))

# raw PTEmu.Bell_Sugi — same raw_params as the Pell call (already includes NB0, MB0, cB1, cB2, avirB)
# numpy path (Python-float params) → Bell_Sugi returns {(l1,l2,L): ndarray(n_pairs,1)}
raw_bell = emu.Bell_Sugi(k_bi, raw_params, ell=list(bi_ells), de_model='lambda')
bk_raw = np.stack([np.squeeze(np.asarray(raw_bell[ell])) for ell in bi_ells], axis=0)

k_diag = k_bi[:, 0]
print('--- bispectrum B000(k=0.05) ---')
k_ref = 0.05
for label_str, bk in [('desilike-jax', bk_comet_jax), ('desilike-numpy', bk_comet_np), ('raw', bk_raw)]:
    val = np.interp(k_ref, k_diag, bk[0])
    print(f'  B000(k={k_ref}) {label_str} = {val:.4f}')

bi_models = [('COMET-desilike-jax', bk_comet_jax, 'C1', '--'),
             ('COMET-desilike-np',  bk_comet_np,  'C0', '-.'),
             ('COMET-raw',          bk_raw,        'C3', ':')]

fig2, axes2 = plt.subplots(2, len(bi_ells), figsize=(10, 7), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))

for col, ell in enumerate(bi_ells):
    ax_top = axes2[0, col]
    ax_bot = axes2[1, col]
    label = bi_ell_labels[ell]

    for name, bk, color, ls in bi_models:
        ax_top.plot(k_diag, k_diag**2 * bk[col], color=color, label=name, lw=1.8, ls=ls)

    ax_top.set_ylabel(rf'$k^2 \, {label[1:-1]}$ [$h^{{-4}}$ Mpc$^4$]')
    ax_top.set_title(label)
    if col == 0:
        ax_top.legend(frameon=False, fontsize=8)

    ref = bk_raw[col]
    ax_bot.axhline(1.0, color='k', lw=0.8, ls=':')
    for name, bk, color, ls in bi_models:
        ratio = bk[col] / np.where(np.abs(ref) > 1e-10, ref, np.nan)
        ax_bot.plot(k_diag, ratio, color=color, lw=1.5, ls=ls)
    ax_bot.set_ylabel('X / raw PTEmu')
    ax_bot.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
    ax_bot.set_ylim(0.5, 1.5)

fig2.suptitle(rf'COMET bispectrum: desilike vs raw PTEmu — $z={Z}$, $b_1={b1_canonical:.3f}$, $k_1=k_2$',
              y=1.01, fontsize=10)
fig2.tight_layout()

outpath2 = os.path.join(os.path.dirname(__file__), 'plot_comet_raw_bispectrum.png')
fig2.savefig(outpath2, bbox_inches='tight')
print(f'Saved {outpath2}')
