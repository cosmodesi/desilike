"""
Compare Kaiser, FOLPS and COMET power spectrum multipoles at z=0.706, prior_basis='physical_aap'.

Both theories are run at default parameter values (b1=1.5, all nuisance=0),
with nbar=3e-4 (Mpc/h)^-3. The bias bases differ between the two models:
FOLPS uses Lagrangian {b1,b2,bs,b3,...} while COMET uses DESI {b1,b2d,bk2,btd,...},
so only b1=1.5 is directly shared; all other nuisance params are left at zero.
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
    FOLPSTracerSpectrum2Poles,
    COMETPTSpectrum2Poles, COMETTracerSpectrum2Poles,
    FOLPSTracerSpectrum3Poles,
    COMETTracerSpectrum3Poles,
)
from desilike.theories.galaxy_clustering.template import DirectSpectrum2Template

Z = 0.706
PRIOR_BASIS = 'physical_aap'
NBAR = 3e-4

k = np.linspace(0.02, 0.30, 80)
ells = (0, 2, 4)
ell_labels = {0: r'$P_0$', 2: r'$P_2$', 4: r'$P_4$'}

B1P = 1.  # shared b1p value; all other nuisance params left at zero

cosmo = CosmoprimoCosmology(engine='class', fiducial='DESI')
template = DirectSpectrum2Template(cosmo=cosmo, z=Z)

theory_kaiser = KaiserTracerSpectrum2Poles(k=k, ells=ells, template=template)
pipe_kaiser = compile(theory_kaiser)
pk_kaiser = np.asarray(pipe_kaiser({'b1': B1P}))

#pt = COMETPTSpectrum2Poles(cosmo=cosmo, z=Z, k=k, ells=ells)
theory_comet = COMETTracerSpectrum2Poles(cosmo=cosmo, pt=False, z=Z, k=k, ells=ells,
                                          prior_basis=PRIOR_BASIS, nbar=NBAR)
pipe_comet = compile(theory_comet)
pk_comet = np.asarray(pipe_comet({'b1p': B1P}))

theory_folps = FOLPSTracerSpectrum2Poles(k=k, ells=ells, prior_basis=PRIOR_BASIS,
                                         template=template, nbar=NBAR)
pipe_folps = compile(theory_folps)
pk_folps = np.asarray(pipe_folps({'b1p': B1P}))

models = [('Kaiser', pk_kaiser, 'C2', '-'),
          ('FOLPS',  pk_folps,  'C0', '-'),
          ('COMET',  pk_comet,  'C1', '--')]

fig, axes = plt.subplots(2, len(ells), figsize=(12, 7), sharex=True,
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
        ax_top.legend(frameon=False)

    ref = pk_kaiser[col]
    ax_bot.axhline(1.0, color='k', lw=0.8, ls=':')
    for name, pk, color, ls in models:
        ratio = pk[col] / np.where(np.abs(ref) > 1e-10, ref, np.nan)
        ax_bot.plot(k, ratio, color=color, lw=1.5, ls=ls)
    ax_bot.set_ylabel('X / Kaiser')
    ax_bot.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
    ax_bot.set_ylim(0.5, 1.5)

fig.suptitle(rf'Kaiser / FOLPS / COMET — $z={Z}$, prior_basis={PRIOR_BASIS!r}, $b_{{1p}}={B1P}$, nuisance=0',
             y=1.01, fontsize=11)
fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), 'plot_comet.png')
fig.savefig(outpath, bbox_inches='tight')
print(f'Saved {outpath}')

# --- Bispectrum comparison: diagonal k1=k2 pairs ---
# COMETTracerSpectrum3Poles does not yet support physical_aap (the propose_params
# select uses non-'p' basenames), so COMET runs in its native EggScoSmi+Comet basis.
# At the DESI fiducial (AsD = A_AP = 1) b1 (EggScoSmi) == b1p (physical_aap).
k_bi = np.column_stack([np.linspace(0.01, 0.1, 11)] * 2)
bi_ells = ((0, 0, 0), (2, 0, 2))
bi_ell_labels = {(0, 0, 0): r'$B_{000}$', (2, 0, 2): r'$B_{202}$'}

theory_folps_bi = FOLPSTracerSpectrum3Poles(k=k_bi, ells=bi_ells, prior_basis=PRIOR_BASIS,
                                             template=template, nbar=NBAR)
pipe_folps_bi = compile(theory_folps_bi)
bk_folps = np.asarray(pipe_folps_bi({'b1p': B1P}))

theory_comet_bi = COMETTracerSpectrum3Poles(cosmo=cosmo, z=Z, k=k_bi, ells=bi_ells,
                                             prior_basis=PRIOR_BASIS, nbar=NBAR)
pipe_comet_bi = compile(theory_comet_bi)
bk_comet = np.asarray(pipe_comet_bi({'b1p': B1P}))

bi_models = [('FOLPS', bk_folps, 'C0', '-'),
             ('COMET', bk_comet, 'C1', '--')]

k_diag = k_bi[:, 0]

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
        ax_top.legend(frameon=False)

    ref = bk_folps[col]
    ax_bot.axhline(1.0, color='k', lw=0.8, ls=':')
    for name, bk, color, ls in bi_models:
        ratio = bk[col] / np.where(np.abs(ref) > 1e-10, ref, np.nan)
        ax_bot.plot(k_diag, ratio, color=color, lw=1.5, ls=ls)
    ax_bot.set_ylabel('X / FOLPS')
    ax_bot.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
    ax_bot.set_ylim(0.5, 1.5)

fig2.suptitle(rf'FOLPS / COMET bispectrum — $z={Z}$, $b_{{1p}}={B1P}$, nuisance=0, $k_1=k_2$',
              y=1.01, fontsize=11)
fig2.tight_layout()

outpath2 = os.path.join(os.path.dirname(__file__), 'plot_comet_bispectrum.png')
fig2.savefig(outpath2, bbox_inches='tight')
print(f'Saved {outpath2}')
