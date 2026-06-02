"""
compare_original_vs_emulator.py
-------------------------------
Complete end-to-end sanity check of the emulator-backed fkpt classes against
the LITERAL original desilike pipeline.

  Method A  "original"  : fkptjaxTracerPowerSpectrumMultipoles
                          + DirectPowerSpectrumTemplate(cosmo=Cosmoprimo(engine='isitgr'))
                          -> linear P(k) from the ISiTGR Boltzmann code; no-wiggle
                             from the template's 'peakaverage' BAO filter.
  Method B  "emulator"  : fkpt_pkemu_TracerPowerSpectrumMultipoles + MgEmulatorCosmology
                          -> linear P(k) from the trained emulator; no-wiggle from
                             folps get_pknow.

Both use mg_variant='binning' with the emulator's training binning constants,
beyond_eds=True, rescale_PS=False, identical (fixed) bias, and the background
held at the fiducial (AP = 1).  Only the binned mu/Sigma vary across cases.

Requires (besides the cosmodesi env):
    export FOLPS_BACKEND=jax
    export PYTHONPATH=\
      <repo>:<cosmoprimo_isitgr>:<isitgr_private>:<fkptjax_muMG/src>:$PYTHONPATH
where
    <cosmoprimo_isitgr> = checkout of cosmodesi/cosmoprimo @ cosmoprimo_isitgr
    <isitgr_private>    = locally-compiled ISiTGR fork
"""

import os
import sys
os.environ.setdefault("FOLPS_BACKEND", "jax")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # this repo's desilike first
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from desilike import parameter
from desilike.theories.primordial_cosmology import Cosmoprimo
from desilike.theories.galaxy_clustering import (
    DirectPowerSpectrumTemplate, fkptjaxTracerPowerSpectrumMultipoles,
    MgEmulatorCosmology, fkpt_pkemu_TracerPowerSpectrumMultipoles)
from cosmoprimo.fiducial import DESI

# --------------------------------------------------------------------------
Z = 0.5
KOUT = np.linspace(0.02, 0.20, 19)
ELLS = (0, 2, 4)
FID = dict(logA=3.044, n_s=0.9649, h=0.6736, omega_b=0.02237, omega_cdm=0.12)
BIAS = dict(b1=2.0, b2=0.0, bs2=0.0, b3nl=0.0,
            alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0,
            alpha0shot=0.0, alpha2shot=0.0)
BINCONST = [('z_div', 1.0), ('z_TGR', 2.0), ('z_tw', 0.05),
            ('k_c', 0.1), ('k_tw', 0.001), ('k_TGR', 0.01), ('k_S', 0.2)]
MGNAMES = ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']

CASES = [
    ("GR (mu=Sigma=1)",        [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mu1=1.5",            [1.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mu=0.5 (weaker G)",  [0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mixed mu/Sigma",     [2.0, 0.5, 1.5, 0.8], [1.5, 0.8, 1.2, 1.0]),
]


def _fix_bias(theory):
    for nm, val in BIAS.items():
        if nm in theory.init.params:
            theory.init.params[nm].update(value=val, fixed=True)


def build_original():
    # IMPORTANT: the binning constants must be passed as ENGINE kwargs (extra
    # params forwarded to ISiTGR), NOT as desilike params -- otherwise the
    # engine silently uses its default k-windows and the linear MG response is
    # wrong.  The fkpt LOOP reads the matching constants from the (now-corrected)
    # _collect_mg_params defaults.
    cosmo = Cosmoprimo(engine='isitgr', redshift_bins=True, scale_bins=True,
                       **dict(BINCONST))
    pn = [p.basename for p in cosmo.init.params]
    for nm, val in FID.items():
        cosmo.init.params[nm].update(value=val, fixed=True)
    if 'm_ncdm' in pn:
        cosmo.init.params['m_ncdm'].update(value=0.06, fixed=True)
    if 'tau_reio' in pn:
        cosmo.init.params['tau_reio'].update(value=0.0568, fixed=True)
    # only the binned mu/Sigma are varying desilike params. The binning
    # CONSTANTS stay as ISiTGR engine kwargs only (appending them as desilike
    # params re-breaks the engine's k-windows); the fkpt LOOP gets the matching
    # constants from the corrected _collect_mg_params defaults.
    for nm in MGNAMES:
        cosmo.init.params.data.append(parameter.Parameter(basename=nm, value=1.0, fixed=False))

    template = DirectPowerSpectrumTemplate(z=Z, fiducial=DESI(), cosmo=cosmo)
    theory = fkptjaxTracerPowerSpectrumMultipoles()
    theory.init.update(prior_basis='standard', tracer='LRG1', template=template,
                       k=KOUT, ells=list(ELLS), model='PHENOM', mg_variant='binning',
                       beyond_eds=True, rescale_PS=False)
    _fix_bias(theory)
    return theory


def build_emulator():
    prov = MgEmulatorCosmology(zs=[Z])
    for nm in MGNAMES:
        prov.init.params[nm].update(fixed=False)
    for nm, val in FID.items():
        prov.init.params[nm].update(value=val, fixed=True)
    theory = fkpt_pkemu_TracerPowerSpectrumMultipoles()
    theory.init.update(prior_basis='standard', tracer='LRG1', cosmo=prov,
                       z=Z, k=KOUT, ells=ELLS, beyond_eds=True)
    _fix_bias(theory)
    return theory


def evaluate(theory, mu, Sigma):
    theory(**dict(zip(MGNAMES, list(mu) + list(Sigma))))
    return np.asarray(theory.power)


def main():
    print("Building original (ISiTGR + fkptjax) pipeline ...")
    th_orig = build_original()
    print("Building emulator pipeline ...")
    th_emu = build_emulator()

    ncase = len(CASES)
    fig, axes = plt.subplots(ncase, 2, figsize=(13, 3.6 * ncase))
    if ncase == 1:
        axes = axes[np.newaxis, :]
    colors = {0: "C0", 2: "C1", 4: "C2"}
    summary = []

    for i, (label, mu, Sigma) in enumerate(CASES):
        print(f"[{i+1}/{ncase}] {label}")
        PA = evaluate(th_orig, mu, Sigma)
        PB = evaluate(th_emu, mu, Sigma)

        axL, axR = axes[i, 0], axes[i, 1]
        worst = 0.0
        for j, ell in enumerate(ELLS):
            c = colors[ell]
            axL.plot(KOUT, KOUT * PA[j], color=c, lw=1.6, label=f"original l={ell}")
            axL.plot(KOUT, KOUT * PB[j], color=c, lw=1.4, ls="--", label=f"emulator l={ell}")
            diff = 100.0 * (PB[j] - PA[j]) / PA[j]
            axR.plot(KOUT, diff, color=c, lw=1.5, label=f"l={ell}")
            worst = max(worst, float(np.max(np.abs(diff))))

        axL.set_title(label, fontsize=10)
        axL.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axL.set_ylabel(r"$k\,P_\ell(k)$")
        axL.legend(fontsize=7, ncol=3); axL.grid(alpha=0.3)
        axR.axhline(0, color="k", lw=0.8)
        for lvl in (1.0, -1.0):
            axR.axhline(lvl, color="gray", lw=0.7, ls=":")
        axR.set_title(f"emulator - original  (max |diff| = {worst:.2f}%)", fontsize=10)
        axR.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axR.set_ylabel("difference [%]")
        axR.legend(fontsize=8); axR.grid(alpha=0.3)
        summary.append((label, worst))

    fig.suptitle(f"fkpt multipoles: original (ISiTGR cosmoprimo) vs emulator class "
                 f"(z={Z}, AP=1, b1={BIAS['b1']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(HERE, "compare_original_vs_emulator.png")
    fig.savefig(out, dpi=150)
    print("\nSaved", out)
    print("\nMax |P_ell emulator - original| per case:")
    for label, worst in summary:
        print(f"  {label:28s} {worst:6.2f} %")


if __name__ == "__main__":
    main()
