"""
compare_multipoles.py
---------------------
Compare fkpt redshift-space power-spectrum multipoles computed two ways:

  Method 1  "ISiTGR + fkpt"  : linear P(k)/P_nw + scalars from a DIRECT ISiTGR
                               computation  (IsitgrCosmology provider)
  Method 2  "emulator + fkpt" : linear P(k)/P_nw + scalars from the trained MG
                               emulator       (MgEmulatorCosmology provider)

BOTH feed the *identical* fkpt loop (Kfuncs_to_tables) and bias machinery via
``fkpt_pkemu_TracerPowerSpectrumMultipoles`` -- the only difference is the
source of the linear inputs.  So the comparison isolates the emulator accuracy.

The cosmological background is held at the fiducial for every case (so AP = 1
and we isolate the modified-gravity response); only the binned mu/Sigma vary.

GR convention: ISiTGR binned muSigma has GR at mu_i = Sigma_i = 1.

Run (inside the cosmodesi env, with fkptjax_muMG on PYTHONPATH and FOLPS_BACKEND=jax):
    python compare_multipoles.py
"""

import os
import sys
os.environ.setdefault("FOLPS_BACKEND", "jax")

HERE = os.path.dirname(os.path.abspath(__file__))
# Make sure THIS repo's desilike (with the emulator classes) wins over any
# cosmodesi-installed desilike, regardless of cwd.
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from desilike.theories.galaxy_clustering import (
    MgEmulatorCosmology, fkpt_pkemu_TracerPowerSpectrumMultipoles)
from cosmologies import IsitgrCosmology

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
Z = 0.5
KOUT = np.linspace(0.02, 0.20, 19)
ELLS = (0, 2, 4)

# fiducial background (matches MgEmulatorCosmology.DEFAULT_FIDUCIAL) -> AP = 1
FID = dict(logA=3.044, n_s=0.9649, h=0.6736, omega_b=0.02237, omega_cdm=0.12)

# identical, fixed bias for both methods (standard basis); only MG differs
BIAS = dict(b1=2.0, b2=0.0, bs2=0.0, b3nl=0.0,
            alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0,
            alpha0shot=0.0, alpha2shot=0.0)

# (label, [mu1..mu4], [Sigma1..Sigma4])
CASES = [
    ("GR (mu=Sigma=1)",        [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mu1=1.5",            [1.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mu=0.5 (weaker G)",  [0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mixed mu/Sigma",     [2.0, 0.5, 1.5, 0.8], [1.5, 0.8, 1.2, 1.0]),
]

MGNAMES = ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']


def build_theory(provider_cls):
    """One provider + one tracer; mu/Sigma left free so cases reuse the jit."""
    prov = provider_cls(zs=[Z])
    for nm in MGNAMES:
        prov.init.params[nm].update(fixed=False)
    for nm, val in FID.items():
        prov.init.params[nm].update(value=val, fixed=True)

    theory = fkpt_pkemu_TracerPowerSpectrumMultipoles()
    theory.init.update(prior_basis="standard", tracer="LRG1",
                       cosmo=prov, z=Z, k=KOUT, ells=ELLS, beyond_eds=True)
    for nm, val in BIAS.items():
        if nm in theory.init.params:
            theory.init.params[nm].update(value=val, fixed=True)
    return theory


def evaluate(theory, mu, Sigma):
    vals = dict(zip(MGNAMES, list(mu) + list(Sigma)))
    theory(**vals)
    return np.asarray(theory.power)


def main():
    print("Building ISiTGR-truth pipeline ...")
    th_isitgr = build_theory(IsitgrCosmology)
    print("Building emulator pipeline ...")
    th_emu = build_theory(MgEmulatorCosmology)

    ncase = len(CASES)
    fig, axes = plt.subplots(ncase, 2, figsize=(13, 3.6 * ncase))
    if ncase == 1:
        axes = axes[np.newaxis, :]

    colors = {0: "C0", 2: "C1", 4: "C2"}
    summary = []

    for i, (label, mu, Sigma) in enumerate(CASES):
        print(f"[{i+1}/{ncase}] {label}")
        P1 = evaluate(th_isitgr, mu, Sigma)   # ISiTGR
        P2 = evaluate(th_emu, mu, Sigma)      # emulator

        axL, axR = axes[i, 0], axes[i, 1]
        worst = 0.0
        for j, ell in enumerate(ELLS):
            c = colors[ell]
            axL.plot(KOUT, KOUT * P1[j], color=c, lw=1.6,
                     label=f"ISiTGR  l={ell}")
            axL.plot(KOUT, KOUT * P2[j], color=c, lw=1.4, ls="--",
                     label=f"emulator l={ell}")
            diff = 100.0 * (P2[j] - P1[j]) / P1[j]
            axR.plot(KOUT, diff, color=c, lw=1.5, label=f"l={ell}")
            worst = max(worst, float(np.max(np.abs(diff))))

        axL.set_title(f"{label}", fontsize=10)
        axL.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$")
        axL.set_ylabel(r"$k\,P_\ell(k)$")
        axL.legend(fontsize=7, ncol=3)
        axL.grid(alpha=0.3)

        axR.axhline(0, color="k", lw=0.8)
        for lvl in (1.0, -1.0):
            axR.axhline(lvl, color="gray", lw=0.7, ls=":")
        axR.set_title(f"emulator - ISiTGR  (max |diff| = {worst:.2f}%)", fontsize=10)
        axR.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$")
        axR.set_ylabel("difference [%]")
        axR.legend(fontsize=8)
        axR.grid(alpha=0.3)
        summary.append((label, worst))

    fig.suptitle(f"fkpt multipoles: ISiTGR vs emulator linear inputs "
                 f"(z={Z}, AP=1, b1={BIAS['b1']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(HERE, "compare_multipoles.png")
    fig.savefig(out, dpi=150)
    print("\nSaved", out)
    print("\nMax |P_ell emulator - ISiTGR| per case:")
    for label, worst in summary:
        print(f"  {label:28s} {worst:6.2f} %")


if __name__ == "__main__":
    main()
