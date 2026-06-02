"""
test_nw_methods.py
------------------
Compare the two no-wiggle prescriptions now supported by MgEmulatorCosmology:

  - nw_method='gr_ratio' (NEW default): pnw_MG = pnw_GR * plin_MG / plin_GR,
    with pnw_GR/plin_GR the GR wiggle ratio from the original GR emulators.
  - nw_method='folps'   (previous):     folps get_pknow on the emulated plin.

For GR + a few MG cases it plots (left) k*P for plin / pnw_gr_ratio / pnw_folps,
and (right) the percent difference pnw_gr_ratio vs pnw_folps.

Run:
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    export FOLPS_BACKEND=jax
    export PYTHONPATH=/global/homes/p/prakharb/fkptjax_muMG/src:$PYTHONPATH
    cd emulator_comparison && python test_nw_methods.py
"""

import os
import sys
os.environ.setdefault("FOLPS_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # this repo's desilike first

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from desilike.theories.galaxy_clustering import MgEmulatorCosmology

Z = 0.5
FID = dict(logA=3.044, n_s=0.9649, h=0.6736, omega_b=0.02237, omega_cdm=0.12)
MGNAMES = ['mu1', 'mu2', 'mu3', 'mu4', 'Sigma1', 'Sigma2', 'Sigma3', 'Sigma4']

CASES = [
    ("GR (mu=Sigma=1)",       [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mu1=1.5",           [1.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mu=0.5 (weaker G)", [0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]),
    ("MG: mixed mu/Sigma",    [2.0, 0.5, 1.5, 0.8], [1.5, 0.8, 1.2, 1.0]),
]


def build(nw_method):
    prov = MgEmulatorCosmology(zs=[Z], nw_method=nw_method)
    for nm in MGNAMES:
        prov.init.params[nm].update(fixed=False)
    for nm, val in FID.items():
        prov.init.params[nm].update(value=val, fixed=True)
    return prov


def evaluate(prov, mu, Sigma):
    prov(**dict(zip(MGNAMES, list(mu) + list(Sigma))))
    r = prov.get_at_z(Z)
    return np.asarray(r['k']), np.asarray(r['pk_dd']), np.asarray(r['pknow'])


def main():
    prov_gr = build('gr_ratio')
    prov_fp = build('folps')

    ncase = len(CASES)
    fig, axes = plt.subplots(ncase, 2, figsize=(13, 3.4 * ncase))
    kmask = None
    print("max |pnw(gr_ratio) - pnw(folps)| / pnw(folps) over 0.01<k<0.5 h/Mpc:")
    for i, (label, mu, Sigma) in enumerate(CASES):
        k, plin, pnw_gr = evaluate(prov_gr, mu, Sigma)
        _, _,   pnw_fp  = evaluate(prov_fp, mu, Sigma)
        if kmask is None:
            kmask = (k >= 0.01) & (k <= 0.5)
        kp = k[kmask]

        axL, axR = axes[i, 0], axes[i, 1]
        axL.plot(kp, kp * plin[kmask],   "k",    lw=1.6, label="plin (MG emu)")
        axL.plot(kp, kp * pnw_gr[kmask], "C0",   lw=1.6, label="pnw — gr_ratio (default)")
        axL.plot(kp, kp * pnw_fp[kmask], "C1--", lw=1.6, label="pnw — folps")
        axL.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axL.set_ylabel(r"$k\,P(k)$")
        axL.set_title(label, fontsize=10); axL.legend(fontsize=8); axL.grid(alpha=0.3)

        diff = 100.0 * (pnw_gr[kmask] - pnw_fp[kmask]) / pnw_fp[kmask]
        worst = float(np.max(np.abs(diff)))
        axR.plot(kp, diff, "C3", lw=1.5)
        axR.axhline(0, color="k", lw=0.8)
        axR.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axR.set_ylabel("pnw diff [%]")
        axR.set_title(f"gr_ratio - folps  (max |diff| = {worst:.2f}%)", fontsize=10)
        axR.grid(alpha=0.3)
        print(f"  {label:24s} {worst:6.2f} %")

    fig.suptitle(f"MgEmulatorCosmology no-wiggle: gr_ratio (default) vs folps (z={Z})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(HERE, "test_nw_methods.png")
    fig.savefig(out, dpi=150)
    print("\nSaved", out)


if __name__ == "__main__":
    main()
