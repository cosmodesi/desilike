"""
compare_original_vs_emulator_p0p2.py
------------------------------------
Same end-to-end comparison as compare_original_vs_emulator.py (literal original
fkptjax + ISiTGR cosmoprimo  vs  emulator class), but restricted to the
monopole and quadrupole (P0, P2) -- dropping the noisy hexadecapole P4.

Run with the same environment / PYTHONPATH as compare_original_vs_emulator.py.
"""

import os
import sys
os.environ.setdefault("FOLPS_BACKEND", "jax")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_original_vs_emulator as C

# Only monopole + quadrupole.
C.ELLS = (0, 2)
ELLS = C.ELLS


def main():
    print("Building original (ISiTGR + fkptjax) pipeline ...")
    th_orig = C.build_original()
    print("Building emulator pipeline ...")
    th_emu = C.build_emulator()

    ncase = len(C.CASES)
    fig, axes = plt.subplots(ncase, 2, figsize=(13, 3.4 * ncase))
    if ncase == 1:
        axes = axes[np.newaxis, :]
    colors = {0: "C0", 2: "C1"}
    summary = []

    for i, (label, mu, Sigma) in enumerate(C.CASES):
        print(f"[{i+1}/{ncase}] {label}")
        PA = C.evaluate(th_orig, mu, Sigma)
        PB = C.evaluate(th_emu, mu, Sigma)

        axL, axR = axes[i, 0], axes[i, 1]
        worst = 0.0
        for j, ell in enumerate(ELLS):
            c = colors[ell]
            axL.plot(C.KOUT, C.KOUT * PA[j], color=c, lw=1.6, label=f"original l={ell}")
            axL.plot(C.KOUT, C.KOUT * PB[j], color=c, lw=1.4, ls="--", label=f"emulator l={ell}")
            diff = 100.0 * (PB[j] - PA[j]) / PA[j]
            axR.plot(C.KOUT, diff, color=c, lw=1.5, label=f"l={ell}")
            worst = max(worst, float(np.max(np.abs(diff))))

        axL.set_title(label, fontsize=10)
        axL.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axL.set_ylabel(r"$k\,P_\ell(k)$")
        axL.legend(fontsize=8, ncol=2); axL.grid(alpha=0.3)
        axR.axhline(0, color="k", lw=0.8)
        for lvl in (1.0, -1.0):
            axR.axhline(lvl, color="gray", lw=0.7, ls=":")
        axR.set_title(f"emulator - original  (max |diff| = {worst:.2f}%)", fontsize=10)
        axR.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); axR.set_ylabel("difference [%]")
        axR.legend(fontsize=9); axR.grid(alpha=0.3)
        summary.append((label, worst))

    fig.suptitle(f"fkpt P0/P2: original (ISiTGR cosmoprimo) vs emulator class "
                 f"(z={C.Z}, AP=1, b1={C.BIAS['b1']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(HERE, "compare_original_vs_emulator_p0p2.png")
    fig.savefig(out, dpi=150)
    print("\nSaved", out)
    print("\nMax |P_ell emulator - original| per case (P0,P2 only):")
    for label, worst in summary:
        print(f"  {label:28s} {worst:6.2f} %")


if __name__ == "__main__":
    main()
