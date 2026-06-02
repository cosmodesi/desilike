"""
test_same_nw.py
---------------
Controlled test: is the original-vs-emulator multipole difference coming from
the no-wiggle (BAO) prescription?

For each case we compare the original class (ISiTGR cosmoprimo template, which
uses the 'peakaverage' no-wiggle) against the emulator class run TWO ways:

  (i)  emulator plin + folps get_pknow   -- the production no-wiggle  ("folps")
  (ii) emulator plin + the SAME peakaverage P_nw the original class used
       (extracted from its template, interpolated onto the emulator k-grid)  ("same-nw")

If (ii) collapses the difference relative to (i), the residual was the no-wiggle
prescription, not the emulator.

Run with the same env / PYTHONPATH as compare_original_vs_emulator.py.
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
from desilike.theories.galaxy_clustering import (
    MgEmulatorCosmology, fkpt_pkemu_TracerPowerSpectrumMultipoles)

ELLS = C.ELLS  # (0, 2, 4)


_FID_COSMO = [None]   # cache the cosmoprimo fiducial used by the peakaverage filter


def _fid_cosmo():
    if _FID_COSMO[0] is None:
        from cosmoprimo.fiducial import DESI
        _FID_COSMO[0] = DESI()
    return _FID_COSMO[0]


class MgEmulatorCosmologyPeakAvgNW(MgEmulatorCosmology):
    """Emulator provider whose no-wiggle P_nw is computed with cosmoprimo's
    'peakaverage' BAO filter applied to the EMULATOR's own plin -- i.e. the SAME
    no-wiggle prescription the original class uses (it filters the template plin
    the same way).  Everything else identical to MgEmulatorCosmology."""

    def calculate(self, logA, n_s, h, omega_b, omega_cdm,
                  mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4, **kwargs):
        from cosmoprimo import PowerSpectrumBAOFilter, PowerSpectrumInterpolator1D
        fid = _fid_cosmo()
        H0 = h * 100.0
        Omega_m = (omega_b + omega_cdm + self.M_NCDM / 93.14) / h**2
        plist = [logA, n_s, H0, omega_b, omega_cdm, self.M_NCDM]
        self._results = {}
        for z in self.zs:
            k, plin, _pnw_emu, scalars = self._emu.predict_all(
                z=z, ln10As=logA, ns=n_s, H0=H0, ombh2=omega_b, omch2=omega_cdm,
                mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4,
                Sigma1=Sigma1, Sigma2=Sigma2, Sigma3=Sigma3, Sigma4=Sigma4)
            k = np.asarray(k); plin = np.asarray(plin)
            # same prescription (peakaverage) as the original class, on emu plin
            interp = PowerSpectrumInterpolator1D(k, plin)
            filt = PowerSpectrumBAOFilter(interp, engine='peakaverage',
                                          cosmo=fid, cosmo_fid=fid)
            pnw = np.asarray(filt.smooth_pk_interpolator()(k))
            chi_fid, e_fid = self._fid_scalars[z]
            qper = h * scalars['chi_z'] / (chi_fid * self.fiducial['h'])
            qpar = e_fid / scalars['e_z']
            self._results[z] = dict(
                k=k, pk_dd=plin, pknow=pnw, Omega_m=Omega_m, h=h,
                sigma8=scalars['sigma8_z'], qper=qper, qpar=qpar,
                mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4, plist=plist)

    def get_at_z(self, z):
        return self._results[z]


def build_emulator_with(provider):
    theory = fkpt_pkemu_TracerPowerSpectrumMultipoles()
    theory.init.update(prior_basis="standard", tracer="LRG1", cosmo=provider,
                       z=C.Z, k=C.KOUT, ells=ELLS, beyond_eds=True)
    C._fix_bias(theory)
    return theory


def build_emu_provider(cls):
    prov = cls(zs=[C.Z])
    for nm in C.MGNAMES:
        prov.init.params[nm].update(fixed=False)
    for nm, val in C.FID.items():
        prov.init.params[nm].update(value=val, fixed=True)
    return prov


def main():
    print("Building original (ISiTGR + fkptjax) ...")
    th_orig = C.build_original()
    print("Building emulator (folps pnw) ...")
    th_folps = build_emulator_with(build_emu_provider(MgEmulatorCosmology))
    print("Building emulator (peakaverage pnw, same prescription) ...")
    prov_ext = build_emu_provider(MgEmulatorCosmologyPeakAvgNW)
    th_ext = build_emulator_with(prov_ext)

    ncase = len(C.CASES)
    fig, axes = plt.subplots(ncase, 2, figsize=(13, 3.4 * ncase))
    if ncase == 1:
        axes = axes[np.newaxis, :]
    colors = {0: "C0", 2: "C1", 4: "C2"}
    summary = []

    for i, (label, mu, Sigma) in enumerate(C.CASES):
        print(f"[{i+1}/{ncase}] {label}")
        PA = C.evaluate(th_orig, mu, Sigma)
        Pf = C.evaluate(th_folps, mu, Sigma)
        Pe = C.evaluate(th_ext, mu, Sigma)

        axL, axR = axes[i, 0], axes[i, 1]
        interior = C.KOUT < 0.18   # exclude the last couple of high-k edge bins
        worst_f = worst_e = worst_e_in = 0.0
        for j, ell in enumerate(ELLS):
            c = colors[ell]
            df = 100.0 * (Pf[j] - PA[j]) / PA[j]
            de = 100.0 * (Pe[j] - PA[j]) / PA[j]
            axL.plot(C.KOUT, df, color=c, lw=1.4, ls=":", label=f"folps  l={ell}")
            axR.plot(C.KOUT, de, color=c, lw=1.6, label=f"same-nw l={ell}")
            worst_f = max(worst_f, float(np.max(np.abs(df))))
            worst_e = max(worst_e, float(np.max(np.abs(de))))
            worst_e_in = max(worst_e_in, float(np.max(np.abs(de[interior]))))

        for ax, ttl in ((axL, f"emulator(folps pnw) - original  (max {worst_f:.2f}%)"),
                        (axR, f"emulator(same pnw) - original  (max {worst_e:.2f}%)")):
            ax.axhline(0, color="k", lw=0.8)
            for lvl in (1.0, -1.0):
                ax.axhline(lvl, color="gray", lw=0.7, ls=":")
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); ax.set_ylabel("difference [%]")
            ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
        # share y-limits per row so the collapse is visually obvious
        ylim = max(abs(np.array(axL.get_ylim())).max(), abs(np.array(axR.get_ylim())).max())
        axL.set_ylim(-ylim, ylim); axR.set_ylim(-ylim, ylim)
        axL.text(0.02, 0.92, label, transform=axL.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", fc="w", alpha=0.7))
        summary.append((label, worst_f, worst_e, worst_e_in))

    fig.suptitle("Does the residual come from the no-wiggle prescription?  "
                 f"(z={C.Z}, AP=1, b1={C.BIAS['b1']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(HERE, "test_same_nw.png")
    fig.savefig(out, dpi=150)
    print("\nSaved", out)
    print("\nMax |emulator - original| per case:")
    print(f"  {'case':26s} {'folps-pnw':>10s} {'same-nw':>10s} {'same-nw(k<0.18)':>16s}")
    for label, wf, we, we_in in summary:
        print(f"  {label:26s} {wf:9.2f}% {we:9.2f}% {we_in:15.2f}%")


if __name__ == "__main__":
    main()
