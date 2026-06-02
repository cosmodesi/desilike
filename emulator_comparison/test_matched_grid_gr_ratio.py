"""
test_matched_grid_gr_ratio.py
-----------------------------
Same as test_matched_grid_folps.py (emulated plin resampled onto the original
pipeline's loop k-grid, geomspace(1e-3, 1, 500)), but the no-wiggle P_nw is the
NEW default GR wiggle-ratio approximation  pnw_MG = pnw_GR * plin_MG / plin_GR
(nw_method='gr_ratio') -- not folps get_pknow nor the original class's
peakaverage.

With the loop grid matched, this isolates the no-wiggle prescription so the
three matched-grid tests form a clean set against the original class
(peakaverage):
  - test_matched_grid.py        -> peakaverage  (same prescription as original)
  - test_matched_grid_folps.py  -> folps get_pknow
  - test_matched_grid_gr_ratio.py (this) -> GR wiggle ratio (new default)

Two plots: with P4 and without P4.  Same env / PYTHONPATH as the others.
"""

import os
import sys
os.environ.setdefault("FOLPS_BACKEND", "jax")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np

import compare_original_vs_emulator as C
import test_matched_grid as TM          # reuse make_plot + ELLS=(0,2,4)
from desilike.theories.galaxy_clustering import (
    MgEmulatorCosmology, fkpt_pkemu_TracerPowerSpectrumMultipoles)

ELLS = TM.ELLS


class MgEmulatorCosmologyMatchedGrRatio(MgEmulatorCosmology):
    """Emulated plin resampled onto the template loop grid, P_nw via the new
    GR wiggle-ratio approximation (nw_method='gr_ratio') on that grid."""

    TARGET_K = np.geomspace(1e-3, 1.0, 500)

    def calculate(self, logA, n_s, h, omega_b, omega_cdm,
                  mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4, **kwargs):
        kt = self.TARGET_K
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
            # GR wiggle-ratio pnw on the NATIVE wide grid (1e-4..10) via the
            # parent helper, then resample plin AND pnw onto the template grid.
            pnw_native = self._pknow(z, k, plin, logA, n_s, H0,
                                     omega_b, omega_cdm, h)
            plin_t = np.exp(np.interp(np.log(kt), np.log(k), np.log(plin)))
            pnw_t = np.exp(np.interp(np.log(kt), np.log(k),
                                     np.log(np.maximum(pnw_native, 1e-300))))
            chi_fid, e_fid = self._fid_scalars[z]
            qper = h * scalars['chi_z'] / (chi_fid * self.fiducial['h'])
            qpar = e_fid / scalars['e_z']
            self._results[z] = dict(
                k=kt, pk_dd=plin_t, pknow=pnw_t, Omega_m=Omega_m, h=h,
                sigma8=scalars['sigma8_z'], qper=qper, qpar=qpar,
                mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4, plist=plist)

    def get_at_z(self, z):
        return self._results[z]


def build_emulator_matched_gr_ratio():
    # nw_method defaults to 'gr_ratio' -> the GR emulators are loaded.
    prov = MgEmulatorCosmologyMatchedGrRatio(zs=[C.Z])
    for nm in C.MGNAMES:
        prov.init.params[nm].update(fixed=False)
    for nm, val in C.FID.items():
        prov.init.params[nm].update(value=val, fixed=True)
    theory = fkpt_pkemu_TracerPowerSpectrumMultipoles()
    theory.init.update(prior_basis="standard", tracer="LRG1", cosmo=prov,
                       z=C.Z, k=C.KOUT, ells=ELLS, beyond_eds=True)
    C._fix_bias(theory)
    return theory


def main():
    print("Building original ...")
    th_orig = C.build_original()
    print("Building emulator (matched grid + gr_ratio nw) ...")
    th_emu = build_emulator_matched_gr_ratio()

    results = []
    for label, mu, Sigma in C.CASES:
        print(f"  {label}")
        PA = C.evaluate(th_orig, mu, Sigma)
        PB = C.evaluate(th_emu, mu, Sigma)
        results.append((label, PA, PB))

    s_all = TM.make_plot(results, (0, 2, 4),
                         "test_matched_grid_gr_ratio_P0P2P4.png",
                         f"matched loop grid + gr_ratio nw: original vs emulator "
                         f"(z={C.Z}, AP=1)  — with P4")
    s_p02 = TM.make_plot(results, (0, 2),
                         "test_matched_grid_gr_ratio_P0P2.png",
                         f"matched loop grid + gr_ratio nw: original vs emulator "
                         f"(z={C.Z}, AP=1)  — P0,P2 only")

    print("\nMax |emulator - original| per case (matched grid, gr_ratio nw):")
    print(f"  {'case':26s} {'with P4':>9s} {'P0,P2 only':>12s}")
    for (label, wa), (_, wp) in zip(s_all, s_p02):
        print(f"  {label:26s} {wa:8.2f}% {wp:11.2f}%")


if __name__ == "__main__":
    main()
