"""
cosmologies.py
--------------
A drop-in "truth" cosmology provider, ``IsitgrCosmology``, that mirrors the
interface of ``desilike...full_shape.MgEmulatorCosmology`` but computes the
linear spectra + scalars **directly with ISiTGR** (the same code the emulator
was trained against), on the SAME k-grid as the emulator.

Feeding this provider to ``fkpt_pkemu_PowerSpectrumMultipoles`` instead of
``MgEmulatorCosmology`` reuses the *identical* fkpt loop (``Kfuncs_to_tables``)
and bias machinery, so a multipole comparison between the two isolates exactly
one thing: the emulator's accuracy in reproducing the ISiTGR linear inputs.

GR convention: ISiTGR binned muSigma has GR at mu_i = Sigma_i = 1 (verified
empirically — the spectrum is unchanged from LCDM there). mu_i = 0 is already
modified gravity.
"""

import numpy as np

from desilike.theories.galaxy_clustering.full_shape import (
    MgEmulatorCosmology, _MG_EMU_BINNING)

# Fixed ISiTGR settings used during emulator training (see data_generation_pk_mg.jl
# and testing_MG/test_mg.py).
_MNU, _TAU, _OMK = 0.06, 0.0568, 0.0


def _isitgr_plin_pnw_scalars(z, ln10As, ns, H0, ombh2, omch2,
                             mu, Sigma, k_grid):
    """ISiTGR reference plin, pnw, scalars on ``k_grid`` [h/Mpc].

    Ported from testing_MG/test_mg.py:compute_isitgr_plin_pnw_scalars, using
    the binned muSigma parameterisation with the emulator's training constants.
    """
    import isitgr
    import folps as folpsv2

    B = _MG_EMU_BINNING
    As = np.exp(ln10As) / 1e10

    pars = isitgr.CAMBparams()
    pars.set_cosmology(
        H0=H0, ombh2=ombh2, omch2=omch2, mnu=_MNU, omk=_OMK, tau=_TAU,
        MG_parameterization="muSigma",
        redshift_bins=True, scale_bins=bool(B["scale_bins"]),
        mu1=mu[0], mu2=mu[1], mu3=mu[2], mu4=mu[3],
        Sigma1=Sigma[0], Sigma2=Sigma[1], Sigma3=Sigma[2], Sigma4=Sigma[3],
        z_div=B["z_div"], z_TGR=B["z_TGR"], z_tw=B["z_tw"],
        k_c=B["k_c"], k_tw=B["k_tw"], k_TGR=B["k_TGR"], k_S=B["k_S"],
    )
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    # Pass z first so get_sigma8() returns [sigma8(z), sigma8(0)].
    pars.set_matter_power(redshifts=[z, 0.0], kmax=20.0)
    pars.NonLinear = isitgr.model.NonLinear_none

    results = isitgr.get_results(pars)
    k_out, _, pk_mat = results.get_matter_power_spectrum(
        minkh=1e-5, maxkh=20.0, npoints=1000,
        var1='delta_nonu', var2='delta_nonu',
    )
    plin = np.interp(k_grid, k_out, pk_mat[1])     # row 1 = z_sample
    # No-wiggle from folps' get_pknow (same as the emulator provider), so the
    # multipole comparison isolates the linear P(k) source.
    k_nw, pknow_ext = folpsv2.get_pknow(k=np.asarray(k_grid), pk=np.asarray(plin), h=H0 / 100.0)
    pnw = np.interp(np.asarray(k_grid), np.asarray(k_nw), np.asarray(pknow_ext))

    s8 = results.get_sigma8()
    scalars = dict(
        sigma8_z=float(s8[0]), sigma8_0=float(s8[1]),
        da_z=float(results.angular_diameter_distance(z)),
        chi_z=float(results.comoving_radial_distance(z)),
        e_z=float(results.hubble_parameter(z)) / H0,
    )
    return np.maximum(plin, 0.0), np.maximum(pnw, 0.0), scalars


class IsitgrCosmology(MgEmulatorCosmology):
    """ISiTGR-truth provider with the same interface as ``MgEmulatorCosmology``.

    Replaces the trained emulator with direct ISiTGR calls; everything
    downstream (AP factors, fkpt loop, bias) is identical.
    """

    def initialize(self, zs, fiducial=None, **kwargs):
        self.zs = list(zs)
        self.fiducial = dict(self.DEFAULT_FIDUCIAL)
        if fiducial is not None:
            self.fiducial.update(fiducial)
        # Use the emulator's native k-grid so both providers feed fkpt the
        # SAME k sampling (only the P(k) values differ).
        self._kgrid = np.load(f"{self.EMU_PLIN_PATH}/k.npy")
        self._build_fid_scalars()

    def _build_fid_scalars(self):
        fid = self.fiducial
        self._fid_scalars = {}
        for z in self.zs:
            _, _, sc = _isitgr_plin_pnw_scalars(
                z, fid['logA'], fid['n_s'], fid['h'] * 100.0,
                fid['omega_b'], fid['omega_cdm'],
                mu=[1.0, 1.0, 1.0, 1.0], Sigma=[1.0, 1.0, 1.0, 1.0],
                k_grid=self._kgrid)
            self._fid_scalars[z] = (sc['chi_z'], sc['e_z'])

    def calculate(self, logA, n_s, h, omega_b, omega_cdm,
                  mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4, **kwargs):
        H0 = h * 100.0
        Omega_m = (omega_b + omega_cdm + self.M_NCDM / 93.14) / h**2
        plist = [logA, n_s, H0, omega_b, omega_cdm, self.M_NCDM]
        mu, Sigma = [mu1, mu2, mu3, mu4], [Sigma1, Sigma2, Sigma3, Sigma4]

        self._results = {}
        for z in self.zs:
            plin, pnw, sc = _isitgr_plin_pnw_scalars(
                z, logA, n_s, H0, omega_b, omega_cdm, mu, Sigma, self._kgrid)
            chi_fid, e_fid = self._fid_scalars[z]
            qper = h * sc['chi_z'] / (chi_fid * self.fiducial['h'])
            qpar = e_fid / sc['e_z']
            self._results[z] = dict(
                k=np.asarray(self._kgrid), pk_dd=np.asarray(plin), pknow=np.asarray(pnw),
                Omega_m=Omega_m, h=h, sigma8=sc['sigma8_z'],
                qper=qper, qpar=qpar,
                mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4, plist=plist,
            )
