"""
BAO power spectrum template for galaxy clustering.

Classes
-------
BAOSpectrum2Template
    Fiducial-cosmology BAO template: power spectra and AP distances computed once from
    cosmoprimo at compile time; scaled at evaluation time by free AP and growth-rate params.
"""

import numpy as np
import jax.numpy as jnp

from ...base import Calculator
from ...parameter import Parameter


_c_kms = 299792.458  # speed of light [km/s]
_kw_pk = dict(extrap_kmin=1e-7, extrap_kmax=1e2)  # cosmoprimo pk_interpolator kwargs


# ── cosmoprimo helper ─────────────────────────────────────────────────────────

def _get_fiducial(fiducial):
    """Return a cosmoprimo Cosmology from a name string, (name, kwargs) tuple, dict, or Cosmology."""
    import cosmoprimo
    if fiducial is None:
        raise ValueError('fiducial cosmology is required')
    # duck-type: already a Cosmology-like object
    if hasattr(fiducial, 'get_fourier'):
        return fiducial
    if isinstance(fiducial, str):
        fiducial = (fiducial, {})
    if isinstance(fiducial, tuple):
        name, kw = fiducial
        return getattr(cosmoprimo.fiducial, name)(**kw)
    if isinstance(fiducial, dict):
        return cosmoprimo.Cosmology(**fiducial)
    raise ValueError(f'Cannot parse fiducial cosmology: {fiducial!r}')


# ── AP distortion ─────────────────────────────────────────────────────────────

def _ap_k_mu(k, mu, qpar, qper):
    """Alcock-Paczynski distortion of (k, mu) grid.

    k   : (...,) or broadcastable
    mu  : (...,) or broadcastable
    Returns (jac, kap, muap) of same broadcast shape.
    """
    qap = qpar / qper
    jac = 1. / (qpar * qper**2)
    factorap = jnp.sqrt(1. + mu**2 * (1. / qap**2 - 1.))
    kap = k / qper * factorap
    muap = mu / qap / factorap
    return jac, kap, muap


# ── BAO template ──────────────────────────────────────────────────────────────

class BAOSpectrum2Template(Calculator):
    r"""
    BAO power spectrum template based on a fixed fiducial cosmology.

    The fiducial power spectra, growth rates, and BAO distances are computed once from
    cosmoprimo at compile time (``__post_init__``). At evaluation time (``__call__``),
    power spectra are copied from fiducial arrays and the growth rate and distances are
    scaled by the free parameters.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to np.logspace(-3, 1, 400).
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology. A string is looked up as ``cosmoprimo.fiducial.<name>()``.
    with_now : str or False, default='peakaverage'
        Engine for the BAO-filtered smooth power spectrum ('peakaverage', 'wallish2018').
        Set to False to skip (pknow_dd is set equal to pk_dd).
    only_now : bool, default=False
        Replace pk_dd with pknow_dd so wiggles are absent from the model.
    apmode : str, default='qparqper'
        AP parameterization. One of:

        - 'qparqper': free parameters ``qpar`` (LOS scaling) and ``qper`` (transverse scaling).
        - 'qisoqap':  free parameters ``qiso`` and ``qap = qpar / qper``.
        - 'qiso':     single isotropic parameter ``qiso``.
        - 'qap':      single AP parameter ``qap``.
    eta : float, default=1./3.
        Exponent in  qiso = qpar**eta * qper**(1 - eta).

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth (no-wiggle) power spectra at ``self.k``.
    f, f0, fk : float or ndarray
        Growth rate f = d ln D / d ln a;  f0 is the k->0 limit, fk is k-dependent.
    DH_over_rd, DM_over_rd, DV_over_rd : float
        BAO distance ratios scaled by the AP parameters.
    """

    def __init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                 only_now=False, apmode='qparqper', eta=1. / 3.):
        # AP parameters — created here (Option B) so they appear in __dict__ for graph scan.
        _apmode = str(apmode)
        _ap_prior = dict(limits=[0.5, 2.])
        _ap_ref = dict(dist='norm', loc=1., scale=0.05)
        if _apmode == 'qparqper':
            self.qpar = Parameter('qpar', value=1., prior=_ap_prior, ref=_ap_ref,
                                  latex=r'q_\parallel')
            self.qper = Parameter('qper', value=1., prior=_ap_prior, ref=_ap_ref,
                                  latex=r'q_\perp')
        elif _apmode == 'qisoqap':
            self.qiso = Parameter('qiso', value=1., prior=_ap_prior, ref=_ap_ref,
                                  latex=r'q_\mathrm{iso}')
            self.qap = Parameter('qap', value=1., prior=_ap_prior, ref=_ap_ref,
                                 latex=r'q_\mathrm{ap}')
        elif _apmode == 'qiso':
            self.qiso = Parameter('qiso', value=1., prior=_ap_prior, ref=_ap_ref,
                                  latex=r'q_\mathrm{iso}')
        elif _apmode == 'qap':
            self.qap = Parameter('qap', value=1., prior=_ap_prior, ref=_ap_ref,
                                 latex=r'q_\mathrm{ap}')
        else:
            raise ValueError(f"apmode must be one of 'qparqper', 'qisoqap', 'qiso', 'qap'; got {apmode!r}")
        self.df = Parameter('df', value=1., prior=dict(limits=[0., 2.]),
                            ref=dict(dist='norm', loc=1., scale=0.05), latex=r'\delta f')

    def __post_init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                      only_now=False, apmode='qparqper', eta=1. / 3.):
        from cosmoprimo import PowerSpectrumBAOFilter

        self._apmode = str(apmode)
        self._eta = float(eta)
        self._only_now = bool(only_now)

        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)

        fid = _get_fiducial(fiducial)
        self._fiducial = fid  # kept so downstream can read e.g. fid.rs_drag

        fo = fid.get_fourier()
        sigma8 = fo.sigma8_z(z, of='delta_cb')
        fsigma8 = fo.sigma8_z(z, of='theta_cb')
        self._f_fid = float(fsigma8 / sigma8)

        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=z)
        ptt_interp = fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=z)

        k0 = 1e-3  # low-k limit for f0
        self._f0_fid = float(np.sqrt(ptt_interp(k0) / pk_interp(k0)))
        self._fk_fid = np.sqrt(ptt_interp(self.k) / pk_interp(self.k))
        self._pk_dd_fid = pk_interp(self.k)

        if with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=fid, cosmo_fid=fid)
            self._pknow_dd_fid = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self._pknow_dd_fid = self._pk_dd_fid

        # Fiducial BAO distance ratios
        rd = fid.rs_drag
        DH_fid = _c_kms / (100. * fid.efunc(z))
        DM_fid = fid.comoving_angular_distance(z)
        DV_fid = DH_fid**eta * DM_fid**(1. - eta) * z**(1. / 3.)
        self._DH_over_rd_fid = float(DH_fid / rd)
        self._DM_over_rd_fid = float(DM_fid / rd)
        self._DV_over_rd_fid = float(DV_fid / rd)

    def _qpar_qper(self):
        """Convert current apmode parameter values to (qpar, qper)."""
        if self._apmode == 'qparqper':
            return self.qpar.value, self.qper.value
        if self._apmode == 'qiso':
            q = self.qiso.value
            return q, q
        if self._apmode == 'qap':
            qap = self.qap.value
            return qap ** (1. - self._eta), qap ** (-self._eta)
        # qisoqap
        qiso, qap = self.qiso.value, self.qap.value
        return qiso * qap ** (1. - self._eta), qiso * qap ** (-self._eta)

    def ap_k_mu(self, k, mu):
        """Apply AP distortion to a (k, mu) grid; returns (jac, kap, muap)."""
        qpar, qper = self._qpar_qper()
        return _ap_k_mu(k, mu, qpar, qper)

    def __call__(self):
        # Power spectra: fixed at fiducial (no cosmo call at eval time).
        self.pk_dd = self._pk_dd_fid
        self.pknow_dd = self._pknow_dd_fid
        if self._only_now:
            self.pk_dd = self._pknow_dd_fid

        # Growth rate scaled by df.
        df = self.df.value
        self.f = self._f_fid * df
        self.f0 = self._f0_fid * df
        self.fk = self._fk_fid * df

        # BAO distances scaled by AP parameters.
        qpar, qper = self._qpar_qper()
        self.DH_over_rd = qpar * self._DH_over_rd_fid
        self.DM_over_rd = qper * self._DM_over_rd_fid
        self.DV_over_rd = qpar ** self._eta * qper ** (1. - self._eta) * self._DV_over_rd_fid

        return self.pk_dd

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd,
                 jnp.asarray(self.f), jnp.asarray(self.f0), jnp.asarray(self.fk)],
                {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk = children
        obj.k = aux['k']
        return obj
