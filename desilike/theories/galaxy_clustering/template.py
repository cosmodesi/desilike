"""
BAO power spectrum template for galaxy clustering.

Classes
-------
CosmoprimoCosmology
    ExternalCalculator wrapping a cosmoprimo Boltzmann call with free cosmological parameters.
BAOSpectrum2Template
    Fiducial-cosmology BAO template: power spectra and AP distances computed once from
    cosmoprimo at compile time; scaled at evaluation time by free AP and growth-rate params.
ShapeFitSpectrum2Template
    BAO template with ShapeFit tilt parameterisation (dm, dn).
DirectSpectrum2Template
    Direct template: power spectrum computed at every evaluation from a
    :class:`CosmoprimoCosmology` dependency.
"""

import numpy as np
import jax.numpy as jnp

from ...base import Calculator, ExternalCalculator
from ...parameter import Parameter
from ..primordial_cosmology import CosmoprimoCosmology, _get_fiducial


_kw_pk = dict(extrap_kmin=1e-7, extrap_kmax=1e2)  # cosmoprimo pk_interpolator kwargs


# ── Base class ────────────────────────────────────────────────────────────────

class Spectrum2Template:
    """Marker base class for all 2-point power-spectrum template calculators.

    Subclassed by :class:`BAOSpectrum2Template`, :class:`ShapeFitSpectrum2Template`,
    and :class:`DirectSpectrum2Template` so that code can use ``isinstance`` checks
    rather than duck-typing on constructor kwargs.
    """


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

class BAOSpectrum2Template(Spectrum2Template, Calculator):
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

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

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
        from cosmoprimo import PowerSpectrumBAOFilter, constants

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
        DH_fid = constants.c / 1e3 / (100. * fid.efunc(z))
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
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk], {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk = children
        obj.k = aux['k']
        return obj


class ShapeFitSpectrum2Template(Spectrum2Template, Calculator):
    r"""
    ShapeFit power spectrum template.

    Multiplies the fiducial power spectrum by a k-dependent tilt factor controlled by ``dm`` and ``dn``.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to np.logspace(-3, 1, 400).
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology.
    with_now : str or False, default='peakaverage'
        Engine for the no-wiggle power spectrum ('peakaverage', 'wallish2018').
        Set to False to skip (pknow_dd is set equal to pk_dd).
    only_now : bool, default=False
        Replace pk_dd with pknow_dd so wiggles are absent.
    apmode : str, default='qparqper'
        AP parameterization: 'qparqper', 'qisoqap', 'qiso', or 'qap'.
    eta : float, default=1./3.
        Exponent in qiso = qpar**eta * qper**(1 - eta).
    kp : float, default=0.03
        Pivot wavenumber [h/Mpc] for the ShapeFit parameterization.
    a : float, default=0.6
        Steepness parameter in the ShapeFit tilt function.

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth (no-wiggle) power spectra.
    f, f0, fk : float or ndarray
        Growth rate.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    def __init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                 only_now=False, apmode='qparqper', eta=1. / 3., kp=0.03, a=0.6):
        _apmode = str(apmode)
        _ap_prior = dict(limits=[0.5, 2.])
        _ap_ref = dict(dist='norm', loc=1., scale=0.05)
        if _apmode == 'qparqper':
            self.qpar = Parameter('qpar', value=1., prior=_ap_prior, ref=_ap_ref, latex=r'q_\parallel')
            self.qper = Parameter('qper', value=1., prior=_ap_prior, ref=_ap_ref, latex=r'q_\perp')
        elif _apmode == 'qisoqap':
            self.qiso = Parameter('qiso', value=1., prior=_ap_prior, ref=_ap_ref, latex=r'q_\mathrm{iso}')
            self.qap = Parameter('qap', value=1., prior=_ap_prior, ref=_ap_ref, latex=r'q_\mathrm{ap}')
        elif _apmode == 'qiso':
            self.qiso = Parameter('qiso', value=1., prior=_ap_prior, ref=_ap_ref, latex=r'q_\mathrm{iso}')
        elif _apmode == 'qap':
            self.qap = Parameter('qap', value=1., prior=_ap_prior, ref=_ap_ref, latex=r'q_\mathrm{ap}')
        else:
            raise ValueError(f"apmode must be one of 'qparqper', 'qisoqap', 'qiso', 'qap'; got {apmode!r}")
        self.df = Parameter('df', value=1., prior=dict(limits=[0., 2.]),
                            ref=dict(dist='norm', loc=1., scale=0.05), latex=r'\delta f')
        self.dm = Parameter('dm', value=0., prior=dict(limits=[-0.5, 0.5]),
                            ref=dict(dist='norm', loc=0., scale=0.05), latex=r'\delta m')
        self.dn = Parameter('dn', value=0., fixed=True, prior=dict(limits=[-0.5, 0.5]),
                            ref=dict(dist='norm', loc=0., scale=0.05), latex=r'\delta n')

    def __post_init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                      only_now=False, apmode='qparqper', eta=1. / 3., kp=0.03, a=0.6):
        from cosmoprimo import PowerSpectrumBAOFilter

        self._apmode = str(apmode)
        self._eta = float(eta)
        self._only_now = bool(only_now)
        self._kp = float(kp)
        self._a = float(a)

        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)

        fid = _get_fiducial(fiducial)
        self._fiducial = fid

        fo = fid.get_fourier()
        sigma8 = fo.sigma8_z(z, of='delta_cb')
        fsigma8 = fo.sigma8_z(z, of='theta_cb')
        self._f_fid = float(fsigma8 / sigma8)

        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=z)
        ptt_interp = fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=z)

        k0 = 1e-3
        self._f0_fid = float(np.sqrt(ptt_interp(k0) / pk_interp(k0)))
        self._fk_fid = np.sqrt(ptt_interp(self.k) / pk_interp(self.k))
        self._pk_dd_fid = pk_interp(self.k)

        if with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=fid, cosmo_fid=fid)
            self._pknow_dd_fid = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self._pknow_dd_fid = self._pk_dd_fid

    def _qpar_qper(self):
        if self._apmode == 'qparqper':
            return self.qpar.value, self.qper.value
        if self._apmode == 'qiso':
            q = self.qiso.value
            return q, q
        if self._apmode == 'qap':
            qap = self.qap.value
            return qap ** (1. - self._eta), qap ** (-self._eta)
        qiso, qap = self.qiso.value, self.qap.value
        return qiso * qap ** (1. - self._eta), qiso * qap ** (-self._eta)

    def ap_k_mu(self, k, mu):
        """Apply AP distortion to a (k, mu) grid; returns (jac, kap, muap)."""
        qpar, qper = self._qpar_qper()
        return _ap_k_mu(k, mu, qpar, qper)

    def __call__(self):
        dm = self.dm.value
        dn = self.dn.value
        df = self.df.value
        factor = jnp.exp(dm / self._a * jnp.tanh(self._a * jnp.log(self.k / self._kp))
                         + dn * jnp.log(self.k / self._kp))
        self.pk_dd = self._pk_dd_fid * factor
        self.pknow_dd = self._pknow_dd_fid * factor
        if self._only_now:
            self.pk_dd = self.pknow_dd
        self.f = self._f_fid * df
        self.f0 = self._f0_fid * df
        self.fk = self._fk_fid * df
        return self.pk_dd

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk], {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk = children
        obj.k = aux['k']
        return obj


class DirectSpectrum2Template(Spectrum2Template, ExternalCalculator):
    r"""
    Direct power spectrum template: power spectrum evaluated at each pipeline call from a
    :class:`CosmoprimoCosmology` dependency.

    AP parameters (qpar, qper) are computed from the ratio of current to fiducial distances.
    By default a :class:`CosmoprimoCosmology` calculator is created internally; an existing
    instance may be passed via ``cosmo`` to share cosmological parameters across theories.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to ``np.logspace(-3, 1, 400)``.
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology for AP ratio denominator and fiducial PK/no-wiggle PK.
    engine : str, default='camb'
        Boltzmann solver engine forwarded to the internal :class:`CosmoprimoCosmology`
        (ignored when ``cosmo`` is supplied).
    with_now : str or False, default=False
        No-wiggle filter engine ('peakaverage', 'wallish2018'); ``False`` to skip.
    only_now : bool, default=False
        Replace ``pk_dd`` with ``pknow_dd`` so wiggles are absent.
    cosmo : CosmoprimoCosmology or None, default=None
        External cosmology calculator to use as a dep.  When ``None`` a fresh
        :class:`CosmoprimoCosmology` is created with the given ``engine`` and
        ``fiducial`` defaults.

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth (no-wiggle) power spectra.
    f, f0, fk : float or ndarray
        Growth rate.
    sigma8, fsigma8 : float
        Normalisation and growth-rate normalisation.
    qpar, qper : float
        AP distortion ratios (current / fiducial distances).
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    def __init__(self, k=None, z=1., fiducial='DESI', engine='camb', with_now=False, only_now=False, cosmo=None):
        # Nodes: the CosmoprimoCosmology dep goes in __init__.
        if cosmo is None:
            cosmo = CosmoprimoCosmology(engine=engine, fiducial=fiducial)
        self.cosmo = cosmo

    def __post_init__(self, k=None, z=1., fiducial='DESI', engine='camb', with_now=False, only_now=False, cosmo=None):
        # Non-node setup: fiducial distances and fiducial PK (fixed at compile time).
        from cosmoprimo import PowerSpectrumBAOFilter, constants
        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)
        self._with_now = with_now
        self._only_now = bool(only_now)

        fid = _get_fiducial(fiducial)
        self._fiducial = fid
        self._DH_fid = float(constants.c / 1e3 / (100. * fid.efunc(self.z)))
        self._DM_fid = float(fid.comoving_angular_distance(self.z))

        # Fiducial PK arrays (used by e.g. ResummedBAOWigglesPTSpectrum2Poles for damping scales).
        fo = fid.get_fourier()
        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=self.z)
        self._pk_dd_fid = pk_interp(self.k)
        if with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=fid, cosmo_fid=fid)
            self._pknow_dd_fid = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self._pknow_dd_fid = self._pk_dd_fid

    def __call__(self):
        from cosmoprimo import PowerSpectrumBAOFilter, constants
        # self.cosmo.cosmo is the populated cosmoprimo.Cosmology from the dep's __call__.
        cosmo = self.cosmo.cosmo
        fo = cosmo.get_fourier()
        pk_interp  = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=self.z)
        ptt_interp = fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=self.z)

        self.pk_dd = pk_interp(self.k)
        if self._with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=self._with_now,
                                                cosmo=cosmo, cosmo_fid=self._fiducial)
            self.pknow_dd = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self.pknow_dd = self.pk_dd.copy()
        if self._only_now:
            self.pk_dd = self.pknow_dd

        sigma8  = float(fo.sigma8_z(self.z, of='delta_cb'))
        fsigma8 = float(fo.sigma8_z(self.z, of='theta_cb'))
        self.sigma8  = sigma8
        self.fsigma8 = fsigma8
        self.f = fsigma8 / sigma8
        k0 = 1e-3
        self.f0 = float(np.sqrt(ptt_interp(k0) / pk_interp(k0)))
        self.fk = np.sqrt(ptt_interp(self.k) / pk_interp(self.k))

        DH = float(constants.c / 1e3 / (100. * cosmo.efunc(self.z)))
        DM = float(cosmo.comoving_angular_distance(self.z))
        self.qpar = DH / self._DH_fid
        self.qper = DM / self._DM_fid
        return self.pk_dd

    def ap_k_mu(self, k, mu):
        """Apply AP distortion; works in JAX context after tree_unflatten."""
        return _ap_k_mu(k, mu, self.qpar, self.qper)

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk,
                 self.qpar, self.qper, self.sigma8, self.fsigma8],
                {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk, obj.qpar, obj.qper, obj.sigma8, obj.fsigma8 = children
        obj.k = aux['k']
        return obj
