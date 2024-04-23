import numpy as np
from scipy import special

from desilike.jax import numpy as jnp
from desilike.jax import interp1d, jit
from desilike.cosmo import is_external_cosmo
from desilike.theories.primordial_cosmology import get_cosmo, Cosmoprimo, constants
from desilike.base import BaseCalculator
from desilike import plotting, utils


class BaseTheoryPowerSpectrumMultipoles(BaseCalculator):

    """Base class for theory power spectrum multipoles."""

    def initialize(self, k=None, ells=(0, 2, 4)):
        if k is None: k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BaseTheoryCorrelationFunctionMultipoles(BaseCalculator):

    """Base class for theory correlation function multipoles."""

    def initialize(self, s=None, ells=(0, 2, 4)):
        if s is None: s = np.linspace(20., 200, 101)
        self.s = np.array(s, dtype='f8')
        self.ells = tuple(ells)

    def __getstate__(self):
        state = {}
        for name in ['s', 'z', 'ells', 'corr', 'fiducial']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    """Base class for theory correlation function from power spectrum multipoles."""
    _initialize_with_namespace = True

    def initialize(self, s=None, power=None, interp_order=1, **kwargs):
        if s is None: s = np.linspace(20., 200, 101)
        self.s = np.array(s, dtype='f8')
        self.interp_order = {'linear': 1, 'cubic': 3}.get(interp_order, interp_order)
        allowed_interp_order = [1, 3]
        if self.interp_order not in allowed_interp_order:
            raise ValueError('interp_order must be one of {}'.format(allowed_interp_order))
        if power is None:
            from .full_shape import KaiserTracerPowerSpectrumMultipoles
            power = KaiserTracerPowerSpectrumMultipoles()
        self.power = power
        self.k = np.logspace(-4., 3., 2048)
        self.power.init.update(**kwargs)
        kin = self.power.init.get('k', None)
        # Important to have high enough sampling, otherwise wiggles can be seen at small s
        if kin is None: self.kin = np.geomspace(self.k[0], 0.6, int(300. / self.interp_order + 0.5))  # kmax = 1. may be better
        else: self.kin = np.array(kin, dtype='f8')
        self.power.init['k'] = self.kin
        mask = self.k > self.kin[-1]
        self.logk_high = np.log10(self.k[mask] / self.kin[-1])
        self.damp_high = np.exp(-(self.k[mask] / self.kin[-1] - 1.)**2 / (2. * (10.)**2))
        #self.k_high = self.k[mask] / self.kin[-1]
        #self.damp = np.exp(-(self.k / 10.)**2)
        self.k_mid = self.k[~mask]
        self.ells = self.power.ells
        from cosmoprimo import PowerToCorrelation
        self.fftlog = PowerToCorrelation(self.k, ell=self.ells, q=0, lowring=True)
        self.set_params()

    def set_params(self):
        self.power.init.params = self.init.params.copy()
        self.init.params.clear()

    # Below several methods for pk -> xi. In the end, differences do not matter for s > 20 Mpc / h.
    """
    def get_corr(self, power):
        tmp = []
        print(power[0].sum(), power[1].sum(), power[2].sum(), self.kin[0], self.kin[-1], self.kin.shape)
        for pk in power:
            slope_high = np.log10(np.abs(pk[-1] / pk[-2])) / np.log10(self.kin[-1] / self.kin[-2])
            r = -2
            from scipy.misc import derivative
            from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
            pki = interpolate(self.kin, pk, k=5)
            slope_high2 = derivative(pki, self.kin[r], dx=self.kin[r] * 1e-6, order=9) * self.kin[r] / pk[r]
            print(slope_high, slope_high2)
            interp = interp1d(np.log10(self.k_mid), np.log10(self.kin), pk, method=self.interp_order)
            #tmp.append(jnp.concatenate([interp, (pk[-1] + slope_high * self.k_high) * self.damp_high], axis=-1))
            tmp.append(jnp.concatenate([interp, pk[-1] * self.k_high**slope_high2], axis=-1) * self.damp)
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for tt in tmp:
            ax.loglog(self.k, tt)
        plt.show()
        #s, corr = self.fftlog(jnp.vstack(tmp))
        from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SphericalBesselTransformNP
        sphr = SphericalBesselTransformNP(self.k, L=5, fourier=True)
        ss0, xi0 = sphr.sph(0, tmp[0])
        ss2, xi2 = sphr.sph(2, tmp[1]); xi2 *= -1
        ss4, xi4 = sphr.sph(4, tmp[2])
        s = [ss0, ss2, ss4]
        corr = [xi0, xi2, xi4]
        return jnp.array([jnp.interp(self.s, ss, cc) for ss, cc in zip(s, corr)])

    @jit(static_argnums=[0])
    def get_corr(self, power):
        tmp = []
        for pk in power:
            slope_high = jnp.log10(jnp.abs(pk[-1] / pk[-2])) / np.log10(self.kin[-1] / self.kin[-2])
            interp = interp1d(np.log10(self.k_mid), np.log10(self.kin), pk, method=self.interp_order)
            #tmp.append(jnp.concatenate([interp, (pk[-1] + slope_high * self.k_high) * self.damp_high], axis=-1))
            #print(np.array(pk[-1]), np.array(slope_high))
            tmp.append(jnp.concatenate([interp, pk[-1] * self.k_high**slope_high], axis=-1) * self.damp)
        s, corr = self.fftlog(jnp.vstack(tmp))
        return jnp.array([jnp.interp(self.s, ss, cc) for ss, cc in zip(s, corr)])
    """
    @jit(static_argnums=[0])
    def get_corr(self, power):  # least terrible solution, others fail when pk2[-2] ~ 0 and pk2[-1] < 0
        tmp = []
        for pk in power:
            slope_high = (pk[-1] - pk[-2]) / np.log10(self.kin[-1] / self.kin[-2])
            interp = interp1d(np.log10(self.k_mid), np.log10(self.kin), pk, method=self.interp_order)
            tmp.append(jnp.concatenate([interp, (pk[-1] + slope_high * self.logk_high) * self.damp_high], axis=-1))
            #tmp.append(jnp.concatenate([interp, (pk[-1] + slope_high * self.logk_high)], axis=-1) * self.damp)
        s, corr = self.fftlog(jnp.vstack(tmp))
        return jnp.array([jnp.interp(self.s, ss, cc) for ss, cc in zip(s, corr)])

    def calculate(self):
        self.corr = self.get_corr(self.power.power)

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot comparison to brute-force (non-fftlog) computation.
        We see convergence towards brute-force when decreasing damping sigma.
        Difference between fftlog and brute-force comes from the effect of truncation / damping.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1 + len(self.ells)`` axes.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        corr = []
        weights = utils.weights_trapz(np.log(self.kin))
        for ill, ell in enumerate(self.ells):
            # Integration in log, adding a k
            tmp = np.sum(self.kin**3 * self.power.power[ill] * weights * special.spherical_jn(ell, self.s[:, None] * self.kin), axis=-1)
            corr.append((-1) ** (ell // 2) / (2. * np.pi**2) * tmp)
        from matplotlib import pyplot as plt
        if fig is None:
            height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
            figsize = (6, 1.5 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
            fig.subplots_adjust(hspace=0)
        else:
            lax = fig.axes
        lax[0].plot([], [], linestyle='-', color='k', label='fftlog')
        lax[0].plot([], [], linestyle='--', color='k', label='brute-force')
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            lax[0].plot(self.s, self.s**2 * self.corr[ill], color=color, linestyle='-', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.s, self.s**2 * corr[ill], linestyle='--', color=color)
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.s, self.s**2 * (self.corr[ill] - corr[ill]), color='C{:d}'.format(ill))
            lax[ill + 1].set_ylabel(r'$\Delta s^{{2}}\xi_{{{0:d}}}$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        lax[0].set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig


class BaseTheoryPowerSpectrumMultipolesFromWedges(BaseTheoryPowerSpectrumMultipoles):

    """Base class for theory correlation function multipoles computed from theory power spectrum multipoles."""

    def initialize(self, *args, mu=20, method='leggauss', **kwargs):
        super(BaseTheoryPowerSpectrumMultipolesFromWedges, self).initialize(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, method=method, ells=self.ells)

    def set_k_mu(self, k, mu=20, method='leggauss', ells=(0, 2, 4)):
        self.k = np.asarray(k, dtype='f8')
        self.mu, wmu = utils.weights_mu(mu, method=method)
        self.wmu = np.array([wmu * (2 * ell + 1) * special.legendre(ell)(self.mu) for ell in ells])

    @jit(static_argnums=[0])
    def to_poles(self, pkmu):
        return jnp.sum(pkmu * self.wmu[:, None, :], axis=-1)


@jit
def ap_k_mu(k, mu, qpar=1., qper=1.):
    qap = qpar / qper
    jac = 1. / (qpar * qper**2)
    factorap = jnp.sqrt(1 + mu**2 * (1. / qap**2 - 1))
    # Beutler 2016 (arXiv: 1607.03150v1) eq 44
    kap = k[..., None] / qper * factorap
    # Beutler 2016 (arXiv: 1607.03150v1) eq 45
    muap = mu / qap / factorap
    return jac, kap, muap


@jit
def ap_s_mu(s, mu, qpar=1., qper=1.):
    qap = qpar / qper
    # Compared to Fourier space, qpar -> 1/qpar, qper -> 1/qper
    factorap = jnp.sqrt(1 + mu**2 * (qap**2 - 1))
    sap = s[..., None] * qper * factorap
    muap = mu * qap / factorap
    return 1., sap, muap


class APEffect(BaseCalculator):
    """
    Alcock-Paczynski effect.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator, required only if ``mode`` is 'geometry' or 'bao';
        defaults to ``Cosmoprimo(fiducial=fiducial)``.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    mode : str, default='geometry'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)
        - 'geometry': scaling parameters computed from the ratio of ``cosmo`` to ``fiducial`` cosmology distances
        - 'bao': scaling parameters computed from the ratio of ``cosmo`` to ``fiducial`` cosmology distances, normalized by the :math:`r_{\mathrm{drag}}` coordinates.

    eta : float, default=1. / 3.
        Relation between 'qpar', 'qper' and 'qiso', 'qap' parameters:
        ``qiso = qpar ** eta * qper ** (1 - eta)``.


    Reference
    ---------
    https://ui.adsabs.harvard.edu/abs/1979Natur.281..358A/abstract
    """
    config_fn = 'base.yaml'

    def initialize(self, z=1., cosmo=None, fiducial='DESI', mode='geometry', eta=1. / 3.):
        self.z = float(z)
        if fiducial is None:
            raise ValueError('Provide fiducial cosmology')
        self.fiducial = get_cosmo(fiducial)
        self.eta = float(eta)
        self.efunc_fid = self.fiducial.efunc(self.z)
        self.DM_fid = self.fiducial.comoving_angular_distance(self.z)
        self.mode = mode
        self.cosmo_requires = {}
        if self.mode == 'qiso':
            self.params = self.params.select(basename=['qiso'])
        elif self.mode == 'qap':
            self.params = self.params.select(basename=['qap'])
        elif self.mode == 'qisoqap':
            self.params = self.params.select(basename=['qiso', 'qap'])
        elif self.mode == 'qparqper':
            self.params = self.params.select(basename=['qpar', 'qper'])
        elif self.mode == 'qisobeta':
            self.params = self.params.select(basename=['qiso', 'betaphi'])
        elif self.mode == 'qparqperbeta': 
            self.params = self.params.select(basename=['qpar', 'qper', 'betaphi'])
        elif self.mode in ['geometry', 'bao']:
            self.params = self.params.clear()
            if is_external_cosmo(cosmo):
                self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
                if self.mode == 'bao': self.cosmo_requires['thermodynamics'] = {'rs_drag': None}
        else:
            raise ValueError('unknown mode {}; it must be one of ["qiso", "qap", "qisoqap", "qparqper", "qisobeta", "qparqperbeta", "geometry", "bao"]'.format(self.mode))
        self.cosmo = cosmo
        if self.mode in ['geometry', 'bao']:
            if cosmo is None:
                self.cosmo = Cosmoprimo(fiducial=self.fiducial)
        else:
            self.cosmo = self.fiducial
        if self.mode in ['geometry', 'bao']:
            self.DH_fid = (constants.c / 1e3) / (100. * self.fiducial.efunc(self.z))
            self.DM_fid = self.fiducial.comoving_angular_distance(self.z)
            self.DH_over_DM_fid = self.DH_fid / self.DM_fid
            self.DV_fid = (self.DH_fid * self.DM_fid**2 * self.z)**(1. / 3.)
            if self.mode == 'bao':
                rs_drag_fid = self.fiducial.rs_drag
                self.DH_over_rd_fid = self.DH_fid / rs_drag_fid
                self.DM_over_rd_fid = self.DM_fid / rs_drag_fid
                self.DV_over_rd_fid = self.DV_fid / rs_drag_fid

    def calculate(self, **params):
        if self.mode in ['geometry', 'bao']:
            self.DH = (constants.c / 1e3) / (100. * self.cosmo.efunc(self.z))
            self.DM = self.cosmo.comoving_angular_distance(self.z)
            self.DH_over_DM = self.DH / self.DM
            self.DV = (self.DH * self.DM**2 * self.z)**(1. / 3.)
            if self.mode == 'bao':
                rs_drag = self.cosmo.rs_drag
                self.DH_over_rd = self.DH / rs_drag
                self.DM_over_rd = self.DM / rs_drag
                self.DV_over_rd = self.DV / rs_drag
                qpar, qper = self.DH_over_rd / self.DH_over_rd_fid, self.DM_over_rd / self.DM_over_rd_fid
            else:  # geometry
                qpar, qper = self.DH / self.DH_fid, self.DM / self.DM_fid
        elif self.mode == 'qiso':
            qpar = qper = params['qiso']
        elif self.mode == 'qap':
            qap = params['qap']  # qpar / qper
            qpar, qper = qap**(1 - self.eta), qap**(-self.eta)
        elif self.mode == 'qisoqap':
            qiso, qap = params['qiso'], params['qap']  # qpar / qper
            qpar, qper = qiso * qap**(1 - self.eta), qiso * qap**(-self.eta)
        elif self.mode == 'qisobeta':
            qap, betaphi = params['qiso'], params['betaphi']
        elif self.mode == 'qparperbeta': 
            qpar, qper, betaphi = params['qpar'], params['qper'], params['betaphi']
        else:
            qpar, qper = params['qpar'], params['qper']
        self.qpar, self.qper = qpar, qper
        self.qap = self.qpar / self.qper
        self.qiso = self.qpar**self.eta * self.qper**(1. - self.eta)
        self.betaphi = betaphi if self.mode in ['qisobeta','qparperbeta'] else 1.0 
        
    def ap_k_mu(self, k, mu):
        return ap_k_mu(k, mu, qpar=self.qpar, qper=self.qper)

    def ap_s_mu(self, s, mu):
        return ap_s_mu(s, mu, qpar=self.qpar, qper=self.qper)

    def ap_beta(self):
        return self.betaphi