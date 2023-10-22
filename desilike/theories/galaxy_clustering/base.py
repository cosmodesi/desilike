import numpy as np
from scipy import special

from desilike.jax import numpy as jnp
from desilike.jax import InterpolatedUnivariateSpline
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
        self.interp_order = int(interp_order)
        allowed_interp_order = [1, 2, 3]
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
        if kin is None: self.kin = np.geomspace(self.k[0], 0.6, int(300. / self.interp_order + 0.5))
        else: self.kin = np.array(kin, dtype='f8')
        self.power.init['k'] = self.kin
        mask = self.k > self.kin[-1]
        self.k_high = np.log10(self.k[mask] / self.kin[-1])
        self.pad_high = np.exp(-(self.k[mask] / self.kin[-1] - 1.)**2 / (2. * (10.)**2))
        self.k_mid = self.k[~mask]
        self.ells = self.power.ells
        from cosmoprimo import PowerToCorrelation
        self.fftlog = PowerToCorrelation(self.k, ell=self.ells, q=0, lowring=True)
        self.set_params()

    def set_params(self):
        self.power.init.params = self.init.params.copy()
        self.init.params.clear()

    def calculate(self):
        power = []
        for pk in self.power.power:
            slope_high = (pk[-1] - pk[-2]) / np.log10(self.kin[-1] / self.kin[-2])
            if self.interp_order == 1:
                interp = jnp.interp(np.log10(self.k_mid), np.log10(self.kin), pk)
            else:
                interp = InterpolatedUnivariateSpline(np.log10(self.kin), pk, k=self.interp_order)(np.log10(self.k_mid))
            power.append(jnp.concatenate([interp, (pk[-1] + slope_high * self.k_high) * self.pad_high], axis=-1))
        s, corr = self.fftlog(jnp.vstack(power))
        self.corr = jnp.array([jnp.interp(self.s, ss, cc) for ss, cc in zip(s, corr)])

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

    def to_poles(self, pkmu):
        return np.sum(pkmu * self.wmu[:, None, :], axis=-1)


class APEffect(BaseCalculator):
    """
    Alcock-Paczynski effect.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator, required only if ``mode`` is 'distances';
        defaults to ``Cosmoprimo(fiducial=fiducial)``.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    mode : str, default='distances'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)
        - 'distances': scaling parameters computed from the ratio of ``cosmo`` to ``fiducial`` cosmologies.

    eta : float, default=1./3.
        Relation between 'qpar', 'qper' and 'qiso', 'qap' parameters:
        ``qiso = qpar ** eta * qper ** (1 - eta)``.


    Reference
    ---------
    https://ui.adsabs.harvard.edu/abs/1979Natur.281..358A/abstract
    """
    config_fn = 'base.yaml'

    def initialize(self, z=1., cosmo=None, fiducial='DESI', mode='distances', eta=1. / 3.):
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
        elif self.mode == 'distances':
            self.params = self.params.clear()
            if is_external_cosmo(cosmo):
                self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
        else:
            raise ValueError('Unknown mode {}; it must be one of ["qiso", "qap", "qisoqap", "qparqper", "distances"]'.format(self.mode))
        self.cosmo = cosmo
        if self.mode == 'distances':
            if cosmo is None:
                self.cosmo = Cosmoprimo(fiducial=self.fiducial)
        else:
            self.cosmo = self.fiducial
        if self.mode == 'distances':
            self.DH_fid = (constants.c / 1e3) / (100. * self.fiducial.efunc(self.z))
            self.DM_fid = self.fiducial.comoving_angular_distance(self.z)
            self.DH_over_DM_fid = self.DH_fid / self.DM_fid
            self.DV_fid = (self.DH_fid * self.DM_fid**2 * self.z)**(1. / 3.)

    def calculate(self, **params):
        if self.mode == 'distances':
            self.DH = (constants.c / 1e3) / (100. * self.cosmo.efunc(self.z))
            self.DM = self.cosmo.comoving_angular_distance(self.z)
            self.DH_over_DM = self.DH / self.DM
            self.DV = (self.DH * self.DM**2 * self.z)**(1. / 3.)
            qpar, qper = self.DH / self.DH_fid, self.DM / self.DM_fid
        elif self.mode == 'qiso':
            qpar = qper = params['qiso']
        elif self.mode == 'qap':
            qap = params['qap']  # qpar / qper
            qpar, qper = qap**(1 - self.eta), qap**(-self.eta)
        elif self.mode == 'qisoqap':
            qiso, qap = params['qiso'], params['qap']  # qpar / qper
            qpar, qper = qiso * qap**(1 - self.eta), qiso * qap**(-self.eta)
        else:
            qpar, qper = params['qpar'], params['qper']
        self.qpar, self.qper = qpar, qper
        self.qap = self.qpar / self.qper
        self.qiso = self.qpar**self.eta * self.qper**(1. - self.eta)

    def ap_k_mu(self, k, mu):
        jac = 1. / (self.qpar * self.qper**2)
        factorap = jnp.sqrt(1 + mu**2 * (1. / self.qap**2 - 1))
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        kap = k[..., None] / self.qper * factorap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        muap = mu / self.qap / factorap
        return jac, kap, muap

    def ap_s_mu(self, s, mu):
        # Compared to Fourier space, qpar -> 1/qpar, qper -> 1/qper
        factorap = jnp.sqrt(1 + mu**2 * (self.qap**2 - 1))
        sap = s[..., None] * self.qper * factorap
        muap = mu * self.qap / factorap
        return 1., sap, muap
