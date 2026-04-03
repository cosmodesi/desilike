import numpy as np
from scipy import special

from desilike.jax import numpy as jnp
from desilike.jax import interp1d, jit
from desilike.cosmo import is_external_cosmo
from desilike.theories.primordial_cosmology import get_cosmo, Cosmoprimo, constants
from desilike.base import BaseCalculator
from desilike import plotting, utils
from desilike.utils import BaseClass


class SpectrumToCorrelationMultipoles(BaseClass):

    """Helper class to compute correlation function multipoles from power spectrum multipoles using FFTLog."""

    def __init__(self, s=None, spectrum=None, interp_order=1):
        """
        Initialize the class.

        Parameters
        ----------
        s : array-like, default=None
            Array of scales where to compute correlation function multipoles.
        spectrum : Base Calculator, default=None
            Power spectrum multipoles calculator, required to get the multipole orders and k sampling.
             If None, it is assumed that the power spectrum multipoles will be provided with the same k sampling as self.k.
        interp_order : str or int, default=1
            Interpolation order for extrapolation of power spectrum multipoles at high k.
        """
        self.s = np.array(s, dtype='f8')
        self.interp_order = {'linear': 1, 'cubic': 3}.get(interp_order, interp_order)
        allowed_interp_order = [1, 3]
        if self.interp_order not in allowed_interp_order:
            raise ValueError('interp_order must be one of {}'.format(allowed_interp_order))
        self.k = np.logspace(-4., 3., 2048)
        kin = spectrum.init.get('k', None)
        # Important to have high enough sampling, otherwise wiggles can be seen at small s
        if kin is None:
            self.kin = np.geomspace(self.k[0], 0.6, int(300. / self.interp_order + 0.5))  # kmax = 1. may be better
        else: self.kin = np.array(kin, dtype='f8')
        spectrum.init['k'] = self.kin
        mask = self.k > self.kin[-1]
        self.logk_high = np.log10(self.k[mask] / self.kin[-1])
        self.damp_high = np.exp(-(self.k[mask] / self.kin[-1] - 1.)**2 / (2. * (10.)**2))
        #self.k_high = self.k[mask] / self.kin[-1]
        #self.damp = np.exp(-(self.k / 10.)**2)
        self.k_mid = self.k[~mask]
        self.ells = spectrum.ells
        from cosmoprimo import PowerToCorrelation
        self.fftlog = PowerToCorrelation(self.k, ell=self.ells, q=0, lowring=True)

    def __call__(self, poles):
        tmp = []
        for pole in poles:
            slope_high = (pole[-1] - pole[-2]) / np.log10(self.kin[-1] / self.kin[-2])
            interp = interp1d(np.log10(self.k_mid), np.log10(self.kin), pole, method=self.interp_order)
            tmp.append(jnp.concatenate([interp, (pole[-1] + slope_high * self.logk_high) * self.damp_high], axis=-1))
            #tmp.append(jnp.concatenate([interp, (pole[-1] + slope_high * self.logk_high)], axis=-1) * self.damp)
        s, corr = self.fftlog(jnp.vstack(tmp))
        return jnp.array([jnp.interp(self.s, ss, cc) for ss, cc in zip(s, corr)])

    @plotting.plotter
    def plot(self, poles, fig=None):
        """
        Plot comparison to brute-force (non-fftlog) computation.
        We see convergence towards brute-force when decreasing damping sigma.
        Difference between fftlog and brute-force comes from the effect of truncation / damping.

        Parameters
        ----------
        poles : list of arrays
            List of power spectrum multipoles, in the same order as :attr:`ells`.
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
            tmp = np.sum(self.kin**3 * poles[ill] * weights * special.spherical_jn(ell, self.s[:, None] * self.kin), axis=-1)
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


class ProjectToMultipoles(BaseClass):

    """Helper class to compute multipoles from wedges using Legendre polynomials."""

    def __init__(self, mu=20, method='leggauss', ells=(0, 2, 4)):
        self.mu, wmu = utils.weights_mu(mu, method=method)
        self.wmu = np.array([wmu * (2 * ell + 1) * special.legendre(ell)(self.mu) for ell in ells])

    def __call__(self, fmu):
        return jnp.sum(fmu * self.wmu[:, None, :], axis=-1)

    def __getstate__(self):
        state = {}
        for name in ['mu', 'wmu']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


@jit
def ap_k_mu(k, mu, qpar=1., qper=1.):
    qpar, qper = map(jnp.asarray, (qpar, qper))
    mu = mu[(Ellipsis,) + (None,) * qpar.ndim]
    k = k[(Ellipsis,) + (None,) * mu.ndim]
    qap = qpar / qper
    jac = 1. / (qpar * qper**2)
    factorap = jnp.sqrt(1 + mu**2 * (1. / qap**2 - 1))
    # Beutler 2016 (arXiv: 1607.03150v1) eq 44
    kap = k / qper * factorap
    # Beutler 2016 (arXiv: 1607.03150v1) eq 45
    muap = mu / qap / factorap
    return jac, kap, muap


@jit
def ap_s_mu(s, mu, qpar=1., qper=1.):
    qpar, qper = map(jnp.asarray, (qpar, qper))
    mu = mu[(Ellipsis,) + (None,) * qpar.ndim]
    s = s[(Ellipsis,) + (None,) * mu.ndim]
    qap = qpar / qper
    # Compared to Fourier space, qpar -> 1/qpar, qper -> 1/qper
    factorap = jnp.sqrt(1 + mu**2 * (qap**2 - 1))
    sap = s * qper * factorap
    muap = mu * qap / factorap
    return 1., sap, muap


class APEffect(BaseCalculator):
    r"""
    Alcock-Paczynski effect: compute 'qpar', 'qper', 'qap' and 'qiso' parameters, either as free parameters
    or from the ratio of distances in ``cosmo`` and ``fiducial`` cosmologies.

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
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    mode : str, default='geometry'
        Alcock-Paczynski parameterization:
        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)
        - 'geometry': scaling parameters computed from the ratio of ``cosmo`` to ``fiducial`` cosmology distances
    eta : float, default=1. / 3.
        Relation between 'qpar', 'qper' and 'qiso', 'qap' parameters:
        ``qiso = qpar ** eta * qper ** (1 - eta)``.


    Reference
    ---------
    https://ui.adsabs.harvard.edu/abs/1979Natur.281..358A/abstract
    """
    config_fn = 'base.yaml'

    def initialize(self, z=1., cosmo=None, fiducial='DESI', mode='geometry', eta=1. / 3.):
        self.z = np.asarray(z)
        if fiducial is None:
            raise ValueError('Provide fiducial cosmology')
        self.fiducial = get_cosmo(fiducial)
        self.eta = float(eta)
        self.efunc_fid = self.fiducial.efunc(self.z)
        self.DM_fid = self.fiducial.comoving_angular_distance(self.z)
        self.mode = mode
        self.cosmo = cosmo
        self.cosmo_requires = {}
        if self.mode == 'qiso':
            varied = ['qiso']
        elif self.mode == 'qap':
            varied = ['qap']
        elif self.mode == 'qisoqap':
            varied = ['qiso', 'qap']
        elif self.mode == 'qparqper':
            varied = ['qpar', 'qper']
        elif self.mode in ['geometry', 'bao']:
            varied = []
        else:
            raise ValueError('unknown mode {}; it must be one of ["qiso", "qap", "qisoqap", "qparqper", "geometry", "bao"]'.format(self.mode))
        self.init.params = self.init.params.select(basename=varied) + self.init.params.select(derived=True)
        if self.mode in ['geometry', 'bao']:
            if is_external_cosmo(cosmo):
                self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
                if self.mode == 'bao': self.cosmo_requires['thermodynamics'] = {'rs_drag': None}
            elif cosmo is None:
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
                qpar = self.DH_over_rd / self.DH_over_rd_fid
                qper = jnp.where(self.z == 0, qpar, self.DM_over_rd / self.DM_over_rd_fid)
            else:  # geometry
                qpar = self.DH / self.DH_fid
                qper = jnp.where(self.z == 0, qpar, self.DM / self.DM_fid)
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
        return ap_k_mu(k, mu, qpar=self.qpar, qper=self.qper)

    def ap_s_mu(self, s, mu):
        return ap_s_mu(s, mu, qpar=self.qpar, qper=self.qper)


_registered_legendre = [None] * 11
_registered_legendre[0] = lambda x: jnp.ones_like(x)
_registered_legendre[1] = lambda x: x
_registered_legendre[2] = lambda x: 3*x**2/2 - 1/2
_registered_legendre[3] = lambda x: 5*x**3/2 - 3*x/2
_registered_legendre[4] = lambda x: 35*x**4/8 - 15*x**2/4 + 3/8
_registered_legendre[5] = lambda x: 63*x**5/8 - 35*x**3/4 + 15*x/8
_registered_legendre[6] = lambda x: 231*x**6/16 - 315*x**4/16 + 105*x**2/16 - 5/16
_registered_legendre[7] = lambda x: 429*x**7/16 - 693*x**5/16 + 315*x**3/16 - 35*x/16
_registered_legendre[8] = lambda x: 6435*x**8/128 - 3003*x**6/32 + 3465*x**4/64 - 315*x**2/32 + 35/128
_registered_legendre[9] = lambda x: 12155*x**9/128 - 6435*x**7/32 + 9009*x**5/64 - 1155*x**3/32 + 315*x/128
_registered_legendre[10] = lambda x: 46189*x**10/256 - 109395*x**8/256 + 45045*x**6/128 - 15015*x**4/128 + 3465*x**2/256 - 63/256


def get_legendre(ell):
    return _registered_legendre[ell]