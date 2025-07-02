"""Warning: not tested!"""

import re

import numpy as np
from scipy import special, integrate

from desilike.base import BaseCalculator
from desilike.cosmo import is_external_cosmo
from desilike import plotting
from desilike.jax import numpy as jnp
from desilike.jax import jit, interp1d
from .power_template import BAOPowerSpectrumTemplate
from .base import (BaseTheoryPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles)


def _interp(template, name, k):
    return interp1d(jnp.log10(k), jnp.log10(template.k), getattr(template, name), method='cubic')
    #return getattr(template, name + '_interpolator')(k)



def _get_orders(base, params, ells):
    orders = {ell: {} for ell in ells}
    for param in params.select(basename=base + '*_*'):
        name = param.basename
        ell = None
        if name == base + '0':
            ell, ind = 0, 0
        else:
            match = re.match(base + '(.*)_(.*)', name)
            if match:
                ell, ind = int(match.group(1)), int(match.group(2))
        if ell is not None:
            if ell in ells:
                orders[ell][name] = ind
            else:
                del params[param]
    return orders


def _kernel_func(x, kernel='tsc'):
    toret = np.zeros_like(x)
    if kernel == 'ngp':
        mask = x < 0.5
        np.add.at(toret, mask, 1.)
    elif kernel == 'cic':
        mask = x < 1.
        np.add.at(toret, mask, 1. - x[mask])
    elif kernel == 'tsc':
        mask = x < 0.5
        np.add.at(toret, mask, 3. / 4. - x[mask]**2)
        mask = (x >= 0.5) & (x < 1.5)
        np.add.at(toret, mask, 1. / 2. * (3. / 2. - x[mask])**2)
    elif kernel == 'pcs':
        mask = x < 1.
        np.add.at(toret, mask, 1. / 6. * (4. - 6. * x[mask]**2 + 3. * x[mask]**3))
        mask = (x >= 1.) & (x < 2.)
        np.add.at(toret, mask, 1. / 6. * (2. - x[mask])**3)
    return toret


class BaseBAOWigglesPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    """Base class for theory BAO power spectrum multipoles, without broadband terms."""

    _klim = (1e-4, 1., 2000)

    def initialize(self, *args, template=None, mode='', smoothing_radius=15., ells=(0, 2), **kwargs):
        super(BaseBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, ells=ells, **kwargs)
        self.mode = str(mode)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('Reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        if template is None:
            template = BAOPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(self._klim[0], self.k[0] / 2, self.template.init.get('k', [1.])[0]), max(self._klim[1], self.k[-1] * 2, self.template.init.get('k', [0.])[0]), self._klim[2])  # margin for AP effect
        self.template.init.update(k=kin)
        self.template.init.setdefault('with_now', 'peakaverage', if_none=True)
        self.z = self.template.z
        self.rs_drag_fid = self.template.fiducial.rs_drag
        if tuple(self.ells) == (0,):  # one should be able to initialize pt without parameters  --- just to k and ells
            for param in self.init.params.select(basename=['dbeta']):
                param.update(fixed=True)

    def calculate(self):
        self.z = self.template.z
        self.rs_drag_fid = self.template.fiducial.rs_drag

    def __getstate__(self):
        state = super(BaseBAOWigglesPowerSpectrumMultipoles, self).__getstate__()
        for name in ['rs_drag_fid']:
            state[name] = getattr(self, name)
        return state


class DampedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    """
    Theory BAO power spectrum multipoles, without broadband terms,
    used in the BOSS DR12 BAO analysis by Beutler et al. 2017.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """
    def initialize(self, *args, mu=10, method='leggauss', model='standard', **kwargs):
        super(DampedBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.model = str(model)
        self.set_k_mu(k=self.k, mu=mu, method=method, ells=self.ells)
        if self.template.only_now:
            for param in self.init.params.select(basename=['sigmapar', 'sigmaper']):
                param.update(fixed=True)

    def calculate(self, b1=1., dbeta=1., sigmas=0., sigmapar=9., sigmaper=6.):
        super(DampedBAOWigglesPowerSpectrumMultipoles, self).calculate()
        f = dbeta * self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknowap = _interp(self.template, 'pknow_dd', kap)
        pkap = _interp(self.template, 'pk_dd', kap)
        if self.model == 'standard':  # Chen 2023
            k, mu = self.k[:, None], self.mu
            pkwap = pkap - pknowap
            sigma_nl2ap = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
            sk = 0.
            if self.mode == 'reciso': sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)  # taken at fiducial coordinates
            Cap = (b1 + f * muap**2 * (1 - sk))**2 * jnp.exp(-sigma_nl2ap / 2.)
            fog = 1. / (1. + (sigmas * k * mu)**2 / 2.)**2.
            B = (b1 + f * mu**2 * (1 - sk))**2 * fog
            pknow = _interp(self.template, 'pknow_dd', k)
            pkmu = B * pknow + Cap * pkwap
            self.power = self.to_poles(pkmu)
        else:
            if 'fix-damping' in self.model: k, mu = self.k[:, None], self.mu
            else: k, mu = kap, muap
            sigma_nl2 = k**2 * (sigmapar**2 * mu**2 + sigmaper**2 * (1. - mu**2))
            damped_wiggles = (pkap - pknowap) / pknowap * jnp.exp(-sigma_nl2 / 2.)
            if 'move-all' in self.model: k, mu = kap, muap
            else: k, mu = self.k[:, None], self.mu
            pknow = _interp(self.template, 'pknow_dd', k)
            fog = 1. / (1. + (sigmas * k * mu)**2 / 2.)**2.
            sk = 0.
            if self.mode == 'reciso': sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
            pksmooth = (b1 + f * mu**2 * (1 - sk))**2 * pknow
            if 'fog-damping' in self.model:  # Beutler2016
                pkmu = pksmooth * fog * (1. + damped_wiggles)
            else:  # Howlett 2023
                pkmu = pksmooth * (fog + damped_wiggles)
        self.power = self.to_poles(pkmu)


class SimpleBAOWigglesPowerSpectrumMultipoles(DampedBAOWigglesPowerSpectrumMultipoles):
    r"""
    As :class:`DampedBAOWigglesPowerSpectrumMultipoles`, but moving only BAO wiggles (and not damping, fog, or RSD terms)
    with scaling parameters.
    """
    def initialize(self, *args, model='fix-damping', **kwargs):
        #import warnings
        #warnings.warn('SimpleBAOWigglesPowerSpectrumMultipoles is deprecated. Use DampedBAOWigglesPowerSpectrumMultipoles instead, with model="fix-damping.")
        super(SimpleBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, model=model, **kwargs)


class ResummedPowerSpectrumWiggles(BaseCalculator):
    r"""
    Resummed BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    def initialize(self, template=None, mode='', smoothing_radius=15., shotnoise=0.):
        self.mode = str(mode)
        self.shotnoise = float(shotnoise)
        available_modes = ['', 'recsym', 'reciso']
        if self.mode not in available_modes:
            raise ValueError('reconstruction mode {} must be one of {}'.format(self.mode, available_modes))
        self.smoothing_radius = float(smoothing_radius)
        if template is None:
            template = BAOPowerSpectrumTemplate()
        self.template = template
        self.template.runtime_info.initialize()
        if is_external_cosmo(self.template.cosmo):
            self.cosmo_requires = {'thermodynamics': {'rs_drag': None}}
        self.z = self.template.z

    def calculate(self):
        self.z = self.template.z
        k = self.template.k
        pklin = self.template.pknow_dd
        q = self.template.cosmo.rs_drag
        j0 = special.jn(0, q * k)
        sk = 0.
        if self.mode: sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        # https://www.overleaf.com/project/633e1b59130591a7bf55a9cd eq. 17
        skc = 1. - sk
        self.sigma_sn2 = 1. / self.smoothing_radius / 6 / np.pi**(3. / 2.)
        self.sigma_nl2 = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * pklin, k)
        self.sigma_dd2 = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * skc**2 * pklin, k)
        if self.mode == 'reciso':
            self.sigma_x2 = 1. / (3. * np.pi**2) * integrate.simps((1. - j0) * skc * pklin, k)

    def wiggles(self, k, mu, b1=1., f=0., d=1.):
        # b1 Eulerian bias, d scaling the growth factor, sigmas FoG
        wiggles = _interp(self.template, 'pk_dd', k) - _interp(self.template, 'pknow_dd', k)
        ksq = (1 + f * (f + 2) * mu**2) * k**2
        d2 = d**2
        sigma_dd2 = self.sigma_dd2 + self.shotnoise * self.sigma_sn2 / b1**2
        sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        skc = 1. - sk
        if self.mode == 'recsym':
            resummed_wiggles = (b1 + f * mu**2)**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_dd2)
        elif self.mode == 'reciso':
            resummed_wiggles = (b1 + f * mu**2 * skc - sk)**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_dd2)
            sigma_ds2 = (1. + f * mu**2) * sigma_dd2 + f * (1. + f) * mu**2 * self.sigma_x2
            resummed_wiggles += 2. * (b1 + f * mu**2 * skc - sk) * (1 + f * mu**2) * sk * jnp.exp(-1. / 2. * ksq * d2 * sigma_ds2)
            sigma_ss2 = sigma_dd2 + f**2 * mu**2 * self.sigma_nl2 + 2 * f * mu**2 * self.sigma_x2
            resummed_wiggles += (1 + f * mu**2)**2 * sk**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_ss2)
        else:  # redshift-space, no reconstruction
            resummed_wiggles = (b1 + f * mu**2)**2 * jnp.exp(-1. / 2. * ksq * d2 * sigma_dd2)
        return resummed_wiggles * wiggles


class ResummedBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    Theory BAO power spectrum multipoles, without broadband terms, with resummation of BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    _default_options = dict(shotnoise=0.)  # to be given shot noise by window matrix

    def initialize(self, *args, mu=10, method='leggauss', model='standard', **kwargs):
        shotnoise = kwargs.pop('shotnoise', self._default_options['shotnoise'])
        super(ResummedBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.model = str(model)
        self.set_k_mu(k=self.k, mu=mu, method=method, ells=self.ells)
        self.wiggles = ResummedPowerSpectrumWiggles(mode=self.mode, template=self.template,
                                                    smoothing_radius=self.smoothing_radius,
                                                    shotnoise=shotnoise)
        if self.template.only_now:
            for param in self.init.params.select(basename=['q']):
                param.update(fixed=True)

    def calculate(self, b1=1., dbeta=1., sigmas=0., d=1., **kwargs):
        super(ResummedBAOWigglesPowerSpectrumMultipoles, self).calculate()
        f = dbeta * self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = _interp(self.template, 'pknow_dd', kap)
        damped_wiggles = 0. if self.template.only_now else self.wiggles.wiggles(kap, muap, b1=b1, f=f, d=d, **kwargs) / pknow
        if 'move-all' in self.model: k, mu = kap, muap
        else: k, mu = self.k[:, None], self.mu
        pknow = _interp(self.template, 'pknow_dd', k)
        fog = 1. / (1. + (sigmas * k * mu)**2 / 2.)**2.
        sk = 0.
        if self.mode == 'reciso': sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        pksmooth = (b1 + f * mu**2 * (1 - sk))**2 * pknow
        if 'fog-damping' in self.model:  # Beutler2016
            pkmu = pksmooth * fog * (1. + damped_wiggles)
        else:  # Howlett 2023
            pkmu = pksmooth * (fog + damped_wiggles)
        self.power = self.to_poles(pkmu)


class FlexibleBAOWigglesPowerSpectrumMultipoles(BaseBAOWigglesPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    Theory BAO power spectrum multipoles with terms multiplying the wiggles; no damping parameter (BAO damping or Finger-of-God).
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    wiggles : str, default='pcs'
        Multiplicative wiggles kernels, one of ['cic', 'tsc', 'pcs', 'power'].
        'power' corresponds to :math:`k^{n}` wiggles terms.

    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.

    """
    @staticmethod
    def _params(params, wiggles='pcs'):
        ells = [0, 2, 4]
        if wiggles == 'power':
            for ell in ells:
                for pow in range(-3, 2):
                    params['ml{:d}_{:d}'.format(ell, pow)] = dict(value=0., ref=dict(limits=[-1e2, 1e2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, pow))
        else:
            for ell in ells:
                for ik in range(-2, 10):  # should be more than enough
                    # We are adding a very loose prior just to regularize the fit --- parameters at the high-k end can be e.g. poorly constrained
                    # because these modes are given zero weight by the window matrix
                    params['ml{:d}_{:d}'.format(ell, ik)] = dict(value=0., prior=dict(dist='norm', loc=0., scale=1e4), ref=dict(limits=[-1e-2, 1e-2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, ik))
        return params

    def initialize(self, *args, mu=10, method='leggauss', model='standard', wiggles='pcs', kp=None, **kwargs):
        super(FlexibleBAOWigglesPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=mu, method=method, ells=self.ells)
        self.model = str(model)
        self.wiggles = str(wiggles)
        if kp is None: self.kp = 2. * np.pi / self.rs_drag_fid
        else: self.kp = float(kp)
        self.set_params()
        if self.template.only_now:
            for param in self.init.params.select(basename='ml*_*'):
                param.update(fixed=True)

    def set_params(self):
        self.wiggles_orders = _get_orders('ml', self.init.params, self.ells)
        self.wiggles_matrix = {}
        if self.wiggles == 'power':
            for ell in self.ells:
                self.wiggles_matrix[ell] = jnp.array([(self.k / self.kp)**pow for pow in self.wiggles_orders[ell].values()])
        elif self.wiggles in ['ngp', 'cic', 'tsc', 'pcs']:
            for ell in self.ells:
                tmp, bb_orders = [], {}
                for name, ik in self.wiggles_orders[ell].items():
                    kernel = _kernel_func(np.abs(self.k / self.kp - ik), kernel=self.wiggles)
                    if not np.allclose(kernel, 0., rtol=0., atol=1e-8):
                        tmp.append(kernel)
                        bb_orders[name] = ik
                self.wiggles_orders[ell] = bb_orders
                self.wiggles_matrix[ell] = jnp.array(tmp)
        else:
            raise ValueError('Unknown kernel: {}'.format(self.wiggles))
        bb_params = ['b1', 'dbeta']
        for params in self.wiggles_orders.values(): bb_params += list(params)
        self.init.params = self.init.params.select(basename=bb_params)

    @jit(static_argnums=[0])
    def get_wiggles(self, wiggles, **kwargs):
        damped_wiggles = 0.
        for ell in self.ells:
            mult = jnp.array([kwargs[name] for name in self.wiggles_orders[ell]]).dot(self.wiggles_matrix[ell])
            if ell == 0: mult += 1.
            leg = special.legendre(ell)(self.mu)
            damped_wiggles += wiggles * mult[:, None] * leg
        return damped_wiggles

    def calculate(self, b1=1., dbeta=1., **kwargs):
        super(FlexibleBAOWigglesPowerSpectrumMultipoles, self).calculate()
        f = dbeta * self.template.f
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pknow = _interp(self.template, 'pknow_dd', kap)
        pk = _interp(self.template, 'pk_dd', kap)
        damped_wiggles = self.get_wiggles(pk - pknow, **kwargs) / pknow
        if 'move-all' in self.model: k, mu = kap, muap
        else: k, mu = self.k[:, None], self.mu
        pknow = _interp(self.template, 'pknow_dd', k)
        sk = 0.
        if self.mode == 'reciso': sk = jnp.exp(-1. / 2. * (k * self.smoothing_radius)**2)
        pksmooth = (b1 + f * mu**2 * (1 - sk))**2 * pknow
        pkmu = pksmooth * (1. + damped_wiggles)
        self.power = self.to_poles(pkmu)

    def get(self):
        return self.power

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k * self.power[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


class BaseBAOWigglesTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):
    r"""
    Base class for theory BAO power spectrum multipoles, with broadband terms.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.

    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.
    """
    config_fn = 'bao.yaml'
    _initialize_with_namespace = True  # to properly forward parameters to pt

    @staticmethod
    def _params(params, broadband='power'):
        broadband = str(broadband)
        ells = [0, 2, 4]
        if 'power' in broadband:
            for ell in ells:
                for pow in range(-3, 2):
                    param = dict(value=0., ref=dict(limits=[-1e2, 1e2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, pow))
                    if broadband == 'power3' and (pow not in [-2, -1, 0]): param.update(fixed=True)
                    params['al{:d}_{:d}'.format(ell, pow)] = param
        else:
            for ell in ells:
                for ik in range(-2, 10):  # should be more than enough for k < 0.4 h/Mpc
                    # We are adding a very loose prior just to regularize the fit --- parameters at the high-k end can be e.g. poorly constrained
                    # because these modes are given zero weight by the window matrix
                    params['al{:d}_{:d}'.format(ell, ik)] = dict(value=0., prior=dict(dist='norm', loc=0., scale=1e4), ref=dict(limits=[-1e-2, 1e-2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, ik))
        return params

    def initialize(self, k=None, ells=(0, 2), broadband='power', kp=None, pt=None, **kwargs):
        super(BaseBAOWigglesTracerPowerSpectrumMultipoles, self).initialize(k=k, ells=ells)
        if pt is None:
            pt = globals()[self.__class__.__name__.replace('Tracer', '')]()
        self.pt = pt
        self.pt.init.update(k=self.k, ells=self.ells, **kwargs)
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.broadband = str(broadband)
        if kp is None: self.kp = 2. * np.pi / self.pt.rs_drag_fid
        else: self.kp = float(kp)
        self.set_params()

    def set_params(self):
        self.broadband_orders = _get_orders('al', self.init.params, self.ells)
        self.broadband_matrix = {}
        pt_params = self.init.params.copy()
        bb_params = []
        for params in self.broadband_orders.values(): bb_params += list(params)
        self.init.params = self.init.params.select(basename=bb_params)
        for param in list(pt_params):
            if param.basename in bb_params or (param.derived is True):
                del pt_params[param]
        self.pt.init.params.update(pt_params, basename=True)
        if 'power' in self.broadband:  # even-power for the correlation function
            for ell in self.ells:
                self.broadband_matrix[ell] = jnp.array([(self.k / self.kp)**pow for pow in self.broadband_orders[ell].values()])
        elif self.broadband in ['ngp', 'cic', 'tsc', 'pcs']:
            pk_now = lambda k: _interp(self.template, 'pknow_dd_fid', k)
            for ell in self.ells:
                tmp, bb_orders = [], {}
                for name, ik in self.broadband_orders[ell].items():  # iterate over nodes
                    kernel = _kernel_func(np.abs(self.k / self.kp - ik), kernel=self.broadband)  # the kernel function
                    if not np.allclose(kernel, 0., rtol=0., atol=1e-8):
                        tmp.append(kernel * pk_now(np.clip(ik * self.kp, self.k[0], self.k[-1])))  # scale kernel by typical amplitude of Pnowiggle
                        bb_orders[name] = ik
                self.broadband_orders[ell] = bb_orders
                self.broadband_matrix[ell] = jnp.array(tmp)
        else:
            raise ValueError('Unknown kernel: {}'.format(self.broadband))
        self.init.params = self.init.params.select(basename=bb_params)

    @jit(static_argnums=[0])
    def get_broadband(self, **params):
        return jnp.array([jnp.array([params.get(name, 0.) for name in self.broadband_orders[ell]]).dot(self.broadband_matrix[ell]) for ell in self.ells])

    def calculate(self, **params):
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.power = self.pt.power.copy() + self.get_broadband(**params)

    @property
    def template(self):
        return self.pt.template

    def get(self):
        return self.power

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k * self.power[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


class DampedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles, with broadband terms, used in the BOSS DR12 BAO analysis by Beutler et al. 2017.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
        'fog-damping_move-all' to use the model of https://arxiv.org/abs/1607.03149.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.

    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.

    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """


class SimpleBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    As :class:`DampedBAOWigglesTracerPowerSpectrumMultipoles`, but moving only BAO wiggles (and not damping or RSD terms)
    with scaling parameters; essentially used for Fisher forecasts.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.

    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.
    """


class ResummedBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles, with broadband terms, with resummation of BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.

    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.

    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    _default_options = dict(shotnoise=0.)  # to be given shot noise by window matrix


class FlexibleBAOWigglesTracerPowerSpectrumMultipoles(BaseBAOWigglesTracerPowerSpectrumMultipoles):
    r"""
    Theory BAO power spectrum multipoles, with broadband terms, with flexible BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    model : str, default='standard'
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.

    wiggles : str, default='pcs'
        Multiplicative wiggles kernels, one of ['cic', 'tsc', 'pcs', 'power'].
        'power' corresponds to :math:`k^{n}` wiggles terms.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`k`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels.

    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.
    """


class BaseBAOWigglesCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):
    """
    Base class that implements theory BAO correlation function multipoles, without broadband terms,
    as Hankel transforms of the theory power spectrum multipoles.
    """
    def initialize(self, s=None, ells=(0, 2), **kwargs):
        power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')](**kwargs)
        super(BaseBAOWigglesCorrelationFunctionMultipoles, self).initialize(s=s, ells=ells, power=power)
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))

    def calculate(self):
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))
        super(BaseBAOWigglesCorrelationFunctionMultipoles, self).calculate()

    @property
    def template(self):
        return self.power.template

    def get(self):
        return self.corr


class DampedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class SimpleBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class ResummedBAOWigglesCorrelationFunctionMultipoles(BaseBAOWigglesCorrelationFunctionMultipoles):

    pass


class BaseBAOWigglesTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    Base class that implements theory BAO correlation function multipoles, with broadband terms.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'even-power' for powers of :math:`s^{2}` (motivated theoretically by Stephen Chen),
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.

    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.
    """
    config_fn = 'bao.yaml'

    @staticmethod
    def _params(params, broadband='power'):
        broadband = str(broadband)
        ells = [0, 2, 4]
        if 'power' in broadband:
            for ell in ells:
                for pow in range(-2, 3):
                    param = dict(value=0., ref=dict(limits=[-1e-3, 1e-3]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, pow))
                    if broadband == 'power3' and (pow not in [-2, -1, 0]): param.update(fixed=True)
                    if broadband == 'even-power' and (pow not in [0, 2]): param.update(fixed=True)
                    params['al{:d}_{:d}'.format(ell, pow)] = param
        else:
            for ell in ells:
                for ik in range(-2, 3):  # should be more than enough
                    # Infinite prior
                    param = dict(value=0., prior=None, ref=dict(limits=[-1e2, 1e2]), delta=0.005, latex='a_{{{:d}, {:d}}}'.format(ell, ik))
                    if broadband == 'pcs2' and (ell == 0 or ik not in [0, 1]): param.update(fixed=True)
                    params['al{:d}_{:d}'.format(ell, ik)] = param
                for ik in [0, 2]:
                    params['bl{:d}_{:d}'.format(ell, ik)] = dict(value=0., ref=dict(limits=[-1e-3, 1e-3]), delta=0.005, latex='b_{{{:d}, {:d}}}'.format(ell, ik))
        return params

    def initialize(self, s=None, ells=(0, 2), sp=None, broadband='power', pt=None, **kwargs):
        self.broadband = str(broadband)
        if sp is None: self.sp = 2. * np.pi / 0.02
        else: self.sp = float(sp)
        if 'power' in self.broadband:
            if pt is None:
                pt = globals()[self.__class__.__name__.replace('TracerCorrelationFunction', 'PowerSpectrum')](**kwargs)
            power = pt
            self.broadband = 'power'
        else:
            self.broadband = self.broadband[:3]  # remove e.g. -2 from pcs2
            power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')](broadband=self.broadband, pt=pt, **kwargs)
        super(BaseBAOWigglesTracerCorrelationFunctionMultipoles, self).initialize(s=s, ells=ells, power=power)
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))

    def set_params(self):
        if 'power' in self.broadband:
            self.k, self.kp = self.s, self.sp
            # other model parameters, e.g. bias
            BaseBAOWigglesTracerPowerSpectrumMultipoles.set_params(self)
            del self.k, self.kp
        else:
            self.broadband_orders = _get_orders('bl', self.init.params, self.ells)
            self.broadband_matrix = {}

            for ell in self.ells:
                self.broadband_matrix[ell] = jnp.array([(self.s / self.sp)**pow for pow in self.broadband_orders[ell].values()])
            power_params = self.init.params.copy()
            bb_params = []
            for params in self.broadband_orders.values(): bb_params += list(params)
            self.init.params = self.init.params.select(basename=bb_params)
            for param in list(power_params):
                if param.basename in bb_params: del power_params[param]
            self.power.params = power_params

    def calculate(self, **params):
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))
        super(BaseBAOWigglesTracerCorrelationFunctionMultipoles, self).calculate()
        self.corr += jnp.array([jnp.array([params.get(name, 0.) for name in self.broadband_orders[ell]]).dot(self.broadband_matrix[ell]) for ell in self.ells])

    @property
    def pt(self):
        return getattr(self.power, 'pt', self.power)

    @property
    def template(self):
        return self.power.template

    @property
    def wiggle(self):
        return self.power.wiggle

    @wiggle.setter
    def wiggle(self, wiggle):
        self.power.wiggle = wiggle

    def get(self):
        return self.corr

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot correlation function multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self.s, self.s**2 * self.corr[ill], color='C{:d}'.format(ill), linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig


class DampedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    Theory BAO correlation function multipoles, with broadband terms.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.
        'fog-damping_move-all' to use the model of https://arxiv.org/abs/1607.03149.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.

    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.


    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """


class SimpleBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    As :class:`DampedBAOWigglesTracerCorrelationFunctionMultipoles`, but moving only BAO wiggles (and not damping or RSD terms)
    with scaling parameters; essentially used for Fisher forecasts.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.

    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.


    Reference
    ---------
    https://arxiv.org/abs/1607.03149
    """


class ResummedBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    Theory BAO correlation function multipoles, with broadband terms, with resummation of BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    model : str, default='standard'
        'fog-damping' to apply Finger-of-God to the wiggle part (in addition to the no-wiggle part).
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.

    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.


    Reference
    ---------
    https://arxiv.org/abs/1907.00043
    """
    _default_options = dict(shotnoise=0.)  # to be given shot noise by window matrix


class FlexibleBAOWigglesTracerCorrelationFunctionMultipoles(BaseBAOWigglesTracerCorrelationFunctionMultipoles):
    r"""
    Theory BAO correlation function multipoles, with broadband terms, with flexible BAO wiggles.
    Supports pre-, reciso, recsym, real (f = 0) and redshift-space reconstruction.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2)
        Multipoles to compute.

    mu : int, default=20
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    mode : str, default=''
        Reconstruction mode:

        - '': no reconstruction
        - 'recsym': recsym reconstruction (both data and randoms are shifted with RSD displacements)
        - 'reciso': reciso reconstruction (data only is shifted with RSD displacements)

    smoothing_radius : float, default=15
        Smoothing radius used in reconstruction.

    template : BasePowerSpectrumTemplate, default=None
        Power spectrum template. If ``None``, defaults to :class:`BAOPowerSpectrumTemplate`.

    model : str, default='standard'
        'move-all' to move the no-wiggle part (in addition to the wiggle part) with scaling parameters.

    wiggles : str, default='pcs'
        Multiplicative wiggles kernels, one of ['cic', 'tsc', 'pcs', 'power'].
        'power' corresponds to :math:`k^{n}` wiggles terms.

    kp : float, default=None
        For 'power' kernel, the pivot :math:`k`.
        For other kernels, their :math:`k`-period.
        Defaults to :math:`2 \pi / r_{d}`.

    broadband : str, default='power'
        Broadband parameterization: 'power' for powers of :math:`s`,
        'ngp', 'cic', 'tsc' or 'pcs' for the sum of corresponding kernels in Fourier space.

    sp : float, default=None
        The pivot :math:`s`. Defaults to :math:`2 \pi / 0.02`.
    """