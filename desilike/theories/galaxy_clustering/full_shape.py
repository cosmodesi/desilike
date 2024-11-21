import re

import numpy as np
from scipy import interpolate

from desilike.jax import numpy as jnp
from desilike.jax import jit, interp1d
from desilike import plotting, utils, BaseCalculator
from .base import BaseTheoryPowerSpectrumMultipolesFromWedges
from .base import BaseTheoryPowerSpectrumMultipoles, BaseTheoryCorrelationFunctionMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles
from .power_template import DirectPowerSpectrumTemplate, StandardPowerSpectrumTemplate


class BasePTPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    """Base class for perturbation theory matter power spectrum multipoles."""
    _default_options = dict()
    _klim = (1e-3, 1., 500)  # klim < 1e-3 h/Mpc causes problems in velocileptors and folps when Omega_k ~ 0.1

    def initialize(self, *args, template=None, z=None, **kwargs):
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        super(BasePTPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        if template is None:
            template = DirectPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(self._klim[0], self.k[0] / 2, self.template.init.get('k', [1.])[0]), max(self._klim[1], self.k[-1] * 2, self.template.init.get('k', [0.])[0]), self._klim[2])  # margin for AP effect
        self.template.init.update(k=kin)
        if z is not None: self.template.init.update(z=z)
        self.z = self.template.z

    def calculate(self):
        self.z = self.template.z


class BasePTCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    _default_options = dict()
    _klim = (1e-3, 1., 500)

    def initialize(self, *args, template=None, **kwargs):
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        super(BasePTCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        if template is None:
            template = DirectPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(self._klim[0], 1 / self.s[-1] / 2, self.template.init.get('k', [1.])[0]), max(self._klim[1], 1 / self.s[0] * 2, self.template.init.get('k', [0.])[0]), self._klim[2])  # margin for AP effect
        self.template.init.update(k=kin)
        self.z = self.template.z

    def calculate(self):
        self.z = self.template.z


class BaseTracerPowerSpectrumMultipoles(BaseCalculator):

    """Base class for perturbation theory tracer power spectrum multipoles."""
    config_fn = 'full_shape.yaml'
    _initialize_with_namespace = True  # to properly forward parameters to pt
    _default_options = dict()

    def initialize(self, pt=None, template=None, **kwargs):
        self.options = self._default_options.copy()
        shotnoise = kwargs.get('shotnoise', 1e4)
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        self.nd = 1. / float(shotnoise)
        if pt is None:
            pt = globals()[getattr(self, 'pt_cls', self.__class__.__name__.replace('Tracer', ''))]()
        self.pt = pt
        if template is not None:
            self.pt.init.update(template=template)
        for name, value in self.pt._default_options.items():
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
            elif name in self.options:
                self.pt.init.update({name: self.options[name]})
        for name in ['method', 'mu']:
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
        self.required_bias_params, self.optional_bias_params = {}, {}
        self.pt.init.update(kwargs)
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.set_params()

    def set_params(self, pt_params=None):
        all_bias_params = list(self.required_bias_params.keys()) + list(self.optional_bias_params.keys())
        if pt_params is None:
            for param in self.init.params:
                if param.basename not in all_bias_params and not (param.derived is True):
                    pt_params.append(param.basename)
        self.pt.init.params.update([param for param in self.init.params if param.basename in pt_params], basename=True)
        self.init.params = self.init.params.select(basename=[param.basename for param in self.init.params if param.basename in all_bias_params or (param.derived is True)])

    def calculate(self):
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))

    def get(self):
        return self.power

    @property
    def template(self):
        return self.pt.template

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'nd', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

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


class BaseTracerCorrelationFunctionMultipoles(BaseCalculator):

    """Base class for perturbation theory tracer correlation function multipoles."""
    config_fn = 'full_shape.yaml'
    _initialize_with_namespace = True  # to properly forward parameters to pt
    _default_options = dict()

    def initialize(self, pt=None, template=None, **kwargs):
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        if pt is None:
            pt = globals()[getattr(self, 'pt_cls', self.__class__.__name__.replace('Tracer', ''))]()
        self.pt = pt
        if template is not None:
            self.pt.init.update(template=template)
        for name, value in self.pt._default_options.items():
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
            elif name in self.options:
                self.pt.init.update({name: self.options[name]})
        self.required_bias_params, self.optional_bias_params = dict.fromkeys(self.init.params.basenames()), {}
        self.pt.init.update(kwargs)
        for name in ['z', 's', 'ells']:
            setattr(self, name, getattr(self.pt, name))
        self.set_params()

    def set_params(self, pt_params=None):
        all_bias_params = list(self.required_bias_params) + list(self.optional_bias_params)
        if pt_params is None:
            for param in self.init.params:
                if param.basename not in all_bias_params and not (param.derived is True):
                    pt_params.append(param.basename)
        self.pt.init.params.update([param for param in self.init.params if param.basename in pt_params], basename=True)
        self.init.params = self.init.params.select(basename=[param.basename for param in self.init.params if param.basename in all_bias_params or (param.derived is True)])

    def calculate(self):
        for name in ['z', 's', 'ells']:
            setattr(self, name, getattr(self.pt, name))

    def get(self):
        return self.corr

    @property
    def template(self):
        return self.pt.template

    def __getstate__(self):
        state = {}
        for name in ['s', 'z', 'ells', 'corr']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

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


class BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):

    """Base class for perturbation theory tracer correlation function multipoles as Hankel transforms of the power spectrum multipoles."""
    config_fn = 'full_shape.yaml'

    def initialize(self, *args, pt=None, template=None, **kwargs):
        power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')]()
        if pt is not None: power.init.update(pt=pt)
        if template is not None: power.init.update(template=template)
        super(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles, self).initialize(*args, power=power, **kwargs)
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))

    def calculate(self):
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))
        super(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles, self).calculate()

    @property
    def pt(self):
        return self.power.pt

    @property
    def template(self):
        return self.power.template

    def get(self):
        return self.corr


class SimpleTracerPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    Kaiser tracer power spectrum multipoles, with fixed damping, essentially used for Fisher forecasts.
    For the matter (unbiased) power spectrum, set b1=1 and sn0=0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`StandardPowerSpectrumTemplate`.

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).
    """
    config_fn = 'full_shape.yaml'

    def initialize(self, *args, mu=8, method='leggauss', template=None, shotnoise=1e4, **kwargs):
        self.nd = 1. / float(shotnoise)
        if template is None:
            template = StandardPowerSpectrumTemplate()
        super(SimpleTracerPowerSpectrumMultipoles, self).initialize(*args, template=template, mu=mu, method=method, **kwargs)

    def calculate(self, b1=1., sn0=0., sigmapar=0., sigmaper=0.):
        super(SimpleTracerPowerSpectrumMultipoles, self).calculate()
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f
        sigmanl2 = self.k[:, None]**2 * (sigmapar**2 * self.mu**2 + sigmaper**2 * (1. - self.mu**2))
        damping = jnp.exp(-sigmanl2 / 2.)
        #pkmu = jac * damping * (b1 + f * muap**2)**2 * jnp.interp(jnp.log10(kap), jnp.log10(self.template.k), self.template.pk_dd) + sn0 / self.nd
        pkmu = jac * damping * (b1 + f * muap**2)**2 * interp1d(jnp.log10(kap), jnp.log10(self.template.k), self.template.pk_dd, method='cubic') + sn0 / self.nd
        self.power = self.to_poles(pkmu)

    def get(self):
        return self.power

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'nd', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

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


class KaiserPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    Kaiser power spectrum multipoles.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    """
    _params = {'sigmapar': {'value': 0., 'fixed': True}, 'sigmaper': {'value': 0, 'fixed': True}}

    def initialize(self, *args, mu=8, **kwargs):
        super(KaiserPowerSpectrumMultipoles, self).initialize(*args, mu=mu, method='leggauss', **kwargs)
        #self.template.init.update(k=np.logspace(-4, 2, 1000))

    def calculate(self, sigmapar=0., sigmaper=0.):
        super(KaiserPowerSpectrumMultipoles, self).calculate()
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f
        sigmanl2 = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1. - muap**2))
        damping = jnp.exp(-sigmanl2 / 2.)

        self.pktable = []
        self.k11 = self.template.k
        self.pk11 = self.template.pk_dd
        pktable = jac * damping * interp1d(jnp.log10(kap), jnp.log10(self.k11), self.pk11, method='cubic')
        self.pktable = {'pk_dd': self.to_poles(pktable), 'pk_dt': self.to_poles(f * muap**2 * pktable), 'pk_tt': self.to_poles(f**2 * muap**4 * pktable)}
        self.pktable['pk11'] = self.pktable['pk_dd']

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self.pktable:
            state[name] = self.pktable[name]
        state['names'] = list(self.pktable.keys())
        return state

    def __setstate__(self, state):
        state = dict(state)
        self.pktable = {name: state.pop(name, None) for name in state['names']}
        super(KaiserPowerSpectrumMultipoles, self).__setstate__(state)


class KaiserTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    r"""
    Kaiser tracer power spectrum multipoles.
    For the matter (unbiased) power spectrum, set b1=1 and sn0=0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    """
    def set_params(self):
        self.required_bias_params.update(dict(b1=1., sn0=0.))
        super().set_params(pt_params=['sigmapar', 'sigmaper'])

    def calculate(self, b1=1., sn0=0.):
        super(KaiserTracerPowerSpectrumMultipoles, self).calculate()
        sn0 = np.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None] * sn0 / self.nd
        self.power = b1**2 * self.pt.pktable['pk_dd'] + 2. * b1 * self.pt.pktable['pk_dt'] + self.pt.pktable['pk_tt'] + sn0


class KaiserTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    Kaiser tracer correlation function multipoles.
    For the matter (unbiased) correlation function, set b1=1 and sn0=0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """


class BaseEFTLikeTracerPowerSpectrumMultipoles(object):
    r"""
    Base class for tracer power spectrum multipoles with EFT-like counter and stochastic terms.
    Can be exactly marginalized over counter terms and stochastic parameters ct*, sn*.
    """
    def initialize(self, *args, **kwargs):
        self.pt_cls = self.__class__.__name__.replace('EFTLike', '').replace('Tracer', '')
        super(BaseEFTLikeTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)

    def set_params(self):
        self.kp = 1.

        def get_params_matrix(base):
            coeffs = {ell: {} for ell in self.ells}
            for param in self.init.params.select(basename=base + '*_*'):
                name = param.basename
                match = re.match(base + '(.*)_(.*)', name)
                if match:
                    ell, pow = int(match.group(1)), int(match.group(2))
                    if ell in self.ells:
                        coeffs[ell][name] = (self.k / self.kp)**pow
                    else:
                        del self.init.params[param]
            for param in self.init.params.select(basename=base + '0'):
                ell, name = 0, param.basename
                if ell in self.ells:
                    if name + '_0' in coeffs[ell]:
                        raise ValueError('Choose between {} and {}'.format(name, name + '_0'))
                    coeffs[ell][name] = 1.
                else:
                    del self.init.params[param]
            params = [name for ell in self.ells for name in coeffs[ell]]
            if not params:
                return params, jnp.array([], dtype='f8')
            matrix = []
            for ell in self.ells:
                row = [np.zeros_like(self.k) for i in range(len(params))]
                for name, k_i in coeffs[ell].items():
                    row[params.index(name)][:] = k_i
                matrix.append(np.column_stack(row))
            matrix = jnp.array(matrix)
            return params, matrix

        self.counterterm_params, self.counterterm_matrix = get_params_matrix('ct')
        self.stochastic_params, self.stochastic_matrix = get_params_matrix('sn')
        params = self.counterterm_params + self.stochastic_params
        self.required_bias_params = dict(**self.required_bias_params, **dict(zip(params, [0] * len(params))))
        super().set_params()

    def calculate(self, **params):
        counterterm_values = jnp.array([params.pop(name, 0.) for name in self.counterterm_params])
        stochastic_values = jnp.array([params.pop(name, 0.) for name in self.stochastic_params]) / self.nd
        super(BaseEFTLikeTracerPowerSpectrumMultipoles, self).calculate(**params)
        self.power += self.counterterm_matrix.dot(counterterm_values) * self.pt.pktable['pk11'][self.pt.ells.index(0)]
        self.power += self.stochastic_matrix.dot(stochastic_values)


class EFTLikeKaiserTracerPowerSpectrumMultipoles(BaseEFTLikeTracerPowerSpectrumMultipoles, KaiserTracerPowerSpectrumMultipoles):
    r"""
    Kaiser tracer power spectrum multipoles with EFT-like counter and stochastic terms.
    Can be exactly marginalized over counter terms and stochastic parameters ct*, sn*.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).
    """


class EFTLikeKaiserTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    EFT-like Kaiser tracer correlation function multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters ct*, sn*.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """


def tns_kernels(k, q, wq):
    jq = q**2 * wq / (4. * np.pi**2)
    k = k[:, None]
    x = q / k
    kernels = [None] * 3
    # Integral of F3(q, -q, k) over mu cosine angle between k and q
    def kernel_ff(x):
        x = np.array(x)
        toret = (6. / x**2 - 79. + 50. * x**2 - 21. * x**4 + 0.75 * (1. / x - x)**3 * (2. + 7. * x**2) * 2 * np.log(np.abs((x - 1.) / (x + 1.)))) / 504.
        mask = x > 10.
        toret[mask] = - 61. / 630. + 2. / 105. / x[mask]**2 - 10. / 1323. / x[mask]**4
        dx = x - 1.
        mask = np.abs(dx) < 0.01
        toret[mask] = - 11. / 126. + dx[mask] / 126. - 29. / 252. * dx[mask]**2
        return toret / x**2

    def kernel_gg(x):
        x = np.array(x)
        toret = (6. / x**2 - 41. + 2. * x**2 - 3. * x**4 + 0.75 * (1. / x - x)**3 * (2. + x**2) * 2 * np.log(np.abs((x - 1.) / (x + 1.)))) / 168.
        mask = x > 10.
        toret[mask] = - 3. / 10. + 26. / 245. / x[mask]**2 - 38. / 2205. / x[mask]**4
        dx = x - 1.
        mask = np.abs(dx) < 0.01
        toret[mask] = - 3. / 14. - 5. / 42. * dx[mask] - 1. / 84. * dx[mask]**2
        return toret / x**2

    kernels[0] = 2 * jq * kernel_ff(x)
    kernels[1] = 2 * jq * kernel_gg(x)

    def kernel_a(x):
        toret = np.zeros((5,) + x.shape, dtype='f8')
        logx = np.zeros_like(x)
        mask = np.abs(x - 1) > 1e-16
        logx[mask] = np.log(np.abs((x[mask] + 1) / (x[mask] - 1)))
        toret[0] = -1. / 84. / x * (2 * x * (19 - 24 * x**2 + 9 * x**4) - 9 * (x**2 - 1)**3 * logx)
        toret[1] = 1. / 112. / x**3 * (2 * x * (x**2 + 1) * (3 - 14 * x**2 + 3 * x**4) - 3 * (x**2 - 1)**4 * logx)
        toret[2] = 1. / 336. / x**3 * (2 * x * (9 - 185 * x**2 + 159 * x**4 - 63 * x**6) + 9 * (x**2 - 1)**3 * (7 * x**2 + 1) * logx)
        toret[4] = 1. / 336. / x**3 * (2 * x * (9 - 109 * x**2 + 63 * x**4 - 27 * x**6) + 9 * (x**2 - 1)**3 * (3 * x**2 + 1) * logx)

        mask = x < 1e-4
        xm = x[mask]
        toret[0][mask] = 8 * xm**8 / 735 + 24 * xm**6 / 245 - 24 * xm**4 / 35 + 8 * xm**2 / 7 - 2. / 3
        toret[1][mask] = - 16 * xm**8 / 8085 - 16 * xm**6 / 735 + 48 * xm**4 / 245 - 16 * xm**2 / 35
        toret[2][mask] = 32 * xm**8 / 1617 + 128 * xm**6 / 735 - 288 * xm**4 / 245 + 64 * xm**2 / 35 - 4. / 3
        toret[4][mask] = 24 * xm**8 / 2695 + 8 * xm**6 / 105 - 24 * xm**4 / 49 + 24 * xm**2 / 35 - 2. / 3

        mask = x > 1e2
        xm = x[mask]
        toret[0][mask] = 2. / 105 - 24 / (245 * xm**2) - 8 / (735 * xm**4) - 8 / (2695 * xm**6) - 8 / (7007 * xm**8)
        toret[1][mask] = -16. / 35 + 48 / (245 * xm**2) - 16 / (735 * xm**4) - 16 / (8085 * xm**6) - 16 / (35035 * xm**8)
        toret[2][mask] = -44. / 105 - 32 / (735 * xm**4) - 64 / (8085 * xm**6) - 96 / (35035 * xm**8)
        toret[4][mask] = -46. / 105 + 24 / (245 * xm**2) - 8 / (245 * xm**4) - 8 / (1617 * xm**6) - 8 / (5005 * xm**8)

        toret[3] = toret[1]
        return toret / x**2

    kernels[2] = jq * kernel_a(x)
    return kernels


@jit
def tns_pt(k, q, wq, pk_q, kernel13_d, kernel13_t, kernel_a):
    # We could have a speed-up with FFTlog, see https://arxiv.org/pdf/1603.04405.pdf
    k11 = k
    k = k[:, None]
    jq = q**2 * wq / (4. * np.pi**2)
    x = q / k

    mus, wmus = utils.weights_mu(10, method='leggauss', sym=False)

    # Compute P22
    pk22_dd, pk22_dt, pk22_tt = (0.,) * 3
    pk_b2d, pk_bs2d, pk_b2t, pk_bs2t, sig3sq, pk_b22, pk_b2s2, pk_bs22 = (0.,) * 8
    A = jnp.zeros((5,) + k11.shape, dtype='f8')
    B = [jnp.zeros(k11.shape, dtype='f8') for i in range(12)]
    kernel_A, kernel_tA = ([jnp.zeros(x.shape, dtype='f8') for i in range(5)] for j in range(2))
    pk_k = jnp.interp(k11, q, pk_q)

    for mu, wmu in zip(mus, wmus):
        kdq = k * q * mu  # k \cdot q
        kq2 = k**2 - 2. * kdq + q**2  # |k - q|^2
        qdkq = kdq - q**2   # k \cdot (k - q)
        F2_d = 5. / 7. + 1. / 2. * qdkq * (1. / q**2 + 1. / kq2) + 2. / 7. * qdkq**2 / (q**2 * kq2)
        F2_t = 3. / 7. + 1. / 2. * qdkq * (1. / q**2 + 1. / kq2) + 4. / 7. * qdkq**2 / (q**2 * kq2)
        # https://arxiv.org/pdf/0902.0991.pdf
        S = (qdkq)**2 / (q**2 * kq2) - 1. / 3.
        D = 2. / 7. * (mu**2 - 1.)
        pk_kq = jnp.interp(kq2**0.5, q, pk_q, left=0., right=0.)
        jq_pk_q_pk_kq = jq * pk_q * pk_kq
        pk_b2d += wmu * jnp.sum(jq_pk_q_pk_kq * F2_d, axis=-1)
        pk_bs2d += wmu * jnp.sum(jq_pk_q_pk_kq * F2_d * S, axis=-1)
        pk_b2t += wmu * jnp.sum(jq_pk_q_pk_kq * F2_t, axis=-1)
        pk_bs2t += wmu * jnp.sum(jq_pk_q_pk_kq * F2_t * S, axis=-1)
        sig3sq += wmu * jnp.sum(105. / 16. * jq * pk_q * (D * S + 8. / 63.), axis=-1)
        pk_b22 += wmu / 2. * jnp.sum(jq * pk_q * (pk_kq - pk_q), axis=-1)
        pk_b2s2 += wmu / 2. * jnp.sum(jq * pk_q * (pk_kq * S - 2. / 3. * pk_q), axis=-1)
        pk_bs22 += wmu / 2. * jnp.sum(jq * pk_q * (pk_kq * S**2 - 4. / 9. * pk_q), axis=-1)
        pk22_dd += 2 * wmu * jnp.sum(F2_d**2 * jq_pk_q_pk_kq, axis=-1)
        pk22_dt += 2 * wmu * jnp.sum(F2_d * F2_t * jq_pk_q_pk_kq, axis=-1)
        pk22_tt += 2 * wmu * jnp.sum(F2_t * F2_t * jq_pk_q_pk_kq, axis=-1)

        xmu = kq2 / k**2
        kernel_A[0] = - x**3 / 7. * (mu + 6 * mu**3 + x**2 * mu * (-3 + 10 * mu**2) + x * (-3 + mu**2 - 12 * mu**4))
        kernel_A[1] = x**4 / 14. * (mu**2 - 1) * (-1 + 7 * x * mu - 6 * mu**2)
        kernel_A[2] = x**3 / 14. * (x**2 * mu * (13 - 41 * mu**2) - 4 * (mu + 6 * mu**3) + x * (5 + 9 * mu**2 + 42 * mu**4))
        kernel_A[3] = kernel_A[1]
        kernel_A[4] = x**3 / 14. * (1 - 7 * x * mu + 6 * mu**2) * (-2 * mu + x * (-1 + 3 * mu**2))
        kernel_tA[0] = 1. / 7. * (mu + x - 2 * x * mu**2) * (3 * x + 7 * mu - 10 * x * mu**2)
        kernel_tA[1] = x / 14. * (mu**2 - 1) * (3 * x + 7 * mu - 10 * x * mu**2)
        kernel_tA[2] = 1. / 14. * (28 * mu**2 + x * mu * (25 - 81 * mu**2) + x**2 * (1 - 27 * mu**2 + 54 * mu**4))
        kernel_tA[3] = x / 14. * (1 - mu**2) * (x - 7 * mu + 6 * x * mu**2)
        kernel_tA[4] = 1. / 14. * (x - 7 * mu + 6 * x * mu**2) * (-2 * mu - x + 3 * x * mu**2)
        # Taruya 2010 (arXiv 1006.0699v1) eq A3
        A += wmu * jnp.sum(jq / x**2 * (jnp.array(kernel_A) * pk_k[:, None] + jnp.array(kernel_tA) * pk_q) * pk_kq / xmu**2, axis=-1)

        jq_pk_q_pk_kq /= x**2 * xmu
        B[0] += wmu * jnp.sum(x**2 * (mu**2 - 1.) / 2. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,1,1
        B[1] += wmu * jnp.sum(3. * x**2 * (mu**2 - 1.)**2 / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,1,2
        B[2] += wmu * jnp.sum(3. * x**4 * (mu**2 - 1.)**2 / xmu / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,2,1
        B[3] += wmu * jnp.sum(5. * x**4 * (mu**2 - 1.)**3 / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,2,2
        B[4] += wmu * jnp.sum(x * (x + 2. * mu - 3. * x * mu**2) / 2. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,1,1
        B[5] += wmu * jnp.sum(- 3. * x * (mu**2 - 1.) * (-x - 2. * mu + 5. * x * mu**2) / 4. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,1,2
        B[6] += wmu * jnp.sum(3. * x**2 * (mu**2 - 1.) * (-2. + x**2 + 6. * x * mu - 5. * x**2 * mu**2) / xmu / 4. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,2,1
        B[7] += wmu * jnp.sum(- 3. * x**2 * (mu**2 - 1.)**2 * (6. - 5. * x**2 - 30. * x * mu + 35. * x**2 * mu**2) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,2,2
        B[8] += wmu * jnp.sum(x * (4. * mu * (3. - 5. * mu**2) + x * (3. - 30. * mu**2 + 35. * mu**4)) / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 3,1,2
        B[9] += wmu * jnp.sum(x * (-8. * mu + x * (-12. + 36. * mu**2 + 12. * x * mu * (3. - 5. * mu**2) + x**2 * (3. - 30. * mu**2 + 35. * mu**4))) / xmu / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 3,2,1
        B[10] += wmu * jnp.sum(3. * x * (mu**2 - 1.) * (-8. * mu + x * (-12. + 60. * mu**2 + 20. * x * mu * (3. - 7. * mu**2) + 5. * x**2 * (1. - 14. * mu**2 + 21. * mu**4))) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 3,2,2
        B[11] += wmu * jnp.sum(x * (8. * mu * (-3. + 5. * mu**2) - 6. * x * (3. - 30. * mu**2 + 35. * mu**4) + 6. * x**2 * mu * (15. - 70. * mu**2 + 63 * mu**4) + x**3 * (5. - 21. * mu**2 * (5. - 15. * mu**2 + 11. * mu**4))) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 4,2,2

    A += pk_k * jnp.sum(kernel_a * pk_q, axis=-1)
    B = jnp.vstack(B)
    pk11 = pk_k
    pk13_dd = 2. * jnp.sum(kernel13_d * pk_q, axis=-1) * pk_k
    pk13_tt = 2. * jnp.sum(kernel13_t * pk_q, axis=-1) * pk_k
    pk13_dt = (pk13_dd + pk13_tt) / 2.
    pk_sig3sq = sig3sq * pk_k
    pk_dd = pk11 + pk22_dd + pk13_dd
    pk_dt = pk11 + pk22_dt + pk13_dt
    pk_tt = pk11 + pk22_tt + pk13_tt

    return [pk11, pk_dd, pk_b2d, pk_bs2d, pk_sig3sq, pk_b22, pk_b2s2, pk_bs22, pk_dt, pk_b2t, pk_bs2t, pk_tt, A, B]


class TNSPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):
    r"""
    TNS power spectrum multipoles.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    """
    _default_options = dict(nloop=1, fog='lorentzian')
    _klim = (1e-3, 2., 500)

    def initialize(self, *args, mu=8, **kwargs):
        super(TNSPowerSpectrumMultipoles, self).initialize(*args, mu=mu, method='leggauss', **kwargs)
        self.nloop = int(self.options['nloop'])
        if self.nloop not in [1]:
            raise ValueError('nloop must be 1 (1-loop)')
        if self.options['fog'] not in ['lorentzian', 'gaussian']:
            raise ValueError('fog must be lorentzian or gaussian')

    def calculate(self, sigmav=0):
        super(TNSPowerSpectrumMultipoles, self).calculate()
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f

        if self.options['fog'] == 'lorentzian':
            damping = 1. / (1. + (sigmav * kap * muap)**2 / 2.)**2.
        else:
            damping = jnp.exp(-(sigmav * kap * muap)**2)

        k11 = np.linspace(self.k[0] * 0.7, self.k[-1] * 1.3, int(len(self.k) * 1.6 + 0.5))
        q = self.template.k
        wq = utils.weights_trapz(q)
        jq = q**2 * wq / (4. * np.pi**2)
        if getattr(self, 'kernels', None) is None:
            self.kernels = tns_kernels(k11, q, wq)

        pktable = tns_pt(k11, q, wq, self.template.pk_dd, *self.kernels)
        names = ['pk11', 'pk_dd', 'pk_b2d', 'pk_bs2d', 'pk_sig3sq', 'pk_b22', 'pk_b2s2', 'pk_bs22', 'pk_dt', 'pk_b2t', 'pk_bs2t', 'pk_tt', 'A', 'B']
        pktable = jnp.concatenate([array[None, :] for array in pktable[:-2]] + pktable[-2:], axis=0)
        pktable = jac * damping * jnp.moveaxis(interp1d(jnp.log10(kap), np.log10(k11), pktable.T, method='cubic'), [0, 1], [1, 2])
        A = pktable[12:]
        B = pktable[17:]
        #self._A = A
        #self._B = np.array([B[0], -(B[1] + B[2]), B[3], B[4], -(B[5] + B[6]), B[7], -(B[8] + B[9]), B[10], B[11]])
        A = jnp.array([f * A[0] * muap**2, f**2 * (A[1] * muap**2 + A[2] * muap**4), f**3 * (A[3] * muap**4 + A[4] * muap**6)])  # for b1^2, b1, 1
        B = jnp.array([f**2 * (B[0] * muap**2 + B[4] * muap**4),
                       -f**3 * ((B[1] + B[2]) * muap**2 + (B[5] + B[6]) * muap**4 + (B[8] + B[9]) * muap**6),
                       f**4 * (B[3] * muap**2 + B[7] * muap**4 + B[10] * muap**6 + B[11] * muap**8)])   # for b1^2, b1, 1

        pktable = [self.to_poles(pktable[:8, None]), self.to_poles(f * muap**2 * pktable[8:11, None]), self.to_poles(f**2 * muap**4 * pktable[11:12, None])]
        self.pktable = {}
        for pkt in pktable:
            for pk in pkt: self.pktable[names[len(self.pktable)]] = pk
        self.pktable['A'] = self.to_poles(A[:, None, ...])
        self.pktable['B'] = self.to_poles(B[:, None, ...])

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'nloop','fog']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self.pktable:
            state[name] = self.pktable[name]
        state['names'] = list(self.pktable.keys())
        return state

    def __setstate__(self, state):
        state = dict(state)
        self.pktable = {name: state.pop(name, None) for name in state['names']}
        super(TNSPowerSpectrumMultipoles, self).__setstate__(state)


class TNSTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    r"""
    TNS tracer power spectrum multipoles.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).
    """
    _default_options = dict(freedom=None)

    def set_params(self):
        self.required_bias_params.update(dict(b1=1., b2=0., bs=0., b3=0., sn0=0.))
        super().set_params(pt_params=['sigmav'])
        freedom = self.options.get('freedom', None)
        fix = []
        if freedom == 'max':
            for param in self.init.params.select(basename=['b1', 'b2', 'bs', 'b3']):
                param.update(fixed=False)
            fix += ['alpha6']
        if freedom == 'min':
            fix += ['b3', 'bs']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)

    def calculate(self, b1=1., b2=0., bs=0., b3=0., sn0=0.):
        super(TNSTracerPowerSpectrumMultipoles, self).calculate()
        self.power = b1**2 * self.pt.pktable['pk_dd'] + 2. * b1 * self.pt.pktable['pk_dt'] + self.pt.pktable['pk_tt'] + sn0 / self.nd
        bs2 = bs - 4. / 7. * (b1 - 1.)
        b3nl = b3 + 32. / 315. * (b1 - 1.)
        #bs2 = b3nl = 0.
        self.power += 2 * b1 * b2 * self.pt.pktable['pk_b2d'] + 2. * b1 * bs2 * self.pt.pktable['pk_bs2d']\
                      + 2 * b1 * b3nl * self.pt.pktable['pk_sig3sq'] + b2**2 * self.pt.pktable['pk_b22']\
                      + 2 * b2 * bs2 * self.pt.pktable['pk_b2s2'] + bs2**2 * self.pt.pktable['pk_bs22']\
                      + b2 * self.pt.pktable['pk_b2t'] + b3nl * self.pt.pktable['pk_sig3sq']
        self.power += b1**2 * (self.pt.pktable['A'][0] + self.pt.pktable['B'][0])
        self.power += b1 * (self.pt.pktable['A'][1] + self.pt.pktable['B'][1])
        self.power += (self.pt.pktable['A'][2] + self.pt.pktable['B'][2])


class TNSTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    TNS tracer correlation function multipoles.
    For the matter (unbiased) correlation function, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """


class EFTLikeTNSTracerPowerSpectrumMultipoles(BaseEFTLikeTracerPowerSpectrumMultipoles, TNSTracerPowerSpectrumMultipoles):
    r"""
    TNS tracer power spectrum multipoles with EFT-like counter and stochastic terms.
    Can be exactly marginalized over counter terms and stochastic parameters ct*, sn*.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).
    """


class EFTLikeTNSTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    TNS tracer correlation function multipoles with EFT-like counter and stochastic terms.
    Can be exactly marginalized over counter terms and stochastic parameters ct*, sn*.
    For the matter (unbiased) correlation function, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """


def get_nthreads(nthreads=None):
    if nthreads is None:
        import os
        nthreads = os.getenv('OMP_NUM_THREADS', '1')
    return int(nthreads)


class BaseVelocileptorsPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):

    """Base class for velocileptors-based matter power spectrum multipoles."""
    _default_options = dict()

    def initialize(self, *args, **kwargs):
        super(BaseVelocileptorsPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.options['threads'] = get_nthreads(self.options.pop('nthreads', None))

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/sfschen/velocileptors')

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'wmu', 'sigma8', 'fsigma8']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._pt_attrs:
            if hasattr(self.pt, name):
                state[name] = getattr(self.pt, name)
        return state


def get_physical_stochastic_settings(tracer=None):
    if tracer is not None:
        tracer = str(tracer).upper()
        # Mark Maus, Ruiyang Zhao
        settings = {'BGS': {'fsat': 0.15, 'sigv': 150*(10)**(1/3)*(1+0.2)**(1/2)/70.},
                    'LRG': {'fsat': 0.15, 'sigv': 150*(10)**(1/3)*(1+0.8)**(1/2)/70.},
                    'ELG': {'fsat': 0.10, 'sigv': 150*2.1**(1/2)/70.},
                    'QSO': {'fsat': 0.03, 'sigv': 150*(10)**(0.7/3)*(2.4)**(1/2)/70.}}
        try:
            settings = settings[tracer]
        except KeyError:
            raise ValueError('unknown tracer: {}, please use any of {}'.format(tracer, list(settings.keys())))
    else:
        settings = {'fsat': 0.1, 'sigv': 5.}
    return settings


class BaseVelocileptorsTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):

    """Base class for velocileptors-based tracer power spectrum multipoles."""

    @staticmethod
    def _params(params, freedom=None, prior_basis='physical'):
        fix = []
        if freedom == 'max':
            for param in params.select(basename=['b1', 'b2', 'bs', 'b3']):
                param.update(fixed=False)
            for param in params.select(basename=['b2', 'bs', 'b3']):
                param.update(prior=dict(limits=[-15., 15.]))
            for param in params.select(basename=['alpha*', 'sn*']):
                param.update(prior=None)
            fix += ['alpha6']  #, 'sn4']
        if freedom == 'min':
            fix += ['b3', 'bs', 'alpha6']  #, 'sn4']
            for param in params.select(basename=['b2']):
                param.update(prior=dict(dist='norm', loc=0., scale=10.))
            for param in params.select(basename=['alpha*', 'sn*']):
                param.update(prior=None)
        for param in params.select(basename=fix):
            param.update(value=0., fixed=True)
        if prior_basis == 'physical':
            for param in list(params):
                basename = param.basename
                param.update(basename=basename + 'p')
                #params.set({'basename': basename, 'namespace': param.namespace, 'derived': True})
            for param in params.select(basename='b1p'):
                param.update(prior=dict(dist='uniform', limits=[0., 3.]), ref=dict(dist='norm', loc=1., scale=0.1))
            for param in params.select(basename=['b2p', 'bsp', 'b3p']):
                param.update(prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.))
            for param in params.select(basename='b3p'):
                param.update(value=0., fixed=True)
            for param in params.select(basename='alpha*p'):
                param.update(prior=dict(dist='norm', loc=0., scale=12.5), ref=dict(dist='norm', loc=0., scale=1.))  # 50% at k = 0.2 h/Mpc
            for param in params.select(basename='sn*p'):
                param.update(prior=dict(dist='norm', loc=0., scale=2. if 'sn0' in param.basename else 5.), ref=dict(dist='norm', loc=0., scale=1.))
        return params

    def set_params(self):
        self.is_physical_prior = self.options['prior_basis'] == 'physical'
        if self.is_physical_prior:
            for name in list(self.required_bias_params):
                self.required_bias_params[name + 'p'] = self.required_bias_params.pop(name)
            settings = get_physical_stochastic_settings(tracer=self.options['tracer'])
            for name, value in settings.items():
                if self.options[name] is None: self.options[name] = value
            if self.mpicomm.rank == 0:
                self.log_debug('Using fsat, sigv = {:.3f}, {:.3f}.'.format(self.options['fsat'], self.options['sigv']))
        super().set_params(pt_params=[])
        fix = []
        if 4 not in self.ells: fix += ['alpha4*', 'alpha6*', 'sn4*']  # * to capture p
        if 2 not in self.ells: fix += ['alpha2*', 'sn2*']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)
        self.nd = 1e-4
        self.fsat = self.snd = 1.
        if self.is_physical_prior:
            self.fsat, self.snd = self.options['fsat'], self.options['shotnoise'] * self.nd  # normalized by 1e-4


class BaseVelocileptorsCorrelationFunctionMultipoles(BasePTCorrelationFunctionMultipoles):

    """Base class for velocileptors-based matter correlation function multipoles."""

    def initialize(self, *args, **kwargs):
        super(BaseVelocileptorsCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        self.options['threads'] = get_nthreads(self.options.pop('nthreads', None))

    def combine_bias_terms_poles(self, pars, **opts):
        return np.array([self.pt.compute_xi_ell(ss, self.template.f, *pars, apar=self.template.qpar, aperp=self.template.qper, **self.options, **opts) for ss in self.s]).T


class BaseVelocileptorsTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    """Base class for velocileptors-based tracer correlation function multipoles."""

    def calculate(self, **params):
        super(BaseVelocileptorsTracerCorrelationFunctionMultipoles, self).calculate()
        pars = [params.get(name, value) for name, value in self.required_bias_params.items()]
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        self.corr = self.pt.combine_bias_terms_poles(pars, **opts, **self.options)

@jit
def tablevel_combine_bias_terms_poles(pktable, pars, nd=1e-4):
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4 = pars
    bias_monomials = jnp.array([1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3, alpha0, alpha2, alpha4, alpha6, sn0 / nd, sn2 / nd, sn4 / nd])
    return jnp.sum(pktable * bias_monomials, axis=-1)


class LPTVelocileptorsPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(use_Pzel=False, kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=4000, nthreads=None, jn=5)
    # Speed is linear with the number of output k

    def initialize(self, *args, mu=4, **kwargs):
        super(LPTVelocileptorsPowerSpectrumMultipoles, self).initialize(*args, mu=mu, method='leggauss', **kwargs)

    def calculate(self):
        super(LPTVelocileptorsPowerSpectrumMultipoles, self).calculate()

        def interp1d(x, y):
            return interpolate.interp1d(x, y, kind='cubic', assume_sorted=True)  # for AP

        from velocileptors.LPT import lpt_rsd_fftw
        lpt_rsd_fftw.interp1d = interp1d

        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        self.pt = LPT_RSD(np.asarray(self.template.k), np.asarray(self.template.pk_dd), **self.options)
        self.pt.make_pltable(np.asarray(self.template.f), kv=np.asarray(self.k), apar=np.asarray(self.template.qpar), aperp=np.asarray(self.template.qper), ngauss=len(self.mu))
        pktable = {0: self.pt.p0ktable, 2: self.pt.p2ktable, 4: self.pt.p4ktable}
        self.pktable = np.array([pktable[ell] for ell in self.ells])
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.f * self.sigma8

    def combine_bias_terms_poles(self, pars, nd=1e-4):
        return tablevel_combine_bias_terms_poles(self.pktable, pars, nd=nd)

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'pktable', 'sigma8', 'fsigma8']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/sfschen/velocileptors')


class LPTVelocileptorsTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):
    r"""
    Velocileptors Lagrangian perturbation theory (LPT) tracer power spectrum multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn*.
    For the matter (unbiased) power spectrum, set all bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}) \sigma_{8}(z), b_{2}^\prime = b_{2} \sigma_{8}(z)^2, b_{s}^\prime = b_{s} \sigma_{8}(z)^2, b_{3}^\prime = b_{3} \sigma_{8}(z)^3`
        :math:`\alpha_{0} = (1 + b_{1})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}) \alpha_{4}^\prime), \alpha_{6} = f^{2} \alpha_{4}^\prime`.
        :math:`s_{n, 0} = f_{\mathrm{sat}}/\bar{n} s_{n, 0}^\prime, s_{n, 2} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{2} s_{n, 2}^\prime, s_{n, 4} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{4} s_{n, 4}^\prime`.
        In this case, ``use_Pzel = False``.

    tracer : str, default=None
        If ``prior_basis = 'physical'``, tracer to load preset ``fsat`` and ``sigv``. One of ['LRG', 'ELG', 'QSO'].

    fsat : float, default=None
        If ``prior_basis = 'physical'``, satellite fraction to assume.

    sigv : float, default=None
        If ``prior_basis = 'physical'``, velocity dispersion to assume.

    shotnoise : float, default=1e4
        Shot noise, to scale stochastic terms.

    **kwargs : dict
        Velocileptors options, defaults to: ``use_Pzel=False, kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=4000, nthreads=1, jn=5``.


    Reference
    ---------
    - https://arxiv.org/abs/2005.00523
    - https://arxiv.org/abs/2012.04636
    - https://github.com/sfschen/velocileptors
    """
    _default_options = dict(freedom=None, prior_basis='physical', tracer=None, fsat=None, sigv=None, shotnoise=1e4)

    def initialize(self, *args, k=None, **kwargs):
        super(LPTVelocileptorsTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        if k is not None:
            self.k = np.array(k, dtype='f8')
        # Increasing the resolution, necessary
        boost_prec = 2
        kvec = np.concatenate([[min(0.0005, self.k[0])], np.geomspace(0.0015, 0.025, 10 * boost_prec, endpoint=True), np.arange(0.03, max(0.5, self.k[-1]) + 0.015 / boost_prec, 0.01 / boost_prec)])  # margin for interpolation below (and numerical noise in endpoint)
        ells = kwargs.get('ells', None)
        if ells is not None: self.ells = tuple(ells)
        self.pt.init.update(k=kvec, ells=self.ells, use_Pzel=not self.is_physical_prior)

    def set_params(self):
        self.required_bias_params = dict(b1=1., b2=0., bs=0., b3=0., alpha0=0., alpha2=0., alpha4=0., alpha6=0., sn0=0., sn2=0., sn4=0.)
        super().set_params()

    def calculate(self, **params):
        for name in ['z']:
            setattr(self, name, getattr(self.pt, name))
        params = {**self.required_bias_params, **params}
        if self.is_physical_prior:
            sigma8 = self.pt.sigma8
            f = self.pt.fsigma8 / sigma8
            pars = b1L, b2L, bsL, b3L = [params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p'] / sigma8**3]
            pars += [(1 + b1L)**2 * params['alpha0p'], f * (1 + b1L) * (params['alpha0p'] + params['alpha2p']),
                     f * (f * params['alpha2p'] + (1 + b1L) * params['alpha4p']), f**2 * params['alpha4p']]
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2, 4]]
        else:
            pars = [params[name] for name in self.required_bias_params]
        #self.__dict__.update(dict(zip(['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4'], pars)))  # for derived parameters
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        index = np.array([self.pt.ells.index(ell) for ell in self.ells])
        self.power = interp1d(self.k, self.pt.k, self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nd)[index].T).T
        #self.power = self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nd)


class LPTVelocileptorsTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    Velocileptors LPT tracer correlation function multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn*.
    For the matter (unbiased) correlation function, set all bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}) \sigma_{8}(z), b_{2}^\prime = b_{2} \sigma_{8}(z)^2, b_{s}^\prime = b_{s} \sigma_{8}(z)^2, b_{3}^\prime = b_{3} \sigma_{8}(z)^3`
        :math:`\alpha_{0} = (1 + b_{1})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}) \alpha_{4}^\prime), \alpha_{6} = f^{2} \alpha_{4}^\prime`.

    **kwargs : dict
        Velocileptors options, defaults to: ``use_Pzel=False, kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=4000, nthreads=1, jn=5``.


    Reference
    ---------
    - https://arxiv.org/abs/2005.00523
    - https://arxiv.org/abs/2012.04636
    - https://github.com/sfschen/velocileptors
    """
    _params = LPTVelocileptorsTracerPowerSpectrumMultipoles._params


def f_over_f0_EH(z, k, Omega0_m, h, fnu, Nnu=3, Neff=3.044):
    r"""
    Computes f(k)/f0, adapted from https://github.com/henoriega/FOLPS-nu, following H&E (1998).

    Reference
    ---------
    https://arxiv.org/pdf/astro-ph/9710216

    Parameters
    ----------
    z : float
        Redshift.
    k : array
        Wavenumber.
    Omega0_m : float
        :math:`\Omega_\mathrm{b} + \Omega_\mathrm{c} + \Omega_\nu` (dimensionless matter density parameter).
    h : float
        :math:`H_0 / 100`.
    fnu : float
        :math:`\Omega_\nu / \Omega_\mathrm{m}`.
    Nnu : int, default=3
        Number of massive neutrinos.
    Neff : int, default=3.044
        Effective number of relativistic species.

    Returns
    -------
    fk : array
        :math:`f(k) / f0`
    """
    eta = jnp.log(1 / (1 + z))  # log of scale factor
    Omega0_r = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3) * Neff))  # rad: including neutrinos
    aeq = Omega0_r / Omega0_m  # matter-radiation equality

    pcb = 5./4 - jnp.sqrt(1 + 24*(1 - fnu)) / 4  # neutrino supression
    c = 0.7
    theta272 = (1.00)**2  # T_{CMB} = 2.7*(theta272)
    pf = (k * theta272) / (Omega0_m * h**2)
    DEdS = jnp.exp(eta) / aeq  # growth function: EdS cosmology

    fnunonzero = jnp.where(fnu != 0., fnu, 1.)
    yFS = 17.2*fnu*(1 + 0.488*fnunonzero**(-7/6)) * (pf*Nnu / fnunonzero)**2  #yFreeStreaming
    # pcb = 0. and yFS = 0. when fnu = 0.
    rf = DEdS/(1 + yFS)
    return 1 - pcb/(1 + (rf)**c)  # f(k)/f0


class REPTVelocileptorsPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(rbao=110, sbao=None, beyond_gauss=True,
                            one_loop=True, shear=True, cutoff=20, jn=5, N=4000,
                            nthreads=None, extrap_min=-4, extrap_max=3, import_wisdom=False)
    # Speed does not depend on the number of output k

    def initialize(self, *args, mu=4, **kwargs):
        super(REPTVelocileptorsPowerSpectrumMultipoles, self).initialize(*args, mu=mu, method='leggauss', **kwargs)
        self.template.init.update(with_now='peakaverage')

    def _emulator_initialize(self):
        self._emulator_bak = getattr(self, '_emulator_bak', self.emulator)
        self.emulator = self._emulator_bak.deepcopy()
        if 'z' not in self.init: return
        z = np.asarray(self.init['z'])
        allz = self.emulator.fixed['z']
        if np.allclose(z, allz): return
        if np.any((z < allz[0]) | (z > allz[-1])):
            raise ValueError('input z = {} is outside of the range of emub1 lated z: {} - {}'.format(z, *allz[[0, -1]]))
        iz = np.searchsorted(allz, z, side='right') - 1
        izp1 = np.minimum(iz + 1, len(allz) - 1)
        keepiz = np.unique(np.concatenate([iz, izp1], axis=0))
        allz = allz[keepiz]
        self.emulator.fixed['z'] = z
        iz = np.searchsorted(keepiz, iz, side='right') - 1
        izp1 = np.minimum(iz + 1, len(allz) - 1)
        wz = z - allz[iz]

        from desilike.emulators import Operation

        # Keep only the iz predictions we are interested in (for jaxeffort, maybe we should fix this later)
        for name, engine in self.emulator.engines.items():
            for operation in engine.model_operations + engine.yoperations:
                operation.update(locals={name: value[keepiz] for name, value in operation._locals.items()})
            engine.yshape = keepiz.shape + engine.yshape[1:]
        self.emulator.yoperations.insert(0, Operation("", "{name: v[name][..., iz] * (1 - wz) + v[name][..., iz + 1] * wz if name in ['pktable', 'fsigma8', 'sigma8'] else v[name] for name in v}", locals={'wz': wz, 'iz': iz}))

    def calculate(self):
        super(REPTVelocileptorsPowerSpectrumMultipoles, self).calculate()
        from velocileptors.EPT.ept_fullresum_varyDz_nu_fftw import REPT
        #from velocileptors.EPT.ept_fullresum_fftw import REPT
        pk_dd, pknow_dd = self.template.pk_dd, self.template.pknow_dd
        #print('desilike', self.template.k.min(), self.template.k.max(), self.template.k.size, self.template.pk_dd.sum())
        if self.z.ndim: pk_dd, pknow_dd = pk_dd[..., 0], pknow_dd[..., 0]
        self.pt = REPT(np.asarray(self.template.k), np.asarray(pk_dd), pnw=np.asarray(pknow_dd), kmin=self.k[0], kmax=self.k[-1], nk=200, **self.options)
        # print(self.template.f, self.k.shape, self.template.qpar, self.template.qper, self.template.k.shape, self.template.pk_dd.shape)
        pktable = {ell: [] for ell in [0, 2, 4]}
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.f * self.sigma8
        Omega_m, h, fnu, Neff, Nnu = 0.3, 0.7, 0., 3.046, 3
        #cosmo = getattr(self.template, 'cosmo', None)
        #if cosmo is not None:
        #    Omega_m, h, fnu, Nnu, Neff = cosmo['Omega_m'], cosmo['h'], cosmo['Omega_ncdm_tot'] / cosmo['Omega_m'], cosmo['N_ncdm'], cosmo['N_eff']

        f0, qpar, qper = map(np.asarray, [self.template.f0, self.template.qpar, self.template.qper])
        pcb, pcb_nw, pttcb = [10**interpolate.interp1d(np.log10(self.template.k), np.log10(pk), kind='cubic', fill_value='extrapolate', axis=0, assume_sorted=True)(np.log10(np.append(self.pt.kv, 1.))) for pk in [self.template.pk_dd, self.template.pknow_dd, self.template.pk_dd * self.template.fk**2]]
        fk = np.sqrt(pttcb / pcb)[:-1]
        if self.z.ndim:
            for iz, z in enumerate(self.z):
                Dz = np.sqrt(pcb[-1, iz] / pcb[-1, 0])
                #fk = f0[iz] * f_over_f0_EH(z, self.pt.kv, Omega_m, h, fnu, Nnu=Nnu, Neff=Neff)
                #print(Dz, pcb[:-1, iz].sum(), pcb_nw[:-1, iz].sum(), fk[..., iz].sum())
                pks = self.pt.compute_redshift_space_power_multipoles_tables(fk[..., iz], apar=qpar[iz], aperp=qper[iz], ngauss=len(self.mu), pcb=pcb[:-1, iz], pcb_nw=pcb_nw[:-1, iz], Dz=Dz)[1:]
                for ill, ell in enumerate(pktable): pktable[ell].append(pks[ill])
            pktable = {ell: np.concatenate([v[..., None] for v in value], axis=-1) for ell, value in pktable.items()}
        else:
            #fk = f0 * f_over_f0_EH(self.z, self.pt.kv, Omega_m, h, fnu, Nnu=Nnu, Neff=Neff)
            pks = self.pt.compute_redshift_space_power_multipoles_tables(fk, apar=qpar, aperp=qper, ngauss=len(self.mu))[1:]
            for ill, ell in enumerate(pktable): pktable[ell] = pks[ill]
        self.pktable = interpolate.interp1d(self.pt.kv, np.array([pktable[ell] for ell in self.ells]), kind='cubic', fill_value='extrapolate', axis=1, assume_sorted=True)(self.k)

    def combine_bias_terms_poles(self, pars, z=None, nd=1e-4):
        # Add co-evolution part
        pars = list(pars)
        b1 = pars[0]
        pars[2] = pars[2] - (2 / 7) * (b1 - 1.)  # bs
        pars[3] = 3 * pars[3] + (b1 - 1.)  # b3
        #return interpolate.interp1d(self.pt.kv, np.array(self.pt.compute_redshift_space_power_multipoles(pars, self.template.f)[1:]), kind='cubic', fill_value='extrapolate', axis=1, assume_sorted=True)(self.k)
        pktable = self.pktable
        if z is not None: pktable = pktable[..., list(self.z).index(z)]
        return tablevel_combine_bias_terms_poles(pktable, pars, nd=nd)

    def __getstate__(self, varied=True, fixed=True):
        state = {}
        for name in (['k', 'z', 'ells'] if fixed else []) + (['pktable', 'sigma8', 'fsigma8'] if varied else []):
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/sfschen/velocileptors')


class REPTVelocileptorsTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):
    r"""
    Velocileptors resummmed Eulerian perturbation theory (REPT) tracer power spectrum multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn*.
    For the matter (unbiased) power spectrum, set all bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}^{L}) \sigma_{8}(z), b_{2}^\prime = b_{2}^{L} \sigma_{8}(z)^2, b_{s}^\prime = b_{s}^{L} \sigma_{8}(z)^2, b_{3}^\prime = 0`
        with: :math:`b_{1} = 1 + b_{1}^{L}, b_{2} = 8/21 b_{1}^{L} + b_{2}^{L}, b_{s} = b_{s}^{L}, b_{3} = b_{3}^{L}`.
        :math:`\alpha_{0} = (1 + b_{1}^{L})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}^{L}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}^{L}) \alpha_{4}^\prime)`.
        :math:`s_{n, 0} = f_{\mathrm{sat}}/\bar{n} s_{n, 0}^\prime, s_{n, 2} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{2} s_{n, 2}^\prime, s_{n, 4} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{4} s_{n, 4}^\prime`.

    tracer : str, default=None
        If ``prior_basis = 'physical'``, tracer to load preset ``fsat`` and ``sigv``. One of ['LRG', 'ELG', 'QSO'].

    fsat : float, default=None
        If ``prior_basis = 'physical'``, satellite fraction to assume.

    sigv : float, default=None
        If ``prior_basis = 'physical'``, velocity dispersion to assume.

    shotnoise : float, default=1e4
        Shot noise, to scale stochastic terms.

    **kwargs : dict
        Velocileptors options, defaults to: ``rbao=110, sbao=None, beyond_gauss=True, one_loop=True, shear=True, cutoff=20, jn=5, N=4000, nthreads=None, extrap_min=-4, extrap_max=3``.


    Reference
    ---------
    - https://arxiv.org/abs/2005.00523
    - https://arxiv.org/abs/2012.04636
    - https://github.com/sfschen/velocileptors
    """
    _default_options = dict(freedom=None, prior_basis='physical', tracer=None, fsat=None, sigv=None, shotnoise=1e4)

    def initialize(self, *args, k=None, z=None, **kwargs):
        super(REPTVelocileptorsTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        if k is not None:
            self.k = np.array(k, dtype='f8')
        # Increasing the resolution, necessary
        boost_prec = 4
        kvec = np.concatenate([[min(0.0005, self.k[0])], np.geomspace(0.0015, 0.025, 10 * boost_prec, endpoint=True), np.arange(0.03, max(0.5, self.k[-1]) + 0.015 / boost_prec, 0.01 / boost_prec)])  # margin for interpolation below (and numerical noise in endpoint)
        ells = kwargs.get('ells', None)
        if ells is not None: self.ells = tuple(ells)
        self.pt.init.update(k=kvec, ells=self.ells)
        if z is not None:
            self.z = float(z)
            z = self.pt.init.get('z', [])
            if self.z not in z: z.append(self.z)
            self.pt.init.update(z=sorted(z))

    def set_params(self):
        self.required_bias_params = dict(b1=1., b2=0., bs=0., b3=0., alpha0=0., alpha2=0., alpha4=0., alpha6=0., sn0=0., sn2=0., sn4=0.)
        super().set_params()

    def calculate(self, **params):
        if self.pt.z.ndim == 0: self.z = self.pt.z
        params = {**self.required_bias_params, **params}
        if self.is_physical_prior:
            sigma8 = self.pt.sigma8
            f = self.pt.fsigma8 / sigma8
            if self.pt.z.ndim:
                iz = list(self.pt.z).index(self.z)
                sigma8, f = sigma8[iz], f[iz]
            # b1_E = 1+b1_L
            # b2_E = b2_L + (8/21)*b1_L
            # bs_E = bs_L - (2/7)*b1_L
            # b3_E = 3*b3_L + b1_L
            pars = b1L, b2L, bsL, b3L = [params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p'] / sigma8**3]
            pars = [1. + b1L, 8. / 21. * b1L + b2L, bsL, b3L]
            pars += [(1 + b1L)**2 * params['alpha0p'], f * (1 + b1L) * (params['alpha0p'] + params['alpha2p']),
                     f * (f * params['alpha2p'] + (1 + b1L) * params['alpha4p']), f**2 * params['alpha4p']]
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2, 4]]
        else:
            pars = [params[name] for name in self.required_bias_params]
        #self.__dict__.update(dict(zip(['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4'], pars)))  # for derived parameters
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        index = np.array([self.pt.ells.index(ell) for ell in self.ells])
        if self.pt.z.ndim: opts['z'] = self.z
        self.power = interp1d(self.k, self.pt.k, self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nd)[index].T).T
        #self.power = self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nd)


class REPTVelocileptorsTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    Velocileptors REPT tracer correlation function multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn*.
    For the matter (unbiased) correlation function, set all bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}^{L}) \sigma_{8}(z), b_{2}^\prime = b_{2}^{L} \sigma_{8}(z)^2, b_{s}^\prime = b_{s}^{L} \sigma_{8}(z)^2, b_{3}^\prime = 0`
        with: :math:`b_{1} = 1 + b_{1}^{L}, b_{2} = 8/21 b_{1}^{L} + b_{2}^{L}, b_{s} = b_{s}^{L}, b_{3} = b_{3}^{L}`.
        :math:`\alpha_{0} = (1 + b_{1}^{L})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}^{L}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}^{L}) \alpha_{4}^\prime)`.

    **kwargs : dict
        Velocileptors options, defaults to: ``rbao=110, sbao=None, beyond_gauss=True, one_loop=True, shear=True, cutoff=20, jn=5, N=4000, nthreads=None, extrap_min=-4, extrap_max=3``.


    Reference
    ---------
    - https://arxiv.org/abs/2005.00523
    - https://arxiv.org/abs/2012.04636
    - https://github.com/sfschen/velocileptors
    """
    _params = REPTVelocileptorsTracerPowerSpectrumMultipoles._params


class PyBirdPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    _default_options = dict(km=0.7, kr=0.25, accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False, with_stoch=True, with_resum='full', eft_basis='eftoflss')
    _klim = (1e-3, 11., 3000)  # numerical instability in pybird's fftlog at 10.
    _pt_attrs = ['co', 'f', 'eft_basis', 'with_stoch', 'with_nnlo_counterterm', 'with_tidal_alignments',
                 'P11l', 'Ploopl', 'Pctl', 'Pstl', 'Pnnlol', 'C11l', 'Cloopl', 'Cctl', 'Cstl', 'Cnnlol']

    def initialize(self, *args, **kwargs):
        super(PyBirdPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        # self.co is fixed, so we can just export it in __getstate__
        from pybird.common import Common
        from pybird.nonlinear import NonLinear
        from pybird.nnlo import NNLO_counterterm
        from pybird.resum import Resum
        from pybird.projection import Projection
        eft_basis = self.options.get('eft_basis', None)
        if eft_basis in [None, 'velocileptors']: eft_basis = 'eftoflss'
        # nd used by combine_bias_terms_poles only
        #self.co = Common(Nl=len(self.ells), kmin=self.k[0] * 0.8, kmax=self.k[-1] * 1.2, km=self.options['km'], kr=self.options['kr'], nd=1e-4,
        # No way to go below kmin = 1e-3 h/Mpc (nan)
        if self.k[0] * 0.8 < 1e-3:
            import warnings
            warnings.warn('pybird does not predict P(k) for k < 0.001 h/Mpc; nan will be replaced by 0')
        self.co = Common(Nl=len(self.ells), kmin=1e-3, kmax=self.k[-1] * 1.3, km=self.options['km'], kr=self.options['kr'], nd=1e-4,
                         eft_basis=eft_basis, halohalo=True, with_cf=False,
                         with_time=True, accboost=float(self.options['accboost']), optiresum=self.options['with_resum'] == 'opti', with_uvmatch=False,
                         exact_time=False, quintessence=False, with_tidal_alignments=False, nonequaltime=False, keep_loop_pieces_independent=False)
        #print(dict(Nl=len(self.ells), kmin=1e-3, kmax=self.k[-1] * 1.3, km=self.options['km'], kr=self.options['kr'], nd=1e-4,
        #                 eft_basis=eft_basis, halohalo=True, with_cf=False,
        #                 with_time=True, accboost=float(self.options['accboost']), optiresum=self.options['with_resum'] == 'opti',
        #                 exact_time=False, quintessence=False, with_tidal_alignments=False, nonequaltime=False, keep_loop_pieces_independent=False))
        self.nonlinear = NonLinear(load=False, save=False, NFFT=256 * int(self.options['fftaccboost']), fftbias=self.options['fftbias'], co=self.co)
        #print(dict(load=False, save=False, NFFT=256 * int(self.options['fftaccboost']), fftbias=self.options['fftbias'], co=self.co))
        self.resum = Resum(co=self.co)
        self.nnlo_counterterm = None
        if self.options['with_nnlo_counterterm']:
            self.nnlo_counterterm = NNLO_counterterm(co=self.co)
            self.template.init.update(with_now='peakaverage')
        self.projection = Projection(self.k, with_ap=True, H_fid=None, D_fid=None, co=self.co)  # placeholders for H_fid and D_fid, as we will provide q's

    def calculate(self):
        super(PyBirdPowerSpectrumMultipoles, self).calculate()
        from pybird.bird import Bird
        cosmo = {'kk': self.template.k, 'pk_lin': self.template.pk_dd, 'pk_lin_2': None, 'f': self.template.f, 'DA': 1., 'H': 1.}
        self.pt = Bird(cosmo, with_bias=False, eft_basis=self.co.eft_basis, with_stoch=self.options['with_stoch'], with_nnlo_counterterm=self.nnlo_counterterm is not None, co=self.co)

        if self.nnlo_counterterm is not None:  # we use smooth power spectrum since we don't want spurious BAO signals
            from scipy import interpolate
            self.nnlo_counterterm.Ps(self.pt, interpolate.interp1d(np.log(self.template.k), np.log(self.template.pknow_dd), fill_value='extrapolate', assume_sorted=True))

        self.nonlinear.PsCf(self.pt)
        self.pt.setPsCfl()

        if self.options['with_resum']:
            self.resum.PsCf(self.pt, makeIR=True, makeQ=True, setIR=True, setPs=True, setCf=False)

        self.projection.AP(self.pt, q=(self.template.qper, self.template.qpar))
        self.projection.xdata(self.pt)

    def combine_bias_terms_poles(self, params, nd=1e-4):
        from pybird import bird
        bird.np = jnp
        self.pt.co.nd = nd
        self.pt.setreducePslb(params, what='full')
        bird.np = np
        return jnp.nan_to_num(self.pt.fullPs, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._pt_attrs:
            if hasattr(self.pt, name):
                state[name] = getattr(self.pt, name)
        return state

    def __setstate__(self, state):
        for name in ['k', 'z', 'ells']:
            if name in state: setattr(self, name, state.pop(name))
        from pybird import bird
        self.pt = bird.Bird.__new__(bird.Bird)
        self.pt.with_bias = False
        self.pt.__dict__.update(state)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/pierrexyz/pybird')


class PyBirdTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    """
    Pybird tracer power spectrum multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters c* and bias term b3*.
    For the matter (unbiased) power spectrum, set b1=1, b2=1, b3=1 (eft_basis='eftoflss') and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).

    **kwargs : dict
        Pybird options, defaults to: ``with_nnlo_higher_derivative=False, with_nnlo_counterterm=False, with_stoch=True, with_resum='full'``.


    Reference
    ---------
    - https://arxiv.org/abs/2003.07956
    - https://github.com/pierrexyz/pybird
    """
    _default_options = dict(with_nnlo_counterterm=False, with_stoch=True, eft_basis=None, freedom=None, shotnoise=1e4)

    @staticmethod
    def _params(params, freedom=None):
        fix = []
        if freedom in ['min', 'max']:
            for param in params.select(basename=['b1']):
                param.update(prior=dict(limits=[0., 4.]))
            for param in params.select(basename=['b4']):
                param.update(prior=dict(limits=[-15., 15.]))
            for param in params.select(basename=['b2', 'b3', 'bs', 'b2p4', 'b2m4', 'b2t', 'b2g', 'b3g', 'c*']):
                param.update(prior=None)
        if freedom == 'max':
            for param in params.select(basename=['b1', 'b2', 'b3', 'b4', 'bs', 'b2p4', 'b2m4', 'b2t', 'b2g', 'b3g']):
                param.update(fixed=False)
            fix += ['ce1']
        if freedom == 'min':
            fix += ['b2', 'b3', 'ce1']
        for param in params.select(basename=fix):
            param.update(value=0., fixed=True)
        return params

    def set_params(self):
        freedom = self.options.get('freedom', None)
        if self.options['eft_basis'] is None:
            self.options['eft_basis'] = 'eftoflss' if freedom == 'min' else 'westcoast'
        allowed_eft_basis = ['eftoflss', 'velocileptors', 'eastcoast', 'westcoast']
        if self.options['eft_basis'] not in allowed_eft_basis:
            raise ValueError('eft_basis must be one of {}'.format(allowed_eft_basis))
        if freedom == 'min' and self.options['eft_basis'] != 'eftoflss':
            raise ValueError('freedom = "min" only defined in eft_basis = "eftoflss"')
        # in pybird:
        # - westcoast: c2, c4 are b2p4, b2m4
        # - eastcoast: b2t, b2g, b3g are bt2, bG2, bGamma3
        if self.options['eft_basis'] == 'eftoflss':
            self.required_bias_params = ['b1', 'b2', 'b3', 'b4']
        if self.options['eft_basis'] == 'velocileptors':
            self.required_bias_params = ['b1', 'b2', 'bs', 'b3']
        if self.options['eft_basis'] == 'westcoast':
            self.required_bias_params = ['b1', 'b2p4', 'b3', 'b2m4']
        if self.options['eft_basis'] == 'eastcoast':
            self.required_bias_params = ['b1', 'b2t', 'b2g', 'b3g']
        self.pt.init.update(eft_basis=self.options['eft_basis'])
        # now EFT parameters
        if self.options['eft_basis'] in ['eftoflss', 'velocileptors', 'westcoast']:
            self.required_bias_params += ['cct', 'cr1', 'cr2']
            if self.options['with_nnlo_counterterm']: self.required_bias_params += ['cr4', 'cr6']
        else:
            self.required_bias_params += ['c0', 'c2', 'c4']
            if self.options['with_nnlo_counterterm']: self.required_bias_params += ['ct']
        # now shotnoise
        if self.options['with_stoch']:
            self.required_bias_params += ['ce0', 'ce1', 'ce2']
        default_values = {'b1': 1.6}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        BaseTracerPowerSpectrumMultipoles.set_params(self, pt_params=[])  # not super, for PyBirdTracerCorrelationFunctionMultipoles
        fix = []
        if 4 not in self.ells: fix += ['cr2', 'c4']
        if 2 not in self.ells: fix += ['cr1', 'c2', 'ce2']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)

    def transform_params(self, **params):
        if self.options['eft_basis'] == 'westcoast':
            b2p4, b2m4 = [params.pop(name) for name in ['b2p4', 'b2m4']]
            params['b2'] = (b2p4 + b2m4) / 2.**0.5
            params['b4'] = (b2p4 - b2m4) / 2.**0.5
        elif self.options['eft_basis'] == 'eastcoast':
            b2g, b2t, b3g = [params.pop(name) for name in ['b2g', 'b2t', 'b3g']]
            params['b2'] = params['b1'] + 7. / 2. * b2g
            params['b3'] = params['b1'] + 15. * b2g + 6. * b3g
            params['b4'] = 1 / 2. * b2t - 7. / 2. * b2g
        elif self.options['eft_basis'] == 'velocileptors':
            b1v, b2v, bsv, b3v = [params.pop(name) for name in ['b1', 'b2', 'bs', 'b3']]
            params['b1'] = b1v # + 1 - 1
            params['b2'] = 1. + 7. / 2. * bsv
            params['b3'] = 7. / 441. * (42. - 145. * b1v - 21. * b3v + 630. * bsv)
            params['b4'] = -7. / 5. * (params['b1'] - 1.) - 7. / 10. * b2v
        if self.options['freedom'] == 'min':
            params['b2'] = 1.
            params['b3'] = (294. - 1015. * (params['b1'] - 1.)) / 441.
        return params

    def calculate(self, **params):
        super(PyBirdTracerPowerSpectrumMultipoles, self).calculate()
        self.power = self.pt.combine_bias_terms_poles(self.transform_params(**params), nd=self.nd)


class PyBirdCorrelationFunctionMultipoles(BasePTCorrelationFunctionMultipoles):

    _default_options = dict(km=0.7, kr=0.25, accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False, with_stoch=False, with_resum='full', eft_basis='eftoflss')
    _klim = (1e-3, 11., 3000)  # numerical instability in pybird's fftlog at 10.
    _pt_attrs = ['co', 'f', 'eft_basis', 'with_stoch', 'with_nnlo_counterterm', 'with_tidal_alignments',
                 'P11l', 'Ploopl', 'Pctl', 'Pstl', 'Pnnlol', 'C11l', 'Cloopl', 'Cctl', 'Cstl', 'Cnnlol']

    def initialize(self, *args, **kwargs):
        super(PyBirdCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        from pybird.common import Common
        from pybird.nonlinear import NonLinear
        from pybird.nnlo import NNLO_counterterm
        from pybird.resum import Resum
        from pybird.projection import Projection
        eft_basis = self.options.get('eft_basis', None)
        if eft_basis in [None, 'velocileptors']: eft_basis = 'eftoflss'
        # nd used by combine_bias_terms_poles only
        self.co = Common(Nl=len(self.ells), kmin=1e-3, kmax=0.25, km=self.options['km'], kr=self.options['kr'], nd=1e-4,
                         eft_basis=eft_basis, halohalo=True, with_cf=True,
                         with_time=True, accboost=float(self.options['accboost']), optiresum=self.options['with_resum'] == 'opti', with_uvmatch=False,
                         exact_time=False, quintessence=False, with_tidal_alignments=False, nonequaltime=False, keep_loop_pieces_independent=False)
        #print(dict(Nl=len(self.ells), kmin=1e-3, kmax=0.25, km=self.options['km'], kr=self.options['kr'], nd=1e-4,
        #                 eft_basis=eft_basis, halohalo=True, with_cf=True,
        #                 with_time=True, accboost=float(self.options['accboost']), optiresum=self.options['with_resum'] == 'opti', with_uvmatch=False,
        #                 exact_time=False, quintessence=False, with_tidal_alignments=False, nonequaltime=False, keep_loop_pieces_independent=False))
        self.nonlinear = NonLinear(load=False, save=False, NFFT=256 * int(self.options['fftaccboost']), fftbias=self.options['fftbias'], co=self.co)  # NFFT=256, fftbias=-1.6
        #print(dict(load=False, save=False, NFFT=256 * int(self.options['fftaccboost']), fftbias=self.options['fftbias'], co=self.co))
        self.resum = Resum(co=self.co)  # LambdaIR=.2, NFFT=192
        self.nnlo_counterterm = None
        if self.options['with_nnlo_counterterm']:
            self.nnlo_counterterm = NNLO_counterterm(co=self.co)
            self.template.init.update(with_now='peakaverage')
        self.projection = Projection(self.s, with_ap=True, H_fid=None, D_fid=None, co=self.co)  # placeholders for H_fid and D_fid, as we will provide q's

    def calculate(self):
        super(PyBirdCorrelationFunctionMultipoles, self).calculate()
        from pybird.bird import Bird
        cosmo = {'kk': self.template.k, 'pk_lin': self.template.pk_dd, 'pk_lin_2': None, 'f': self.template.f, 'DA': 1., 'H': 1.}
        self.pt = Bird(cosmo, with_bias=False, eft_basis=self.co.eft_basis, with_stoch=self.options['with_stoch'], with_nnlo_counterterm=self.nnlo_counterterm is not None, co=self.co)
        #print(dict(with_bias=False, eft_basis=self.co.eft_basis, with_stoch=self.options['with_stoch'], with_nnlo_counterterm=self.nnlo_counterterm is not None, co=self.co))
        if self.nnlo_counterterm is not None:  # we use smooth power spectrum since we don't want spurious BAO signals
            from scipy import interpolate
            self.nnlo_counterterm.Cf(self.pt, interpolate.interp1d(np.log(self.template.k), np.log(self.template.pknow_dd), fill_value='extrapolate', assume_sorted=True))

        self.nonlinear.PsCf(self.pt)
        self.pt.setPsCfl()

        if self.options['with_resum']:
            self.resum.PsCf(self.pt, makeIR=True, makeQ=True, setIR=True, setPs=True, setCf=True)

        self.projection.AP(self.pt, q=(self.template.qper, self.template.qpar))
        self.projection.xdata(self.pt)

    def combine_bias_terms_poles(self, params, nd=1e-4):
        from pybird import bird
        bird.np = jnp
        self.pt.co.nd = nd
        self.pt.setreduceCflb(params, what='full')
        bird.np = np
        return self.pt.fullCf

    def __getstate__(self):
        state = {}
        for name in ['s', 'z', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._pt_attrs:
            if hasattr(self.pt, name):
                state[name] = getattr(self.pt, name)
        return state

    def __setstate__(self, state):
        for name in ['s', 'z', 'ells']:
            if name in state: setattr(self, name, state.pop(name))
        from pybird import bird
        self.pt = bird.Bird.__new__(bird.Bird)
        self.pt.with_bias = False
        self.pt.__dict__.update(state)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/pierrexyz/pybird')


class PyBirdTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):
    """
    Pybird tracer correlation function multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters c* and bias term b3*.
    For the matter (unbiased) correlation function, set b1=1, b2=1, b3=1 (eft_basis='eftoflss') and all other bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    **kwargs : dict
        Pybird options, defaults to: ``with_nnlo_higher_derivative=False, with_nnlo_counterterm=False, with_stoch=False, with_resum='full'``.
    """
    _default_options = dict(with_nnlo_counterterm=False, with_stoch=False, eft_basis=None, freedom=None)

    _params = PyBirdTracerPowerSpectrumMultipoles._params

    set_params = PyBirdTracerPowerSpectrumMultipoles.set_params

    transform_params = PyBirdTracerPowerSpectrumMultipoles.transform_params

    def calculate(self, **params):
        super(PyBirdTracerCorrelationFunctionMultipoles, self).calculate()
        self.corr = self.pt.combine_bias_terms_poles(self.transform_params(**params))


class Namespace(object):

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)

@jit
def folps_combine_bias_terms_pkmu(k, mu, jac, f0, table, table_now, sigma2t, pars, nd=1e-4):
    import FOLPSnu as FOLPS
    pars = list(pars) + [1. / nd]  # add shot noise
    b1 = pars[0]
    # Add co-evolution part
    pars[2] = pars[2] - 4. / 7. * (b1 - 1.)  # bs
    pars[3] = pars[3] + 32. / 315. * (b1 - 1.)  # b3
    FOLPS.f0 = f0
    fk = table[1] * f0
    pkl, pkl_now, sigma2t = table[0], table_now[0], sigma2t
    pkmu = jac * ((b1 + fk * mu**2)**2 * (pkl_now + jnp.exp(-k**2 * sigma2t)*(pkl - pkl_now)*(1 + k**2 * sigma2t))
                   + jnp.exp(-k**2 * sigma2t) * FOLPS.PEFTs(k, mu, pars, table)
                   + (1 - jnp.exp(-k**2 * sigma2t)) * FOLPS.PEFTs(k, mu, pars, table_now))
    return pkmu


class FOLPSPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):

    _default_options = dict(kernels='fk')
    _pt_attrs = ['kap', 'muap', 'table', 'table_now', 'sigma2t', 'f0', 'jac']

    def initialize(self, *args, mu=6, **kwargs):
        super(FOLPSPowerSpectrumMultipoles, self).initialize(*args, mu=mu, method='leggauss', **kwargs)
        import FOLPSnu as FOLPS
        FOLPS.Matrices()
        self.matrices = Namespace(**{name: getattr(FOLPS, name) for name in ['M22matrices', 'M13vectors', 'bnu_b', 'N']})

    def calculate(self):
        super(FOLPSPowerSpectrumMultipoles, self).calculate()
        import FOLPSnu as FOLPS
        FOLPS.__dict__.update(self.matrices.__dict__)
        # [z, omega_b, omega_cdm, omega_ncdm, h]
        # only used for neutrinos
        # sensitive to omega_b + omega_cdm, not omega_b, omega_cdm separately
        cosmo_params = [self.z, 0.022, 0.12, 0., 0.7]
        cosmo = getattr(self.template, 'cosmo', None)
        if cosmo is not None:
            cosmo_params = [self.z, cosmo['omega_b'], cosmo['omega_cdm'], cosmo['omega_ncdm_tot'], cosmo['h']]
        FOLPS.NonLinear([self.template.k, self.template.pk_dd], cosmo_params, kminout=self.k[0] * 0.7, kmaxout=self.k[-1] * 1.3, nk=max(len(self.k), 120),
                        EdSkernels=self.options['kernels'] == 'eds')
        #FOLPS.NonLinear([self.template.k, self.template.pk_dd], cosmo_params, kminout=0.001, kmaxout=0.5, nk=120,
        #                EdSkernels=self.options['kernels'] == 'eds')
        k = FOLPS.kTout
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        FOLPS.f0 = f0 = self.template.f0  # for Sigma2Total
        table = FOLPS.Table_interp(kap, k, FOLPS.TableOut_interp(k))
        table_now = FOLPS.TableOut_NW_interp(k)
        sigma2t = FOLPS.Sigma2Total(k, muap, table_now)
        table_now = FOLPS.Table_interp(kap, k, table_now)
        self.pt = Namespace(kap=kap, muap=muap, table=table, table_now=table_now, sigma2t=sigma2t, f0=f0, jac=jac)
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.f * self.sigma8

    def combine_bias_terms_poles(self, pars, nd=1e-4):
        return self.to_poles(folps_combine_bias_terms_pkmu(self.pt.kap, self.pt.muap, self.pt.jac, self.pt.f0,
                                                           self.pt.table, self.pt.table_now, self.pt.sigma2t, pars, nd=nd))

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'wmu', 'sigma8', 'fsigma8']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._pt_attrs:
            if hasattr(self.pt, name):
                state[name] = getattr(self.pt, name)
        return state

    def __setstate__(self, state):
        for name in ['k', 'z', 'ells', 'wmu', 'sigma8', 'fsigma8']:
            if name in state: setattr(self, name, state.pop(name))
        self.pt = Namespace(**state)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/henoriega/FOLPS-nu')


class FOLPSTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    r"""
    FOLPS tracer power spectrum multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn* and bias term b3*.
    By default, bs and b3 are fixed to 0, following co-evolution.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}^{L}) \sigma_{8}(z), b_{2}^\prime = b_{2}^{L} \sigma_{8}(z)^2, b_{s}^\prime = b_{s}^{L} \sigma_{8}(z)^2, b_{3}^\prime = 0`
        with: :math:`b_{1} = 1 + b_{1}^{L}, b_{2} = 8/21 b_{1}^{L} + b_{2}^{L}, b_{s} = -4/7 b_{1}^{L} + b_{s}^{L}`.
        :math:`\alpha_{0} = (1 + b_{1}^{L})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}^{L}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}^{L}) \alpha_{4}^\prime)`.
        :math:`s_{n, 0} = f_{\mathrm{sat}}/\bar{n} s_{n, 0}^\prime, s_{n, 2} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{2} s_{n, 2}^\prime, s_{n, 4} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{4} s_{n, 4}^\prime`.

    tracer : str, default=None
        If ``prior_basis = 'physical'``, tracer to load preset ``fsat`` and ``sigv``. One of ['LRG', 'ELG', 'QSO'].

    fsat : float, default=None
        If ``prior_basis = 'physical'``, satellite fraction to assume.

    sigv : float, default=None
        If ``prior_basis = 'physical'``, velocity dispersion to assume.

    Reference
    ---------
    - https://arxiv.org/abs/2208.02791
    - https://github.com/henoriega/FOLPS-nu
    """
    _default_options = dict(freedom=None, prior_basis='standard', tracer=None, fsat=None, sigv=None, shotnoise=1e4)

    @staticmethod
    def _params(params, freedom=None, prior_basis='standard'):
        fix = []
        if freedom in ['min', 'max']:
            for param in params.select(basename=['b1']):
                param.update(prior=dict(limits=[0., 10.]))
            for param in params.select(basename=['b2']):
                param.update(prior=dict(limits=[-50., 50.]))
            for param in params.select(basename=['bs', 'b3', 'alpha*', 'sn*']):
                param.update(prior=None)
        if freedom == 'max':
            for param in params.select(basename=['b1', 'b2', 'bs', 'b3']):
                param.update(fixed=False)
            fix += ['ct']
        if freedom == 'min':
            fix += ['b3', 'bs', 'ct']
        for param in params.select(basename=fix):
            param.update(value=0., fixed=True)
        if prior_basis == 'physical':
            for param in list(params):
                basename = param.basename
                param.update(basename=basename + 'p')
                #params.set({'basename': basename, 'namespace': param.namespace, 'derived': True})
            for param in params.select(basename='b1p'):
                param.update(prior=dict(dist='uniform', limits=[0., 3.]), ref=dict(dist='norm', loc=1., scale=0.1))
            for param in params.select(basename=['b2p', 'bsp', 'b3p']):
                param.update(prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.))
            for param in params.select(basename='b3p'):
                param.update(value=0., fixed=True)
            for param in params.select(basename='alpha*p'):
                param.update(prior=dict(dist='norm', loc=0., scale=12.5), ref=dict(dist='norm', loc=0., scale=1.))  # 50% at k = 0.2 h/Mpc
            for param in params.select(basename='sn*p'):
                param.update(prior=dict(dist='norm', loc=0., scale=2. if 'sn0' in param.basename else 5.), ref=dict(dist='norm', loc=0., scale=1.))
        return params

    def set_params(self):
        self.required_bias_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'ct', 'sn0', 'sn2']
        default_values = {'b1': 2.}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        self.is_physical_prior = self.options['prior_basis'] == 'physical'
        if self.is_physical_prior:
            for name in list(self.required_bias_params):
                self.required_bias_params[name + 'p'] = self.required_bias_params.pop(name)
            settings = get_physical_stochastic_settings(tracer=self.options['tracer'])
            for name, value in settings.items():
                if self.options[name] is None: self.options[name] = value
            if self.mpicomm.rank == 0:
                self.log_debug('Using fsat, sigv = {:.3f}, {:.3f}.'.format(self.options['fsat'], self.options['sigv']))
        super().set_params(pt_params=[])
        fix = []
        if 4 not in self.ells: fix += ['alpha4']
        if 2 not in self.ells: fix += ['alpha2', 'sn2']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)
        self.nd = 1e-4
        self.fsat = self.snd = 1.
        if self.is_physical_prior:
            self.fsat, self.snd = self.options['fsat'], self.options['shotnoise'] * self.nd  # normalized by 1e-4

    def calculate(self, **params):
        super(FOLPSTracerPowerSpectrumMultipoles, self).calculate()
        params = {**self.required_bias_params, **params}
        if self.is_physical_prior:
            sigma8 = self.pt.sigma8
            f = self.pt.fsigma8 / sigma8
            # b1E = b1L + 1
            # b2E = 8/21 * b1L + b2L
            # bsE = -4/7 b1L + bsL
            b1L, b2L, bsL, b3 = params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p']
            pars = [1. + b1L, b2L + 8. / 21. * b1L, bsL, b3]  # compensate bs by 4. / 7. * b1L as it is removed by combine_bias_terms_poles below
            pars += [(1 + b1L)**2 * params['alpha0p'], f * (1 + b1L) * (params['alpha0p'] + params['alpha2p']),
                     f * (f * params['alpha2p'] + (1 + b1L) * params['alpha4p']), 0.]
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2]]
        else:
            pars = [params[name] for name in self.required_bias_params]
        #self.__dict__.update(dict(zip(['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2'], pars)))  # for derived parameters
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        self.power = self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nd)


class FOLPSTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    FOLPS tracer correlation function multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn* and bias term b3*.
    By default, bs and b3 are fixed to 0, following co-evolution.
    For the matter (unbiased) correlation function, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    prior_basis : str, default='physical'
        :math:`b_{1}^\prime = (1 + b_{1}^{L}) \sigma_{8}(z), b_{2}^\prime = b_{2}^{L} \sigma_{8}(z)^2, b_{s}^\prime = b_{s}^{L} \sigma_{8}(z)^2, b_{3}^\prime = 0`
        with: :math:`b_{1} = 1 + b_{1}^{L}, b_{2} = 8/21 b_{1}^{L} + b_{2}^{L}, b_{s} = -4/7 b_{1}^{L} + b_{s}^{L}`.
        :math:`\alpha_{0} = (1 + b_{1}^{L})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}^{L}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}^{L}) \alpha_{4}^\prime)`.

    Reference
    ---------
    - https://arxiv.org/abs/2208.02791
    - https://github.com/cosmodesi/folpsax
    """
    _params = FOLPSTracerPowerSpectrumMultipoles._params


class FOLPSAXPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, BaseTheoryPowerSpectrumMultipolesFromWedges):

    _default_options = dict(kernels='fk', rbao=104.)
    _pt_attrs = ['jac', 'kap', 'muap', 'table', 'table_now', 'scalars', 'scalars_now']

    def initialize(self, *args, mu=6, **kwargs):
        super(FOLPSAXPowerSpectrumMultipoles, self).initialize(*args, mu=mu, method='leggauss', **kwargs)
        from folpsax import get_mmatrices
        self.matrices = get_mmatrices()
        self.template.init.update(with_now='peakaverage')

    def calculate(self):
        super(FOLPSAXPowerSpectrumMultipoles, self).calculate()
        # [z, omega_b, omega_cdm, omega_ncdm, h]
        # only used for neutrinos
        # sensitive to omega_b + omega_cdm, not omega_b, omega_cdm separately
        #cosmo_params = {'z': self.z, 'fnu': 0., 'Omega_m': 0.3, 'h': 0.7}
        #cosmo = getattr(self.template, 'cosmo', None)
        #if cosmo is not None:
        #    cosmo_params['fnu'] = cosmo['Omega_ncdm_tot'] / cosmo['Omega_m']
        #    cosmo_params['Omega_m'] = cosmo['Omega_m']
        #    cosmo_params['h'] = cosmo['h']
        #    cosmo_params['Nnu'] = cosmo['N_ncdm']
        #    cosmo_params['Neff'] = cosmo['N_eff']
        #cosmo_params['f0'] = self.template.f0
        cosmo_params = {}
        cosmo_params['pkttlin'] = self.template.pk_dd * self.template.fk**2

        if getattr(self, '_get_non_linear', None) is None:

            from folpsax import get_non_linear

            def _get_non_linear(pk_dd, pknow_dd, **cosmo_params):
                return get_non_linear(self.template.k, pk_dd, self.matrices, pknow=pknow_dd,
                                      kminout=self.k[0] * 0.7, kmaxout=self.k[-1] * 1.3, nk=max(len(self.k), 150),
                                      kernels=self.options['kernels'], rbao=self.options['rbao'], **cosmo_params)

            self._get_non_linear = jit(_get_non_linear)

        table, table_now = self._get_non_linear(self.template.pk_dd, self.template.pknow_dd, **cosmo_params)

        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        self.pt = Namespace(jac=jac, kap=kap, muap=muap, table=table[1:26], table_now=table_now[1:26], scalars=table[26:], scalars_now=table_now[26:])
        self.kt = table[0]
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.f * self.sigma8

    def combine_bias_terms_poles(self, pars, nd=1e-4):
        table = (self.kt,) + tuple(self.pt.table) + tuple(self.pt.scalars)
        table_now = (self.kt,) + tuple(self.pt.table_now) + tuple(self.pt.scalars_now)
        pars = list(pars) + [1. / nd]  # add shot noise
        b1 = pars[0]
        # add co-evolution part
        pars[2] = pars[2] - 4. / 7. * (b1 - 1.)  # bs
        pars[3] = pars[3] + 32. / 315. * (b1 - 1.)  # b3
        ncols = len(table)

        if getattr(self, '_get_poles', None) is None:

            from folpsax import get_rsd_pkmu

            def _get_poles(jac, kap, muap, pars, *table):
                return self.to_poles(jac * get_rsd_pkmu(kap, muap, pars, table[:ncols], table[ncols:]))

            self._get_poles = jit(_get_poles)
        return self._get_poles(self.pt.jac, self.pt.kap, self.pt.muap, jnp.array(pars), *table, *table_now)
        #pkmu = self.pt.jac * get_rsd_pkmu(self.pt.kap, self.pt.muap, pars, table, table_now)
        #return self.to_poles(pkmu)

    def __getstate__(self, varied=True, fixed=True):
        state = {}
        for name in (['k', 'z', 'ells', 'wmu', 'kt'] if fixed else []) + (['sigma8', 'fsigma8'] if varied else []):
            if hasattr(self, name):
                state[name] = getattr(self, name)
        if varied:
            for name in self._pt_attrs:
                if hasattr(self.pt, name):
                    state[name] = getattr(self.pt, name)
        return state

    def __setstate__(self, state):
        for name in ['k', 'z', 'ells', 'wmu', 'kt', 'sigma8', 'fsigma8']:
            if name in state: setattr(self, name, state.pop(name))
        if not hasattr(self, 'pt'): self.pt = Namespace()
        self.pt.update(**state)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/folpsax')


class FOLPSAXTracerPowerSpectrumMultipoles(FOLPSTracerPowerSpectrumMultipoles):
    r"""
    FOLPS tracer power spectrum multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn* and bias term b3*.
    By default, bs and b3 are fixed to 0, following co-evolution.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}^{L}) \sigma_{8}(z), b_{2}^\prime = b_{2}^{L} \sigma_{8}(z)^2, b_{s}^\prime = b_{s}^{L} \sigma_{8}(z)^2, b_{3}^\prime = 0`
        with: :math:`b_{1} = 1 + b_{1}^{L}, b_{2} = 8/21 b_{1}^{L} + b_{2}^{L}, b_{s} = -4/7 b_{1}^{L} + b_{s}^{L}`.
        :math:`\alpha_{0} = (1 + b_{1}^{L})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}^{L}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}^{L}) \alpha_{4}^\prime)`.
        :math:`s_{n, 0} = f_{\mathrm{sat}}/\bar{n} s_{n, 0}^\prime, s_{n, 2} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{2} s_{n, 2}^\prime, s_{n, 4} = f_{\mathrm{sat}}/\bar{n} \sigma_{v}^{4} s_{n, 4}^\prime`.

    tracer : str, default=None
        If ``prior_basis = 'physical'``, tracer to load preset ``fsat`` and ``sigv``. One of ['LRG', 'ELG', 'QSO'].

    fsat : float, default=None
        If ``prior_basis = 'physical'``, satellite fraction to assume.

    sigv : float, default=None
        If ``prior_basis = 'physical'``, velocity dispersion to assume.


    Reference
    ---------
    - https://arxiv.org/abs/2208.02791
    - https://github.com/cosmodesi/folpsax
    """


class FOLPSAXTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):
    r"""
    FOLPS tracer correlation function multipoles.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn* and bias term b3*.
    By default, bs and b3 are fixed to 0, following co-evolution.
    For the matter (unbiased) correlation function, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}^{L}) \sigma_{8}(z), b_{2}^\prime = b_{2}^{L} \sigma_{8}(z)^2, b_{s}^\prime = b_{s}^{L} \sigma_{8}(z)^2, b_{3}^\prime = 0`
        with: :math:`b_{1} = 1 + b_{1}^{L}, b_{2} = 8/21 b_{1}^{L} + b_{2}^{L}, b_{s} = -4/7 b_{1}^{L} + b_{s}^{L}`.
        :math:`\alpha_{0} = (1 + b_{1}^{L})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}^{L}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}^{L}) \alpha_{4}^\prime)`.


    Reference
    ---------
    - https://arxiv.org/abs/2208.02791
    - https://github.com/cosmodesi/folpsax
    """
    _params = FOLPSAXTracerPowerSpectrumMultipoles._params