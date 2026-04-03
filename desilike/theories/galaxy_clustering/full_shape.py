
"""
Full-shape power spectrum, correlation function and bispectrum multipoles with various perturbation theory models.

This module provides theory predictions for full-shape power spectrum and bispectrum
multipoles in galaxy clustering analyses. It supports multiple perturbation theory
implementations with flexible bias parameterizations.

Key Classes
-----------
Base Classes
    - BasePTPowerSpectrumMultipoles: Base perturbation theory (PT) power spectrum multipoles
    - BasePTCorrelationFunctionMultipoles: Base perturbation theory correlation function multipoles
    - BaseTracerPowerSpectrumMultipoles: Base class for tracer power spectrum multipoles
    - BaseTracerCorrelationFunctionMultipoles: Base class for tracer correlation function multipoles
    - BaseTracerPowerSpectrumMultipoles: Base class for PT-based tracer power spectrum multipoles
    - BaseTracerCorrelationFunctionMultipoles: Base class for PT-based tracer correlation function multipoles
    - BaseTracerBispectrumMultipoles: Base class for tracer bispectrum multipoles
    - BaseTracerPTBispectrumMultipoles: Base class for PT-based tracer bispectrum multipoles

Tracer Power Spectrum
    - SimpleTracerPowerSpectrumMultipoles: Simple Kaiser model with fixed damping
    - KaiserTracerPowerSpectrumMultipoles: Kaiser with bias and shot noise
    - TNSTracerPowerSpectrumMultipoles: TNS 1-loop tracer P(k)
    - LPTVelocileptorsTracerPowerSpectrumMultipoles: Velocileptors LPT tracer P(k)
    - REPTVelocileptorsTracerPowerSpectrumMultipoles: Velocileptors REPT tracer P(k)
    - PyBirdTracerPowerSpectrumMultipoles: PyBird EFT-based tracer P(k)
    - FOLPSTracerPowerSpectrumMultipoles: FOLPS infrared-resummed tracer P(k)
    - FOLPSv2TracerPowerSpectrumMultipoles: FOLPS v2 tracer P(k)
    - JAXEffortTracerPowerSpectrumMultipoles: JAXEffort emulator-based tracer P(k)

Tracer Correlation Function
    - KaiserTracerCorrelationFunctionMultipoles: Kaiser tracer xi(s)
    - TNSTracerCorrelationFunctionMultipoles: TNS tracer xi(s)
    - LPTVelocileptorsTracerCorrelationFunctionMultipoles: Velocileptors LPT tracer xi(s)
    - REPTVelocileptorsTracerCorrelationFunctionMultipoles: Velocileptors REPT tracer xi(s)
    - PyBirdTracerCorrelationFunctionMultipoles: PyBird tracer xi(s)
    - FOLPSTracerCorrelationFunctionMultipoles: FOLPS tracer xi(s)
    - FOLPSv2TracerBispectrumMultipoles: FOLPS bispectrum

Bispectrum Models
    - GeoFPTAXTracerBispectrumMultipoles: GeoFPTAX bispectrum model
    - FOLPSv2TracerBispectrumMultipoles: FOLPS v2 bispectrum

Class MultitracerBiasParameters handlses bias parameter namespacing for multitracer analyses.
To implement a new power spectrum, correlation function or bispectrum model::
- if it is a PT-based model, implement the PT part into a class inheriting from BasePTPowerSpectrumMultipoles or BasePTBispectrumMultipoles,
and then implement a tracer version inheriting from BaseTracerPTPowerSpectrumMultipoles or BaseTracerPTBispectrumMultipoles.
An example of this is the KaiserTracerPowerSpectrumMultipoles class, which inherits from BaseTracerPTPowerSpectrumMultipoles and uses the KaiserPowerSpectrumMultipoles as PT module.
- if it is not a PT-based model, implement it directly into a class inheriting from BaseTracerPowerSpectrumMultipoles, BaseTracerCorrelationFunctionMultipoles or BaseTracerBispectrumMultipoles.
"""

import os
import re
import functools

import numpy as np
from scipy import interpolate
import time
from desilike.jax import numpy as jnp
from desilike.jax import jit, interp1d
from desilike import jax
from desilike import plotting, utils, BaseCalculator
from .base import APEffect, SpectrumToCorrelationMultipoles, ProjectToMultipoles, get_legendre
from .power_template import DirectPowerSpectrumTemplate, StandardPowerSpectrumTemplate, Cosmoprimo, get_cosmo


class BasePTPowerSpectrumMultipoles(BaseCalculator):

    """Base class for perturbation theory matter power spectrum multipoles."""
    _default_options = dict()

    def initialize(self, k=None, ells=(0, 2, 4), template=None, z=None, **kwargs):
        self._set_options(k=k, ells=ells, **kwargs)
        self._set_template(template=template, z=z)

    def _set_options(self, k=None, ells=(0, 2, 4), **kwargs):
        if k is None: k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)

    def _set_template(self, template=None, z=None, klim=(1e-3, 1., 500)):
        # klim < 1e-3 h/Mpc causes problems in velocileptors and folps when Omega_k ~ 0.1
        if template is None:
            template = DirectPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(klim[0], self.k[0] / 2, self.template.init.get('k', [1.])[0]), max(klim[1], self.k[-1] * 2, self.template.init.get('k', [0.])[0]), klim[2])  # margin for AP effect
        self.template.init.update(k=kin)
        if z is not None: self.template.init.update(z=z)
        self.z = self.template.z

    def calculate(self):
        self.z = self.template.z

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BasePTCorrelationFunctionMultipoles(BaseCalculator):

    _default_options = dict()

    def initialize(self, s=None, ells=(0, 2, 4), template=None, z=None, **kwargs):
        self._set_options(s=s, ells=ells, **kwargs)
        self._set_template(template=template, z=z)

    def _set_options(self, s=None, ells=(0, 2, 4), **kwargs):
        if s is None: s = np.linspace(20., 200, 101)
        self.s = np.array(s, dtype='f8')
        self.ells = tuple(ells)
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)

    def _set_template(self, template=None, z=None, klim=(1e-3, 1., 500)):
        # klim < 1e-3 h/Mpc causes problems in velocileptors and folps when Omega_k ~ 0.1
        if template is None:
            template = DirectPowerSpectrumTemplate()
        self.template = template
        kin = np.geomspace(min(klim[0], 1 / self.s[-1] / 2, self.template.init.get('k', [1.])[0]), max(klim[1], 1 / self.s[0] * 2, self.template.init.get('k', [0.])[0]), klim[2])  # margin for AP effect
        self.template.init.update(k=kin)
        if z is not None: self.template.init.update(z=z)
        self.z = self.template.z

    def calculate(self):
        self.z = self.template.z

    def __getstate__(self):
        state = {}
        for name in ['s', 'z', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


from desilike import base


class MultitracerBiasParameters(object):

    """Class to handle multitracer bias parameters, with support for deterministic and stochastic parameters,
    and automatic namespace handling for auto and cross correlations."""

    delimiter = base.namespace_delimiter

    def __init__(self, tracers=None, deterministic=None, stochastic=None, ntracers=2):
        """
        Initialize multitracer bias parameters.

        Parameters
        ----------
        tracers : list of str, optional
            List of tracer names. If not provided, it defaults to an empty list, which corresponds to the standard single-tracer case.
        deterministic : list of str, optional
            List of deterministic parameter basenames. These parameters will be duplicated for each tracer and automatically namespaced for auto correlations.
        stochastic : list of str, optional
            List of stochastic parameter basenames. These parameters will be shared across tracers and automatically namespaced for cross correlations.
        ntracers : int, optional
            Maximum number of tracers supported. If the number of tracers provided exceeds this value, an error is raised. Default is 2.
        """
        if tracers is None:
            tracers = []
        else:
            if isinstance(tracers, str):
                tracers = [tracers]
            tracers = tuple(tracers)
        self.tracers = tracers
        assert not any(self.delimiter in tracer for tracer in self.tracers), f'tracers cannot contain {self.delimiter}'
        assert all(self.tracers), f'tracers must be non-empty strings'
        self.ntracers = int(ntracers)
        if len(self.tracers) > self.ntracers:
            raise ValueError(f'{len(self.tracers):d} not supported; max is {self.ntracers:d}')
        self.deterministic = list(deterministic or [])
        self.stochastic = list(stochastic or [])
        if len(self.tracers) == 0:
            # default auto correlation
            *self.auto_namespaces, self.cross_namespace = [''] * (self.ntracers + 1)
        elif len(self.tracers) == 1:
            # auto correlation
            *self.auto_namespaces, self.cross_namespace = [self.tracers[0]] * (self.ntracers + 1)
        else:
            # cross correlation
            self.auto_namespaces, self.cross_namespace = list(self.tracers), 'x'.join(self.tracers)

    def _params(self, params):
        """Process input parameters to handle multitracer namespaces for deterministic and stochastic parameters."""
        if not self.tracers:
            return params

        for param in list(params):
            if param.basename in self.deterministic:
                param = params.pop(param)
                for namespace in self.auto_namespaces:
                    params.set(param.clone(namespace=(param.namespace, namespace)))
            elif param.basename in self.stochastic:
                param.update(namespace=(param.namespace, self.cross_namespace))
        return params

    def __call__(self, params, defaults=None):
        """
        Return a dictionary of parameter values with parameter basenames as keys,
        and tuple of values for deterministic parameters, if more than a single tracer is supported.
        """
        defaults = dict(defaults or {})
        # In case there is input namespace, we take the last part as the parameter name (basename)
        toret = defaults | {name.split(self.delimiter)[-1]: value for name, value in params.items()}
        if self.ntracers > 1:
            nnamespace = int(bool(self.cross_namespace))
            auto_namespaces, cross_namespace = self.auto_namespaces, self.cross_namespace
            if nnamespace:
                auto_namespaces = [namespace + self.delimiter for namespace in self.auto_namespaces]
                cross_namespace = self.cross_namespace + self.delimiter
            params = {self.delimiter.join(name.split(self.delimiter)[-(nnamespace + 1):]): value for name, value in params.items()}
            for param in self.deterministic:
                toret[param] = tuple(params.get(f'{namespace}{param}', defaults.get(param, None)) for namespace in auto_namespaces)
            for param in self.stochastic:
                toret[param] = params.get(f'{cross_namespace}{param}', defaults.get(param, None))
        return toret


class BaseTracerPowerSpectrumMultipoles(BaseCalculator):

    """Base class for theory tracer power spectrum multipoles."""

    config_fn = 'full_shape.yaml'
    _default_options = dict(shotnoise=1e4)
    _initialize_with_namespace = True
    _calculate_with_namespace = True

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers, ntracers=1)

    @classmethod
    def _params(cls, params, tracers=None):
        return cls._get_multitracer(tracers=tracers)._params(params)

    def initialize(self, k=None, ells=(0, 2, 4), tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_options(self, k=None, ells=(0, 2, 4), tracers=None, **kwargs):
        # Wavenumber and multipoles
        if k is None: k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)
        self.tracers = tracers
        # First set shotnoise, useful for rescaling stochastic terms
        shotnoise = kwargs.get('shotnoise', 1e4)
        if np.size(shotnoise) > 1:
            # cross correlation: geometric mean
            shotnoise = np.prod(shotnoise)**(1. / len(shotnoise))
        shotnoise = np.array(shotnoise).item()
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        if 'shotnoise' in self.options:
            self.options['shotnoise'] = shotnoise
        # The quantity used for the rescaling
        self.nbar = 1. / float(shotnoise)

    def calculate(self, **params):
        params = self.decode_params(params)
        # params['b1'] is a single parameter value in standard case
        # a tuple if multitracer support

    def get(self):
        # Return power spectrum multipoles
        return self.power

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'nbar', 'power']:
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
            ax.plot(self.k, self.k * self.power[ill], color=f'C{ill:d}', linestyle='-', label=rf'$\ell = {ell:d}$')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


class BaseTracerCorrelationFunctionMultipoles(BaseCalculator):

    """Base class for tracer correlation function multipoles."""

    config_fn = 'full_shape.yaml'
    _default_options = dict(shotnoise=1e4)
    _initialize_with_namespace = True  # for multitracer
    _calculate_with_namespace = True  # for multitracer

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers, ntracers=1)

    @classmethod
    def _params(cls, params, tracers=None):
        return cls._get_multitracer(tracers=tracers)._params(params)

    def initialize(self, s=None, ells=(0, 2, 4), tracers=None, **kwargs):
        self._set_options(s=s, ells=ells, tracers=tracers, **kwargs)
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_options(self, s=None, ells=(0, 2, 4), tracers=None, **kwargs):
        # Wavenumber and multipoles
        if s is None: s = np.linspace(20., 200, 101)
        self.s = np.array(s, dtype='f8')
        self.ells = tuple(ells)
        self.tracers = tracers
        # First set shotnoise, useful for rescaling stochastic terms
        shotnoise = kwargs.get('shotnoise', 1e4)
        if np.size(shotnoise) > 1:
            # cross correlation: geometric mean
            shotnoise = np.prod(shotnoise)**(1. / len(shotnoise))
        shotnoise = np.array(shotnoise).item()
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        if 'shotnoise' in self.options:
            self.options['shotnoise'] = shotnoise
        # The quantity used for the rescaling
        self.nbar = 1. / float(shotnoise)

    def calculate(self, **params):
        params = self.decode_params(params)
        # params['b1'] is a single parameter value in standard case
        # a tuple if multitracer support

    def get(self):
        return self.corr

    def __getstate__(self):
        state = {}
        for name in ['s', 'z', 'ells', 'nbar', 'corr']:
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
            ax.plot(self.s, self.s**2 * self.corr[ill], color=f'C{ill:d}', linestyle='-', label=rf'$\ell = {ell:d}$')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^2 \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^2$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig



class BaseTracerPTPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):

    """Base class for theory tracers power spectrum multipoles, using a perturbation theory (PT) module."""

    config_fn = 'full_shape.yaml'
    _default_options = dict(shotnoise=1e4)

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_pt(self, pt=None, template=None, **kwargs):
        # Perturbation theory module
        if pt is None:
            _pt_cls = getattr(self, '_pt_cls', None)
            if _pt_cls is None:
                _pt_cls = globals()[self.__class__.__name__.replace('Tracer', '')]
            pt = _pt_cls()
        self.pt = pt
        # Linear power spectrum
        if template is not None:
            self.pt.init.update(template=template)
        # Transfer options to PT module
        for name, value in self.pt._default_options.items():
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
            elif name in self.options:
                self.pt.init.update({name: self.options[name]})
        # mu-integration for multipoles
        for name in ['mu']:
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
        self.pt.init.update({name: kwargs[name] for name in kwargs if name not in self._default_options})
        self.pt.init.update(k=self.k)

    def _set_from_pt(self):
        # Update z, k, ells from pt
        for name in ['z', 'k', 'ells']:
            setattr(self, name, getattr(self.pt, name))

    def _set_params(self, pt_params=None):
        if pt_params is not None:
            self.pt.init.params.update([param for param in self.init.params if param.basename in pt_params], basename=True)
            self.init.params = self.init.params.select(basename=[param.basename for param in self.init.params if param.basename not in pt_params])

    def calculate(self):
        self._set_from_pt()


class BaseTracerPTCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    """Base class for tracer correlation function multipoles, with perturbation theory (PT) natively in configuration space."""

    config_fn = 'full_shape.yaml'
    _default_options = dict(shotnoise=1e4)

    def initialize(self, s=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(s=s, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_pt(self, pt=None, template=None, **kwargs):
        # Perturbation theory module
        if pt is None:
            pt = globals()[getattr(self, '_pt_cls', self.__class__.__name__.replace('Tracer', ''))]()
        self.pt = pt
        # Linear power spectrum
        if template is not None:
            self.pt.init.update(template=template)
        # Transfer options to PT module
        for name, value in self.pt._default_options.items():
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
            elif name in self.options:
                self.pt.init.update({name: self.options[name]})
        # mu-integration for multipoles
        for name in ['mu']:
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
        self.pt.init.update({name: kwargs[name] for name in kwargs if name not in self._default_options})
        self.pt.init.update(s=self.s)

    def _set_from_pt(self):
        # Update z, k, ells from pt
        for name in ['z', 's', 'ells']:
            setattr(self, name, getattr(self.pt, name))

    def _set_params(self, pt_params=None):
        if pt_params is not None:
            self.pt.init.params.update([param for param in self.init.params if param.basename in pt_params], basename=True)
            self.init.params = self.init.params.select(basename=[param.basename for param in self.init.params if param.basename not in pt_params])

    def calculate(self):
        self._set_from_pt()

    def get(self):
        return self.corr


class BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles(BaseTracerCorrelationFunctionMultipoles):

    """Base class for tracers correlation function multipoles as Hankel transforms of the power spectrum multipoles."""

    config_fn = 'full_shape.yaml'

    def initialize(self, s=None, ells=(0, 2, 4), tracers=None, pt=None, template=None, **kwargs):
        power = self._power_cls()
        if pt is not None: power.init.update(pt=pt)
        if template is not None: power.init.update(template=template)
        power.init.update(**kwargs)
        self.power = power
        self._set_options(s=s, ells=ells, tracers=tracers)
        self.to_correlation = SpectrumToCorrelationMultipoles(s=self.s, spectrum=self.power)
        self.power.init.params = self.init.params.copy()
        self.init.params.clear()
        for name in ['z', 'ells', 'options']:
            setattr(self, name, getattr(self.power, name))

    def calculate(self):
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.power, name))
        self.corr = self.to_correlation(self.power.power)

    @property
    def pt(self):
        return self.power.pt


class SimpleTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    r"""
    Kaiser tracer power spectrum multipoles, with fixed damping, essentially used for Fisher forecasts.
    For the matter (unbiased) power spectrum, set b1=1 and sn0=0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.
    tracers : str or list of str, default=None
        Tracer name(s). Namespace added to bias parameters. If 2 tracers are provided, cross-correlation is included.
    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`StandardPowerSpectrumTemplate`.
    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).
    """
    config_fn = 'full_shape.yaml'

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers, deterministic=['b1'], stochastic=['sn0'], ntracers=2)

    def initialize(self, k=None, ells=(0, 2, 4), mu=8, tracers=None, z=None, template=None):
        self._set_options(k=k, ells=ells, tracers=tracers)
        if template is None:
            template = StandardPowerSpectrumTemplate()
        BasePTPowerSpectrumMultipoles._set_template(self, template=template, z=z)
        self.z = self.template.z
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        self.decode_params = self._get_multitracer(tracers=tracers)

    def calculate(self, sigmapar=0., sigmaper=0., **params):
        self.z = self.template.z
        params = self.decode_params(params)
        (b1X, b1Y), sn0 = params['b1'], params['sn0']
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f
        sigmanl2 = self.k[:, None]**2 * (sigmapar**2 * self.mu**2 + sigmaper**2 * (1. - self.mu**2))
        damping = jnp.exp(-sigmanl2 / 2.)
        #pkmu = jac * damping * (b1X + f * muap**2) * (b1Y + f * muap**2) * jnp.interp(jnp.log10(kap), jnp.log10(self.template.k), self.template.pk_dd) + sn0 / self.nbar
        pkmu = jac * damping * (b1X + f * muap**2) * (b1Y + f * muap**2) * interp1d(jnp.log10(kap), jnp.log10(self.template.k), self.template.pk_dd, method='cubic') + sn0 / self.nbar
        self.power = self.to_poles(pkmu)


class KaiserPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):
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
    # Extra PT parameters
    _params = {'sigmapar': {'value': 0., 'fixed': True}, 'sigmaper': {'value': 0, 'fixed': True}}

    def initialize(self, s=None, ells=(0, 2, 4), template=None, z=None, mu=8, **kwargs):
        self._set_options(s=s, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu

    def calculate(self, sigmapar=0., sigmaper=0.):
        # PT computation
        self.z = self.template.z
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
        state = super().__getstate__()
        for name in self.pktable:
            state[name] = self.pktable[name]
        state['names'] = list(self.pktable.keys())
        return state

    def __setstate__(self, state):
        state = dict(state)
        self.pktable = {name: state.pop(name, None) for name in state['names']}
        super().__setstate__(state)


class KaiserTracerPowerSpectrumMultipoles(BaseTracerPTPowerSpectrumMultipoles):
    r"""
    Kaiser tracer power spectrum multipoles.
    For the matter (unbiased) power spectrum, set b1=1 and sn0=0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.
    tracers : str or list of str, default=None
        Tracer name(s). Namespace added to bias parameters. If 2 tracers are provided, cross-correlation is included.
    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    """
    _default_options = dict(shotnoise=1e4)

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers, deterministic=['b1'], stochastic=['sn0'], ntracers=2)

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self._set_params(pt_params=['sigmapar', 'sigmaper'])
        self.decode_params = self._get_multitracer(tracers=tracers)

    def calculate(self, **params):
        self._set_from_pt()
        params = self.decode_params(params, defaults={'sn0': 0.})  # default sn0 for correlation function
        (b1X, b1Y), sn0 = params['b1'], params['sn0']
        sn0 = np.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None] * sn0 / self.nbar
        self.power = b1X * b1Y * self.pt.pktable['pk_dd'] + (b1X + b1Y) * self.pt.pktable['pk_dt'] + self.pt.pktable['pk_tt'] + sn0


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
    tracers : str or list of str, default=None
        Tracer name(s). Namespace added to bias parameters. If 2 tracers are provided, cross-correlation is included.
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """
    _power_cls = KaiserTracerPowerSpectrumMultipoles

    @classmethod
    def _params(cls, params, tracers=None):
        return cls._power_cls._params(params, tracers=tracers)


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

    mus, wmus = utils.weights_mu(10, method='leggauss')

    # Compute P22
    pk22_dd, pk22_dt, pk22_tt = (0.,) * 3
    pk_b2d, pk_bs2d, pk_b2t, pk_bs2t, sig3sq, pk_b22, pk_b2s2, pk_bs22 = (0.,) * 8
    A = jnp.zeros((5,) + k11.shape, dtype='f8')
    B = [jnp.zeros(k11.shape, dtype='f8') for i in range(12)]
    pk_k = jnp.interp(k11, q, pk_q)

    def get_terms(mu, wmu):
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

        pk_b2d = wmu * jnp.sum(jq_pk_q_pk_kq * F2_d, axis=-1)
        pk_bs2d = wmu * jnp.sum(jq_pk_q_pk_kq * F2_d * S, axis=-1)
        pk_b2t = wmu * jnp.sum(jq_pk_q_pk_kq * F2_t, axis=-1)
        pk_bs2t = wmu * jnp.sum(jq_pk_q_pk_kq * F2_t * S, axis=-1)
        sig3sq = wmu * jnp.sum(105. / 16. * jq * pk_q * (D * S + 8. / 63.), axis=-1)
        pk_b22 = wmu / 2. * jnp.sum(jq * pk_q * (pk_kq - pk_q), axis=-1)
        pk_b2s2 = wmu / 2. * jnp.sum(jq * pk_q * (pk_kq * S - 2. / 3. * pk_q), axis=-1)
        pk_bs22 = wmu / 2. * jnp.sum(jq * pk_q * (pk_kq * S**2 - 4. / 9. * pk_q), axis=-1)
        pk22_dd = 2 * wmu * jnp.sum(F2_d**2 * jq_pk_q_pk_kq, axis=-1)
        pk22_dt = 2 * wmu * jnp.sum(F2_d * F2_t * jq_pk_q_pk_kq, axis=-1)
        pk22_tt = 2 * wmu * jnp.sum(F2_t * F2_t * jq_pk_q_pk_kq, axis=-1)

        xmu = kq2 / k**2
        kernel_A, kernel_tA = [0] * 5, [0] * 5
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
        A = wmu * jnp.sum(jq / x**2 * (jnp.array(kernel_A) * pk_k[:, None] + jnp.array(kernel_tA) * pk_q) * pk_kq / xmu**2, axis=-1)

        jq_pk_q_pk_kq /= x**2 * xmu
        B = [0.] * 12
        B[0] = wmu * jnp.sum(x**2 * (mu**2 - 1.) / 2. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,1,1
        B[1] = wmu * jnp.sum(3. * x**2 * (mu**2 - 1.)**2 / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,1,2
        B[2] = wmu * jnp.sum(3. * x**4 * (mu**2 - 1.)**2 / xmu / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,2,1
        B[3] = wmu * jnp.sum(5. * x**4 * (mu**2 - 1.)**3 / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 1,2,2
        B[4] = wmu * jnp.sum(x * (x + 2. * mu - 3. * x * mu**2) / 2. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,1,1
        B[5] = wmu * jnp.sum(- 3. * x * (mu**2 - 1.) * (-x - 2. * mu + 5. * x * mu**2) / 4. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,1,2
        B[6] = wmu * jnp.sum(3. * x**2 * (mu**2 - 1.) * (-2. + x**2 + 6. * x * mu - 5. * x**2 * mu**2) / xmu / 4. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,2,1
        B[7] = wmu * jnp.sum(- 3. * x**2 * (mu**2 - 1.)**2 * (6. - 5. * x**2 - 30. * x * mu + 35. * x**2 * mu**2) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 2,2,2
        B[8] = wmu * jnp.sum(x * (4. * mu * (3. - 5. * mu**2) + x * (3. - 30. * mu**2 + 35. * mu**4)) / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 3,1,2
        B[9] = wmu * jnp.sum(x * (-8. * mu + x * (-12. + 36. * mu**2 + 12. * x * mu * (3. - 5. * mu**2) + x**2 * (3. - 30. * mu**2 + 35. * mu**4))) / xmu / 8. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 3,2,1
        B[10] = wmu * jnp.sum(3. * x * (mu**2 - 1.) * (-8. * mu + x * (-12. + 60. * mu**2 + 20. * x * mu * (3. - 7. * mu**2) + 5. * x**2 * (1. - 14. * mu**2 + 21. * mu**4))) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 3,2,2
        B[11] = wmu * jnp.sum(x * (8. * mu * (-3. + 5. * mu**2) - 6. * x * (3. - 30. * mu**2 + 35. * mu**4) + 6. * x**2 * mu * (15. - 70. * mu**2 + 63 * mu**4) + x**3 * (5. - 21. * mu**2 * (5. - 15. * mu**2 + 11. * mu**4))) / xmu / 16. * jq_pk_q_pk_kq, axis=-1)  # n,a,b = 4,2,2
        return jnp.stack([pk_b2d, pk_bs2d, pk_b2t, pk_bs2t, sig3sq, pk_b22, pk_b2s2, pk_bs22, pk22_dd, pk22_dt, pk22_tt] + list(A) + B)

    res = jnp.sum(jax.vmap(get_terms)(mus, wmus), axis=0)
    pk_b2d, pk_bs2d, pk_b2t, pk_bs2t, sig3sq, pk_b22, pk_b2s2, pk_bs22, pk22_dd, pk22_dt, pk22_tt = res[:11]
    A, B = res[11:16], res[16:]
    A += pk_k * jnp.sum(kernel_a * pk_q, axis=-1)
    pk11 = pk_k
    pk13_dd = 2. * jnp.sum(kernel13_d * pk_q, axis=-1) * pk_k
    pk13_tt = 2. * jnp.sum(kernel13_t * pk_q, axis=-1) * pk_k
    pk13_dt = (pk13_dd + pk13_tt) / 2.
    pk_sig3sq = sig3sq * pk_k
    pk_dd = pk11 + pk22_dd + pk13_dd
    pk_dt = pk11 + pk22_dt + pk13_dt
    pk_tt = pk11 + pk22_tt + pk13_tt

    return [pk11, pk_dd, pk_b2d, pk_bs2d, pk_sig3sq, pk_b22, pk_b2s2, pk_bs22, pk_dt, pk_b2t, pk_bs2t, pk_tt, A, B]


class TNSPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):
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

    def initialize(self, k=None, ells=(0, 2, 4), template=None, z=None, mu=8, **kwargs):
        self._set_options(k=k, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        self.nloop = int(self.options['nloop'])
        if self.nloop not in [1]:
            raise ValueError('nloop must be 1 (1-loop)')
        if self.options['fog'] not in ['lorentzian', 'gaussian']:
            raise ValueError('fog must be lorentzian or gaussian')

    def calculate(self, sigmav=0):
        self.z = self.template.z
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f

        if self.options['fog'] == 'lorentzian':
            damping = 1. / (1. + (sigmav * kap * muap)**2 / 2.)**2.
        else:
            damping = jnp.exp(-(sigmav * kap * muap)**2)

        k11 = np.linspace(self.k[0] * 0.7, self.k[-1] * 1.3, int(len(self.k) * 1.6 + 0.5))
        q = self.template.k
        wq = utils.weights_trapz(q)
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
        state = super().__getstate__()
        for name in ['nloop', 'fog']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self.pktable:
            state[name] = self.pktable[name]
        state['names'] = list(self.pktable.keys())
        return state

    def __setstate__(self, state):
        state = dict(state)
        self.pktable = {name: state.pop(name, None) for name in state['names']}
        super().__setstate__(state)


class TNSTracerPowerSpectrumMultipoles(BaseTracerPTPowerSpectrumMultipoles):
    r"""
    TNS tracer power spectrum multipoles.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
    mu : int, default=8
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    shotnoise : float, default=1e4
        Shot noise (which is usually marginalized over).
    """
    _default_options = dict(freedom=None, shotnoise=1e4)

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self._set_params()
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_params(self):
        super()._set_params(pt_params=['sigmav'])
        freedom = self.options.get('freedom', None)
        fix = []
        if freedom == 'max':
            for param in self.init.params.select(basename=['b1', 'b2', 'bs', 'b3']):
                param.update(fixed=False)
        if freedom == 'min':
            fix += ['b3', 'bs']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)

    def calculate(self, **params):
        self._set_from_pt()
        params = self.decode_params(params)
        b1, b2, bs, b3, sn0 = [params[name] for name in ['b1', 'b2', 'bs', 'b3', 'sn0']]
        self.power = b1**2 * self.pt.pktable['pk_dd'] + 2. * b1 * self.pt.pktable['pk_dt'] + self.pt.pktable['pk_tt'] + sn0 / self.nbar
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
    TNS tracers correlation function multipoles.
    For the matter (unbiased) correlation function, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.
    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """
    _power_cls = TNSTracerPowerSpectrumMultipoles

    @classmethod
    def _params(cls, params, tracers=None):
        return cls._power_cls._params(params, tracers=tracers)


def get_nthreads(nthreads=None):
    if nthreads is None:
        import os
        nthreads = os.getenv('OMP_NUM_THREADS', '1')
    return int(nthreads)


class BaseVelocileptorsPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    """Base class for velocileptors-based matter power spectrum multipoles."""
    _default_options = dict()

    def initialize(self, k=None, ells=(0, 2, 4), template=None, z=None, mu=4, **kwargs):
        self._set_options(k=k, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
        self.nmu = int(mu)
        self.options['threads'] = get_nthreads(self.options.pop('nthreads', None))

    def calculate(self):
        self.z = self.template.z

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/sfschen/velocileptors')

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'pktable', 'sigma8', 'fsigma8']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
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


class BaseVelocileptorsTracerPowerSpectrumMultipoles(BaseTracerPTPowerSpectrumMultipoles):

    """Base class for velocileptors-based tracer power spectrum multipoles."""

    @classmethod
    def _get_multitracer(cls, tracers=None, prior_basis='physical'):
        deterministic = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6']
        stochastic = ['sn0', 'sn2', 'sn4']
        if prior_basis == 'physical':
            deterministic = [name + 'p' for name in deterministic]
            stochastic = [name + 'p' for name in stochastic]
        return MultitracerBiasParameters(tracers=tracers, deterministic=deterministic, stochastic=stochastic, ntracers=1)

    @classmethod
    def _params(cls, params, freedom=None, prior_basis='physical', tracers=None):
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
        params = cls._get_multitracer(tracers=tracers, prior_basis=prior_basis)._params(params)
        return params

    def _set_params(self):
        self.is_physical_prior = self.options['prior_basis'] == 'physical'
        if self.is_physical_prior:
            settings = get_physical_stochastic_settings(tracer=self.options['tracer'])
            for name, value in settings.items():
                if self.options[name] is None: self.options[name] = value
            if self.mpicomm.rank == 0:
                self.log_debug(f"Using fsat, sigv = {self.options['fsat']:.3f}, {self.options['sigv']:.3f}.")
        super()._set_params(pt_params=[])
        fix = []
        if 4 not in self.ells: fix += ['alpha4*', 'alpha6*', 'sn4*']  # * to capture p
        if 2 not in self.ells: fix += ['alpha2*', 'sn2*']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)
        self.nbar = 1e-4
        self.fsat = self.snd = 1.
        if self.is_physical_prior:
            self.fsat, self.snd = self.options['fsat'], self.options['shotnoise'] * self.nbar  # normalized by 1e-4


class BaseVelocileptorsCorrelationFunctionMultipoles(BasePTCorrelationFunctionMultipoles):

    """Base class for velocileptors-based matter correlation function multipoles."""

    def initialize(self, s=None, ells=(0, 2, 4), template=None, z=None, **kwargs):
        self._set_options(s=s, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
        self.options['threads'] = get_nthreads(self.options.pop('nthreads', None))

    def combine_bias_terms_poles(self, pars, **opts):
        return np.array([self.pt.compute_xi_ell(ss, self.template.f, *pars, apar=self.template.qpar, aperp=self.template.qper, **self.options, **opts) for ss in self.s]).T


class BaseVelocileptorsTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    """Base class for velocileptors-based tracer correlation function multipoles."""

    def calculate(self, **params):
        params = self.decode_params(params, defaults=self.required_bias_params | self.optional_bias_params)
        pars = [params[name] for name in self.required_bias_params]
        opts = {name: params[name] for name in self.optional_bias_params}
        self.corr = self.pt.combine_bias_terms_poles(pars, **opts, **self.options)

@jit
def tablevel_combine_bias_terms_poles(pktable, pars, nd=1e-4):
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4 = pars
    bias_monomials = jnp.array([1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3, alpha0, alpha2, alpha4, alpha6, sn0 / nd, sn2 / nd, sn4 / nd])
    return jnp.sum(pktable * bias_monomials, axis=-1)


class LPTVelocileptorsPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(use_Pzel=False, kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=4000, nthreads=None, jn=5)
    # Speed is linear with the number of output k

    def initialize(self, **kwargs):
        super().initialize(**kwargs)

    def calculate(self):
        super().calculate()

        def interp1d(x, y):
            return interpolate.interp1d(x, y, kind='cubic', assume_sorted=True)  # for AP

        from velocileptors.LPT import lpt_rsd_fftw
        lpt_rsd_fftw.interp1d = interp1d

        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        self.pt = LPT_RSD(np.asarray(self.template.k), np.asarray(self.template.pk_dd), **self.options)
        self.pt.make_pltable(np.asarray(self.template.f), kv=np.asarray(self.k), apar=np.asarray(self.template.qpar), aperp=np.asarray(self.template.qper), ngauss=self.nmu)
        pktable = {0: self.pt.p0ktable, 2: self.pt.p2ktable, 4: self.pt.p4ktable}
        self.pktable = np.array([pktable[ell] for ell in self.ells])
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.f * self.sigma8

    def combine_bias_terms_poles(self, pars, nd=1e-4):
        return tablevel_combine_bias_terms_poles(self.pktable, pars, nd=nd)

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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
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

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_params()
        boost_prec = 2
        kvec = np.concatenate([[min(0.0005, self.k[0])], np.geomspace(0.0015, 0.025, 10 * boost_prec, endpoint=True), np.arange(0.03, max(0.5, self.k[-1]) + 0.015 / boost_prec, 0.01 / boost_prec)])  # margin for interpolation below (and numerical noise in endpoint)
        self.pt.init.update(k=kvec, ells=self.ells, use_Pzel=not self.is_physical_prior)
        self._set_from_pt()
        self.decode_params = self._get_multitracer(tracers=tracers, prior_basis=self.options['prior_basis'])

    def _set_from_pt(self):
        # Update z, ells from pt
        for name in ['z', 'ells']:
            setattr(self, name, getattr(self.pt, name))

    def calculate(self, **params):
        self._set_from_pt()
        if self.is_physical_prior:
            params = self.decode_params(params, defaults={f'sn{i:d}p': 0. for i in [0, 2, 4]})  # defaults for correlation function
            sigma8 = self.pt.sigma8
            f = self.pt.fsigma8 / sigma8
            pars = b1L, b2L, bsL, b3L = [params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p'] / sigma8**3]
            pars += [(1 + b1L)**2 * params['alpha0p'], f * (1 + b1L) * (params['alpha0p'] + params['alpha2p']),
                     f * (f * params['alpha2p'] + (1 + b1L) * params['alpha4p']), f**2 * params['alpha4p']]
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2, 4]]
        else:
            params = self.decode_params(params, defaults={f'sn{i:d}': 0. for i in [0, 2, 4]})
            pars = [params[name] for name in ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4']]
        #self.__dict__.update(dict(zip(['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4'], pars)))  # for derived parameters
        opts = {}
        index = np.array([self.pt.ells.index(ell) for ell in self.ells])
        self.power = interp1d(self.k, self.pt.k, self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nbar)[index].T).T
        #self.power = self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nbar)


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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
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
    _power_cls = LPTVelocileptorsTracerPowerSpectrumMultipoles

    @classmethod
    def _params(cls, params, tracers=None, prior_basis='physical'):
        return cls._power_cls._params(params, tracers=tracers, prior_basis=prior_basis)


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

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.template.init.update(with_now='peakaverage')

    def calculate(self):
        super().calculate()
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
        #Omega_m, h, fnu, Neff, Nnu = 0.3, 0.7, 0., 3.046, 3
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
                pks = self.pt.compute_redshift_space_power_multipoles_tables(fk[..., iz], apar=qpar[iz], aperp=qper[iz], ngauss=self.nmu, pcb=pcb[:-1, iz], pcb_nw=pcb_nw[:-1, iz], Dz=Dz)[1:]
                for ill, ell in enumerate(pktable): pktable[ell].append(pks[ill])
            pktable = {ell: np.concatenate([v[..., None] for v in value], axis=-1) for ell, value in pktable.items()}
        else:
            #fk = f0 * f_over_f0_EH(self.z, self.pt.kv, Omega_m, h, fnu, Nnu=Nnu, Neff=Neff)
            pks = self.pt.compute_redshift_space_power_multipoles_tables(fk, apar=qpar, aperp=qper, ngauss=self.nmu)[1:]
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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
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

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, z=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_params()
        boost_prec = 2
        kvec = np.concatenate([[min(0.0005, self.k[0])], np.geomspace(0.0015, 0.025, 10 * boost_prec, endpoint=True), np.arange(0.03, max(0.5, self.k[-1]) + 0.015 / boost_prec, 0.01 / boost_prec)])  # margin for interpolation below (and numerical noise in endpoint)
        self.pt.init.update(k=kvec, ells=self.ells, use_Pzel=not self.is_physical_prior)
        if z is not None:  # share the same PT
            self.z = float(z)
            z = self.pt.init.get('z', [])
            if self.z not in z: z.append(self.z)
            self.pt.init.update(z=sorted(z))
        self._set_from_pt()
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_from_pt(self):
        # Update ells from pt
        for name in ['ells']:
            setattr(self, name, getattr(self.pt, name))
        if self.pt.z.ndim == 0: self.z = self.pt.z

    def calculate(self, **params):
        self._set_from_pt()
        if self.is_physical_prior:
            params = self.decode_params(params, defaults={f'sn{i:d}p': 0. for i in [0, 2, 4]})  # defaults for correlation function
            sigma8 = self.pt.sigma8
            f = self.pt.fsigma8 / sigma8
            if self.pt.z.ndim:
                iz = list(self.pt.z).index(self.z)
                sigma8, f = sigma8[iz], f[iz]
            # b1_E = 1 + b1_L
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
            params = self.decode_params(params, defaults={f'sn{i:d}': 0. for i in [0, 2, 4]})
            pars = [params[name] for name in ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4']]
        opts = {}
        index = np.array([self.pt.ells.index(ell) for ell in self.ells])
        if self.pt.z.ndim: opts['z'] = self.z
        self.power = interp1d(self.k, self.pt.k, self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nbar)[index].T).T
        #self.power = self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nbar)


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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
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
    _power_cls = REPTVelocileptorsTracerPowerSpectrumMultipoles

    @classmethod
    def _params(cls, params, tracers=None, prior_basis='physical'):
        return cls._power_cls._params(params, tracers=tracers, prior_basis=prior_basis)


class PyBirdPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    _default_options = dict(km=0.7, kr=0.25, accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False, with_stoch=True, with_resum='full', with_ap=True, eft_basis='eftoflss')
    _klim = (1e-3, 11., 3000)  # numerical instability in pybird's fftlog at 10.
    _pt_attrs = ['co', 'f', 'eft_basis', 'with_stoch', 'with_nnlo_counterterm', 'with_tidal_alignments',
                 'P11l', 'Ploopl', 'Pctl', 'Pstl', 'Pnnlol', 'C11l', 'Cloopl', 'Cctl', 'Cstl', 'Cnnlol']

    def initialize(self, k=None, ells=(0, 2, 4), template=None, z=None, **kwargs):
        self._set_options(k=k, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
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
        for name in ['km', 'kr']:
            self.options[name] = tuple(self.options[name]) if utils.is_sequence(self.options[name]) else (self.options[name],) * 2
        self.km = self.options['km']
        self.kr = self.options['kr']
        self.co = Common(Nl=len(self.ells), kmin=1e-3, kmax=self.k[-1] * 1.3, km=min(self.options['km']), kr=min(self.options['kr']), nd=1e-4,
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
        self.projection = Projection(self.k, with_ap=self.options['with_ap'], H_fid=None, D_fid=None, co=self.co)  # placeholders for H_fid and D_fid, as we will provide q's

    def calculate(self):
        self.z = self.template.z
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

        if self.options['with_ap']:
            self.projection.AP(self.pt, q=(self.template.qper, self.template.qpar))
        self.projection.xdata(self.pt)

    def combine_bias_terms_poles(self, params, nd=1e-4):
        from pybird import bird
        bird.np = jnp
        self.pt.co.nbar = nd
        self.pt.setreducePslb(params, what='full')
        bird.np = np
        return jnp.nan_to_num(self.pt.fullPs, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)

    def combine_bias_terms_poles_for_cross(self, biasX, biasY, nd=1e-4, km=(0.7, 0.7), kr=(0.25, 0.25)):
        # Follows https://arxiv.org/abs/2308.06206 eq(13), except that stochastic terms are scaled by geometric means of nd and km
        bird = self.pt
        f = bird.f
        b1X, b2X, b3X, b4X = (biasX[f'b{i:d}'] for i in [1, 2, 3, 4])
        b1Y, b2Y, b3Y, b4Y = (biasY[f'b{i:d}'] for i in [1, 2, 3, 4])
        kmX, kmY = km
        krX, krY = kr
        if bird.eft_basis in ["eftoflss", "westcoast"]:
            b5X, b6X, b7X = (biasX[name] / ks**2 for name, ks in zip(["cct", "cr1", "cr2"], [kmX, krX, krX]))
            b5Y, b6Y, b7Y = (biasY[name] / ks**2 for name, ks in zip(["cct", "cr1", "cr2"], [kmY, krY, krY]))
        elif bird.eft_basis == 'eastcoast': # inversion of (2.23) of 2004.10607
            ct0X = biasX["c0"] - f/3. * biasX["c2"] + 3/35. * f**2 * biasX["c4"]
            ct2X = biasX["c2"] - 6/7. * f * biasX["c4"]
            ct4X = biasX["c4"]
            ct0Y = biasY["c0"] - f/3. * biasY["c2"] + 3/35. * f**2 * biasY["c4"]
            ct2Y = biasY["c2"] - 6/7. * f
            ct4Y = biasY["c4"]
        b11 = jnp.array([b1X * b1Y, (b1X + b1Y) * f, f**2])
        if bird.eft_basis in ["eftoflss", "westcoast"]:
            bct = jnp.array([b1X * b5Y + b1Y * b5X, b1Y * b6X + b1X * b6Y, b1Y * b7X + b1X * b7Y, (b5X + b5Y) * f, (b6X + b6Y) * f, (b7X + b7Y) * f])
        elif bird.eft_basis == 'eastcoast':
            bct = - np.array([ct0X + ct0Y, f * (ct2X + ct2Y), f**2 * (ct4X + ct4Y)])
        if bird.with_nnlo_counterterm:
            raise NotImplementedError("PyBird cross-power spectrum with nnlo counterterm is not implemented yet.")
        #     if bird.eft_basis in ["eftoflss", "westcoast"]: cnnlo = 0.25 * jnp.array([b1X**2 * biasX["cr4"], b1X * biasX["cr6"]]) / kr[0]**4
        #     elif bird.eft_basis == "eastcoast": cnnlo = - biasX["ct"] * f**4 * jnp.array([b1X**2, 2. * b1X * f, f**2])   # these are not divided by kr^4 according to eastcoast definition; the prior is adjusted accordingly
        bloop = jnp.array([1., 0.5*(b1X+b1Y), 0.5*(b2X+b2Y), 0.5*(b3X+b3Y), 0.5*(b4X+b4Y), b1X*b1Y, 0.5*(b1X*b2Y+b1Y*b2X), 0.5*(b1X*b3Y+b1Y*b3X), 0.5*(b1X*b4Y+b1Y*b4X), b2X*b2Y, 0.5*(b2X*b4Y+b2Y*b4X), b4X*b4Y])
        if bird.with_stoch:
            # ces in biasX and biasY refer to the same jnp object
            bst = jnp.array([biasX["ce0"], biasX["ce1"] / (km[0] * km[1]), biasX["ce2"] / (km[0] * km[1])]) / nd

        Ps = [None] * 3
        Ps[0] = jnp.einsum('b,lbx->lx', b11, bird.P11l)
        Ps[1] = jnp.einsum('b,lbx->lx', bloop, bird.Ploopl) + jnp.einsum('b,lbx->lx', bct, bird.Pctl)
        if bird.with_stoch: Ps[1] += jnp.einsum('b,lbx->lx', bst, bird.Pstl)
        # if bird.with_nnlo_counterterm: Ps[2] = jnp.einsum('b,lbx->lx', cnnlo, bird.Pnnlol)
        if Ps[2] is None:
            Ps[2] = jnp.zeros_like(Ps[0])
        Ps = jnp.array(Ps)
        fullPs = jnp.sum(Ps, axis=0)
        return jnp.nan_to_num(fullPs, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'km', 'kr']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._pt_attrs:
            if hasattr(self.pt, name):
                state[name] = getattr(self.pt, name)
        return state

    def __setstate__(self, state):
        for name in ['k', 'z', 'ells', 'km', 'kr']:
            if name in state: setattr(self, name, state.pop(name))
        from pybird import bird
        self.pt = bird.Bird.__new__(bird.Bird)
        self.pt.with_bias = False
        self.pt.__dict__.update(state)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/pierrexyz/pybird')


class PyBirdTracerPowerSpectrumMultipoles(BaseTracerPTPowerSpectrumMultipoles):
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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. If 2 tracers are provided, cross-correlation is included.
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

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers,
        deterministic=['b1', 'b2', 'b3', 'b4', 'bs', 'b2p4', 'b2m4', 'b2t', 'b2g', 'b3g', 'cct', 'cr1', 'cr2', 'cr4', 'cr6', 'c0', 'c2', 'c4', 'ct'],
        stochastic=['ce0', 'ce1', 'ce2'], ntracers=2)

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self._set_params()
        self.decode_params = self._get_multitracer(tracers=tracers)

    @classmethod
    def _params(cls, params, freedom=None, tracers=None):
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
        return cls._get_multitracer(tracers=tracers)._params(params)

    def _set_params(self):
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
        self._set_from_pt()
        params = self.decode_params(params)
        if len(self.decode_params.tracers) > 1:
            paramsX, paramsY = {}, {}
            for k, v in params.items():
                if isinstance(v, tuple):
                    paramsX[k], paramsY[k] = v
                else:
                    paramsX[k] = paramsY[k] = v  # stochastic terms
            paramsX, paramsY = self.transform_params(**paramsX), self.transform_params(**paramsY)
            self.power = self.pt.combine_bias_terms_poles_for_cross(paramsX, paramsY, nd=self.nbar, km=self.pt.km, kr=self.pt.kr)
        else:
            params = {k: v[0] if isinstance(v, tuple) else v for k, v in params.items()}
            self.power = self.pt.combine_bias_terms_poles(self.transform_params(**params), nd=self.nbar)


class PyBirdCorrelationFunctionMultipoles(BasePTCorrelationFunctionMultipoles):

    _default_options = dict(km=0.7, kr=0.25, accboost=1, fftaccboost=1, fftbias=-1.6, with_nnlo_counterterm=False, with_stoch=False, with_resum='full', with_ap=True, eft_basis='eftoflss')
    _klim = (1e-3, 11., 3000)  # numerical instability in pybird's fftlog at 10.
    _pt_attrs = ['co', 'f', 'eft_basis', 'with_stoch', 'with_nnlo_counterterm', 'with_tidal_alignments',
                 'P11l', 'Ploopl', 'Pctl', 'Pstl', 'Pnnlol', 'C11l', 'Cloopl', 'Cctl', 'Cstl', 'Cnnlol']

    def initialize(self, s=None, ells=(0, 2, 4), template=None, z=None, **kwargs):
        self._set_options(s=s, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
        from pybird.common import Common
        from pybird.nonlinear import NonLinear
        from pybird.nnlo import NNLO_counterterm
        from pybird.resum import Resum
        from pybird.projection import Projection
        eft_basis = self.options.get('eft_basis', None)
        if eft_basis in [None, 'velocileptors']: eft_basis = 'eftoflss'
        # nd used by combine_bias_terms_poles only
        for name in ['km', 'kr']:
            self.options[name] = self.options[name] if utils.is_sequence(self.options[name]) else (self.options[name],) * 2
        self.co = Common(Nl=len(self.ells), kmin=1e-3, kmax=0.25, km=min(self.options['km']), kr=min(self.options['kr']), nd=1e-4,
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
        self.projection = Projection(self.s, with_ap=self.options['with_ap'], H_fid=None, D_fid=None, co=self.co)  # placeholders for H_fid and D_fid, as we will provide q's

    def calculate(self):
        self.z = self.template.z
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

        if self.options['with_ap']:
            self.projection.AP(self.pt, q=(self.template.qper, self.template.qpar))
        self.projection.xdata(self.pt)

    def combine_bias_terms_poles(self, params, nd=1e-4):
        from pybird import bird
        bird.np = jnp
        self.pt.co.nbar = nd
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


class PyBirdTracerCorrelationFunctionMultipoles(BaseTracerPTCorrelationFunctionMultipoles):
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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    **kwargs : dict
        Pybird options, defaults to: ``with_nnlo_higher_derivative=False, with_nnlo_counterterm=False, with_stoch=False, with_resum='full'``.
    """
    _default_options = dict(with_nnlo_counterterm=False, with_stoch=False, eft_basis=None, freedom=None)

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers,
        deterministic=['b1', 'b2', 'b3', 'b4', 'bs', 'b2p4', 'b2m4', 'b2t', 'b2g', 'b3g', 'cct', 'cr1', 'cr2', 'cr4', 'cr6', 'c0', 'c2', 'c4', 'ct'],
        stochastic=['ce0', 'ce1', 'ce2'], ntracers=1)

    _params = classmethod(PyBirdTracerPowerSpectrumMultipoles._params.__func__)
    _set_params = PyBirdTracerPowerSpectrumMultipoles._set_params
    transform_params = PyBirdTracerPowerSpectrumMultipoles.transform_params

    def calculate(self, **params):
        self._set_from_pt()
        params = self.decode_params(params)
        self.corr = self.pt.combine_bias_terms_poles(self.transform_params(**params), nd=self.nbar)


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
    # pars[2] = pars[2] - 4. / 7. * (b1 - 1.)  # bs
    pars[3] = pars[3] + 32. / 315. * (b1 - 1.)  # b3
    FOLPS.f0 = f0
    fk = table[1] * f0
    pkl, pkl_now, sigma2t = table[0], table_now[0], sigma2t
    pkmu = jac * ((b1 + fk * mu**2)**2 * (pkl_now + jnp.exp(-k**2 * sigma2t)*(pkl - pkl_now)*(1 + k**2 * sigma2t))
                   + jnp.exp(-k**2 * sigma2t) * FOLPS.PEFTs(k, mu, pars, table)
                   + (1 - jnp.exp(-k**2 * sigma2t)) * FOLPS.PEFTs(k, mu, pars, table_now))
    return pkmu


class FOLPSPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    _default_options = dict(kernels='fk')
    _pt_attrs = ['kap', 'muap', 'table', 'table_now', 'sigma2t', 'f0', 'jac']

    def initialize(self, k=None, ells=(0, 2, 4), mu=6, template=None, z=None, **kwargs):
        self._set_options(k=k, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
        self.template.init.update(with_now='peakaverage')
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        import FOLPSnu as FOLPS
        FOLPS.Matrices()
        self.matrices = Namespace(**{name: getattr(FOLPS, name) for name in ['M22matrices', 'M13vectors', 'bnu_b', 'N']})

    def calculate(self):
        self.z = self.template.z
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
        jac, kap, muap = self.template.ap_k_mu(self.k, self.to_poles.mu)
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
        state = self.to_poles.__getstate__()  # mu, wmu
        for name in ['k', 'z', 'ells', 'sigma8', 'fsigma8']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._pt_attrs:
            if hasattr(self.pt, name):
                state[name] = getattr(self.pt, name)
        return state

    def __setstate__(self, state):
        for name in ['k', 'z', 'ells', 'sigma8', 'fsigma8']:
            if name in state: setattr(self, name, state.pop(name))
        self.to_poles = ProjectToMultipoles.from_state({name: state.pop(name) for name in ['mu', 'wmu']})
        self.pt = Namespace(**state)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/henoriega/FOLPS-nu')


class FOLPSTracerPowerSpectrumMultipoles(BaseTracerPTPowerSpectrumMultipoles):
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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
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
    _default_options = dict(freedom=None, prior_basis='physical', tracer=None, fsat=None, sigv=None, shotnoise=1e4)

    @classmethod
    def _get_multitracer(cls, tracers=None, prior_basis='physical'):
        deterministic = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'ct']
        stochastic = ['sn0', 'sn2']
        if prior_basis == 'physical':
            deterministic = [name + 'p' for name in deterministic]
            stochastic = [name + 'p' for name in stochastic]
        return MultitracerBiasParameters(tracers=tracers, deterministic=deterministic, stochastic=stochastic, ntracers=1)

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self._set_params()
        self.decode_params = self._get_multitracer(tracers=tracers, prior_basis=self.options['prior_basis'])

    @classmethod
    def _params(cls, params, freedom=None, prior_basis='physical', tracers=None):
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
        params = cls._get_multitracer(tracers=tracers, prior_basis=prior_basis)._params(params)
        return params

    def _set_params(self):
        self.is_physical_prior = self.options['prior_basis'] == 'physical'
        if self.is_physical_prior:
            settings = get_physical_stochastic_settings(tracer=self.options['tracer'])
            for name, value in settings.items():
                if self.options[name] is None: self.options[name] = value
            if self.mpicomm.rank == 0:
                self.log_debug('Using fsat, sigv = {:.3f}, {:.3f}.'.format(self.options['fsat'], self.options['sigv']))
        super()._set_params(pt_params=[])
        fix = []
        if 4 not in self.ells: fix += ['alpha4']
        if 2 not in self.ells: fix += ['alpha2', 'sn2']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)
        self.nbar = 1e-4
        self.fsat = self.snd = 1.
        if self.is_physical_prior:
            self.fsat, self.snd = self.options['fsat'], self.options['shotnoise'] * self.nbar  # normalized by 1e-4

    def calculate(self, **params):
        self._set_from_pt()
        if self.is_physical_prior:
            # defaults for correlation function
            params = self.decode_params(params, defaults={f'sn{i:d}p': 0. for i in [0, 2]})
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
            params = self.decode_params(params, defaults={f'sn{i:d}': 0. for i in [0, 2]})
            pars = [params[name] for name in ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'ct', 'sn0', 'sn2']]
        opts = {}
        self.power = self.pt.combine_bias_terms_poles(pars, **opts, nd=self.nbar)


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
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
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
    _power_cls = FOLPSTracerPowerSpectrumMultipoles

    @classmethod
    def _params(cls, params, tracers=None, prior_basis='physical'):
        return cls._power_cls._params(params, tracers=tracers, prior_basis=prior_basis)


def pt_kernel(k, q, wq):
    jq = q**2 * wq / (4. * np.pi**2)
    k = k[:, None]
    x = q / k
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

    return 2 * jq * kernel_ff(x)


@jit
def pt_pk_1loop(k, q, wq, pk_q, kernel13_d):
    # We could have a speed-up with FFTlog, see https://arxiv.org/pdf/1603.04405.pdf
    k11 = k
    k = k[:, None]
    jq = q**2 * wq / (4. * np.pi**2)

    mus, wmus = utils.weights_mu(10, method='leggauss')

    # Compute P22
    pk_k = jnp.interp(k11, q, pk_q)

    def get_pk22_dd(mu, wmu):
        kdq = k * q * mu  # k \cdot q
        kq2 = k**2 - 2. * kdq + q**2  # |k - q|^2
        qdkq = kdq - q**2   # k \cdot (k - q)
        F2_d = 5. / 7. + 1. / 2. * qdkq * (1. / q**2 + 1. / kq2) + 2. / 7. * qdkq**2 / (q**2 * kq2)
        pk_kq = jnp.interp(kq2**0.5, q, pk_q, left=0., right=0.)
        jq_pk_q_pk_kq = jq * pk_q * pk_kq
        return 2 * wmu * jnp.sum(F2_d**2 * jq_pk_q_pk_kq, axis=-1)

    pk22_dd = jnp.sum(jax.vmap(get_pk22_dd)(mus, wmus), axis=0)
    pk11 = pk_k
    pk13_dd = 2. * jnp.sum(kernel13_d * pk_q, axis=-1) * pk_k
    pk_dd = pk11 + pk22_dd + pk13_dd
    return pk_dd


class JAXEffortTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):
    r"""
    Wrapper to JAXEffort emulator.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn* and bias term b3*.
    By default, bs and b3 are fixed to 0, following co-evolution.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate multipoles.
    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
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
    https://github.com/CosmologicalEmulators/jaxeffort
    """
    _default_options = dict(freedom=None, prior_basis='physical', tracer=None, fsat=None, sigv=None, shotnoise=1e4)

    @classmethod
    def _get_model_cls(cls, model='velocileptors_rept_mnuw0wacdm'):
        if 'velocileptors_lpt' in model:
            return LPTVelocileptorsTracerPowerSpectrumMultipoles
        elif 'velocileptors_rept' in model:
            return REPTVelocileptorsTracerPowerSpectrumMultipoles
        else:
            raise NotImplementedError

    @classmethod
    def _get_multitracer(cls, model='velocileptors_rept_mnuw0wacdm', prior_basis='physical', tracers=None):
        return cls._get_model_cls(model=model)._get_multitracer(prior_basis=prior_basis, tracers=tracers)

    @classmethod
    def _params(cls, params, model='velocileptors_rept_mnuw0wacdm', freedom=None, prior_basis='physical', tracers=None):
        from desilike.base import get_calculator_config
        model_cls = cls._get_model_cls(model=model)
        params = get_calculator_config(model_cls)[-1]
        if 'velocileptors_lpt' in model:
            return model_cls._params(params, freedom=freedom, prior_basis=prior_basis, tracers=tracers)
        elif 'velocileptors_rept' in model:
            return model_cls._params(params, freedom=freedom, prior_basis=prior_basis, tracers=tracers)
        else:
            raise NotImplementedError

    def _set_params(self):
        if 'velocileptors' in self.model:
            REPTVelocileptorsTracerPowerSpectrumMultipoles._set_params(self)
        else:
            raise NotImplementedError

    def transform_params(self, cosmo, **params):
        if 'velocileptors' in self.model:
            # FIXME sigma8 not provided
            if self.is_physical_prior:
                raise NotImplementedError
                sigma8 = 1.
                f = 0.
                pars = b1L, b2L, bsL, b3L = [params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p'] / sigma8**3]
                pars += [(1 + b1L)**2 * params['alpha0p'], f * (1 + b1L) * (params['alpha0p'] + params['alpha2p']),
                        f * (f * params['alpha2p'] + (1 + b1L) * params['alpha4p']), f**2 * params['alpha4p']]
                sigv = self.options['sigv']
                pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2, 4]]
            else:
                pars = [params[name] for name in self.required_bias_params]
            if 'rept' in self.model:
                pars = list(pars)
                b1 = pars[0]
                pars[2] = pars[2] - (2 / 7) * (b1 - 1.)  # bs
                pars[3] = 3 * pars[3] + (b1 - 1.)  # b3
            return pars
        else:
            raise NotImplementedError
        return

    def initialize(self, k=None, ells=(0, 2, 4), tracers=None, mu=8, model='velocileptors_rept_mnuw0wacdm', cosmo=None, fiducial='DESI', **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo = cosmo
        if cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, mode='geometry', cosmo=self.cosmo)
        self.model = model
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        self._set_params()
        self.decode_params = self._get_multitracer(tracers=tracers)
        import jaxeffort
        self.emulators = [jaxeffort.trained_emulators[model][f"{ell:d}"] for ell in self.ells]

    def calculate(self, **params):
        cosmo_dict = {'ln10As': self.cosmo['logA'], 'ns': self.cosmo['n_s'], 'h': self.cosmo['H0'] / 100.,
                      'omega_b': self.cosmo['omega_b'], 'omega_c': self.cosmo['omega_cdm'], 'm_nu': self.cosmo['m_ncdm_tot'],
                      'w0': self.cosmo['w0_fld'], 'wa': self.cosmo['wa_fld']}
        import jaxeffort
        cosmo_jaxeffort = jaxeffort.W0WaCDMCosmology(**cosmo_dict)
        theta = jnp.array([self.z, cosmo_dict["ln10As"], cosmo_dict["ns"], 100. * cosmo_dict["h"], cosmo_dict["omega_b"], cosmo_dict["omega_c"], cosmo_dict["m_nu"], cosmo_dict["w0"], cosmo_dict["wa"]])
        D = cosmo_jaxeffort.D_z(self.z)
        bias = self.transform_params(cosmo_jaxeffort, **params)
        poles = [emulator.get_Pl(theta, bias, D) for emulator in self.emulators]
        jac, kap, muap = self.apeffect.ap_k_mu(self.k, self.mu)
        pkmu = sum(pole[:, None] * get_legendre(ell)(muap) for ell, pole in zip(self.ells, poles))
        func = lambda kap, pkmu: interp1d(kap, self.emulators[0].P11.k_grid, pkmu)
        pkmu = jac * jax.vmap(func, in_axes=1, out_axes=1)(kap, pkmu)
        self.power = self.to_poles(pkmu)

    def get(self):
        return self.power

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/jaxeffort')


# ============================================================================
# Bispectrum
# ============================================================================

class BaseTracerBispectrumMultipoles(BaseCalculator):

    """Base class for theory tracer power spectrum multipoles."""
    config_fn = 'full_shape.yaml'
    _default_options = dict(shotnoise=1e4)
    _initialize_with_namespace = True
    _calculate_with_namespace = True

    @classmethod
    def _get_multitracer(cls, tracers=None):
        return MultitracerBiasParameters(tracers=tracers, ntracers=1)

    @classmethod
    def _params(cls, params, tracers=None):
        return cls._get_multitracer(tracers=tracers)._params(params)

    def initialize(self, k=None, ells=((0, 0, 0), (2, 0, 2)), tracers=None, basis='sugiyama', **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, basis=basis, **kwargs)
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_options(self, k=None, ells=((0, 0, 0), (2, 0, 2)), tracers=None, basis='sugiyama', **kwargs):
        # Wavenumber and multipoles
        if k is None:
            if basis == 'soccimarro':
                # Default k-bins (k1, k2, k3) in Soccimarro basis
                k = np.linspace(0.01, 0.1, 11)
                k = np.meshgrid(k, k, k, indexing='ij')
                k = np.column_stack([kk.ravel() for kk in k])
                # Impose triangular condition
                mask = (k[:, 0] <= k[:, 1] + k[:, 2]) | (k[:, 1] <= k[:, 0] + k[:, 2]) | (k[:, 2] <= k[:, 0] + k[:, 1])
                k = k[mask]
            else:
                k = np.column_stack([np.linspace(0.01, 0.1, 11)] * 2)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)
        self.tracers = tracers
        # First set shotnoise, useful for rescaling stochastic terms
        shotnoise = np.atleast_1d(kwargs.get('shotnoise', 1e4))
        if shotnoise.shape[-1] > 1:
            # cross correlation: geometric mean
            shotnoise = np.prod(shotnoise)**(1. / len(shotnoise))
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        if 'shotnoise' in self.options:
            self.options['shotnoise'] = shotnoise
        # The quantity used for the rescaling
        self.nbar = 1. / shotnoise

    def calculate(self, **params):
        params = self.decode_params(params)
        # params['b1'] is a single parameter value in standard case
        # a tuple if multitracer support

    def get(self):
        # Return power spectrum multipoles
        return self.power

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'ells', 'nbar', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot bispectrum multipoles.

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
            height_ratios = [3, 1]
            figsize = (6, 1.5 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
            fig.subplots_adjust(hspace=0.1)
        else:
            lax = fig.axes
        ax = lax[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(np.arange(len(self.k)), self.k.prod(axis=-1) * self.power[ill], color=f'C{ill:d}', label=rf'$\ell = {ell}$')
        ax.set_xlabel('bin index')
        ax.grid(True)
        ax.legend()
        if 'scoccimarro' in self.basis:
            ax.set_ylabel(r'$k_1 k_2 k_3 B_{\ell}(k_1, k_2, k_3)$ [$(\mathrm{Mpc}/h)^{6}$]')
        else:
            ax.set_ylabel(r'$k_1 k_2 B_{\ell_1 \ell_2 \ell_3}(k_1, k_2)$ [$(\mathrm{Mpc}/h)^{4}$]')
        for i in range(self.k.shape[1]):
            lax[1].plot(np.arange(len(self.k)), self.k[..., i], color=f'C{i:d}', label=f'$k_{i:d}$')
        return fig


class BaseTracerPTBispectrumMultipoles(BaseTracerBispectrumMultipoles):

    """Base class for theory tracer power spectrum multipoles."""
    config_fn = 'full_shape.yaml'
    _default_options = dict(shotnoise=1e4)

    def initialize(self, k=None, ells=((0, 0, 0), (2, 0, 2)), pt=None, template=None, tracers=None, basis='sugiyama', **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, basis=basis, **kwargs)
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self.decode_params = self._get_multitracer(tracers=tracers)

    def _set_pt(self, pt=None, template=None, **kwargs):
        # Perturbation theory module
        if pt is None:
            _pt_cls = getattr(self, '_pt_cls', None)
            if _pt_cls is None:
                _pt_cls = globals()[self.__class__.__name__.replace('Tracer', '')]
            pt = _pt_cls()
        self.pt = pt
        # Linear power spectrum
        if template is not None:
            self.pt.init.update(template=template)
        # Transfer options to PT module
        for name, value in self.pt._default_options.items():
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
            elif name in self.options:
                self.pt.init.update({name: self.options[name]})
        # mu-integration for multipoles
        for name in ['mu']:
            if name in kwargs:
                self.pt.init.update({name: kwargs.pop(name)})
        self.pt.init.update({name: kwargs[name] for name in kwargs if name not in self._default_options})

    def _set_from_pt(self):
        # Update z, k, ells from pt
        for name in ['z']:
            setattr(self, name, getattr(self.pt, name))

    def _set_params(self, pt_params=None):
        if pt_params is not None:
            self.pt.init.params.update([param for param in self.init.params if param.basename in pt_params], basename=True)
            self.init.params = self.init.params.select(basename=[param.basename for param in self.init.params if param.basename not in pt_params])

    def calculate(self):
        self._set_from_pt()



class GeoFPTAXTracerBispectrumMultipoles(BaseTracerBispectrumMultipoles):
    r"""
    GeoFPTAX bispectrum multipoles.
    Can be exactly marginalized over stochastic parameters sn*.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Note
    ----
    This is the bispectrum in the scoccimarro basis. Not really supported for now.

    Parameters
    ----------
    k : tuple of arrays, default=None
        Triangles of wavenumbers of shape (nk, 3) where to evaluate multipoles.
    ells : tuple, default=(0, 2)
        Multipoles to compute.
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    pt : str, default=None
        Order of :math:`P(k)` fed into the bispectrum calculation.
        If ``None``, linear :math:`P(k)`.
        If '1loop', use 1-loop standard PT.
    shotnoise : array, default=1e4
        Shot noise for each of the multipoles. Same length as ``k``.
    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters:
        :math:`b_{1}^\prime = (b_{1}^{E}) \sigma_{8}(z), b_{2}^\prime = b_{2}^{E} \sigma_{8}(z)^2

    Reference
    ---------
    - https://arxiv.org/pdf/2303.15510v1
    - https://github.com/dforero0896/geofptax
    """
    config_fn = 'full_shape.yaml'
    _default_options = dict(prior_basis='physical', mu=50)

    @classmethod
    def _get_multitracer(cls, tracers=None, prior_basis='physical'):
        deterministic = ['b1', 'b2', 'sigmav']
        stochastic = ['sn0']
        if prior_basis == 'physical':
            deterministic = [name + 'p' for name in deterministic]
            stochastic = [name + 'p' for name in stochastic]
        return MultitracerBiasParameters(tracers=tracers, deterministic=deterministic, stochastic=stochastic, ntracers=1)

    def initialize(self, k=None, ells=((0, 0, 0), (2, 0, 2)), tracers=None, basis='sugiyama', pt=None, template=None, z=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, basis=basis, **kwargs)
        BasePTPowerSpectrumMultipoles._set_template(self, template=template, z=z, klim=(1e-3, 2., 500))
        self.z = self.template.z
        self._set_params()
        self.decode_params = self._get_multitracer(tracers=tracers)
        self.pt = pt
        assert self.pt in [None, '1loop']

    @classmethod
    def _params(cls, params, prior_basis='physical', tracers=None):
        # Prior basis is 'physical' = sampled bias bp is b * sigma8^n
        if prior_basis == 'physical':
            for param in list(params):
                basename = param.basename
                param.update(basename=basename + 'p')
                #params.set({'basename': basename, 'namespace': param.namespace, 'derived': True})
            for param in params.select(basename='b1p'):
                param.update(prior=dict(dist='uniform', limits=[0., 3.]), ref=dict(dist='norm', loc=1., scale=0.1))
            for param in params.select(basename=['b2p']):
                param.update(prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.))
            for param in params.select(basename='sn*p'):
                param.update(prior=dict(dist='norm', loc=0., scale=2. if 'sn0' in param.basename else 5.), ref=dict(dist='norm', loc=0., scale=1.))
        params = cls._get_multitracer(tracers=tracers, prior_basis=prior_basis)._params(params)
        return params

    def _set_params(self):
        # Set parameters (self.init.params)
        self.is_physical_prior = self.options['prior_basis'] == 'physical'
        fix = []
        if 2 not in self.ells: fix += ['sn2']
        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)

    def calculate(self, **params):
        # Calculte the bispectrum (set attribute self.power, see at the end)
        self.z = self.template.z
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.f * self.sigma8
        params = self.decode_params(params)
        pars = []
        # Conversion from "physical" bias parameters to standard basis
        if self.is_physical_prior:
            sigma8 = self.template.sigma8
            f = self.template.fsigma8 / sigma8
            b1E, b2E = params['b1p'] / sigma8, params['b2p'] / sigma8**2
            pars += [b1E, b2E, params['sigmavp'], params['sn0p']]
        else:
            pars = [params[name] for name in ['b1', 'b2', 'sigmav', 'sn0']]
        # b1, b2, A_P, sigma_P, A_B, sigma_B, *_P
        pars = pars[:2] + [1., 4.] + [pars[3], pars[2]]
        # Alock-Paczynski parameters are self.template.qpar, self.template.qper
        all_pars = jnp.array([self.sigma8, self.fsigma8 / self.sigma8, self.template.qpar, self.template.qper] + pars)
        from geofptax.kernels import bk_multip

        kt = self.template.k
        pkt = self.template.pk_dd  # theory linear pk
        if self.pt:  # loop correction: update pkt with 1-loop calculation
            q = kt
            ktmin, ktmax = min(kk.min() for kk in self.k) * 0.7, max(kk.max() for kk in self.k) * 1.3
            kt = jnp.linspace(ktmin, ktmax, 500)
            wq = utils.weights_trapz(q)
            if getattr(self, 'kernel', None) is None:
                # Compute pt kernel the first time only
                self.kernel = pt_kernel(kt, q, wq)
            pkt = pt_pk_1loop(kt, q, wq, pkt, self.kernel)

        # k for bk0, bk200, bk020, bk002
        kk = list(self.k) + [self.k[-1]] * (4 - len(self.k))
        # Compute bk multipoles
        res = bk_multip(*kk, kt, pkt, all_pars, redshift=self.z, num_points=self.options['mu'])
        tells = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
        res = [res[tells.index(ell)] for ell in self.ells]
        # Include shot noise term, rescaling by AP (alpha_par * alpha_per**2)**2
        A_B = all_pars[8] / (all_pars[2] * all_pars[3]**2)**2
        res = [rr + A_B * sn for rr, sn in zip(res, self.shotnoise)]
        self.power = res

    def get(self):
        # Returned value when calling the calculator
        return self.power

    @classmethod
    def install(cls, installer):
        # Dependency
        installer.pip('git+https://github.com/dforero0896/geofptax')




# @jit
@jit(static_argnames=['rsd_class', 'IR_resummation', 'damping'])
def folpsv2_combine_bias_terms_pkmu(k, mu, jac, table, table_now, pars,rsd_class, IR_resummation=True, damping='lor'):
    b1 = pars[0]
    f0 = table[-1]
    fk = table[1] * f0
    pkl, pkl_now = table[0], table_now[0]
    sigma2, delta_sigma2 = table_now[-3:-1]
    # Sigma² tot for IR-resummations, see eq.~ 3.59 at arXiv:2208.02791
    if IR_resummation:
        sigma2t = (1 + f0*mu**2 * (2 + f0)) * sigma2 + (f0*mu)**2 * (mu**2 - 1) * delta_sigma2
    else:
        sigma2t = 0
    pkmu = ((b1 + fk * mu**2)**2 * (pkl_now + jnp.exp(-k**2 * sigma2t)*(pkl - pkl_now)*(1 + k**2 * sigma2t))
                 + jnp.exp(-k**2 * sigma2t) * rsd_class.get_eft_pkmu(k, mu, pars, table, damping)
                 + (1 - jnp.exp(-k**2 * sigma2t)) * rsd_class.get_eft_pkmu(k, mu, pars, table_now, damping))
    return pkmu * jac


def _get_bispectrum_multipoles_folpsv2(
    pars,
    k1k2,
    k_pkl_pklnw_fk,
    f0, qpar, qper,
    multipoles=['B000', 'B202'],
    precision=(8, 10, 10),
    damping='lor',
    interpolation_size=20,
    interpolation_method='linear',
    bias_scheme='folps',
    model='FOLPSD',
    renormalized=True,
):
    import folps as folpsv2
    # folpsv2.MatrixCalculator(A_full=True, use_TNS_model=False)
    # folps_bispectrum_class = folpsv2.BispectrumCalculator_fk(model='FOLPSD')
    f0 = jnp.asarray(f0)
    bpars = jnp.asarray(pars)

    if k1k2.ndim == 1:
        bs = folpsv2.WindowConvolvedBispectrum(model=model)
        results = bs.reduced_Bl1l2L(bpars, None,
                        qpar, qper, k_pkl_pklnw_fk, k1k2, Ssize=interpolation_size,
                        precision_full=[8, 10, 10], precision_diag=[12, 15, 15],
                        f=f0,
                        renormalize=renormalized,
                        interpolation_method_full=interpolation_method,
                        interpolation_method_diag=interpolation_method,
                        use_full_diag=True)
        ells = ['B000', 'B110', 'B220', 'B112', 'B202']
        toret = []
        for ell in multipoles:
            if ell in ells:
                toret.append(results[ells.index(ell)].ravel())
            elif (ell_swap:=ell[0] + ell[2:0:-1] + ell[3:]) in ells:
                toret.append(results[ells.index(ell_swap)].T.ravel())
            else:
                toret.append(np.zeros((k1k2.size,) * 2).ravel())
        folpsv2.BispectrumCalculator._tables_cache = {}  # to avoid leak
        return toret

    bispectrum = folpsv2.BispectrumCalculator(model=model)
    toret = bispectrum.Sugiyama_Bell(
        f=f0,
        bpars=bpars,
        k_pkl_pklnw=k_pkl_pklnw_fk,
        k1k2pairs=k1k2,
        qpar=qpar,
        qper=qper,
        precision=precision,
        damping=damping,
        multipoles=list(multipoles),
        bias_scheme=bias_scheme,
        renormalize=renormalized,
        interpolation_method=interpolation_method
    )
    folpsv2.BispectrumCalculator._tables_cache = {}  # to avoid leak
    return toret


class FOLPSv2PowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    _default_options = dict(kernels='fk', rbao=104., A_full=True, remove_DeltaP=False, backend='jax')

    # 'qpar','qper','f','f0',
    _pt_attrs = ['jac', 'kap', 'muap', 'table', 'table_now', 'scalars', 'scalars_now', 'A_full', 'remove_DeltaP', 'qpar', 'qper', 'f', 'f0', 'pklir']

    def initialize(self, k=None, ells=(0, 2, 4), mu=6, template=None, z=None, **kwargs):
        self._set_options(k=k, ells=ells, **kwargs)
        self._set_template(template=template, z=z)
        self.template.init.update(with_now='peakaverage')
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu
        os.environ.setdefault('FOLPS_BACKEND', self.options['backend'])
        import folps as folpsv2
        folps_matrix_class = folpsv2.MatrixCalculator(A_full=self.options['A_full'], use_TNS_model=self.options['remove_DeltaP'])
        self.matrices = folps_matrix_class.get_mmatrices()

    def calculate(self):
        import folps as folpsv2
        self.z = self.template.z
        cosmo_params = {}
        cosmo_params['pkttlin'] = self.template.pk_dd * self.template.fk**2
        cosmo_params['f0'] = self.template.f0

        if getattr(self, '_get_non_linear', None) is None:
            # from folpsv2 import NonLinearPowerSpectrumCalculator
            # folpsv2.BackendManager(preferred_backend='jax')
            def _get_non_linear(pk_dd, pknow_dd, **cosmo_params):
                #folpsv2.MatrixCalculator(A_full=self.options['A_full'], use_TNS_model=self.options['remove_DeltaP'])
                folps_nlps_class = folpsv2.NonLinearPowerSpectrumCalculator(mmatrices=self.matrices,
                                    kernels=self.options['kernels'], rbao=self.options['rbao'], **cosmo_params)
                # pknow = folpsv2.extrapolate_pklin(k, pknow_dd)
                # folps_nlps_class._initialize_nonwiggle_power_spectrum(pknow=pknow_dd)
                return folps_nlps_class.calculate_loop_table(k=self.template.k, pklin=pk_dd, pknow=pknow_dd, **cosmo_params)

            self._get_non_linear = jit(_get_non_linear) if self.options['backend'] == 'jax' else _get_non_linear
            #Commented out for now, only going ahead with numpy implementation
            # self._get_non_linear = _get_non_linear

        table, table_now = self._get_non_linear(self.template.pk_dd, self.template.pknow_dd, **cosmo_params)
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)

        extra = 6 if self.options['A_full'] else 0
        table_pklir = (table[0], *table[1:28 + extra], *table[28 + extra:])
        table_now_pklir = (table[0], *table_now[1:28 + extra], *table_now[28 + extra:])
        self.pklir = folpsv2.get_linear_ir_ini(table_pklir[0], table_pklir[1], table_now_pklir[1], k_BAO=1. / self.template.cosmo.rs_drag)

        self.pt = Namespace(jac=jac, kap=kap, muap=muap, table=table[1:28 + extra], table_now=table_now[1:28 + extra],
                            scalars=table[28 + extra:], scalars_now=table_now[28 + extra:], A_full=self.options['A_full'],
                            remove_DeltaP=self.options['remove_DeltaP'], f=self.template.f,
                            f0=self.template.f0, qpar=self.template.qpar, qper=self.template.qper, pklir=self.pklir)
        # ,qpar = self.template.qpar, qper=self.template.qper, ,f=self.template.f,f0=self.template.f0
        self.kt = table[0]
        self.sigma8 = self.template.sigma8
        self.fsigma8 = self.template.f * self.sigma8

    @property
    def qpar(self):
        return self.pt.qpar

    @property
    def qper(self):
        return self.pt.qper

    def combine_bias_terms_spectrum_poles(self, pars, nd=1e-4, **kwargs):
        import folps as folpsv2
        table = (self.kt, *self.pt.table, *self.pt.scalars)
        table_now = (self.kt, *self.pt.table_now, *self.pt.scalars_now)
        # Inject shot noise at correct position
        pars = list(pars[:-1]) + [1. / nd, pars[-1]]  #1. / nd
        ncols = len(table)
        # Sync the FOLPSpip-module-level globals to the A_full / remove_DeltaP settings
        import folps.folps as _folps_module
        _folps_module.A_full_status = getattr(self.pt, 'A_full', True)
        _folps_module.use_TNS_model_status = getattr(self.pt, 'remove_DeltaP', True)
        if getattr(self, '_get_poles', None) is None:

            @jit(static_argnums=(4, 5))
            def _get_poles(jac, kap, muap, pars, bias_scheme, damping, *table):
                # print(self.pt.A_full)
                # folpsv2.MatrixCalculator(A_full=getattr(self.pt, "A_full", True), use_TNS_model=getattr(self.pt, "remove_DeltaP", False))
                folps_rsdmps_class = folpsv2.RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
                pars = folps_rsdmps_class.set_bias_scheme(pars=pars, bias_scheme=bias_scheme) #folps
                return self.to_poles(jac * folps_rsdmps_class.get_rsd_pkmu(kap, muap, pars, table[:ncols], table[ncols:], IR_resummation=True, damping=damping))

            # self._get_poles = jit(_get_poles) #Only going ahead with numpy implementation for now
            # self._get_poles = jit(_get_poles)  if kwargs['backend'] == 'jax' else _get_poles
            self._get_poles = _get_poles

        return self._get_poles(self.pt.jac, self.pt.kap, self.pt.muap, jnp.array(pars), kwargs['bias_scheme'], kwargs['damping'], *table, *table_now)

    def combine_bias_terms_bispectrum_poles(self, pars, k1k2, ells=None, **kwargs):
        import folps as folpsv2
        table = (self.kt, *self.pt.table, *self.pt.scalars)
        table_now = (self.kt, *self.pt.table_now, *self.pt.scalars_now)
        k_pkl_pklnw_fk = jnp.array([table[0], table[1], table_now[1], table[2] * self.pt.f0])
        multipoles = tuple(f"B{ell1}{ell2}{ell3}" for (ell1, ell2, ell3) in ells)
        get_bispectrum_multipoles_jit = _get_bispectrum_multipoles_folpsv2
        full = not np.allclose(k1k2[..., 1], k1k2[..., 0])
        if full:
            k1k2 = np.unique(k1k2[..., 0])
        if folpsv2.backend_manager.backend == 'jax':
            get_bispectrum_multipoles_jit = jit(static_argnames=['multipoles', 'precision', 'damping', 'interpolation_method', 'bias_scheme', 'renormalized'])(_get_bispectrum_multipoles_folpsv2)
        poles = get_bispectrum_multipoles_jit(pars, k1k2, k_pkl_pklnw_fk, self.pt.f0, self.pt.qpar, self.pt.qper,
                                              multipoles=multipoles,
                                              **{key: kwargs[key] for key in ['precision', 'damping', 'interpolation_method', 'bias_scheme', 'renormalized'] if key in kwargs})
        poles = jnp.asarray(poles)
        return poles

    def __getstate__(self):
        state = self.to_poles.__getstate__()  # mu, wmu
        for name in ['k', 'z', 'ells', 'kt', 'sigma8', 'fsigma8']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._pt_attrs:
            if hasattr(self.pt, name):
                state['pt-' + name] = getattr(self.pt, name)
        return state

    def __setstate__(self, state):
        for name in ['k', 'z', 'ells', 'kt', 'sigma8', 'fsigma8']:
            if name in state: setattr(self, name, state.pop(name))
        self.to_poles = ProjectToMultipoles.from_state({name: state.pop(name) for name in ['mu', 'wmu']})
        if not hasattr(self, 'pt'): self.pt = Namespace()
        self.pt.update(**{name[3:]: value for name, value in state.items() if name.startswith('pt-')})

    @classmethod
    def install(cls, installer):
         installer.pip('git+https://github.com/cosmodesi/FolpsD')


class FOLPSv2TracerPowerSpectrumMultipoles(BaseTracerPTPowerSpectrumMultipoles):
    r"""
    FOLPS power spectrum multipoles.
    Can be exactly marginalized over stochastic parameters sn*.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    pt : FOLPSv2PowerSpectrumMultipoles, optional
        PT calculator.
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    k : array (N, 2)
        Output wavenumbers.
    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
    shotnoise : array, default=1e4
        Shot noise for each of the multipoles.
    prior_basis : str, default='standard'
        - standard: standard basis as used in folps paper (ArXiv: 2404.07269)
        - physical: physical basis as used in velocileptors paper from DR1
        - physical_aap: physical basis from the 2pt3pt prior document
        - tcm_chudaykin_aap: physical basis with AP scaling along with class-pt basis from Chudaykin et. al.
    """
    _default_options = dict(freedom=None, prior_basis='physical_aap', tracer=None, fsat=None, sigv=None, shotnoise=1e4, model='FOLPSD',
                            bias_scheme='folps', IR_resummation=True, damping='lor',
                            b3_coev=True, backend='jax', sigma8_fid=None, h_fid=None)
    _pt_cls = FOLPSv2PowerSpectrumMultipoles
    # Helpers
    @classmethod
    def _get_multitracer(cls, tracers=None, prior_basis='physical_aap'):
        deterministic = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'ct', 'X_FoG_p']
        stochastic = ['sn0', 'sn2']
        if 'physical' in prior_basis:
            deterministic = [name + 'p' for name in deterministic]
            stochastic = [name + 'p' for name in stochastic]
        return MultitracerBiasParameters(tracers=tracers, deterministic=deterministic, stochastic=stochastic, ntracers=1)

    @staticmethod
    def _rename_prior_basis(prior_basis: str) -> str:
        pb = str(prior_basis).strip()
        aliases = {
            'standard': 'standard_folps',
            'physical': 'physical_velocileptors',
            'physical_aap': 'physical_aap',
            'tcm_chudaykin_aap': 'tcm_chudaykin_aap',
        }
        if pb not in aliases:
            raise ValueError(f"Unknown prior_basis='{prior_basis}'. "
                             "Valid: ['standard', 'physical', 'physical_aap',"
                             "'tcm_chudaykin_aap'].")
        return aliases[pb]

    def initialize(self, k=None, ells=(0, 2, 4), pt=None, template=None, tracers=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, **kwargs)
        self.prior_basis = self._rename_prior_basis(self.options['prior_basis'])
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self._set_params()
        self.decode_params = self._get_multitracer(tracers=tracers, prior_basis=self.prior_basis)

    # Default parameter priors (before initialization)
    @classmethod
    def _params(cls, params, freedom=None, prior_basis='physical_aap', tracers=None):
        prior_basis = cls._rename_prior_basis(prior_basis)
        # freedom logic (pre-rename)
        fix = []
        if freedom in ['min', 'max']:
            for param in params.select(basename=['b1']):
                param.update(prior=dict(limits=[0., 10.]))
            for param in params.select(basename=['b2']):
                param.update(prior=dict(limits=[-50., 50.]))

            # remove priors for the nuisances (let later blocks define them)
            for param in params.select(basename=['bs', 'b3', 'alpha*', 'sn*', 'X_FoG_p']):
                param.update(prior=None)

        if freedom == 'max':
            for param in params.select(basename=['b1', 'b2', 'bs', 'b3', 'X_FoG_p']):
                param.update(fixed=False)
            fix += ['ct']

        if freedom == 'min':
            fix += ['b3', 'bs', 'ct']

        for param in params.select(basename=fix):
            param.update(value=0., fixed=True)

        # physical modes: rename -> add suffix 'p'
        if 'physical' in prior_basis:
            for param in list(params):
                param.update(basename=param.basename + 'p')
            # b1p prior
            for param in params.select(basename='b1p'):
                param.update(prior=dict(dist='uniform', limits=[0., 3.]),
                             ref=dict(dist='norm', loc=1., scale=0.1))
            for param in params.select(basename=['b2p', 'bsp', 'b3p']):
                param.update(prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.))
            for param in params.select(basename='b3p'):
                param.update(value=0., fixed=True)
            for param in params.select(basename='alpha*p'):
                param.update(prior=dict(dist='norm', loc=0., scale=12.5), ref=dict(dist='norm', loc=0., scale=1.))  # 50% at k = 0.2 h/Mpc
            for param in params.select(basename='sn*p'):
                param.update(prior=dict(dist='norm', loc=0., scale=2. if 'sn0' in param.basename else 5.), ref=dict(dist='norm', loc=0., scale=1.))
        params = cls._get_multitracer(tracers=tracers, prior_basis=prior_basis)._params(params)
        return params

    # Process the input parameters (at initialization)
    def _set_params(self):
        self.is_physical_prior = 'physical' in self.prior_basis

        if self.is_physical_prior:
            settings = get_physical_stochastic_settings(tracer=self.options['tracer'])
            for name, value in settings.items():
                if self.options[name] is None:
                    self.options[name] = value

            if self.mpicomm.rank == 0:
                self.log_debug('Using fsat, sigv = {:.3f}, {:.3f}.'.format(self.options['fsat'], self.options['sigv']))

        # fix unused multipole-related params
        fix = []
        if 4 not in self.ells:
            fix += ['alpha4'] if not self.is_physical_prior else ['alpha4p']
        if 2 not in self.ells:
            fix += (['alpha2', 'alpha2shot'] if not self.is_physical_prior else ['alpha2p', 'alpha2shotp'])

        for param in self.init.params.select(basename=fix):
            param.update(value=0., fixed=True)

        self.nbar = 1e-4
        self.fsat = self.snd = 1.
        if self.is_physical_prior:
            self.fsat = self.options['fsat']
            self.snd = self.options['shotnoise'] * self.nbar  # normalized by 1e-4

    # main mapping
    def calculate(self, **params):
        self._set_from_pt()
        params = self.decode_params(params)
        # Case A: STANDARD_FOLPS -> forward directly (Eulerian nuisances)
        if self.prior_basis == 'standard_folps':
            pars = [params[name] for name in ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'ct', 'sn0', 'sn2', 'X_FoG_p']]
            if self.options['b3_coev']:
                b1 = pars[0]
                pars[3] = 32 / 315 * (b1 - 1)
            opts = {}
            self.power = self.pt.combine_bias_terms_spectrum_poles(pars, **opts, nd=self.nbar, model=self.options['model'],
                                                                    bias_scheme=self.options['bias_scheme'], IR_resummation=self.options['IR_resummation'],
                                                                    damping=self.options['damping'], prior_basis=self.options['prior_basis'],
                                                                    b3_coev=self.options['b3_coev'], backend=self.options['backend'])
            return

        # From here on: PHYSICAL modes
        sigma8 = self.pt.sigma8
        f = self.pt.fsigma8 / sigma8
        sigma8_fid = self.options.get('sigma8_fid', None)
        # amplitude rescaling convention (Class-PT style)
        A = (sigma8 / sigma8_fid)**2 if sigma8_fid is not None else 1.0
        qpar, qper = self.pt.qpar, self.pt.qper
        # A_AP = (h_fid / h)**3 / (qper**2 * qpar)
        A_AP = 1 / (qper**2 * qpar)
        sqrt_A_AP = A_AP**0.5
        self.A_AP = A_AP

        # Counterterms mapping
        if self.prior_basis == 'physical_velocileptors':
            # This one need to be fixed
            # --- Lagrangian -> Eulerian ---
            b1L = params['b1p'] / sigma8 - 1.0
            b2L = params['b2p'] / sigma8**2
            bsL = params['bsp'] / sigma8**2
            b1E  = 1.0 + b1L
            b2E  = b2L + 8.0 / 21.0 * b1L
            # defaults (non-APscaling)
            b3L  = params['b3p']
            bsE = -4.0 / 7.0 * b1L + bsL
            b3E  = b3L + 32.0 / 315.0 * b1L
            ctildeE = params.get('ctp', 0.0)
            # interpret alpha?p as the actual EFT alpha0/alpha2/alpha4 coefficients
            # (optionally undo overall A if you want the same convention as your other physical modes)
            alpha0, alpha2, alpha4 = (params[name] / A for name in ['alpha0p', 'alpha2p', 'alpha4p'])
            pars = [b1E, b2E, bsE, b3E, alpha0, alpha2, alpha4, ctildeE]
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2]]
            pars += [params['X_FoG_pp']]
        elif self.prior_basis == 'physical_aap':
                # --- Lagrangian -> Eulerian ---
            b1L = params['b1p'] / sigma8 / sqrt_A_AP - 1.0
            b2L = params['b2p'] / sigma8**2 / sqrt_A_AP
            bK2 = params['bsp'] / sigma8**2 / sqrt_A_AP
            # btd = params['bsp'] / sigma8**3 / sqrt_A_AP
            btd = params['b3p'] / A_AP / sigma8**4
            b1E  = 1.0 + b1L
            # b2E  = b2L + 8.0 / 21.0 * b1L
            b2E  = b2L
            if self.options['b3_coev']:
                # bK2 = -2/7*(b1E-1)
                btd = 23 / 42 * (b1E - 1)
            bsE = 2 * bK2
            b3E = 64 / 105 * (-5 / 4 * bsE - btd)
            # bsE = -4.0 / 7.0 * (b1E-1)
            # b3E  = 32.0 / 315.0 * (b1E-1)
            ctildeE = params.get('ctp', 0.0)
            # interpret alpha?p as tilde-alphas, map to folps alphas
            a0t, a2t, a4t = (params[name] / A_AP / sigma8**2 for name in ['alpha0p', 'alpha2p', 'alpha4p'])
            alpha0 = (b1E**2) * a0t
            alpha2 = (b1E * f) * (a0t + a2t)
            alpha4 = (f**2) * a2t + (b1E * f) * a4t
            pars = [b1E, b2E, bsE, b3E, alpha0, alpha2, alpha4, ctildeE]
            # NOTE: ignores the mu^6 term if your true model has it.
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] / A_AP * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2]]
            pars += [params['X_FoG_pp']]
        elif self.prior_basis == 'tcm_chudaykin':
            # APscaling: include A_AP and decode the table-style priors
            self.options['bias_scheme'] = 'classpt' #As in chudaykin et. al.
            b1L, b2L, bsL, b3 = params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p'] / A
            pars = [1. + b1L, b2L, bsL, b3]   #Class-pt bias free b3
            c0, c2, c4 = (params[name] / (A * A_AP) for name in ['alpha0p', 'alpha2p', 'alpha4p'])
            pars += [-2 / 105 * (105 * c0 - 35 * c2 * f + 9 * c4 * f**2), -2 / 7 * f * (7 * c2 - 6 * f * c4), -2 * f**2 * c4, 0]
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2]]
            pars += [params['X_FoG_pp']]
            # use coevolution b3E = b3E + 32.0/315.0 * b1L ?
        else:
            raise ValueError(f"Internal error: unsupported normalized prior basis '{self.prior_basis}'.")

        opts = {}
        self.power = self.pt.combine_bias_terms_spectrum_poles(pars, **opts, nd=self.nbar, model=self.options['model'], bias_scheme=self.options['bias_scheme'], IR_resummation=self.options['IR_resummation'],
                                                               damping=self.options['damping'], prior_basis=self.options['prior_basis'], b3_coev=self.options['b3_coev'], backend=self.options['backend'])


class FOLPSv2TracerBispectrumMultipoles(BaseTracerPTBispectrumMultipoles):
    r"""
    FOLPS bispectrum multipoles.
    Can be exactly marginalized over stochastic parameters sn*.
    For the matter (unbiased) power spectrum, set b1=1 and all other bias parameters to 0.

    Parameters
    ----------
    pt : FOLPSv2PowerSpectrumMultipoles, optional
        PT calculator.
    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.
    k : array (N, 2)
        Output wavenumbers.
    ells : tuple, default=((0, 0, 0), (2, 0, 2))
        Multipoles to compute.
        Available are (0, 0, 0), (1, 1, 0), (2, 2, 0), (0, 2, 2), (1, 1, 2).
    tracers : str, default=None
        Tracer name. Namespace added to bias parameters. Cross-correlation not supported.
    shotnoise : array, default=1e4
        Shot noise for each of the multipoles.
    prior_basis : str, default='standard'
        - standard: standard basis as used in folps paper (ArXiv: 2404.07269)
        - physical: physical basis as used in velocileptors paper from DR1
        - physical_aap: physical basis from the 2pt3pt prior document
        - tcm_chudaykin_aap: physical basis with AP scaling along with class-pt basis from Chudaykin et. al.
    """
    config_fn = 'full_shape.yaml'
    _klim = (1e-3, 1., 500)
    # _default_options = dict(prior_basis='physical', mu=50)
    _default_options = dict(freedom=None, prior_basis='physical_aap', basis='sugiyama',
                            tracer=None, fsat=None, sigv=None,
                            shotnoise=1e4, h_fid=None, sigma8_fid=None,
                            model='FOLPSD', bias_scheme='folps', IR_resummation=True, damping='lor', rbao=104.,
                            A_full=True, remove_DeltaP=False, precision=(8, 10, 10),
                            renormalized=True, interpolation_method='linear')
    _pt_cls = FOLPSv2PowerSpectrumMultipoles

    @classmethod
    def _get_multitracer(cls, tracers=None, prior_basis='physical_aap'):
        deterministic = ['b1', 'b2', 'bs', 'c1', 'c2', 'X_FoG_b']
        stochastic = ['Pshot', 'Bshot']
        if 'physical' in prior_basis:
            deterministic = [name + 'p' for name in deterministic]
            stochastic = [name + 'p' for name in stochastic]
        return MultitracerBiasParameters(tracers=tracers, deterministic=deterministic, stochastic=stochastic, ntracers=1)

    @classmethod
    def _params(cls, params, tracers=None):
        return cls._get_multitracer(tracers=tracers)._params(params)

    def initialize(self, k=None, ells=((0, 0, 0), (2, 0, 2)), tracers=None, basis='sugiyama', pt=None, template=None, **kwargs):
        self._set_options(k=k, ells=ells, tracers=tracers, basis=basis, **kwargs)
        self.prior_basis = self._rename_prior_basis(self.options['prior_basis'])
        self._set_pt(pt=pt, template=template, **kwargs)
        self._set_from_pt()
        self._set_params()
        self.decode_params = self._get_multitracer(tracers=tracers, prior_basis=self.prior_basis)

    @staticmethod
    def _rename_prior_basis(prior_basis: str) -> str:
        return FOLPSv2TracerPowerSpectrumMultipoles._rename_prior_basis(prior_basis)

    @classmethod
    def _params(cls, params, tracers=None, prior_basis='physical_aap'):
        prior_basis = cls._rename_prior_basis(prior_basis)
        for param in params.select(basename=['b1']):
            param.update(prior=dict(limits=[0., 10.]))
        for param in params.select(basename=['b2']):
            param.update(prior=dict(limits=[-50., 50.]))
        for param in params.select(basename=['bs', 'c1', 'c2', 'Pshot', 'Bshot', 'X_FoG_b']):
            param.update(prior=None)
        for param in params.select(basename=['bs', 'c1', 'c2', 'Pshot', 'Bshot', 'X_FoG_b']):
            param.update(fixed=False)
        # for param in params.select(basename=fix):
        #     param.update(value=0., fixed=True)
        if 'physical' in prior_basis:
            for param in list(params):
                basename = param.basename
                param.update(basename=basename + 'p')
                #params.set({'basename': basename, 'namespace': param.namespace, 'derived': True})
            for param in params.select(basename='b1p'):
                param.update(prior=dict(dist='uniform', limits=[0., 3.]), ref=dict(dist='norm', loc=1., scale=0.1))
            for param in params.select(basename=['b2p', 'bsp']):
                param.update(prior=dict(dist='norm', loc=0., scale=5.), ref=dict(dist='norm', loc=0., scale=1.))
            # Decide the priors for c1 and c2 (Not worrying about it now)
        params = cls._get_multitracer(tracers=tracers, prior_basis=prior_basis)._params(params)
        return params

    def _set_params(self):
        self.is_physical_prior = 'physical' in self.prior_basis

        if self.is_physical_prior:
            settings = get_physical_stochastic_settings(tracer=self.options['tracer'])
            for name, value in settings.items():
                if self.options[name] is None:
                    self.options[name] = value

            if self.mpicomm.rank == 0:
                self.log_debug('Using fsat, sigv = {:.3f}, {:.3f}.'.format(self.options['fsat'], self.options['sigv']))

        # super()._set_params(pt_params=[])
        # fix unused multipole-related params
        self.nbar = 1e-4
        self.fsat = self.snd = 1.
        if self.is_physical_prior:
            self.fsat = self.options['fsat']
            # FIXME: theory modules should take density as input
            self.snd = np.mean(self.options['shotnoise']) * self.nbar  # normalized by 1e-4

    def calculate(self, **params):
        self._set_from_pt()
        params = self.decode_params(params)
        # params = {**self.required_bias_params, **params}
        # import folps as folpsv2
        # folpsv2.MatrixCalculator(A_full=getattr(self.pt, "A_full", True), use_TNS_model=getattr(self.pt, "remove_DeltaP", False))
        #Initialise global variables
        qpar, qper = self.pt.qpar, self.pt.qper
        # Case A: STANDARD_FOLPS -> forward directly (Eulerian nuisances)
        if self.prior_basis == 'standard_folps':
            pars = [params[name] for name in ['b1', 'b2', 'bs', 'c1', 'c2', 'Pshot', 'Bshot', 'X_FoG_b']]
            self.power = self.pt.combine_bias_terms_bispectrum_poles(pars, self.k, precision=self.options['precision'], damping=self.options['damping'],
                                                                basis=self.options['basis'], model=self.options['model'],
                                                                bias_scheme=self.options['bias_scheme'], renormalized=self.options['renormalized'],
                                                                interpolation_method=self.options['interpolation_method'],
                                                                ells=self.ells, qpar=qpar, qper=qper)
            return
        # From here on: PHYSICAL modes
        sigma8 = self.pt.sigma8
        f = self.pt.fsigma8 / sigma8
        sigma8_fid = self.options.get('sigma8_fid', None)
        # Amplitude rescaling convention (Class-PT style)
        A = (sigma8 / sigma8_fid)**2 if sigma8_fid is not None else 1.0
        A_AP = 1 / (qper**2 * qpar)
        sqrt_A_AP = A_AP**0.5
        self.A_AP = A_AP
        # Counterterms mapping
        if self.prior_basis == 'physical_velocileptors':
            # This one needs to be fixed
            # --- Lagrangian -> Eulerian ---
            b1L = params['b1p'] / sigma8 - 1.0
            b2L = params['b2p'] / sigma8**2
            bsL = params['bsp'] / sigma8**2
            b1E  = 1.0 + b1L
            b2E  = b2L + 8.0 / 21.0 * b1L
            # defaults (non-APscaling)
            #b3L  = params['b3p']
            bsE = -4.0 / 7.0 * b1L + bsL
            kNL = 0.3
            c1, c2 = (params[name] / (kNL**2) for name in ['c1p', 'c2p'])
            Pshot = params['Pshotp'] * self.snd
            Bshot = params['Bshotp'] * self.snd
            pars = [b1E, b2E, bsE, c1, c2, Pshot, Bshot, params['X_FoG_bp']]
        elif self.prior_basis == 'physical_aap':
            # --- Lagrangian -> Eulerian ---
            b1L = params['b1p'] / sigma8 / sqrt_A_AP - 1.0
            b2L = params['b2p'] / sigma8**2 / sqrt_A_AP
            # b2L  = params['b2p']
            bK2 = params['bsp'] / sigma8**2 / sqrt_A_AP
            b1E = 1.0 + b1L
            b2E = b2L
            bsE = 2 * bK2
            kNL = 0.3
            c1, c2 = (params[name] / (kNL**2) / (A_AP * sigma8**2) for name in ['c1p', 'c2p'])
            Ashot = A_AP
            Pshot = (params['Pshotp'] / Ashot) * self.snd
            Bshot = (params['Bshotp'] / Ashot) * self.snd
            # c1 = params['c1p']
            # c2 = params['c2p']
            # Pshot = params['Pshotp']
            # Bshot = params['Bshotp']
            pars = [b1E, b2E, bsE, c1, c2, Pshot, Bshot, params['X_FoG_bp']]
        elif self.prior_basis == 'tcm_chudaykin':
            # APscaling: include A_AP and decode the table-style priors
            self.options['bias_scheme'] = 'classpt' #As in chudaykin et. al.
            b1L, b2L, bsL, b3 = params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p']/A
            pars = [1. + b1L, b2L, bsL, b3]   #Class-pt bias free b3
            c0, c2, c4 = (params[name] / (A * A_AP) for name in ['alpha0p', 'alpha2p', 'alpha4p'])
            pars += [-2. / 105 * (105 * c0 - 35 * c2 * f + 9 * c4 * f**2), -2. / 7 * f * (7 * c2 - 6 * f * c4), -2 * f**2 * c4, 0]
            sigv = self.options['sigv']
            pars += [params['sn{:d}p'.format(i)] * self.snd * (self.fsat if i > 0 else 1.) * sigv**i for i in [0, 2]]
            pars += [params['X_FoG_bp']]
            # use coevolution b3E = b3E + 32.0/315.0 * b1L
        # pars = [params[name] for name in self.required_bias_params]
        self.power = self.pt.combine_bias_terms_bispectrum_poles(pars, self.k, precision=self.options['precision'], damping=self.options['damping'], basis=self.options['basis'],
                                                                 model=self.options['model'] ,bias_scheme=self.options['bias_scheme'], renormalized=self.options['renormalized'],
                                                                 interpolation_method=self.options['interpolation_method'], ells=self.ells, qpar=qpar, qper=qper)

    def get(self):
        # Returned value when calling the calculator
        return self.power

    def __getstate__(self):
        # Required only for quick emulation (Taylor expansion)
        state = {}
        for name in ['k', 'z', 'ells', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
