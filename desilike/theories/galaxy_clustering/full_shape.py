import numpy as np
from scipy import interpolate

from desilike.utils import jnp
from .base import TrapzTheoryPowerSpectrumMultipoles
from .base import BaseTheoryPowerSpectrumMultipoles, BaseTheoryCorrelationFunctionMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles
from .power_template import FullPowerSpectrumTemplate  # to add calculator in the registry


class BasePTPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    _default_options = dict()

    def initialize(self, *args, template=None, **kwargs):
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        super(BasePTPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.kin = np.geomspace(min(1e-3, self.k[0] / 2), max(1., self.k[0] * 2), 600)  # margin for AP effect
        if template is None:
            template = FullPowerSpectrumTemplate(k=self.kin)
        self.template = template
        self.template.update(k=self.kin)


class BasePTCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    _default_options = dict()

    def initialize(self, *args, template=None, **kwargs):
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        super(BasePTCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        self.kin = np.geomspace(min(1e-3, 1 / self.s[-1] / 2), max(2., 1 / self.s[0] * 2), 1000)  # margin for AP effect
        if template is None:
            template = FullPowerSpectrumTemplate(k=self.kin)
        self.template = template
        self.template.update(k=self.kin)


class BaseTracerPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    config_fn = 'full_shape.yaml'
    _default_options = dict()

    def initialize(self, *args, template=None, **kwargs):
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        self.pt = globals()[self.__class__.__name__.replace('Tracer', '')](template=template)
        for name, value in self.pt._default_options.items():
            if name in kwargs:
                self.pt.update({name: kwargs.pop(name)})
            elif name in self.options:
                self.pt.update({name: self.options[name]})
        self.required_bias_params, self.optional_bias_params = {}, {}
        super(BaseTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.pt.update(k=self.k, ells=self.ells)
        self.set_params()

    def set_params(self):
        self.params = self.params.select(basename=list(self.required_bias_params.keys()) + list(self.optional_bias_params.keys()))

    def get(self):
        return self.power


class BaseTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    config_fn = 'full_shape.yaml'
    _default_options = dict()

    def __init__(self, *args, template=None, **kwargs):
        self.options = self._default_options.copy()
        for name, value in self._default_options.items():
            self.options[name] = kwargs.pop(name, value)
        self.pt = globals()[self.__class__.__name__.replace('Tracer', '')](template=template)
        for name, value in self.pt._default_options.items():
            if name in kwargs:
                self.pt.update({name: kwargs.pop(name)})
            elif name in self.options:
                self.pt.update({name: self.options[name]})
        self.required_bias_params, self.optional_bias_params = {}, {}
        super(BaseTracerCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        self.pt.update(s=self.s, ells=self.ells)
        self.set_params()

    def set_params(self):
        self.params = self.params.select(basename=list(self.required_bias_params.keys()) + list(self.optional_bias_params.keys()))

    def get(self):
        return self.corr


class BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):

    config_fn = 'full_shape.yaml'

    def initialize(self, *args, template=None, **kwargs):
        power = globals()[self.__class__.__name__.replace('CorrelationFunction', 'PowerSpectrum')]()
        power.update(template=template)
        super(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles, self).initialize(*args, power=power, **kwargs)

    def get(self):
        return self.corr


class KaiserTracerPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles):

    config_fn = 'full_shape.yaml'

    def initialize(self, *args, mu=200, **kwargs):
        super(KaiserTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=self.mu, ells=self.ells)

    def calculate(self, b1=1., sn0=0.):
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f
        pkmu = (b1 + f * muap**2)**2 * np.interp(np.log10(kap), np.log10(self.kin), self.template.pk_dd) + sn0
        self.power = self.to_poles(pkmu)

    def get(self):
        return self.power


class KaiserTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):

    pass


class BaseVelocileptorsPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    _default_options = dict()

    def initialize(self, *args, **kwargs):
        super(BaseVelocileptorsPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.options['threads'] = self.options.pop('nthreads', 1)
        if 'kmin' in self._default_options:
            self._default_options['kmin'] = self.k[0] * 0.8
        if 'kmax' in self._default_options:
            self._default_options['kmax'] = self.k[-1] * 1.2
        if 'nk' in self._default_options:
            self._default_options['nk'] = int(len(self.k) * 1.4 + 0.5)

    def combine_bias_terms_poles(self, pars, **opts):
        tmp = np.array(self.pt.compute_redshift_space_power_multipoles(pars, self.template.f, apar=self.template.qpar, aperp=self.template.qper, **self.options, **opts)[1:])
        return interpolate.interp1d(self.pt.kv, tmp, kind='cubic', axis=-1, copy=False, bounds_error=True, assume_sorted=True)(self.k)

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/sfschen/velocileptors')


class BaseVelocileptorsTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):

    _default_options = dict()

    def calculate(self, **params):
        pars = [params.get(name, value) for name, value in self.required_bias_params.items()]
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        self.power = self.pt.combine_bias_terms_poles(pars, **opts, **self.options)


class BaseVelocileptorsCorrelationFunctionMultipoles(BasePTCorrelationFunctionMultipoles):

    _default_options = dict()

    def initialize(self, *args, **kwargs):
        super(BaseVelocileptorsCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        self.options['threads'] = self.options.pop('nthreads')

    def combine_bias_terms_poles(self, pars, **opts):
        return np.array([self.pt.compute_xi_ell(ss, self.template.f, *pars, apar=self.template.qpar, aperp=self.template.qper, **self.options, **opts) for ss in self.s]).T


class BaseVelocileptorsTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    _default_options = dict()

    def calculate(self, **params):
        pars = [params.get(name, value) for name, value in self.required_bias_params.items()]
        opts = {name: params.get(name, default) for name, default in self.optional_bias_params.items()}
        self.corr = self.pt.combine_bias_terms_poles(pars, **opts, **self.options)


class LPTVelocileptorsPowerSpectrumMultipoles(BaseVelocileptorsPowerSpectrumMultipoles):

    _default_options = dict(kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=4000, nthreads=1, jn=5)
    # Slow, ~ 4 sec per iteration

    def calculate(self):
        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        self.lpt = LPT_RSD(self.kin, self.template.pk_dd, **self.options)
        # print(self.template.f, self.k.shape, self.template.qpar, self.template.qper, self.kin.shape, self.template.pk_dd.shape)
        self.lpt.make_pltable(self.template.f, kv=self.k, apar=self.template.qpar, aperp=self.template.qper, ngauss=3)
        lpttable = {0: self.lpt.p0ktable, 2: self.lpt.p2ktable, 4: self.lpt.p4ktable}
        self.lpttable = np.array([lpttable[ell] for ell in self.ells])

    def combine_bias_terms_poles(self, pars):
        # bias = [b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4]
        # pkells = self.lpt.combine_bias_terms_pkell(bias)[1:]
        # return np.array([pkells[[0, 2, 4].index(ell)] for ell in self.ells])
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4 = pars
        bias_monomials = jnp.array([1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4])
        return jnp.sum(self.lpttable * bias_monomials, axis=-1)

    def __getstate__(self):
        state = {}
        for name in ['k', 'zeff', 'ells', 'lpttable']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/sfschen/velocileptors')


class LPTVelocileptorsTracerPowerSpectrumMultipoles(BaseVelocileptorsTracerPowerSpectrumMultipoles):

    def initialize(self, *args, **kwargs):
        super(LPTVelocileptorsTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)

    def set_params(self):
        self.required_bias_params = dict(b1=0.69, b2=-1.17, bs=-0.71, b3=0., alpha0=0., alpha2=0., alpha4=0., alpha6=0., sn0=0., sn2=0., sn4=0.)
        super(LPTVelocileptorsTracerPowerSpectrumMultipoles, self).set_params()

    def calculate(self, **params):
        return super(LPTVelocileptorsTracerPowerSpectrumMultipoles, self).calculate(**params)


class LPTVelocileptorsTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionFromPowerSpectrumMultipoles):

    pass


class PyBirdPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles):

    _default_options = dict(optiresum=True, nd=None, with_nnlo_higher_derivative=False, with_nnlo_counterterm=False, with_stoch=False, with_resum='opti')
    _bird_attrs = ['co', 'f', 'eft_basis', 'with_stoch', 'with_nnlo_counterterm', 'with_tidal_alignments',
                   'P11l', 'Ploopl', 'Pctl', 'Pstl', 'Pnnlol', 'C11l', 'Cloopl', 'Cctl', 'Cstl', 'Cnnlol']

    def initialize(self, *args, shotnoise=1e4, **kwargs):
        import pybird_dev as pybird
        super(PyBirdPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        if self.options['nd'] is None: self.options['nd'] = 1. / shotnoise
        self.co = pybird.Common(halohalo=True, with_time=True, exact_time=False, quintessence=False, with_tidal_alignments=False, nonequaltime=False,
                                Nl=len(self.ells), kmin=self.k[0] * 0.8, kmax=self.k[-1] * 1.2, optiresum=self.options['with_resum'] == 'opti',
                                nd=self.options['nd'], with_cf=False)
        self.nonlinear = pybird.NonLinear(load=False, save=False, co=self.co)
        self.resum = pybird.Resum(co=self.co)
        self.nnlo_higher_derivative = self.nnlo_counterterm = None
        if self.options['with_nnlo_higher_derivative']:
            self.nnlo_higher_derivative = pybird.NNLO_higher_derivative(self.co.k, with_cf=False, co=self.co)
        if self.options['with_nnlo_counterterm']:
            self.nnlo_counterterm = pybird.NNLO_counterterm(co=self.co)
        self.projection = pybird.Projection(self.k, Om_AP=0.3, z_AP=1., co=self.co)  # placeholders for Om_AP and z_AP, as we will provide q's

    def calculate(self):
        import pybird_dev as pybird
        cosmo = {'k11': self.kin, 'P11': self.template.pk_dd, 'f': self.template.f, 'DA': 1., 'H': 1.}
        self.bird = pybird.Bird(cosmo, with_bias=False, eft_basis='eftoflss', with_stoch=self.options['with_stoch'], with_nnlo_counterterm=self.nnlo_counterterm is not None, co=self.co)

        if self.nnlo_counterterm is not None:  # we use smooth power spectrum since we don't want spurious BAO signals
            self.nnlo_counterterm.Ps(self.bird, np.log(self.template.pknow_dd))

        self.nonlinear.PsCf(self.bird)
        self.bird.setPsCfl()

        if self.options['with_resum']:
            self.resum.Ps(self.bird)

        self.projection.AP(self.bird, q=(self.template.qper, self.template.qpar))
        self.projection.xdata(self.bird)

    def combine_bias_terms_poles(self, **params):
        from pybird_dev import bird
        bird.np = jnp
        self.bird.setreducePslb(params, what='full')
        bird.np = np
        return self.bird.fullPs

    def __getstate__(self):
        state = {}
        for name in self._bird_attrs:
            if hasattr(self.bird, name):
                state[name] = getattr(self.bird, name)
        return state

    def __setstate__(self, state):
        import pybird_dev as pybird
        self.bird = pybird.Bird.__new__(pybird.Bird)
        self.bird.with_bias = False
        self.bird.__dict__.update(state)

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/adematti/pybird@dev')


class PyBirdTracerPowerSpectrumMultipoles(BaseTracerPowerSpectrumMultipoles):

    _default_options = dict(with_nnlo_higher_derivative=False, with_nnlo_counterterm=False, with_stoch=False, eft_basis='eftoflss')

    def set_params(self):
        self.required_bias_params = ['b1', 'b3', 'cct']
        allowed_eft_basis = ['eftoflss', 'westcoast']
        if self.options['eft_basis'] not in allowed_eft_basis:
            raise ValueError('eft_basis must be one of {}'.format(allowed_eft_basis))
        if self.options['eft_basis'] == 'westcoast':
            self.required_bias_params += ['b2p4', 'b2m4']
        else:
            self.required_bias_params += ['b2', 'b4']
        if len(self.ells) >= 2: self.required_bias_params += ['cr1', 'cr2']
        if self.options['with_stoch']:
            self.required_bias_params += ['ce0', 'ce1', 'ce2']
        if self.options['with_nnlo_counterterm']:
            self.required_bias_params += ['cr4', 'cr6']
        default_values = {'b1': 1.69, 'b3': -0.479}
        self.required_bias_params = {name: default_values.get(name, 0.) for name in self.required_bias_params}
        self.params = self.params.select(basename=list(self.required_bias_params.keys()) + list(self.optional_bias_params.keys()))

    def transform_params(self, **params):
        if self.options['eft_basis'] == 'westcoast':
            params['b2'] = (params['b2p4'] + params['b2m4']) / 2.**0.5
            params['b4'] = (params.pop('b2p4') - params.pop('b2m4')) / 2.**0.5
        return params

    def calculate(self, **params):
        self.power = self.pt.combine_bias_terms_poles(**self.transform_params(**params))


class PyBirdCorrelationFunctionMultipoles(BasePTCorrelationFunctionMultipoles):

    _default_options = dict(optiresum=True, nd=None, with_nnlo_higher_derivative=False, with_nnlo_counterterm=False, with_stoch=False, with_resum='opti')
    _bird_attrs = ['co', 'f', 'eft_basis', 'with_stoch', 'with_nnlo_counterterm', 'with_tidal_alignments',
                   'P11l', 'Ploopl', 'Pctl', 'Pstl', 'Pnnlol', 'C11l', 'Cloopl', 'Cctl', 'Cstl', 'Cnnlol']

    def initialize(self, *args, shotnoise=1e4, **kwargs):
        import pybird_dev as pybird
        super(PyBirdCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        if self.options['nd'] is None: self.options['nd'] = 1. / shotnoise
        self.co = pybird.Common(halohalo=True, with_time=True, exact_time=False, quintessence=False, with_tidal_alignments=False, nonequaltime=False,
                                Nl=len(self.ells), kmin=1e-3, kmax=0.25, optiresum=self.options['with_resum'] == 'opti',
                                nd=self.options['nd'], with_cf=True)
        self.nonlinear = pybird.NonLinear(load=False, save=False, co=self.co)
        self.resum = pybird.Resum(co=self.co)
        self.nnlo_higher_derivative = self.nnlo_counterterm = None
        if self.options['with_nnlo_higher_derivative']:
            self.nnlo_higher_derivative = pybird.NNLO_higher_derivative(self.co.k, with_cf=True, co=self.co)
        if self.options['with_nnlo_counterterm']:
            self.nnlo_counterterm = pybird.NNLO_counterterm(co=self.co)
        self.projection = pybird.Projection(self.s, Om_AP=0.3, z_AP=1., co=self.co)  # placeholders for Om_AP and z_AP, as we will provide q's

    def calculate(self):
        import pybird_dev as pybird
        cosmo = {'k11': self.kin, 'P11': self.template.pk_dd, 'f': self.template.f, 'DA': 1., 'H': 1.}
        self.bird = pybird.Bird(cosmo, with_bias=False, eft_basis='eftoflss', with_stoch=self.options['with_stoch'], with_nnlo_counterterm=self.nnlo_counterterm is not None, co=self.co)

        if self.nnlo_counterterm is not None:  # we use smooth power spectrum since we don't want spurious BAO signals
            self.nnlo_counterterm.Cf(self.bird, np.log(self.template.pknow_dd))

        self.nonlinear.PsCf(self.bird)
        self.bird.setPsCfl()

        if self.options['with_resum']:
            self.resum.PsCf(self.bird)

        self.projection.AP(self.bird, q=(self.template.qper, self.template.qpar))
        self.projection.xdata(self.bird)

    def combine_bias_terms_poles(self, **params):
        from pybird_dev import bird
        bird.np = jnp
        self.bird.setreduceCflb(params)
        bird.np = np
        return self.bird.fullCf

    def __getstate__(self):
        state = {}
        for name in self._bird_attrs:
            if hasattr(self.bird, name):
                state[name] = getattr(self.bird, name)
        return state

    def __setstate__(self, state):
        import pybird_dev as pybird
        self.bird = pybird.Bird.__new__(pybird.Bird)
        self.bird.with_bias = False
        self.bird.__dict__.update(state)

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/adematti/pybird@dev')


class PyBirdTracerCorrelationFunctionMultipoles(BaseTracerCorrelationFunctionMultipoles):

    _default_options = dict(with_nnlo_higher_derivative=False, with_nnlo_counterterm=False, with_stoch=False, eft_basis='eftoflss')

    def set_params(self):
        return PyBirdTracerPowerSpectrumMultipoles.set_params(self)

    def transform_params(self, **params):
        return PyBirdTracerPowerSpectrumMultipoles.transform_params(self, **params)

    def calculate(self, **params):
        self.corr = self.pt.combine_bias_terms_poles(**self.transform_params(**params))
