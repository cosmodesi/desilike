import os

import numpy as np

from desilike.io import BaseConfig
from desilike.bindings.base import BaseLikelihoodGenerator, get_likelihood_params, Parameter, ParameterCollection


from desilike.cosmo import Cosmology, BaseExternalEngine, BaseSection, PowerSpectrumInterpolator2D, flatarray, _make_list, get_default, merge


"""Mock up cosmoprimo with cobaya's camb / classy provider."""


class CobayaEngine(BaseExternalEngine):

    @classmethod
    def get_requires(cls, requires):

        convert = {'delta_m': 'delta_tot', 'delta_cb': 'delta_nonu',
                   'theta_m': 'delta_tot', 'theta_cb': 'delta_nonu'}

        toret = {}
        require_f = []
        for section, names in super(CobayaEngine, cls).get_requires(requires).items():
            for name, attrs in names.items():
                if section == 'background':
                    tmp = {'z': attrs['z']}
                    if name == 'efunc':
                        tmp['z'] = np.insert(tmp['z'], 0, 0.)
                        toret['Hubble'] = tmp
                    if name in ['comoving_radial_distance', 'angular_diameter_distance']:
                        toret[name] = tmp
                    if name == 'comoving_angular_distance':
                        toret['angular_diameter_distance'] = tmp
                elif section == 'thermodynamics':  # rs_drag
                    if name == 'rs_drag':
                        toret['rdrag'] = None
                elif section == 'fourier':
                    tmp = {}
                    if name == 'sigma8_z':
                        name = 'pk_interpolator'
                        for aname in ['z', 'k']: attrs[aname] = attrs.get(aname, get_default(aname))
                        attrs['non_linear'] = False
                    if name == 'pk_interpolator':
                        tmp['nonlinear'] = attrs['non_linear']
                        tmp['z'] = attrs['z']
                        tmp['k_max'] = attrs['k'].max()  # 1/Mpc unit
                        tmp['vars_pairs'] = []
                        for pair in attrs['of']:
                            if any('theta' in p for p in pair):
                                require_f.append(tmp['z'])
                            tmp['vars_pairs'].append(tuple(convert[p] for p in pair))
                        Pk_grid = toret.get('Pk_grid', {})
                        if Pk_grid:
                            tmp['nonlinear'] |= Pk_grid['nonlinear']
                            tmp['z'] = merge([tmp['z'], Pk_grid['z']])
                            tmp['k_max'] = max(tmp['k_max'], Pk_grid['k_max'])
                            tmp['vars_pairs'] = list(set(Pk_grid['vars_pairs'] + tmp['vars_pairs']))
                        toret['Pk_grid'] = tmp
        if require_f:
            require_f = merge(require_f)
            # oversampling, to interpolate at z of Pk_grid with classy wrapper
            require_f = merge([require_f, np.linspace(0., require_f[-1] + 1., 20)])
            toret['fsigma8'] = {'z': require_f}
            toret['sigma8_z'] = {'z': require_f}
        if toret or requires.get('params', {}):  # to get /h units
            if 'Hubble' in toret:
                toret['Hubble']['z'] = np.unique(np.insert(tmp['z'], 0, 0.))
            else:
                toret['Hubble'] = {'z': np.array([0.])}
        return toret


class Section(BaseSection):

    def __init__(self, engine):
        self.provider = engine.provider
        self.h = np.squeeze(self.provider.get_Hubble(0.) / 100.)


class Background(Section):

    @flatarray(dtype=np.float64)
    def efunc(self, z):
        return self.provider.get_Hubble(z) / (100. * self.h)

    @flatarray(dtype=np.float64)
    def comoving_radial_distance(self, z):
        return self.provider.get_comoving_radial_distance(z) * self.h

    @flatarray(dtype=np.float64)
    def angular_diameter_distance(self, z):
        return self.provider.get_angular_diameter_distance(z) * self.h

    @flatarray(dtype=np.float64)
    def comoving_angular_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)


class Thermodynamics(Section):

    @property
    def rs_drag(self):
        return self.provider.get_param('rdrag') * self.h


class Fourier(Section):

    def pk_interpolator(self, non_linear=False, of='delta_m', **kwargs):
        convert = {'delta_m': 'delta_tot', 'delta_cb': 'delta_nonu',
                   'theta_m': 'delta_tot', 'theta_cb': 'delta_nonu'}
        of = _make_list(of, length=2)
        var_pair = [convert[of_] for of_ in of]
        k, z, pk = self.provider.get_Pk_grid(var_pair=var_pair, nonlinear=non_linear)
        k = k / self.h
        pk = pk.T * self.h**3
        ntheta = sum('theta' in of_ for of_ in of)
        if ntheta:
            from scipy import interpolate
            collector = {}
            for name in ['sigma8_z', 'fsigma8']:
                provider = self.provider.requirement_providers[name]  # hacky way to get to classy
                collector[name] = interpolate.interp1d(provider.collectors[name].z_pool, provider.current_state[name], kind='cubic', axis=-1, copy=True, bounds_error=False, assume_sorted=False)(z)
            f = collector['fsigma8'] / collector['sigma8_z']
            # Below does not work for classy wrapper, because z does not match requested z...
            # f = self.provider.get_fsigma8(z=z) / self.provider.get_sigma8_z(z=z)
            pk = pk * f**ntheta
        return PowerSpectrumInterpolator2D(k, z, pk, **kwargs)

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)


def cosmoprimo_to_camb_params(params):
    convert = {'H0': 'H0', 'omega_b': 'ombh2', 'omega_cdm': 'omch2', 'A_s': 'As', 'n_s': 'ns', 'N_eff': 'nnu', 'm_ncdm': 'mnu', 'Omega_k': 'omk'}
    toret = ParameterCollection()
    params = params.copy()
    name = 'h'
    if name in params and params[name].varied:
        toret[name] = params.pop(name).copy()#.clone(drop=True)
        toret.set(Parameter('H0', derived='100 * {h}'))
    for name in ['Omega_b', 'Omega_cdm']:
        if name in params and params[name].varied:
            cname = convert[name.lower()]
            toret.set(params.pop(name).clone(basename=name))
            toret.set(Parameter(cname, derived='{{{}}} * ({{H0}} / 100)**2'.format(name)))
    for param in params:
        if param.varied:
            try:
                name = convert[param.name]
            except KeyError as exc:
                raise ValueError('There is no translation for parameter {} to camb; we can only translate {}'.format(param.name, list(convert.keys()))) from exc
            param = param.clone(basename=name)
            toret.set(param)
    return toret


def cosmoprimo_to_classy_params(params):
    convert = {name: name for name in ['H0', 'h', 'A_s', 'ln10^{10}A_s', 'sigma8', 'n_s', 'omega_b', 'Omega_b', 'omega_cdm', 'Omega_cdm', 'omega_m', 'Omega_m', 'Omega_ncdm', 'omega_ncdm', 'm_ncdm', 'omega_k', 'Omega_k']}
    toret = ParameterCollection()
    for param in params:
        if param.varied:
            try:
                name = convert[param.name]
            except KeyError as exc:
                raise ValueError('There is no translation for parameter {} to classy; we can only translate {}'.format(param.name, list(convert.keys()))) from exc
            param = param.clone(basename=name)
            toret.set(param)
    return toret


def camb_or_classy_to_cosmoprimo(fiducial, provider, **params):
    if fiducial: cosmo = Cosmology.from_state(fiducial)
    else: cosmo = Cosmology()
    params = {**provider.params, **params}
    convert = {'H0': 'H0', 'As': 'A_s', 'ns': 'n_s', 'ombh2': 'omega_b', 'omch2': 'omega_cdm', 'nnu': 'N_eff', 'mnu': 'm_ncdm', 'omk': 'Omega_k'}
    convert.update({name: name for name in ['H0', 'h', 'A_s', 'ln10^{10}A_s', 'sigma8', 'n_s', 'omega_b', 'Omega_b', 'omega_cdm', 'Omega_cdm', 'omega_m', 'Omega_m', 'Omega_ncdm', 'omega_ncdm', 'm_ncdm', 'omega_k', 'Omega_k']})
    state = {convert[param]: value for param, value in params.items() if param in convert}  # NEED MORE CHECKS!
    if not any(name in state for name in ['H0', 'h']):
        state['H0'] = np.squeeze(provider.get_Hubble(0.))
    cosmo = cosmo.clone(**state, engine=CobayaEngine)
    cosmo._engine.provider = provider
    return cosmo


def CobayaLikelihoodFactory(cls, kw_like, module=None):

    def initialize(self):
        """Prepare any computation, importing any necessary code, files, etc."""

        self.like = cls(**kw_like)
        self._cosmo_params, self._nuisance_params = get_likelihood_params(self.like)
        """
        import inspect
        kwargs = {name: getattr(self, name) for name in inspect.getargspec(cls).args}
        self.like = cls(**kwargs)
        """

    def get_requirements(self):
        """Return dictionary specifying quantities calculated by a theory code are needed."""
        requires = self.like.runtime_info.pipeline.get_cosmo_requires()
        self._fiducial = requires.get('fiducial', {})
        self._requires = CobayaEngine.get_requires(requires)
        return self._requires

    def logp(self, _derived=None, **params_values):
        """
        Take a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        if self._requires:
            cosmo = camb_or_classy_to_cosmoprimo(self._fiducial, self.provider, **params_values)
            self.like.runtime_info.pipeline.set_cosmo_requires(cosmo)
        return self.like(**{name: value for name, value in params_values.items() if name in self._nuisance_params})

    d = {'initialize': initialize, 'get_requirements': get_requirements, 'logp': logp}
    if module is not None:
        d['__module__'] = module
    from cobaya.likelihood import Likelihood
    return type(Likelihood)(cls.__name__, (Likelihood,), d)


class CobayaLikelihoodGenerator(BaseLikelihoodGenerator):

    """Extend :class:`BaseLikelihoodGenerator` with support for cobaya, writing parameters to a .yaml file."""

    def __init__(self, *args, **kwargs):
        super(CobayaLikelihoodGenerator, self).__init__(CobayaLikelihoodFactory, *args, **kwargs)

    def get_code(self, *args, **kwargs):
        cls, fn, code = super(CobayaLikelihoodGenerator, self).get_code(*args, **kwargs)
        dirname = os.path.dirname(fn)
        params = {}

        def decode_prior(prior):
            di = {}
            di['dist'] = prior.dist
            if prior.is_limited():
                di['min'], di['max'] = prior.limits
            for name in ['loc', 'scale']:
                if hasattr(prior, name):
                    di[name] = getattr(prior, name)
            return di

        cosmo_params, nuisance_params = get_likelihood_params(cls(**self.kw_like))
        # if self.engine == 'camb':
        #     cosmo_params = cosmoprimo_to_camb_params(cosmo_params)
        # elif self.engine == 'classy':
        #     cosmo_params = cosmoprimo_to_classy_params(cosmo_params)
        params = {}
        for param in nuisance_params:
            if param.solved or param.derived and not param.depends: continue
            if param.fixed:
                params[param.name] = param.value
            else:
                di = {'latex': param.latex()}
                if param.depends:
                    names = param.depends.values()
                    for name in names:
                        derived = param.derived.replace('{{{}}}'.format(name), name)
                    di['value'] = 'lambda {}: '.format(', '.join(names)) + derived
                else:
                    di['prior'] = decode_prior(param.prior)
                    if param.ref.is_proper():
                        di['ref'] = decode_prior(param.ref)
                    if param.proposal is not None:
                        di['proposal'] = param.proposal
                #if param.drop: di['drop'] = True
                params[param.name] = di

        BaseConfig(dict(stop_at_error=True, params=params)).write(os.path.join(dirname, cls.__name__ + '.yaml'))

        import_line = 'from .{} import *'.format(os.path.splitext(os.path.basename(fn))[0])
        with open(os.path.join(dirname, '__init__.py'), 'a+') as file:
            file.seek(0, 0)
            lines = file.read()
            if import_line not in lines:
                if lines and lines[-1] != '\n': file.write('\n')
                file.write(import_line + '\n')

        return cls, fn, code
