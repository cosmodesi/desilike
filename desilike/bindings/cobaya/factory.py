import os

import numpy as np

from desilike.io import BaseConfig
from desilike.bindings.base import BaseLikelihoodGenerator, get_likelihood_params, Parameter, ParameterCollection


from desilike.cosmo import Cosmology, BaseExternalEngine, BaseSection, PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D, flatarray, addproperty, _make_list, get_default, merge


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
                    if name in ['comoving_angular_distance', 'luminosity_distance']:
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
            require_f = merge([require_f, [0., require_f[-1] + 1.]])
            toret['fsigma8'] = {'z': require_f}
            toret['sigma8_z'] = {'z': require_f}
        if toret or requires.get('params', {}):  # to get /h units
            if 'Hubble' in toret:
                toret['Hubble']['z'] = np.unique(np.insert(toret['Hubble']['z'], 0, 0.))
            else:
                toret['Hubble'] = {'z': np.array([0.])}
        return toret


@addproperty('h')
class Section(BaseSection):

    def __init__(self, engine):
        self._engine = engine
        self._provider = engine.provider
        self._h = np.squeeze(self._provider.get_Hubble(0.) / 100.)


class Background(Section):

    @flatarray(dtype=np.float64)
    def efunc(self, z):
        return self._provider.get_Hubble(z) / (100. * self._h)

    @flatarray(dtype=np.float64)
    def comoving_radial_distance(self, z):
        return self._provider.get_comoving_radial_distance(z) * self._h

    @flatarray(dtype=np.float64)
    def angular_diameter_distance(self, z):
        return self._provider.get_angular_diameter_distance(z) * self._h

    @flatarray(dtype=np.float64)
    def comoving_angular_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)

    @flatarray(dtype=np.float64)
    def luminosity_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)**2


class Thermodynamics(Section):

    @property
    def rs_drag(self):
        return self._provider.get_param('rdrag') * self._h


@addproperty('A_s', 'n_s', 'alpha_s', 'beta_s')
class Primordial(Section):

    def __init__(self, engine):
        """Initialize :class:`Primordial`."""
        super(Primordial, self).__init__(engine)
        self._A_s = self._engine._derived['A_s']
        #self._A_s = self._provider.get_param('A_s')
        self._n_s = self._engine['n_s']
        self._alpha_s = self._engine['alpha_s']
        self._beta_s = self._engine['beta_s']
        self._rsigma8 = self._engine._rescale_sigma8()

    @property
    def ln_1e10_A_s(self):
        r""":math:`\ln(10^{10}A_s)`, unitless."""
        return np.log(1e10 * self.A_s)

    @property
    def k_pivot(self):
        r"""Primordial power spectrum pivot scale, where the primordial power is equal to :math:`A_{s}`, in :math:`h/\mathrm{Mpc}`."""
        return self._engine['k_pivot'] / self._h

    def pk_k(self, k, mode='scalar'):
        r"""
        The primordial spectrum of curvature perturbations at ``k``, generated by inflation, in :math:`(\mathrm{Mpc}/h)^{3}`.
        For scalar perturbations this is e.g. defined as:

        .. math::

            \mathcal{P_R}(k) = A_s \left (\frac{k}{k_\mathrm{pivot}} \right )^{n_s - 1 + 1/2 \alpha_s \ln(k/k_\mathrm{pivot}) + 1/6 \beta_s \ln(k/k_\mathrm{pivot})^2}

        See also: eq. 2 of `this reference <https://arxiv.org/abs/1303.5076>`_.

        Parameters
        ----------
        k : array_like
            Wavenumbers, in :math:`h/\mathrm{Mpc}`.

        mode : string, default='scalar'
            'scalar' mode.

        Returns
        -------
        pk : array
            The primordial power spectrum.
        """
        index = ['scalar'].index(mode)
        lnkkp = np.log(k / self.k_pivot)
        return self._h**3 * self.A_s * (k / self.k_pivot) ** (self.n_s - 1. + 1. / 2. * self.alpha_s * lnkkp + 1. / 6. * self.beta_s * lnkkp**2)

    def pk_interpolator(self, mode='scalar'):
        """
        Return power spectrum interpolator.

        Parameters
        ----------
        mode : string, default='scalar'
            'scalar', 'vector' or 'tensor' mode.

        Returns
        -------
        interp : PowerSpectrumInterpolator1D
        """
        return PowerSpectrumInterpolator1D.from_callable(pk_callable=lambda k: self.pk_k(k, mode=mode))


class Fourier(Section):

    def pk_interpolator(self, non_linear=False, of='delta_m', **kwargs):
        convert = {'delta_m': 'delta_tot', 'delta_cb': 'delta_nonu',
                   'theta_m': 'delta_tot', 'theta_cb': 'delta_nonu'}
        of = _make_list(of, length=2)
        var_pair = [convert[of_] for of_ in of]
        k, z, pk = self._provider.get_Pk_grid(var_pair=var_pair, nonlinear=non_linear)
        k = k / self._h
        pk = pk.T * self._h**3
        ntheta = sum('theta' in of_ for of_ in of)
        if ntheta:
            from scipy import interpolate
            collector = {}
            for name in ['sigma8_z', 'fsigma8']:
                provider = self._provider.requirement_providers[name]  # hacky way to get to classy
                z_pool = provider.collectors[name].z_pool
                if z_pool is None:
                    z_pool = provider.z_pool_for_perturbations
                collector[name] = interpolate.interp1d(z_pool.values, provider.current_state[name], kind=min(3, len(z_pool.values) - 1), axis=-1, copy=True, fill_value='extrapolate', assume_sorted=False)(z)
            f = collector['fsigma8'] / collector['sigma8_z']
            # Below does not work for classy wrapper, because z does not match requested z...
            # f = self._provider.get_fsigma8(z=z) / self._provider.get_sigma8_z(z=z)
            pk = pk * f**ntheta
        return PowerSpectrumInterpolator2D(k, z, pk, **kwargs)

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)

    @property
    def sigma8_m(self):
        return self.sigma8_z(0., of='delta_m')


_convert_cosmoprimo_to_camb_params = {'H0': 'H0', 'theta_mc': 'cosmomc_theta', 'omega_b': 'ombh2', 'omega_cdm': 'omch2', 'A_s': 'As', 'n_s': 'ns', 'N_eff': 'nnu', 'm_ncdm': 'mnu', 'Omega_k': 'omk', 'w0_fld': 'w', 'wa_fld': 'wa', 'tau_reio': 'tau'}
_convert_cosmoprimo_to_classy_params = {name: name for name in ['H0', 'h', 'A_s', 'sigma8', 'n_s', 'omega_b', 'Omega_b', 'omega_cdm', 'Omega_cdm', 'omega_m', 'Omega_m', 'Omega_ncdm', 'omega_ncdm', 'm_ncdm', 'omega_k', 'Omega_k', 'w0_fld', 'wa_fld', 'tau_reio']}
for name in ['logA', 'ln10^{10}A_s', 'ln10^10A_s', 'ln_A_s_1e10']: _convert_cosmoprimo_to_classy_params[name] = 'ln_A_s_1e10'
_convert_camb_or_classy_to_cosmoprimo_params = {}
for name, value in _convert_cosmoprimo_to_camb_params.items():
    _convert_camb_or_classy_to_cosmoprimo_params[value] = name
for name, value in _convert_cosmoprimo_to_classy_params.items():
    _convert_camb_or_classy_to_cosmoprimo_params[value] = name


def cosmoprimo_to_camb_params(params):
    convert = dict(_convert_cosmoprimo_to_camb_params)
    convert.update({'h': ('H0', '100 * {h}'), 'Omega_b': ('ombh2', '{Omega_b} * ({H0} / 100)**2'), 'Omega_cdm': ('omch2', '{Omega_cdm} * ({H0} / 100)**2'), 'logA': ('As', '1e-10 * np.exp({logA})')})
    toret = ParameterCollection()
    for param in params:
        if param.depends: continue
        if param.derived:
            toret.set(param)
        else:
            try:
                name = convert[param.name]
            except KeyError as exc:
                raise ValueError('There is no translation for parameter {} to camb; we can only translate {}'.format(param.name, list(convert.keys()))) from exc
            if isinstance(name, tuple):
                cname, derived = name
                toret.set(param.copy())
                toret.set(Parameter(cname, derived=derived))
            else:
                param = param.clone(basename=name)
                toret.set(param)
    return toret


def cosmoprimo_to_classy_params(params):
    convert = dict(_convert_cosmoprimo_to_classy_params)
    toret = ParameterCollection()
    for param in params:
        if param.depends: continue
        if param.derived:
            toret.set(param)
        else:
            try:
                name = convert[param.name]
            except KeyError as exc:
                raise ValueError('There is no translation for parameter {} to classy; we can only translate {}'.format(param.name, list(convert.keys()))) from exc
            param = param.clone(basename=name)
            toret.set(param)
    return toret


def camb_or_classy_to_cosmoprimo(fiducial, provider, params, ignore_unknown_params=True, return_input_params=False):
    if fiducial: cosmo = Cosmology.from_state(fiducial)
    else: cosmo = Cosmology()
    convert = dict(_convert_camb_or_classy_to_cosmoprimo_params)
    params = {**provider.params, **params}
    state = {convert[param]: value for param, value in params.items() if param in convert}
    input_params = state.copy()
    A_s = None
    for p in provider.requirement_providers.values():
        if p.__class__.__name__ in ['classy', 'camb']:
            for param, value in p.current_state['params'].items():
                if param in convert:
                    state[convert[param]] = value
                elif not ignore_unknown_params:
                    raise ValueError('cannot translate {} parameter {} to cosmoprimo'.format(p, param))
            try:
                A_s = p.classy.get_current_derived_parameters(['A_s'])['A_s']  # classy
            except AttributeError:
                A_s = p.get_param('As')  # camb
    if not any(name in state for name in ['H0', 'h']):
        state['H0'] = np.squeeze(provider.get_Hubble(0.))
    from cosmoprimo.cosmology import find_conflicts
    conf = {}
    for name in list(state):
        conf[name] = name
        for eq in find_conflicts(name, Cosmology._conflict_parameters):  # always ordered in the same way, e.g. ['A_s', ..., 'sigma8']
            if eq in state:
                conf[name] = eq  # set A_s if in state, else sigma8
                break
    state = {name: value for name, value in state.items() if conf[name] == name}  # prune conflicting parameters
    cosmo = cosmo.clone(**state, engine=CobayaEngine)
    cosmo._engine.provider = provider
    cosmo._engine._derived['A_s'] = A_s
    if return_input_params:
        return cosmo, input_params
    return cosmo


def desilike_to_cobaya_params(params, engine=None):

    def decode_prior(prior, param):
        di = {}
        di['dist'] = prior.dist
        if prior.is_limited():
            di['min'], di['max'] = prior.limits
        elif prior.dist == 'uniform':
            raise ValueError('Provide bounded prior distribution for parameter {}'.format(param))
            di['min'], di['max'] = -np.inf, np.inf
        for name in ['loc', 'scale']:
            if hasattr(prior, name):
                di[name] = getattr(prior, name)
        if prior.dist == 'norm' and prior.is_limited():
            di['dist'] = 'truncnorm'
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            di['a'], di['b'] = (di.pop('min') - di['loc']) / di['scale'], (di.pop('max') - di['loc']) / di['scale']
        return di

    if engine == 'camb':
        params = cosmoprimo_to_camb_params(params)
    elif engine == 'classy':
        params = cosmoprimo_to_classy_params(params)
    elif engine is not None:
        raise ValueError('unknown engine {}'.format(engine))
    toret = {}
    for param in params:
        if param.solved: continue
        if param.derived and (not param.depends) and (param.ndim > 0): continue
        if param.fixed and not param.derived:
            toret[param.name] = param.value
        else:
            di = {'latex': param.latex()}
            if param.depends:
                names = param.depends.values()
                derived = param.derived
                for name in names:
                    derived = derived.replace('{{{}}}'.format(name), name)
                di['value'] = 'lambda {}: '.format(', '.join(names)) + derived
            elif param.derived:
                di['derived'] = True
            elif param.varied:
                di['prior'] = decode_prior(param.prior, param.name)
                if param.ref.is_proper():
                    di['ref'] = decode_prior(param.ref, param.name)
                if param.proposal is not None:
                    di['proposal'] = param.proposal
            # if param.drop: di['drop'] = True
            toret[param.name] = di
    return toret


def cobaya_params(like):
    cosmo_params, nuisance_params = get_likelihood_params(like, derived=0)
    return desilike_to_cobaya_params(nuisance_params)


def CobayaLikelihoodFactory(cls, name_like=None, kw_like=None, module=None, kw_cobaya=None, params=None):
    """
    Pass ``params=True`` for dynamic bindings (when no likelihood *.yaml parameter file is written):

    >>> CobayaBAOLikelihood = CobayaLikelihoodFactory(BAOLikelihood, params=True)

    """
    if name_like is None:
        name_like = cls.__name__
    if kw_like is None:
        kw_like = {}
    if params is not None:
        if params is True:
            params = cobaya_params(cls(**kw_like))
        else:
            params = desilike_to_cobaya_params(ParameterCollection(params))
    kw_cobaya = dict(kw_cobaya or {})

    def initialize(self):
        """Prepare any computation, importing any necessary code, files, etc."""

        _kw_cobaya = {name: getattr(self, name, value) for name, value in kw_cobaya.items()}
        self.cache_size = 2
        self.likes = [cls(**{**kw_like, **_kw_cobaya}) for i in range(self.cache_size)]
        self.ignore_unknown_cosmoprimo_params = getattr(self, 'ignore_unknown_cosmoprimo_params', True)
        self._input_params = [{} for i in range(self.cache_size)]
        from desilike import mpi
        for like in self.likes:
            like.mpicomm = mpi.COMM_SELF  # no likelihood-level MPI-parallelization
            self._cosmo_params, self._nuisance_params = get_likelihood_params(like)
            for param in like.varied_params: param.update(prior=None)  # remove prior on varied parameters (already taken care of by cobaya)
        """
        import inspect
        kwargs = {name: getattr(self, name) for name in inspect.getargspec(cls).args}
        self.like = cls(**kwargs)
        """

    def get_requirements(self):
        """Return dictionary specifying quantities calculated by a theory code are needed."""
        for like in self.likes:
            requires = like.runtime_info.pipeline.get_cosmo_requires()
        self._fiducial = requires.get('fiducial', {})
        self._requires = CobayaEngine.get_requires(requires)
        return self._requires

    def logp(self, _derived=None, **params_values):
        """
        Take a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        if self._requires:
            from desilike.utils import deep_eq
            cosmo, input_params = camb_or_classy_to_cosmoprimo(self._fiducial, self.provider, params_values, ignore_unknown_params=self.ignore_unknown_cosmoprimo_params, return_input_params=True)
            # manual caching
            found = False
            for ilike, ip in enumerate(self._input_params):
                if deep_eq(input_params, ip):
                    found = True
                    break
            if not found:
                ilike = (getattr(self, '_ilike', -1) + 1) % self.cache_size  # set at a new position
                self.likes[ilike].runtime_info.pipeline.set_cosmo_requires(cosmo)
                self._input_params[ilike] = input_params
            #    print('COMPUTE COSMO!') #, input_params, _input_params)
            #else:
                #print('SKIP COSMO!')
        #import time
        #t0 = time.time()
        loglikelihood, derived = self.likes[ilike]({name: value for name, value in params_values.items() if name in self._nuisance_params}, return_derived=True)
        #print(time.time() - t0)
        self._ilike = ilike
        if _derived is not None:
            for value in derived:
                if value.param.ndim == 0:
                    _derived[value.param.name] = float(value[()])
        return float(loglikelihood)

    '''
    @classmethod
    def get_text_file_content(cls, file_name: str) -> Optional[str]:
        """
        Return the content of a file in the directory of the module, if it exists.
        """
        package = inspect.getmodule(cls).__package__  # for __package__ to be non-zero, we need the class module's package to be specified, e.g. cobaya_bindings
        try:
            return resources.read_text(package, file_name)
        except Exception:
            return None
    '''

    d = {'initialize': initialize, 'get_requirements': get_requirements, 'logp': logp}
    if module is not None:
        d['__module__'] = module
    if params is not None:
        d['params'] = params
    from cobaya.likelihood import Likelihood
    return type(Likelihood)(name_like, (Likelihood,), d)


class CobayaLikelihoodGenerator(BaseLikelihoodGenerator):

    """Extend :class:`BaseLikelihoodGenerator` with support for cobaya, writing parameters to a .yaml file."""

    def __init__(self, *args, **kwargs):
        super(CobayaLikelihoodGenerator, self).__init__(CobayaLikelihoodFactory, *args, **kwargs)

    def get_code(self, *args, kw_cobaya=None, **kwargs):
        cls, name_like, fn, code = super(CobayaLikelihoodGenerator, self).get_code(*args, kw_cobaya=kw_cobaya, **kwargs)
        dirname = os.path.dirname(fn)
        params = cobaya_params(cls(**self.kw_like))
        kw_cobaya = dict(kw_cobaya or {})
        BaseConfig(dict(stop_at_error=True, ignore_unknown_cosmoprimo_params=True, params=params, **kw_cobaya)).write(os.path.join(dirname, name_like + '.yaml'))

        import_line = 'from .{} import *'.format(os.path.splitext(os.path.basename(fn))[0])
        with open(os.path.join(dirname, '__init__.py'), 'a+') as file:
            file.seek(0, 0)
            lines = file.read()
            if import_line not in lines:
                if lines and lines[-1] != '\n': file.write('\n')
                file.write(import_line + '\n')

        return cls, name_like, fn, code
