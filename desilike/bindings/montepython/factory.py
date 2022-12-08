import os

import numpy as np

from desilike import utils
from desilike.bindings.base import LikelihoodGenerator, get_likelihood_params, ParameterCollection


from desilike.cosmo import Cosmology, ExternalEngine, BaseSection, PowerSpectrumInterpolator2D, flatarray, _make_list


class MontePythonEngine(ExternalEngine):

    @classmethod
    def get_requires(cls, requires):
        toret = {'output': set()}
        for section, names in super(MontePythonEngine, cls).get_requires(requires).items():
            for name, attrs in names.items():
                if section == 'fourier':
                    toret['output'].add('mPk')
                    if name == 'sigma8_z':
                        toret['z_max_pk'] = max(toret.get('z_max_pk', 0.), attrs['z'].max())
                    if name == 'pk_interpolator':
                        toret['z_max_pk'] = max(toret.get('z_max_pk', 0.), attrs['z'].max())
                        toret['P_k_max_h/Mpc'] = np.max(attrs.get('k', 1))
        toret['output'] = ' '.join(toret['output'])
        return toret


class Section(BaseSection):

    def __init__(self, engine):
        self.classy = engine.classy
        self.h = self.classy.h()


class Background(Section):

    @flatarray(dtype=np.float64)
    def efunc(self, z):
        return self.classy.Hubble(z) / (100. * self.h)

    @flatarray(dtype=np.float64)
    def angular_diameter_distance(self, z):
        return self.classy.angular_distance(z) * self.h

    @flatarray(dtype=np.float64)
    def comoving_angular_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)


class Thermodynamics(Section):

    @property
    def rs_drag(self):
        return self.classy.rs_drag() * self.h


class Fourier(Section):

    def pk_interpolator(self, non_linear=False, of='delta_m', **kwargs):
        of = tuple(_make_list(of, length=2))
        cb = {('delta_cb', 'delta_cb'): True, ('delta_m', 'delta_m'): False,
              ('theta_cb', 'theta_cb'): True, ('theta_m', 'theta_m'): False}[of]
        pk, k, z = self.classy.get_pk_and_k_and_z(nonlinear=non_linear, only_clustering_species=cb)
        k = k / self.h
        pk = pk * self.h**3
        ntheta = sum('theta' in of_ for of_ in of)
        if ntheta:
            f = np.array([self.classy.scale_independent_growth_factor_f(zz) for zz in z])
            #f = self.classy.effective_f_sigma8(z) / self.classy.sigma8(z)  # only available in last class's version
            pk = pk * f**ntheta
        return PowerSpectrumInterpolator2D(k, z, pk, **kwargs)

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)


def cosmoprimo_to_montepython_params(params):
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


def montepython_to_cosmoprimo(fiducial, classy):
    if fiducial: cosmo = Cosmology.from_state(fiducial)
    else: cosmo = Cosmology()
    convert = {name: name for name in ['H0', 'h', 'A_s', 'ln10^{10}A_s', 'sigma8', 'n_s', 'omega_b', 'Omega_b', 'omega_cdm', 'Omega_cdm', 'omega_m', 'Omega_m', 'Omega_ncdm', 'omega_ncdm', 'm_ncdm', 'omega_k', 'Omega_k']}
    cosmo = cosmo.clone(**{convert[name]: value for name, value in classy.pars.items() if name in convert}, engine=MontePythonEngine)
    cosmo._engine.classy = classy
    return cosmo


def MontePythonLikelihoodFactory(cls, kw_like, module=None):

    from montepython.likelihood_class import Likelihood

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        self.like = cls(**kw_like)
        self._cosmo_params, self._nuisance_params = get_likelihood_params(self.like)
        self.nuisance = self.use_nuisance = [param.name for param in self._nuisance_params]  # required by MontePython
        requires = self.like.runtime_info.pipeline.get_cosmo_requires()
        self._fiducial = requires.get('fiducial', {})
        self._requires = MontePythonEngine.get_requires(requires)
        # On two steps, otherwise z_max_pk and P_k_max_h become zero
        self.need_cosmo_arguments(data, {'output': self._requires['output']})
        self.need_cosmo_arguments(data, {name: value for name, value in self._requires.items() if name != 'output'})

    def loglkl(self, classy, data):
        if self._requires:
            cosmo = montepython_to_cosmoprimo(self._fiducial, classy)
            self.like.runtime_info.pipeline.set_cosmo_requires(cosmo)
        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        loglikelihood = self.like(**{name: data.mcmc_parameters[name]['current'] * data.mcmc_parameters[name]['scale'] for name in nuisance_parameter_names})
        return loglikelihood

    d = {'__init__': __init__, 'loglkl': loglkl}
    if module is not None:
        d['__module__'] = module
    return type(Likelihood)(cls.__name__, (Likelihood,), d)


class MontePythonLikelihoodGenerator(LikelihoodGenerator):

    def __init__(self):
        super(MontePythonLikelihoodGenerator, self).__init__(MontePythonLikelihoodFactory)

    def get_code(self, *args, **kwargs):
        cls, fn, code = super(MontePythonLikelihoodGenerator, self).get_code(*args, **kwargs)
        dirname = os.path.join(os.path.dirname(fn), cls.__name__)
        fn = os.path.join(dirname, '__init__.py')

        def decode_prior(prior):
            di = {}
            di['dist'] = prior.dist
            if prior.is_limited():
                di['min'], di['max'] = prior.limits
            else:
                di['min'] = di['max'] = None
            try:
                di['center'] = prior.loc
                di['variance'] = prior.scale ** 2
            except AttributeError:
                pass
            return di

        parameters, likelihood_attrs = {}, {}
        cosmo_params, nuisance_params = get_likelihood_params(cls(**self.kw_like))
        #cosmo_params = cosmoprimo_to_montepython_params(cosmo_params)
        for param in nuisance_params:
            if param.depends:
                raise ValueError('Cannot cope with parameter dependencies')
            prior = decode_prior(param.prior)
            name = '{}.{}'.format(cls.__name__, param.name)
            for attr in ['center', 'variance']:
                if attr in prior:
                    likelihood_attrs['{}_prior_{}'.format(name, attr)] = float(prior[attr])

            proposal = 0. if param.fixed else param.proposal
            if proposal is None:
                raise ValueError('Provide proposal value for {}'.format(param))
            mi, ma = [float(m) if m is not None else None for m in [prior['min'], prior['max']]]
            parameters[name] = [float(param.value), mi, ma, float(proposal), 1., 'nuisance']

        utils.mkdir(dirname)
        with open(os.path.join(dirname, cls.__name__ + '.data'), 'w') as file:
            for name, value in likelihood_attrs.items():
                file.write('{} = {}\n'.format(name, value))

        with open(os.path.join(dirname, cls.__name__ + '.param'), 'w') as file:
            file.write('# To be copy-pasted in the MontePython *.param file\n')
            for name, value in parameters.items():
                file.write("data.parameters['{}'] = {}\n".format(name[len(cls.__name__) + 1:], value))

        return cls, fn, code


if __name__ == '__main__':

    MontePythonLikelihoodGenerator()()
