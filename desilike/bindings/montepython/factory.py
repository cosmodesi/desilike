import os

import numpy as np

from desilike import utils
from desilike.bindings.base import BaseLikelihoodGenerator, get_likelihood_params, ParameterCollection


from desilike.cosmo import Cosmology, BaseExternalEngine, BaseSection, PowerSpectrumInterpolator2D, flatarray, _make_list


"""Mock up cosmoprimo with montepython classy."""


class MontePythonEngine(BaseExternalEngine):

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
                        toret['P_k_max_h/Mpc'] = attrs['k'].max()
        toret['output'] = ' '.join(toret['output'])
        return toret


class Section(BaseSection):

    def __init__(self, engine):
        self.classy = engine.classy
        self.h = self.classy.h()


class Background(Section):

    @flatarray(dtype=np.float64)
    def efunc(self, z):
        return self.classy.Hubble(z) * 2.99792458e5 / (100. * self.h)

    @flatarray(dtype=np.float64)
    def angular_diameter_distance(self, z):
        return self.classy.angular_distance(z) * self.h

    @flatarray(dtype=np.float64)
    def comoving_angular_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)

    @flatarray(dtype=np.float64)
    def luminosity_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)**2


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


def convert_param_name(name):
    return name.replace('.', '_')


def MontePythonLikelihoodFactory(cls, name_like=None, kw_like=None, module=None):

    if name_like is None:
        name_like = cls.__name__
    if kw_like is None:
        kw_like = {}

    from montepython.likelihood_class import Likelihood

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        self.like = cls(**kw_like)
        from desilike import mpi
        self.like.mpicomm = mpi.COMM_SELF  # no likelihood-level MPI-parallelization
        self._cosmo_params, self._nuisance_params = get_likelihood_params(self.like)
        for param in self.like.varied_params: param.update(prior=None)  # remove prior on varied parameters (already taken care of by montepython)
        self._nuisance_params = {convert_param_name(param.name): param.name for param in self._nuisance_params}
        self.nuisance = self.use_nuisance = list(self._nuisance_params.keys())  # required by MontePython
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
        # nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        loglikelihood = self.like({name: data.mcmc_parameters[mname]['current'] * data.mcmc_parameters[mname]['scale'] for mname, name in self._nuisance_params.items()})
        return loglikelihood

    d = {'__init__': __init__, 'loglkl': loglkl}
    if module is not None:
        d['__module__'] = module
    return type(Likelihood)(name_like, (Likelihood,), d)


class MontePythonLikelihoodGenerator(BaseLikelihoodGenerator):

    """Extend :class:`MontePythonLikelihoodGenerator` with support for montepython, generating .data and .param files."""

    def __init__(self, *args, **kwargs):
        super(MontePythonLikelihoodGenerator, self).__init__(MontePythonLikelihoodFactory, *args, **kwargs)

    def get_code(self, *args, **kwargs):
        cls, name_like, fn, code = super(MontePythonLikelihoodGenerator, self).get_code(*args, **kwargs)
        dirname = os.path.join(os.path.dirname(fn), name_like)
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
                di['variance'] = prior.scale**2
            except AttributeError:
                pass
            return di

        parameters, likelihood_attrs = {}, {'{}.name'.format(name_like): name_like}  # useless placeholder, just to avoid MontePython to complain
        cosmo_params, nuisance_params = get_likelihood_params(cls(**self.kw_like))
        # cosmo_params = cosmoprimo_to_montepython_params(cosmo_params)
        for param in nuisance_params:
            if param.depends:
                raise ValueError('Cannot cope with parameter dependencies')
            prior = decode_prior(param.prior)
            name = '{}.{}'.format(name_like, convert_param_name(param.name))
            for attr in ['center', 'variance']:
                if attr in prior:
                    likelihood_attrs['{}_prior_{}'.format(name, attr)] = float(prior[attr])

            proposal = 0. if param.fixed else param.proposal
            if proposal is None:
                raise ValueError('Provide proposal value for {}'.format(param))
            mi, ma = [float(m) if m is not None else None for m in [prior['min'], prior['max']]]
            parameters[name] = [float(param.value), mi, ma, float(proposal), 1., 'nuisance']

        utils.mkdir(dirname)
        with open(os.path.join(dirname, name_like + '.data'), 'w') as file:
            for name, value in likelihood_attrs.items():
                file.write('{} = {}\n'.format(name, value))

        with open(os.path.join(dirname, name_like + '.param'), 'w') as file:
            file.write('# To be copy-pasted in the MontePython *.param file\n')
            for name, value in parameters.items():
                file.write("data.parameters['{}'] = {}\n".format(name[len(name_like) + 1:], value))

        return cls, name_like, fn, code
