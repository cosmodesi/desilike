import os

import numpy as np
from scipy import constants, stats

from desilike import utils
from desilike.bindings.base import LikelihoodGenerator, get_likelihood_params

from desilike.cosmo import ExternalEngine, BaseSection, PowerSpectrumInterpolator2D, flatarray, _make_list


class CosmoSISEngine(ExternalEngine):

    pass


class Section(BaseSection):

    def __init__(self, engine):
        self.block = engine.block
        self.h = self.block['cosmological_parameters', 'h0']


class Background(Section):

    @flatarray(dtype=np.float64)
    def efunc(self, z):
        return np.interp(z, self.block['distances', 'z'], (constants.c / 1e3) * self.block['distances', 'H'] / (100. * self.h))

    @flatarray(dtype=np.float64)
    def angular_diameter_distance(self, z):
        return np.interp(z, self.block['distances', 'z'], self.block['distances', 'D_A'] * self.h)

    @flatarray(dtype=np.float64)
    def comoving_angular_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)


class Thermodynamics(Section):

    @property
    def rs_drag(self):
        return self.block['distances', 'rs_drag'] * self.h


class Fourier(Section):

    def pk_interpolator(self, non_linear=False, of='delta_m', **kwargs):
        of = tuple(_make_list(of, length=2))
        if non_linear:
            raise NotImplementedError('Linear power spectrum only implemented')
        section_name = {('delta_cb', 'delta_cb'): 'cdm_baryon_power_lin', ('delta_m', 'delta_m'): 'matter_power_lin',
                        ('theta_cb', 'theta_cb'): 'cdm_baryon_power_lin', ('theta_m', 'theta_m'): 'matter_power_lin'}[of]
        k = self.block[section_name, 'k_h']
        z = self.block[section_name, 'z']
        pk = self.block[section_name, 'p_k'].T
        ntheta = sum('theta' in of_ for of_ in of)
        if ntheta:
            #f = self.block['growth_parameters', 'fsigma_8'] / self.block['growth_parameters', 'sigma_8']
            f = self.block['growth_parameters', 'f_z']
            pk = pk * f**ntheta
        return PowerSpectrumInterpolator2D(k, z, pk, **kwargs)

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)


desilike_name = 'desi'


def CosmoSISLikelihoodFactory(cls, module=None):

    def __init__(self, options):
        self.like = cls()
        self._params = get_likelihood_params(self.like)
        self._requires = self.like.runtime_info.pipeline.get_cosmo_requires()

    def do_likelihood(self, block):
        if self._requires:
            from cosmoprimo import Cosmology
            section = 'cosmological_parameters'
            params = {name: block[section, name] for _, name in block.keys(section=section)}
            cosmo = Cosmology(h=params['h0'],
                              Omega_b=params['omega_b'],
                              Omega_cdm=params['omega_c'],
                              sigma8=params.get('sigma_8', 0.8),
                              n_s=params.get('n_s', 0.96),
                              Omega_k=params['omega_k'],
                              N_eff=3.046 + params.get('delta_neff', 0.),
                              w0_fld=params.get('w0', -1.),
                              wa_fld=params.get('wa', 0.),
                              engine=CosmoSISEngine)
            cosmo._engine.block = block
            self.like.runtime_info.pipeline.set_cosmo_requires(cosmo)
        loglikelihood = self.like(**{param.name: block[desilike_name, param.name] for param in self._params})
        block['likelihoods', '{}_like'.format(desilike_name)] = loglikelihood

    @classmethod
    def build_module(cls):

        from cosmosis.datablock import SectionOptions

        def setup(options):
            options = SectionOptions(options)
            likelihoodCalculator = cls(options)
            return likelihoodCalculator

        def execute(block, config):
            likelihoodCalculator = config
            likelihoodCalculator.do_likelihood(block)
            return 0

        def cleanup(config):
            pass
            # likelihoodCalculator = config
            # likelihoodCalculator.cleanup()

        return setup, execute, cleanup

    @classmethod
    def as_module(cls, name):
        from cosmosis.runtime import FunctionModule
        setup, execute, cleanup = cls.build_module()
        return FunctionModule(name, setup, execute, cleanup)

    d = {'__init__': __init__, 'do_likelihood': do_likelihood, 'build_module': build_module, 'as_module': as_module}
    if module is not None:
        d['__module__'] = module
    return type(object)(cls.__name__, (object,), d)


class CosmoSISLikelihoodGenerator(LikelihoodGenerator):

    def __init__(self):
        super(CosmoSISLikelihoodGenerator, self).__init__(CosmoSISLikelihoodFactory)

    def get_code(self, likelihood):
        cls, fn, code = super(CosmoSISLikelihoodGenerator, self).get_code(likelihood)
        dirname = os.path.dirname(fn)
        fn = os.path.join(dirname, cls.__name__ + '.py')

        def decode_prior(prior, param):
            limits = list(prior.limits)
            nsigmas = 5
            for ilim, (lim, cdf) in enumerate(zip(limits, stats.norm().cdf([-nsigmas, nsigmas]))):  # 5-sigma limits
                if not np.isfinite(lim):
                    lim = prior.ppf(cdf)
                    limits[ilim] = lim
                    self.log_warning('Infinite prior for parameter {}; setting to {:d}-sigma = {}'.format(param, nsigmas, lim))
            if prior.dist == 'uniform':
                prior = ['uniform'] + limits
            elif prior.dist == 'norm':
                prior = ['gaussian', prior.loc, prior.scale]
            elif prior.dist == 'expon':
                if prior.loc != 0:
                    raise ValueError('Exponential prior must be centered on 0 for parameter {}'.format(param))
                prior = ['exponential', prior.scale]
            else:
                raise ValueError('Prior distribution must be either uniform, norm or expon for parameter {}'.format(param))
            return prior, limits

        values, priors = {}, {}
        for param in get_likelihood_params(cls()):
            prior, limits = decode_prior(param.prior, param.name)
            values[param] = [param.value]
            if param.varied:
                values[param] = [limits[0], param.value, limits[1]]
            priors[param] = prior

        def tostr(li):
            return ' '.join(map(str, li))

        utils.mkdir(dirname)
        with open(os.path.join(dirname, cls.__name__ + '_values.ini'), 'w') as file:
            file.write('[{}]\n'.format(desilike_name))
            for name, value in values.items():
                file.write('{} = {}\n'.format(name, tostr(value)))

        with open(os.path.join(dirname, cls.__name__ + '_priors.ini'), 'w') as file:
            file.write('[{}]\n'.format(desilike_name))
            for name, value in priors.items():
                file.write('{} = {}\n'.format(name, tostr(value)))

        code += '\n\n'
        code += 'setup, execute, cleanup = {}.build_module()'.format(cls.__name__)

        return cls, fn, code


if __name__ == '__main__':

    CosmoSISLikelihoodGenerator()()
