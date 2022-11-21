import os

import numpy as np

from desilike.io import BaseConfig
from desilike.bindings.base import LikelihoodGenerator, get_likelihood_params


from desilike.cosmo import ExternalEngine, BaseSection, PowerSpectrumInterpolator2D, _make_list


class CobayaEngine(ExternalEngine):

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
                    if name == 'pk_interpolator':
                        tmp['nonlinear'] = attrs['non_linear']
                        tmp['z'] = attrs['z']
                        tmp['k_max'] = np.max(attrs.get('k', 1.))  # 1/Mpc unit
                        tmp['vars_pairs'] = []
                        for pair in attrs['of']:
                            if any('theta' in p for p in pair):
                                require_f += [tmp['z']]
                            tmp['vars_pairs'].append([convert[p] for p in pair])
                        toret['Pk_grid'] = tmp
        if require_f:
            toret['fsigma8'] = {'z': require_f}
            toret['sigma8_z'] = {'z': require_f}
        if toret:  # to get /h units
            if 'Hubble' in toret:
                toret['Hubble']['z'] = np.unique(np.insert(tmp['z'], 0, 0.))
            else:
                toret['Hubble'] = {'z': np.array([0.])}
        return toret


class Section(BaseSection):

    def __init__(self, engine):
        self.provider = engine.provider
        self.h = self.provider.get_Hubble(0.) / 100.


class Background(Section):

    def efunc(self, z):
        return self.provider.get_Hubble(z) / (100. * self.h)

    def comoving_radial_distance(self, z):
        return self.provider.get_comoving_radial_distance(z) * self.h

    def angular_diameter_distance(self, z):
        return self.provider.get_angular_diameter_distance(z) * self.h

    def comoving_angular_distance(self, z):
        return self.angular_diameter_distance(z) * (1. + z)


class Thermodynamics(Section):

    @property
    def rs_drag(self):
        return self.provider.get_rdrag() * self.h


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
            f = self.provider.get_fsigma8(z=z) / self.provider.get_sigma8_z(z=z)
            pk = pk * f**ntheta
        return PowerSpectrumInterpolator2D(k, z, pk, **kwargs)

    def sigma_rz(self, r, z, of='delta_m', **kwargs):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`r \mathrm{Mpc}/h`."""
        return self.pk_interpolator(non_linear=False, of=of, **kwargs).sigma_rz(r, z)

    def sigma8_z(self, z, of='delta_m'):
        r"""Return the r.m.s. of `of` perturbations in sphere of :math:`8 \mathrm{Mpc}/h`."""
        return self.sigma_rz(8., z, of=of)


def CobayaLikelihoodFactory(cls, module=None):

    def initialize(self):
        """Prepare any computation, importing any necessary code, files, etc."""

        self.like = cls()
        """
        import inspect
        kwargs = {name: getattr(self, name) for name in inspect.getargspec(cls).args}
        self.like = cls(**kwargs)
        """

    def get_requirements(self):
        """Return dictionary specifying quantities calculated by a theory code are needed."""
        self._requires = CobayaEngine.get_requires(self.like.runtime_info.pipeline.get_cosmo_requires())
        return self._requires

    def logp(self, _derived=None, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        if self._requires:
            from cosmoprimo import Cosmology
            cosmo = Cosmology(engine=CobayaEngine)
            cosmo._engine.provider = self.provider
            self.like.runtime_info.pipeline.set_cosmo_requires(cosmo)
        return self.like(**params_values)

    d = {'initialize': initialize, 'get_requirements': get_requirements, 'logp': logp}
    if module is not None:
        d['__module__'] = module
    from cobaya.likelihood import Likelihood
    return type(Likelihood)(cls.__name__, (Likelihood,), d)


class CobayaLikelihoodGenerator(LikelihoodGenerator):

    def __init__(self):
        super(CobayaLikelihoodGenerator, self).__init__(CobayaLikelihoodFactory)

    def get_code(self, likelihood):
        cls, fn, code = super(CobayaLikelihoodGenerator, self).get_code(likelihood)
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

        for param in get_likelihood_params(cls()):
            if param.derived or param.solved:
                continue
            if param.fixed:
                params[param.name] = param.value
            else:
                di = {'latex': param.latex()}
                di['prior'] = decode_prior(param.prior)
                if param.ref.is_proper():
                    di['ref'] = decode_prior(param.ref)
                if param.proposal is not None:
                    di['proposal'] = param.proposal
                params[param.name] = di

        BaseConfig(dict(stop_at_error=True, params=params)).write(os.path.join(dirname, cls.__name__ + '.yaml'))
        return cls, fn, code


if __name__ == '__main__':

    CobayaLikelihoodGenerator()()
