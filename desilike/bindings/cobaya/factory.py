import os

from numbers import Number

from desilike.io import BaseConfig
from desilike.bindings import LikelihoodGenerator


def CobayaLikelihoodFactory(cls, module=None):

    from cobaya.likelihood import Likelihood

    def initialize(self):
        """Prepare any computation, importing any necessary code, files, etc."""
        self.like = cls()

    def get_requirements(self):
        """Return dictionary specifying quantities calculated by a theory code are needed."""
        return getattr(self.like, 'external_requires', {})

    def logp(self, _derived=None, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        loglikelihood = self.like(**params_values)
        if isinstance(loglikelihood, Number):
            return loglikelihood
        return loglikelihood.loglikelihood

    d = {'initialize': initialize, 'get_requirements': get_requirements, 'logp': logp}
    if module is not None:
        d['__module__'] = module
    return type(Likelihood)(cls.__name__, (Likelihood,), d)


class CobayaLikelihoodGenerator(LikelihoodGenerator):

    def __init__(self):
        super(CobayaLikelihoodGenerator, self).__init__(CobayaLikelihoodFactory)

    def get_code(self, likelihood):
        cls, fn, code = super(CobayaLikelihoodGenerator, self).get_code(likelihood)
        params = {}
        from desilike.parameter import ParameterCollection

        def prior_to_cobaya(prior):
            di = {}
            di['dist'] = prior.dist
            if prior.is_limited():
                di['min'], di['max'] = prior.limits
            for name in ['loc', 'scale']:
                if hasattr(prior, name):
                    di[name] = getattr(prior, name)
            return di

        for param in ParameterCollection(cls().params):
            if param.fixed:
                params[param.name] = param.value
            else:
                di = {'latex': param.latex()}
                di['prior'] = prior_to_cobaya(param.prior)
                if param.ref.is_proper():
                    di['ref'] = prior_to_cobaya(param.ref)
                if param.proposal is not None:
                    di['proposal'] = param.proposal
                params[param.name] = di

        dirname = os.path.dirname(fn)
        BaseConfig(dict(params=params)).write(os.path.join(dirname, cls.__name__ + '.yaml'))
        return cls, fn, code


if __name__ == '__main__':

    CobayaLikelihoodGenerator()()
