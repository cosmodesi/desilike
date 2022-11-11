import os

from numbers import Number

from desilike import utils
from desilike.bindings import LikelihoodGenerator


def MontePythonLikelihoodFactory(cls, module=None):

    from montepython.likelihood_class import Likelihood

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        self.like = cls()
        self.need_cosmo_arguments(data, getattr(self.like, 'external_requires', {}))

    def loglkl(self, cosmo, data):
        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        loglikelihood = self.like(**{name: data.mcmc_parameters[name]['current'] * data.mcmc_parameters[name]['scale'] for name in nuisance_parameter_names})
        if isinstance(loglikelihood, Number):
            return loglikelihood
        return loglikelihood.loglikelihood

    d = {'__init__': __init__, 'loglkl': loglkl}
    if module is not None:
        d['__module__'] = module
    return type(Likelihood)(cls.__name__, (Likelihood,), d)


class MontePythonLikelihoodGenerator(LikelihoodGenerator):

    def __init__(self):
        super(MontePythonLikelihoodGenerator, self).__init__(MontePythonLikelihoodFactory)

    def get_code(self, likelihood):
        cls, fn, code = super(MontePythonLikelihoodGenerator, self).get_code(likelihood)
        dirname = os.path.join(os.path.dirname(fn), cls.__name__)
        fn = os.path.join(dirname, '__init__.py')

        from desilike.parameter import ParameterCollection

        def prior_to_montepython(prior):
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
        for param in ParameterCollection(cls().params):
            prior = prior_to_montepython(param.prior)
            name = '{}.{}'.format(cls.__name__, param.name)
            for attr in ['center', 'variance']:
                if attr in prior:
                    likelihood_attrs['{}_{}'.format(name, attr)] = float(prior[attr])

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
            for name, value in parameters.items():
                file.write("data.parameters['{}'] = {}\n".format(name, value))

        return cls, fn, code


if __name__ == '__main__':

    MontePythonLikelihoodGenerator()()
