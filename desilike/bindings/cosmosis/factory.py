import os

from numbers import Number

from desilike import utils
from desilike.bindings import LikelihoodGenerator


desilike_name = 'desi'


def CosmoSISLikelihoodFactory(cls, module=None):

    from cosmosis.datablock import SectionOptions
    from cosmosis.runtime import FunctionModule

    def __init__(self, options):
        self.like = cls()

    def do_likelihood(self, block):
        loglikelihood = self.like(**{param.name: block[desilike_name, param.name] for param in self.params})
        if not isinstance(loglikelihood, Number):
            loglikelihood = loglikelihood.loglikelihood
        block['likelihoods', '{}_like'.format(desilike_name)] = loglikelihood

    @classmethod
    def build_module(cls):

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
        setup, execute, cleanup = cls.build_module()
        return FunctionModule(name, setup, execute, cleanup)

    d = {'__init__': __init__, 'do_likelihood': do_likelihood, 'build_module': build_module, 'as_module': as_module}
    if module is not None:
        d['__module__'] = module
    return object(cls.__name__, (object,), d)


class CosmoSISLikelihoodGenerator(LikelihoodGenerator):

    def __init__(self):
        super(CosmoSISLikelihoodGenerator, self).__init__(CosmoSISLikelihoodFactory)

    def get_code(self, likelihood):
        cls, fn, code = super(CosmoSISLikelihoodGenerator, self).get_code(likelihood)
        dirname = os.path.dirname(fn)
        fn = os.path.join(dirname, cls.__name__ + '.py')

        from desilike.parameter import ParameterCollection

        def prior_to_cosmosis(prior):
            limits = prior.limits
            if prior.dist == 'uniform':
                prior = ['uniform'] + limits
            elif prior.dist == 'norm':
                prior = ['gaussian', prior.loc, prior.scale]
            elif prior.dist == 'expon':
                if prior.loc != 0:
                    raise ValueError('Exponential prior must be centered on 0')
                prior = ['exponential', prior.scale]
            else:
                raise ValueError('Prior distribution must be either uniform, norm or expon')
            return prior, limits

        values, priors = {}, {}
        for param in ParameterCollection(cls().params):
            prior, limits = prior_to_cosmosis(param.prior)
            values[param] = [param.value]
            if not param.fixed:
                values[param] += limits
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

        return cls, fn, code


if __name__ == '__main__':

    CosmoSISLikelihoodGenerator()()
