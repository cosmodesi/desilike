import os
import sys

from desilike.parameter import ParameterCollection, Parameter
from desilike.utils import BaseClass, import_class, is_sequence


class LikelihoodGenerator(BaseClass):

    line_delimiter = '\n\n'

    def __init__(self, factory):
        self.factory = factory
        self.header = '# NOTE: This is automatically generated code by {}.{}\n'.format(self.__class__.__module__, self.__class__.__name__)
        self.header += 'from {} import {}'.format(self.factory.__module__, self.factory.__name__)

    def get_code(self, likelihood, kw_like=None):
        self.kw_like = kw_like = kw_like or {}
        dirname = os.path.dirname(sys.modules[self.factory.__module__].__file__)
        cls = import_class(likelihood)
        fn = sys.modules[cls.__module__].__file__
        fn = os.path.join(dirname, os.path.relpath(fn, os.path.commonpath([dirname, fn])))
        code = 'from {} import {}\n'.format(cls.__module__, cls.__name__)
        code += '{0} = {1}({0}, {2}, __name__)'.format(cls.__name__, self.factory.__name__, kw_like)
        return cls, fn, code

    def __call__(self, likelihoods, kw_likes=None):
        if not is_sequence(likelihoods):
            likelihoods = [likelihoods]
        if not is_sequence(kw_likes):
            kw_likes = [kw_likes] * len(likelihoods)
        if len(kw_likes) != len(likelihoods):
            raise ValueError('Number of provided likelihood kwargs is not the same as the number of likelihoods')
        txt = {}
        for likelihood, kw_like in zip(likelihoods, kw_likes):
            fn, code = self.get_code(likelihood, kw_like)[1:]
            txt[fn] = txt.get(fn, []) + [code]
        for fn in txt:
            self.log_debug('Saving likelihood in {}'.format(fn))
            with open(fn, 'w') as file:
                file.write(self.header + self.line_delimiter)
                for line in txt[fn]:
                    file.write(line + self.line_delimiter)


def get_likelihood_params(like):
    all_params = like.runtime_info.pipeline.params.select(derived=False, solved=False)
    cosmo_names = like.runtime_info.pipeline.get_cosmo_requires().get('params', {})
    cosmo_params, nuisance_params = ParameterCollection(), ParameterCollection()
    for param in all_params:
        if param.basename in cosmo_names:
            cosmo_params.set(param)
        else:
            nuisance_params.set(param)
    return cosmo_params, nuisance_params
