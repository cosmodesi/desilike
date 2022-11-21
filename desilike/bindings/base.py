import os
import sys

from desilike.utils import BaseClass, import_class, is_sequence


class LikelihoodGenerator(BaseClass):

    line_delimiter = '\n\n'

    def __init__(self, factory):
        self.factory = factory
        self.header = '# NOTE: This is automatically generated code by {}.{}\n'.format(self.__class__.__module__, self.__class__.__name__)
        self.header += 'from {} import {}'.format(self.factory.__module__, self.factory.__name__)

    def get_code(self, likelihood):
        dirname = os.path.dirname(sys.modules[self.factory.__module__].__file__)
        cls = import_class(likelihood)
        fn = sys.modules[cls.__module__].__file__
        fn = os.path.join(dirname, os.path.relpath(fn, os.path.commonpath([dirname, fn])))
        code = 'from {} import {}\n'.format(cls.__module__, cls.__name__)
        code += '{0} = {1}({0}, __name__)'.format(cls.__name__, self.factory.__name__)
        return cls, fn, code

    def __call__(self, likelihoods):
        if not is_sequence(likelihoods): likelihoods = [likelihoods]
        txt = {}
        for likelihood in likelihoods:
            fn, code = self.get_code(likelihood)[1:]
            txt[fn] = txt.get(fn, []) + [code]
        for fn in txt:
            self.log_debug('Saving likelihood in {}'.format(fn))
            with open(fn, 'w') as file:
                file.write(self.header + self.line_delimiter)
                for line in txt[fn]:
                    file.write(line + self.line_delimiter)


def get_likelihood_params(like):
    return like.runtime_info.pipeline.params.select(derived=False, solved=False)
