import os
import sys

from desilike.parameter import ParameterCollection, Parameter
from desilike.utils import BaseClass, import_class, is_sequence


def load_from_file(fn, obj):

    import importlib.util
    spec = importlib.util.spec_from_file_location('bindings', fn)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return getattr(foo, obj)


class LikelihoodGenerator(BaseClass):

    line_delimiter = '\n\n'

    def __init__(self, factory, dirname='.'):
        self.factory = factory
        self.header = '# NOTE: This is automatically generated code by {}.{}\n'.format(self.__class__.__module__, self.__class__.__name__)
        self.header += 'from {} import {}'.format(self.factory.__module__, self.factory.__name__)
        self.dirname = os.path.abspath(os.path.join(dirname, os.path.basename(os.path.dirname(sys.modules[self.factory.__module__].__file__))))

    def get_code(self, likelihood, kw_like=None):
        self.kw_like = kw_like = kw_like or {}
        cls = import_class(likelihood)
        src_fn = sys.modules[cls.__module__].__file__
        src_dir = os.path.dirname(src_fn)
        fn = os.path.join(self.dirname, os.path.relpath(src_fn, os.path.commonpath([self.dirname, src_fn])))
        if os.path.isfile(os.path.join(src_dir, '__init__.py')):  # check if this is a package, then assumed in pythonpath
            code = 'from {} import {}\n'.format(cls.__module__, cls.__name__)
        else:
            #code = 'import sys\n'
            #code += "sys.path.insert(0, '{}')\n".format(src_dir)
            #code += 'from {} import {}\n'.format(os.path.splitext(os.path.basename(fn))[0], cls.__name__)
            code = 'from desilike.bindings.base import load_from_file\n'
            code += "{} = load_from_file('{}', '{}')\n".format(cls.__name__, src_fn, cls.__name__)
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
