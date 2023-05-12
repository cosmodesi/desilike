import os
import sys

from desilike.parameter import ParameterCollection, Parameter
from desilike.utils import BaseClass, import_class, is_sequence


def find_module_from_file(fn):
    """
    Return full module name corresponding to ``fn``.
    It first checks if there is an '__init__.py' file in the same directory as ``fn``,
    and if so, it recursively finds the module name of that directory and combines it
    with the basename of ``fn``.

    >>> find_module_from_file('base.py')
    'desilike.bindings.base'

    If there is no '__init__.py' in the same directory as the given file, it returns ``None``.
    """
    dirname = os.path.dirname(fn)
    if os.path.isfile(os.path.join(dirname, '__init__.py')):
        module = find_module_from_file(dirname) or os.path.basename(dirname)
        basename = os.path.splitext(os.path.basename(fn))[0]
        return '.'.join([module, basename])


def load_from_file(fn, obj):
    """Load object ``obj`` from file ``fn``."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('bindings', fn)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return getattr(foo, obj)


class BaseLikelihoodGenerator(BaseClass):

    """Base class to write necessary files for likelihoods to be imported by external inference codes."""

    line_delimiter = '\n\n'

    def __init__(self, factory, dirname='.'):
        """
        Initialize :class:`BaseLikelihoodGenerator`.

        Parameters
        ----------
        factory : callable
            A callable that returns a likelihood object adapted to the external inference code,
            and takes as input a callable that returns a :class:`BaseLikelihood`,
            a dictionary of optional arguments (passed to the latter callable),
            and the name of the module where it is called.
            and returns a likelihood object adapted to the external inference code.

        dirname : str, Path, default='.'
            Base directory for the file structure: bindings are saved in dirname/{code}/,
            with {code} the directory where ``factory`` is defined: 'cobaya', 'cosmosis', 'montepython'...
        """
        self.factory = factory
        self.header = '# NOTE: This code has been automatically generated by {}.{}\n'.format(self.__class__.__module__, self.__class__.__name__)
        self.header += 'from {} import {}'.format(self.factory.__module__, self.factory.__name__)
        self.dirname = os.path.abspath(os.path.join(dirname, os.path.basename(os.path.dirname(sys.modules[self.factory.__module__].__file__))))

    def get_code(self, likelihood, name_like=None, kw_like=None, module=None):
        """
        Internal method to write code to generate likelihood object to be imported by the external inference code.

        Parameters
        ----------
        likelihood : type, callable
            Callable that returns a :class:`BaseLikelihood`, given some optional arguments (see ``kw_like``).

        name_like : str, default=None
            Likelihood name, defaults to ``likelihood`` name.

        kw_like : dict, default=None
            Optional arguments for ``likelihood``.

        module : str, default=None
            Full module name where ``likelihood`` is defined.
            If ``None``, the full module name is searched with :func:`find_module_from_file`; if not in a package,
            absolute path to file where ``likelihood`` object is defined will be used to import it in the generated code.

        Returns
        -------
        cls, name, fn, code : callable, str, str
            Callable that generates :class:`BaseLikelihood`, likelihood name, file name where the code is to be written, and code itself.
        """
        self.kw_like = kw_like = kw_like or {}
        cls = import_class(likelihood)
        if name_like is None:
            name_like = cls.__name__
        src_fn = sys.modules[cls.__module__].__file__
        # src_dir = os.path.dirname(src_fn)
        fn = os.path.join(self.dirname, os.path.relpath(src_fn, os.path.commonpath([self.dirname, src_fn])))
        if module is None:
            module = find_module_from_file(src_fn)
        if module is not None:  # check if this is a package, then assumed in pythonpath
            code = 'from {} import {}\n'.format(module, cls.__name__)
        else:
            # code = 'import sys\n'
            # code += "sys.path.insert(0, '{}')\n".format(src_dir)
            # code += 'from {} import {}\n'.format(os.path.splitext(os.path.basename(fn))[0], cls.__name__)
            code = 'from desilike.bindings.base import load_from_file\n'
            code += "{} = load_from_file('{}', '{}')\n".format(cls.__name__, src_fn, cls.__name__)
        code += '{} = {}({}, {}, __name__)'.format(name_like, self.factory.__name__, cls.__name__, kw_like)
        return cls, name_like, fn, code

    def __call__(self, likelihood, name_like=None, kw_like=None, module=None, overwrite=True):
        """
        Generate file structure and code containing definition of likelihood such that it can be imported by the external inference code.

        Parameters
        ----------
        likelihood : list, type, callable
            List of (or single) callable(s) that returns a :class:`BaseLikelihood`, given some optional arguments (see ``kw_like``).

        name_like : str, default=None
            Likelihood name, defaults to ``likelihood`` name.

        kw_like : dict, default=None
            Optional arguments for (each of) ``likelihood``.

        module : str, default=None
            Module where ``likelihood`` is defined.
            If ``None``, absolute path to file where ``likelihood`` object is defined
            will be used to import it in the generated code.
        """
        if not is_sequence(likelihood):
            likelihood = [likelihood]
        if not is_sequence(module):
            module = [module] * len(likelihood)
        if len(module) != len(likelihood):
            raise ValueError('Number of provided likelihood modules is not the same as the number of likelihoods')
        if not is_sequence(name_like):
            name_like = [name_like] * len(likelihood)
        if len(name_like) != len(likelihood):
            raise ValueError('Number of provided likelihood names name_like is not the same as the number of likelihoods')
        if not is_sequence(kw_like):
            kw_like = [kw_like] * len(likelihood)
        if len(kw_like) != len(likelihood):
            raise ValueError('Number of provided likelihood kwargs kw_like is not the same as the number of likelihoods')
        txt = {}
        for likelihood, name_like, kw_like, module in zip(likelihood, name_like, kw_like, module):
            fn, code = self.get_code(likelihood, name_like=name_like, kw_like=kw_like, module=module)[2:]
            txt[fn] = txt.get(fn, []) + [code]
        for fn in txt:
            self.log_debug('Saving likelihood in {}'.format(fn))
            try:
                with open(fn, 'r') as file: current = file.read()
            except IOError:
                current = ''
            with open(fn, 'w' if overwrite else 'a') as file:
                for line in [self.header] + txt[fn]:
                    if line not in current:
                        file.write(line + self.line_delimiter)


def get_likelihood_params(likelihood):
    """
    Given a :class:`BaseLikelihood` instance,
    return its cosmological parameters and its "nuisance" parameters.
    """
    all_params = likelihood.runtime_info.pipeline.params.select(derived=False, solved=False)
    cosmo_names = likelihood.runtime_info.pipeline.get_cosmo_requires().get('params', {})
    cosmo_params, nuisance_params = ParameterCollection(), ParameterCollection()
    for param in all_params:
        if param.basename in cosmo_names:
            cosmo_params.set(param)
        else:
            nuisance_params.set(param)
    return cosmo_params, nuisance_params
