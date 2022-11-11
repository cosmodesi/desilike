import os
import re
from collections import UserDict

import numpy as np
import yaml

from . import utils
from .utils import BaseClass, deep_eq


class YamlLoader(yaml.SafeLoader):
    """
    *yaml* loader that correctly parses numbers.
    Taken from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number.
    """


YamlLoader.add_implicit_resolver(u'tag:yaml.org,2002:float',
                                 re.compile(u'''^(?:
                                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                                 |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                                 |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                                 |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                                 |[-+]?\\.(?:inf|Inf|INF)
                                 |\\.(?:nan|NaN|NAN))$''', re.X),
                                 list(u'-+0123456789.'))

YamlLoader.add_implicit_resolver('!none', re.compile('None$'), first='None')


def none_constructor(loader, node):
    return None


YamlLoader.add_constructor('!none', none_constructor)


def yaml_parser(string, index=None):
    """Parse string in *yaml* format."""
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    alls = list(yaml.load_all(string, Loader=YamlLoader))
    if index is not None:
        if isinstance(index, dict):
            match = False
            for config in alls:
                match = all([config.get(name) == value for name, value in index.items()])
                if match: break
            if not match:
                raise IndexError('No match found for index {}'.format(index))
        else:
            config = alls[index]
    else:
        config = yaml.load(string, Loader=YamlLoader)
    data = dict(config)
    return data


class MetaClass(type(BaseClass), type(UserDict)):

    pass


class ConfigError(Exception):

    pass


class BaseConfig(BaseClass, UserDict, metaclass=MetaClass):
    """
    Class that decodes configuration dictionary, taking care of template forms.

    Attributes
    ----------
    data : dict
        Decoded configuration dictionary.

    raw : dict
        Raw (without decoding of template forms) configuration dictionary.

    filename : string
        Path to corresponding configuration file.

    parser : callable
        *yaml* parser.
    """
    _attrs = []
    _key_import = '.import'

    def __init__(self, data=None, string=None, parser=None, decode=True, base_dir=None, **kwargs):
        """
        Initialize :class:`Decoder`.

        Parameters
        ----------
        data : dict, string, default=None
            Dictionary or path to a configuration *yaml* file to decode.

        string : string
            If not ``None``, *yaml* format string to decode.
            Added on top of ``data``.

        parser : callable, default=yaml_parser
            Function that parses *yaml* string into a dictionary.
            Used when ``data`` is string, or ``string`` is not ``None``.

        decode : bool, default=True
            Whether to decode configuration dictionary, i.e. solve template forms.

        kwargs : dict
            Arguments for :func:`parser`.
        """
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return

        self.parser = parser
        if parser is None:
            self.parser = yaml_parser

        datad = {}

        self.base_dir = base_dir
        if isinstance(data, str):
            if string is None: string = ''
            if base_dir is None: self.base_dir = os.path.dirname(data)
            with open(data, 'r') as file:
                string += file.read()
        elif data is not None:
            datad = dict(data)

        if string is not None:
            datad.update(self.parser(string, **kwargs))

        self.data = datad
        if decode: self.decode()

    def decode(self):

        eval_re_pattern = re.compile("e'(.*?)'$")
        format_re_pattern = re.compile("f'(.*?)'$")

        def decode_eval(word):
            m = re.match(eval_re_pattern, word)
            if m:
                word = m.group(1)
                placeholders = re.finditer(r'\$?\{.*?\}', word)
                word_letters = re.sub(r'[^a-zA-Z]', '_', word)
                di = {}
                for placeholder in placeholders:
                    placeholder = placeholder.group()
                    inenv = placeholder.startswith('$')
                    if inenv: placeholder_nobrackets = placeholder[2:-1]
                    else: placeholder_nobrackets = placeholder[1:-1]
                    if placeholder_nobrackets.startswith('{'):
                        word = word.replace(placeholder, '$' * inenv + placeholder_nobrackets)
                    else:
                        if inenv:
                            freplace = os.getenv(placeholder_nobrackets)
                        else:
                            freplace = replace = self.search(placeholder_nobrackets)
                            if isinstance(replace, str):
                                freplace = decode_eval(replace)
                                if freplace is None: freplace = replace
                        #if isinstance(freplace, str):
                        #    word = word.replace(placeholder, freplace)
                        #else:
                        key = '__variable_of_{}_{:d}__'.format(word_letters, len(di) + 1)
                        assert key not in word
                        di[key] = freplace
                        word = word.replace(placeholder, key)
                return utils.evaluate(word, locals=di)
            return None

        def decode_format(word):
            m = re.match(format_re_pattern, word)
            if m:
                word = m.group(1)
                placeholders = re.finditer(r'\$?\{.*?\}', word)
                for placeholder in placeholders:
                    placeholder = placeholder.group()
                    inenv = placeholder.startswith('$')
                    if inenv: placeholder_nobrackets = placeholder[2:-1]
                    else: placeholder_nobrackets = placeholder[1:-1]
                    if placeholder_nobrackets.startswith('{'):
                        word = word.replace(placeholder, '$' * inenv + placeholder_nobrackets)
                    else:
                        keyfmt = re.match(r'^(.*[^:])(:[^:]*)$', placeholder_nobrackets)
                        if keyfmt: key, fmt = keyfmt.groups()
                        else: key, fmt = placeholder_nobrackets, ''
                        if inenv:
                            freplace = os.getenv(key)
                        else:
                            freplace = replace = self.search(key)
                            if isinstance(replace, str):
                                freplace = decode_format(replace)
                                if freplace is None: freplace = replace
                        word = word.replace(placeholder, ('{' + fmt + '}').format(freplace))
                return word
            return None

        def callback(di, decode):
            for key, value in (di.items() if isinstance(di, dict) else enumerate(di)):
                if isinstance(value, (dict, list)):
                    callback(value, decode)
                elif isinstance(value, str):
                    tmp = decode(value)
                    if tmp is not None:
                        di[key] = tmp

        callback(self.data, decode_eval)
        callback(self.data, decode_format)

        def insert_dict(di, key, obj):
            toret = {}
            for k, v in di.items():
                if k == key:
                    toret.update(obj)
                else:
                    toret[k] = v
            return toret

        def insert_list(li, ind, obj):
            return li[:ind] + obj + li[ind:]

        def walk(di):
            for key, value in list(di.items() if isinstance(di, dict) else enumerate(di)):
                yield key, value

        def callback_import(di):

            def add_colon(tmp):
                if '::' not in tmp:
                    return tmp + '::'
                return tmp

            for key, value in walk(di):
                if key == self._key_import:  # dictionary
                    imports = self.search(add_colon(value))
                    if not isinstance(imports, list): imports = [imports]
                    add = {}
                    for imp in imports:
                        if not isinstance(imp, dict):
                            raise ValueError('Imported {} must be dictionaries'.format(value))
                        add.update(imp)
                    di = insert_dict(di, key, add)
                elif not isinstance(di, dict) and isinstance(value, dict) and set(value.keys()) == {self._key_import}:
                    imports = self.search(add_colon(value[self._key_import]))
                    if not isinstance(imports[0], list): imports = [imports]
                    add = []
                    for imp in imports:
                        if not isinstance(imp, list):
                            raise ValueError('Imported {} must be lists'.format(value))
                        add += imp
                    di = insert_list(di, key, add)
                if isinstance(value, (dict, list)):
                    di[key] = callback_import(value)
            return di

        self.data = callback_import(self.data)

    def search(self, namespaces, delimiter=None, fn=None):
        if isinstance(namespaces, str):
            if fn is None:
                try:
                    fn, namespaces = namespaces.split('::', 1)
                except ValueError:
                    pass
            if delimiter is None:
                from .base import namespace_delimiter as delimiter
            isscalar = ',' not in namespaces
            namespaces = [namespace.split(delimiter) for namespace in namespaces.split(',')]
        else:
            isscalar = len(namespaces) == 0 or np.ndim(namespaces[0]) == 0
            if isscalar:
                namespaces = [namespaces]
        if fn is None:
            d = self
        else:
            if self.base_dir is not None:
                fn = os.path.join(self.base_dir, fn)
            d = BaseConfig(fn)

        def search(d, namespaces):
            for namespace in namespaces:
                d = d[namespace]
            return d

        toret = [search(d, ns) for ns in namespaces]
        if isscalar:
            return toret[0]
        return toret

    def update_from_namespace(self, string, value, inherit_type=True, delimiter=None):
        if delimiter is None:
            from .base import namespace_delimiter as delimiter
        namespaces = string.split(delimiter)
        namespaces, basename = namespaces[:-1], namespaces[-1]
        d = self.search(namespaces)
        if inherit_type and basename in d:
            d[basename] = type(d[basename])(value)
        else:
            d[basename] = value

    def __copy__(self):
        import copy
        new = super(BaseConfig, self).__copy__()
        new.data = self.data.copy()
        for name in self._attrs:
            if hasattr(self, name):
                setattr(new, name, copy.copy(getattr(self, name)))
        return new

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def update(self, *args, **kwargs):
        super(BaseConfig, self).update(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], self.__class__):
            self.__dict__.update({name: value for name, value in args[0].__dict__.items() if name != 'data'})

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def select(self, keys=None):
        toret = self.copy()
        if keys is not None:
            for key in list(toret.keys()):
                if key not in keys:
                    del toret[key]
        return toret

    def __eq__(self, other):
        return type(other) == type(self) and deep_eq(self.data, other.data)

    def write(self, fn):
        self.log_info('Saving {}.'.format(fn))
        utils.mkdir(os.path.dirname(fn))
        data = utils.dict_to_yaml(self.data)

        def list_rep(dumper, data):
            return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

        yaml.add_representer(list, list_rep)

        utils.mkdir(os.path.dirname(fn))
        with open(fn, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
