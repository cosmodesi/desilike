import numpy as np

from . import utils

from cosmoprimo.cosmology import Cosmology, BaseEngine, BaseSection, CosmologyError
from cosmoprimo.interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D
from cosmoprimo.utils import flatarray, addproperty


def _make_list(li, length=None, isinst=(list, tuple)):
    if isinstance(li, isinst):
        toret = list(li)
    else:
        toret = [li]
    if length is not None:
        if len(toret) == 1:
            return [toret[0]] * length
        elif length == len(toret):
            return toret
        raise ValueError('Length must be {:d}'.format(length))
    return toret


def get_default(name='z'):
    return {'z': np.linspace(0., 10., 60), 'k': np.logspace(-6., 2., 500)}[name]


def merge(arrays):
    arrays = _make_list(arrays)
    return np.unique(np.concatenate([np.atleast_1d(a) for a in arrays], axis=0))


def is_external_cosmo(cosmo):
    return isinstance(cosmo, str) and cosmo == 'external'


class BaseExternalEngine(BaseEngine):
    """
    A base cosmoprimo's engine class, to be extended for specific external provider of cosmological calculation.
    Used in desilike's bindings to cosmological inference codes.
    """
    def __init__(self, *args, **kwargs):
        super(BaseExternalEngine, self).__init__(*args, **kwargs)

    @classmethod
    def get_requires(cls, *requires):
        """
        Merge input requirements as a dictionary mapping section to method's name and arguments,
        e.g. 'background': {'comoving_radial_distance': {'z': z}}
        """
        def _merge_dict(d1, d2):
            toret = d1.copy()
            for name, value in d2.items():
                if name in d1:
                    if utils.deep_eq(d1[name], value):
                        pass
                    elif isinstance(d1[name], dict) and isinstance(value, dict):
                        value = _merge_dict(d1[name], value)
                    else:
                        value = _make_list(d1[name], isinst=(list,)) + _make_list(value, isinst=(list,))
                toret[name] = value
            return toret

        toret = {}
        for req in requires: toret = _merge_dict(toret, req or {})
        requires = toret
        requires.setdefault('params', {})

        for section, names in requires.items():
            for name, attrs in names.items():
                if section == 'background':
                    attrs = attrs or {}
                    attrs['z'] = merge(attrs.get('z', get_default('z')))
                if section == 'primordial':
                    attrs = attrs or {}
                    attrs['k'] = merge(attrs.get('k', get_default('k')))
                if section == 'fourier':
                    attrs = attrs or {}
                    if name == 'pk_interpolator':
                        attrs['of'] = list(set([tuple(_make_list(of, length=2)) for of in attrs['of']]))
                        for aname in ['z', 'k']: attrs[aname] = merge(attrs.get(aname, get_default(aname)))
                        attrs['non_linear'] = attrs.get('non_linear', False)
                    if name == 'sigma8_z':
                        attrs['of'] = list(set([tuple(_make_list(of, length=2)) for of in attrs['of']]))
                        for aname in ['z']: attrs[aname] = merge(attrs.get(aname, get_default(aname)))
        return requires
