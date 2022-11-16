import numpy as np

from . import utils

from cosmoprimo.cosmology import BaseEngine, BaseSection, CosmologyError
from cosmoprimo.interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D


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


class ExternalEngine(BaseEngine):

    def __init__(self, *args, **kwargs):
        super(ExternalEngine, self).__init__(*args, **kwargs)

    @classmethod
    def get_requires(cls, *requires):

        def merge_dict(d1, d2):
            toret = d1.copy()
            for name, value in d2.items():
                if name in d1:
                    if isinstance(d1[name], dict) and isinstance(value, dict):
                        toret[name] = merge_dict(d1[name], value)
                    else:
                        toret[name] = _make_list(d1[name], isinst=(list,)) + _make_list(value, isinst=(list,))
                else:
                    toret[name] = value
            return toret

        toret = {}
        for req in requires: toret = merge_dict(toret, req or {})
        requires = toret

        def concatenate(arrays):
            arrays = _make_list(arrays)
            return np.unique(np.concatenate([np.atleast_1d(a) for a in arrays], axis=0))

        for section, names in requires.items():
            for name, attrs in names.items():
                if section == 'background':
                    attrs['z'] = concatenate(attrs['z'])
                if section == 'primordial':
                    attrs['k'] = concatenate(attrs['k'])
                if section == 'fourier':
                    if name == 'pk_interpolator':
                        attrs['of'] = [tuple(_make_list(of, length=2)) for of in attrs['of']]
                        for a in ['z', 'k']: attrs[a] = concatenate(attrs[a])
                        attrs['non_linear'] = attrs.get('non_linear', False)
        return requires
