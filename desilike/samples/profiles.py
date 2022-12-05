"""Definition of :class:`Profiles`, to hold products of likelihood profiling."""

import os

import numpy as np

from desilike.mpi import CurrentMPIComm
from desilike.utils import BaseClass
from desilike.parameter import Parameter, ParameterArray, ParameterCollection, BaseParameterCollection, Samples
from desilike import mpi
from . import utils


class ParameterBestFit(Samples):

    _attrs = Samples._attrs + ['_logposterior']

    def __init__(self, *args, logposterior='logposterior', **kwargs):
        self._logposterior = logposterior
        super(ParameterBestFit, self).__init__(*args, **kwargs)

    @property
    def logposterior(self):
        if self._logposterior not in self:
            self[self._logposterior] = np.zeros(self.shape, dtype='f8')
        return self[self._logposterior]

    def choice(self, index='argmax', params=None, return_type='dict', **kwargs):
        if params is None:
            params = self.params(**kwargs)
        if index == 'argmax':
            index = self.logposterior.argmax()
        di = {str(param): self[param][[index]] for param in params}
        if return_type == 'dict':
            return {k: v[0] for k, v in di.items()}
        if return_type == 'nparray':
            return np.array(list(di.values()))
        toret = self.copy()
        toret.data = [ParameterArray(value, param=value.param) for value in di.values()]
        return toret


class ParameterCovariance(BaseClass):

    """Class that represents a parameter covariance."""

    def __init__(self, covariance, params=None, sizes=None):
        """
        Initialize :class:`ParameterCovariance`.

        Parameters
        ----------
        covariance : array
            2D array representing covariance.

        params : list, ParameterCollection
            Parameters corresponding to input ``covariance``.
        """
        if isinstance(covariance, self.__class__):
            self.__dict__.update(covariance.__dict__)
            return
        self._value = np.atleast_2d(covariance)
        if self._value.ndim != 2:
            raise ValueError('Input covariance must be 2D')
        shape = self._value.shape
        if shape[1] != shape[0]:
            raise ValueError('Input covariance must be square')
        if params is None:
            raise ValueError('Provide covariance parameters')
        self._params = ParameterCollection(params)
        if sizes is None:
            sizes = [1] * len(self._params)
        self._sizes = [int(size) for size in sizes]
        if len(self._sizes) != len(self._params):
            raise ValueError('Provide as many sizes as params')
        if sum(self._sizes) != self._value.shape[0]:
            raise ValueError('Input sizes / number of params must match input covariance shape')

    def params(self, *args, **kwargs):
        return self._params.params(*args, **kwargs)

    def select(self, params=None, **kwargs):
        new = self.copy()
        if params is None:
            params = self._params.select(**kwargs)
        new._value = self.cov(params=params)
        new._params = params
        return new

    def cov(self, params=None, return_type='nparray'):
        """Return covariance matrix for input parameters ``params``."""
        if params is None:
            params = self._params
        isscalar = not utils.is_sequence(params)
        if isscalar:
            params = [params]
        idx = np.array([self._params.index(param) for param in params])
        cumsizes = np.cumsum([0] + self._sizes)
        index = np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx])
        toret = self._value[np.ix_(index, index)]
        if return_type == 'nparray':
            if isscalar:
                toret = toret[0, 0]
            return toret
        return ParameterCovariance(toret, params=[self._params[ii] for ii in idx], sizes=[self._sizes[ii] for ii in idx])

    def invcov(self, params=None):
        """Return inverse covariance matrix for input parameters ``params``."""
        return utils.inv(self.cov(params))

    def corrcoef(self, params=None):
        """Return correlation matrix for input parameters ``params``."""
        return utils.cov_to_corrcoef(self.cov(params=params))

    def __contains__(self, name):
        """Has this parameter?"""
        return name in self._params

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in ['value', 'sizes']: state[name] = getattr(self, '_{}'.format(name))
        state['params'] = self._params.__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self._params = ParameterCollection.from_state(state['params'])
        # Backward-compatibility
        state.setdefault('sizes', [1] * len(self._params))
        for name in ['value', 'sizes']:
            setattr(self, '_{}'.format(name), state[name])

    def __repr__(self):
        """Return string representation of parameter covariance, including parameters."""
        return '{}({})'.format(self.__class__.__name__, self._params)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and other.params() == self.params() and np.all(other._value == self._value)

    @classmethod
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        import mpytools as mpy
        if mpicomm is None:
            mpicomm = mpy.CurrentMPIComm.get()
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['value'] = None
        state = mpicomm.bcast(state, root=mpiroot)
        state['value'] = mpy.bcast(value._value if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return cls.from_state(state)


class ParameterContours(BaseParameterCollection):

    """Class that holds samples drawn from likelihood."""

    _type = None
    _attrs = BaseParameterCollection._attrs

    @classmethod
    def _get_name(cls, items):
        toret = []
        for item in items:
            if isinstance(item, str):
                toret.append(item)
                continue
            if isinstance(item, Parameter):
                param = item
            else:
                param = item.param
            toret.append(param.name)
        return tuple(toret)

    @staticmethod
    def _get_param(items):
        return (items[0].param, items[1].param)

    def __init__(self, data=None, params=None, attrs=None):
        self.attrs = dict(attrs or {})
        self.data = []
        if params is not None:
            if len(params) != len(data):
                raise ValueError('Provide as many parameters as arrays')
            for param, value in zip(params, data):
                self[param] = value
        else:
            super(ParameterContours, self).__init__(data=data, attrs=attrs)

    def __setitem__(self, name, item):
        """
        Update parameter in collection (a parameter with same name must already exist).
        See :meth:`set` to set a new parameter.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        item : Parameter
            Parameter.
        """
        if not utils.is_sequence(item):
            raise TypeError('{} is not a tuple')
        items, params = list(item), []
        for ii, item in enumerate(items):
            if not isinstance(item, ParameterArray):
                try:
                    param = self.data[name][ii].param  # list index
                except TypeError:
                    param = Parameter(name[ii])
                params.append(param)
        if params in self:
            tmp = self[params]
            params = [tmp[ii].param.clone(param) for ii, param in enumerate(params)]
        for ii, param in enumerate(params):
            items[ii] = ParameterArray(item, param)
        item = tuple(items)
        try:
            self.data[name] = item  # list index
        except TypeError:
            item_name = self._get_name(item)
            if self._get_name(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

    def setdefault(self, item):
        """Set parameter ``param`` in collection if not already in it."""
        if not utils.is_sequence(item) or not all(isinstance(it, ParameterArray) for it in item):
            raise TypeError('{} is not a tuple of {}.'.format(item, ParameterArray.__call__.__name__))
        item = tuple(item)
        if item not in self:
            self.set(item)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {'data': [tuple(item.__getstate__() for item in items) for items in self]}
        for name in self._attrs:
            # if hasattr(self, name):
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        BaseClass.__setstate__(self, state)
        self.data = [tuple(ParameterArray.from_state(item) for item in items) for items in state['data']]

    def params(self, **kwargs):
        self = self.select(**kwargs)
        return tuple(ParameterCollection([self._get_param(item)[i] for item in self]) for i in range(2))

    def names(self, **kwargs):
        """Return parameter names in collection."""
        return [tuple(param.name for param in params) for params in self.params(**kwargs)]

    def basenames(self, **kwargs):
        """Return base parameter names in collection."""
        return [tuple(param.basename for param in params) for params in self.params(**kwargs)]

    def __repr__(self):
        """Return string representation, including shape and columns."""
        return '{}(params={})'.format(self.__class__.__name__, self.params())

    @classmethod
    @CurrentMPIComm.enable
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['data'] = [tuple(array['param'] for array in arrays) for arrays in state['data']]
        state = mpicomm.bcast(state, root=mpiroot)
        for ivalue, params in enumerate(state['data']):
            state['data'][ivalue] = tuple({'value': mpi.bcast(value.data[ivalue][i] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot), 'param': params[i]} for i in range(2))
        return cls.from_state(state)

    @CurrentMPIComm.enable
    def send(self, dest, tag=0, mpicomm=None):
        state = self.__getstate__()
        state['data'] = [tuple(array['param'] for array in arrays) for arrays in state['data']]
        mpicomm.send(state, dest=dest, tag=tag)
        for arrays in self:
            for array in arrays:
                mpi.send(array, dest=dest, tag=tag, mpicomm=mpicomm)

    @classmethod
    @CurrentMPIComm.enable
    def recv(cls, source=mpi.ANY_SOURCE, tag=mpi.ANY_TAG, mpicomm=None):
        state = mpicomm.recv(source=source, tag=tag)
        for ivalue, params in enumerate(state['data']):
            state['data'][ivalue] = tuple({'value': mpi.recv(source, tag=tag, mpicomm=mpicomm), 'param': params[i]} for i in range(2))
        return cls.from_state(state)

    @classmethod
    @CurrentMPIComm.enable
    def sendrecv(cls, value, source=0, dest=0, tag=0, mpicomm=None):
        if dest == source:
            return value.copy()
        if mpicomm.rank == source:
            value.send(dest=dest, tag=tag, mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == dest:
            toret = cls.recv(source=source, tag=tag, mpicomm=mpicomm)
        return toret


class Profiles(BaseClass):
    r"""
    Class holding results of likelihood profiling.

    Attributes
    ----------
    start : ParamDict
        Initial parameter values.

    bestfit : ParamDict
        Best fit parameters.

    error : ParamDict
        Parameter parabolic errors.

    interval : ParamDict
        Lower and upper errors corresponding to :math:`\Delta \chi^{2} = 1`.

    covariance : ParameterCovariance
        Parameter covariance at best fit.
    """
    _attrs = {'start': Samples, 'bestfit': ParameterBestFit, 'error': Samples, 'covariance': ParameterCovariance,
              'interval': Samples, 'profile': Samples, 'contour': ParameterContours}

    def __init__(self, attrs=None, **kwargs):
        """
        Initialize :class:`Profiles`.

        Parameters
        ----------
        attrs : dict, default=None
            Other attributes.
        """
        self.attrs = attrs or {}
        self.set(**kwargs)

    def set(self, **kwargs):
        for name, cls in self._attrs.items():
            if name in kwargs:
                item = cls(kwargs[name])
                setattr(self, name, item)

    def get(self, *args, **kwargs):
        """Access attribute by name."""
        return getattr(self, *args, **kwargs)

    def __contains__(self, name):
        """Has this attribute?"""
        return hasattr(self, name)

    def __copy__(self):
        new = super(Profiles, self).__copy__()
        import copy
        for name in ['attrs'] + list(self._attrs.keys()):
            if name in self:
                setattr(new, name, copy.copy(getattr(new, name)))
        return new

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def update(self, other):
        self.attrs.update(other.attrs)
        for name in other._attrs:
            if name in other:
                if name in self:
                    self.get(name).update(other.get(name))
                else:
                    self.set(**{name: other.get(name)})

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def items(self):
        toret = []
        for name in self._attrs.keys():
            try:
                toret.append((name, getattr(self, name)))
            except AttributeError:
                pass
        return toret

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate profiles together.

        Parameters
        ----------
        others : list
            List of :class:`Profiles` instances.

        Returns
        -------
        new : Profiles

        Warning
        -------
        :attr:`attrs` of returned profiles contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others: return cls()
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].copy()
        concatenable_attrs = list(new._attrs.keys())[:3]
        attrs = [name for name in new._attrs if name in new and name in concatenable_attrs]
        for other in others:
            if [name for name in other._attrs if name in other and name in concatenable_attrs] != attrs:
                raise ValueError('Cannot concatenate two profiles if both do not have same attributes.')
        for name in attrs:
            setattr(new, name, new._attrs[name].concatenate(*[other.get(name) for other in others]))
        return new

    def extend(self, other):
        """Extend profiles with ``other``."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        state['attrs'] = self.attrs
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.attrs = state.get('attrs', {})
        for name, cls in self._attrs.items():
            if name in state:
                setattr(self, name, cls.from_state(state[name]))

    def to_stats(self, params=None, quantities=None, sigfigs=2, tablefmt='latex_raw', fn=None):
        """
        Export profiling quantities.

        Parameters
        ----------
        params : list, ParameterCollection
            Parameters to export quantities for.
            Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['bestfit','error','interval']``.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : string, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.

        fn : string default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        tab : string
            Summary table.
        """
        import tabulate
        ref_params = self.bestfit.params()
        if params is None: params = ref_params.select(varied=True)
        else: params = [ref_params[param] for param in params]
        data = []
        allowed_quantities = ['bestfit', 'error', 'interval']
        if quantities is None: quantities = [quantity for quantity in allowed_quantities if quantity in self]
        for quantity in quantities:
            if quantity not in allowed_quantities:
                raise ValueError('Unknown quantity {}.'.format(quantity))
        is_latex = 'latex_raw' in tablefmt
        argmax = self.bestfit.logposterior.argmax()

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs, positive_sign='u')[1:]
            if is_latex: return '${{}}_{{{}}}^{{{}}}$'.format(low, up)
            return '{}/{}'.format(low, up)

        for iparam, param in enumerate(params):
            row = []
            if is_latex: row.append(param.latex(inline=True))
            else: row.append(str(param.name))
            row.append(str(param.varied))
            if param in self.error:
                ref_error = self.error[param][argmax]
            else:
                ref_error = None
            for quantity in quantities:
                value = self.get(quantity)
                if param in value:
                    value = value[param]
                else:
                    row.append('')
                    continue
                if quantity in allowed_quantities[:2]:
                    value = value[argmax]
                    value = utils.round_measurement(value, ref_error if ref_error is not None else abs(value), sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                else:
                    row.append(round_errors(*value))
            data.append(row)
        headers = []
        chi2min = '{:.2f}'.format(-2. * self.bestfit.logposterior[argmax])
        headers.append((r'$\chi^{{2}} = {}$' if is_latex else 'chi2 = {}').format(chi2min))
        headers.append('varied')
        headers += [quantity.replace('_', ' ') for quantity in quantities]
        tab = tabulate.tabulate(data, headers=headers, tablefmt=tablefmt)
        if fn is not None:
            utils.mkdir(os.path.dirname(fn))
            self.log_info('Saving to {}.'.format(fn))
            with open(fn, 'w') as file:
                file.write(tab)
        return tab

    @classmethod
    @CurrentMPIComm.enable
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            for name in cls._attrs: state[name] = None
        state = mpicomm.bcast(state, root=mpiroot)
        for name, acls in cls._attrs.items():
            if mpicomm.bcast(name in value if mpicomm.rank == mpiroot else None, root=mpiroot):
                state[name] = acls.bcast(value.get(name) if mpicomm.rank == mpiroot else None, mpiroot=mpiroot).__getstate__()
            else:
                del state[name]
        return cls.from_state(state)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return all(other.get(name, None) == self.get(name, None) for name in self._attrs)
