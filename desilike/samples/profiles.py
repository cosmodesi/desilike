"""Definition of :class:`Profiles`, to hold products of likelihood profiling."""

import os
from collections import UserDict

import numpy as np

from desilike.mpi import CurrentMPIComm
from desilike.utils import BaseClass
from desilike.parameter import Parameter, ParameterArray, ParameterCollection, BaseParameterCollection, ParameterCovariance, Samples, is_parameter_sequence
from desilike import LikelihoodFisher, mpi
from . import utils


class ParameterBestFit(Samples):

    """
    Class holding parameter best fits (in practice, :class:`Samples` with a log-posterior).

    Parameter arrays can be accessed (and updated) as for a dictionary:

    .. code-block:: python

        bestfit = ParameterBestFit([np.ones(10), np.zeros(10), np.linspace(0., 1., 10)], params=['a', 'b', 'logposterior'])
        di = bestfit.choice(index='argmax')  # dictionary of parameter values for best fit with larger log-posterior.
        index = bestfit.logposterior.argmax()
        assert {param.name: bestfit[param][index] for param in bestfit.params()} == di

    """

    _attrs = Samples._attrs + ['_logposterior', '_loglikelihood', '_logprior']

    def __init__(self, *args, logposterior=None, loglikelihood=None, logprior=None, **kwargs):
        """
        Initialize :class:`ParameterBestFit`.

        Parameters
        ----------
        data : list, dict, Samples
            Can be:

            - list of :class:`ParameterArray`, or :class:`np.ndarray` if list of parameters
              (or :class:`ParameterCollection`) is provided in ``params``
            - dictionary mapping parameter to array

        params : list, ParameterCollection
            Optionally, list of parameters.

        logposterior : str, default='logposterior'
            Name of log-posterior in ``data``.

        loglikelihood : str, default='loglikelihood'
            Name of log-likelihood in ``data``.

        logprior : str, default='logprior'
            Name of log-prior in ``data``.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        super(ParameterBestFit, self).__init__(*args, **kwargs)
        for _name in self._attrs[-3:]:
            name = _name[1:]
            value = locals()[name]
            if getattr(self, _name, None) is None or value is not None:  # set only if not previously set, or new value are provided
                setattr(self, _name, name if value is None else str(value))
            value = getattr(self, _name)
            if value in self:
                self[value].param.update(derived=True)

    def __setstate__(self, state):
        # Backward-compatibility
        for name in self._attrs[-3:]:
            state.setdefault(name, name[1:])
        super(ParameterBestFit, self).__setstate__(state)

    @property
    def logposterior(self):
        """Log-posterior."""
        if self._logposterior not in self:
            self[Parameter(self._logposterior, derived=True, latex=utils.outputs_to_latex(self._logposterior))] = np.zeros(self.shape, dtype='f8')
        return self[self._logposterior]

    @logposterior.setter
    def logposterior(self, item):
        """Set log-posterior."""
        self[Parameter(self._logposterior, derived=True, latex=utils.outputs_to_latex(self._logposterior))] = item

    @property
    def chi2min(self):
        r"""Minimum :math:`\chi^{2}` (= -2 * :attr:`logposterior.max()`)"""
        return -2. * float(self.logposterior.max())

    def choice(self, index='argmax', params=None, return_type='dict', **kwargs):
        """
        Return parameter best fit(s).

        Parameters
        ----------
        index : str, default='argmax'
            'argmax' to return best fit (as defined by the point with maximum log-posterior in the samples).

        params : list, ParameterCollection, default=None
            Parameters to compute best fit for. Defaults to all parameters.

        return_dict : default='dict'
            'dict' to return a dictionary mapping parameter names to best fit;
            'nparray' to return an array of parameter best fits;
            ``None`` to return a :class:`ParameterBestFit` instance with a single value.

        **kwargs : dict
            Optional arguments passed to :meth:`params` to select params to return, e.g. ``varied=True, derived=False``.

        Returns
        -------
        toret : dict, array, ParameterBestFit
        """
        if params is None:
            params = self.params(**kwargs)
        if isinstance(index, str) and index == 'argmax':
            index = np.unravel_index(self.logposterior.argmax(), self.shape)
        if not isinstance(index, tuple):
            index = (index,)
        di = {str(param): self[param][index] for param in params}
        if return_type == 'dict':
            return di
        if return_type == 'nparray':
            return np.array(list(di.values()))
        toret = self.copy()
        isscalar = all(np.ndim(ii) == 0 for ii in index)
        toret.data = []
        for param, value in di.items():
            value = np.asarray(value)
            toret.data.append(self[param].clone(value=value[(None,) * self.ndim + (Ellipsis,)] if isscalar else value))
        return toret


class ParameterContour(BaseParameterCollection):

    """Class holding parameter 2D contours (in practice, :class:`BaseParameterCollection` indexed by two parameters)."""

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
        return dict.fromkeys(toret)  # rather than tuple, as (a, b) <-> (b, a)

    @staticmethod
    def _get_param(items):
        return (items[0].param, items[1].param)

    def __init__(self, data=None, params=None, attrs=None):
        """
        Initialize :class:`ParameterContours`.

        Parameters
        ----------
        data : list, dict, ParameterContours
            Can be:

            - list of tuples of arrays if list of parameters
              (or :class:`ParameterCollection`) is provided in ``params``
            - dictionary mapping tuple of parameters to tuple of arrays

        params : list, ParameterCollection
            Optionally, list of tuple of parameters.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        self.attrs = dict(attrs or {})
        self.data = []
        if params is not None:
            if len(params) != len(data):
                raise ValueError('Provide as many parameters as arrays')
            for param, value in zip(params, data):
                self[param] = value
        else:
            super(ParameterContour, self).__init__(data=data, attrs=attrs)

    def __getitem__(self, name):
        """
        Return contour corresponding to parameters ``name``.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.
        """
        try:
            return self.data[name]
        except TypeError:
            name = self._get_name(name)
            items = self.data[self.index(name)]
            item_names = [item.param.name for item in items]
            return tuple(items[item_names.index(name)] for name in name)  # reorder given input names

    def __setitem__(self, name, item):
        """
        Update parameter contour in collection.

        Parameters
        ----------
        name : tuple, list of parameters
            Parameter names.

        item : array, ParameterArray
            Parameter array (with two columns).
        """
        if not is_parameter_sequence(item):
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
        """Set parameter contour in collection if not already in it."""
        if not is_parameter_sequence(item) or not all(isinstance(it, ParameterArray) for it in item):
            raise TypeError('{} is not a tuple of {}.'.format(item, ParameterArray.__call__.__name__))
        item = tuple(item)
        if item not in self:
            self.set(item)

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {'data': [tuple(item.__getstate__() for item in items) for items in self]}
        for name in self._attrs:
            # if hasattr(self, name):
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        BaseClass.__setstate__(self, state)
        self.data = [tuple(ParameterArray.from_state(item) for item in items) for items in state['data']]

    def params(self, **kwargs):
        """Return tuple of parameters."""
        self = self._select(**kwargs)
        return tuple(ParameterCollection([self._get_param(item)[i] for item in self]) for i in range(2))

    def names(self, **kwargs):
        """Return tuple of parameter names in collection."""
        return [tuple(param.name for param in params) for params in self.params(**kwargs)]

    def basenames(self, **kwargs):
        """Return tuple of base parameter names in collection."""
        return [tuple(param.basename for param in params) for params in self.params(**kwargs)]

    def __repr__(self):
        """Return string representation, including shape and columns."""
        return '{}(params={})'.format(self.__class__.__name__, self.params())

    @classmethod
    @CurrentMPIComm.enable
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        """Broadcast input contours ``value`` from rank ``mpiroot`` to other processes."""
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
        """Send ``self`` to rank ``dest``."""
        state = self.__getstate__()
        state['data'] = [tuple(array['param'] for array in arrays) for arrays in state['data']]
        mpicomm.send(state, dest=dest, tag=tag)
        for arrays in self:
            for array in arrays:
                mpi.send(array, dest=dest, tag=tag, mpicomm=mpicomm)

    @classmethod
    @CurrentMPIComm.enable
    def recv(cls, source=mpi.ANY_SOURCE, tag=mpi.ANY_TAG, mpicomm=None):
        """Receive contours from rank ``source``."""
        state = mpicomm.recv(source=source, tag=tag)
        for ivalue, params in enumerate(state['data']):
            state['data'][ivalue] = tuple({'value': mpi.recv(source, tag=tag, mpicomm=mpicomm), 'param': params[i]} for i in range(2))
        return cls.from_state(state)

    @classmethod
    @CurrentMPIComm.enable
    def sendrecv(cls, value, source=0, dest=0, tag=0, mpicomm=None):
        """Send samples from rank ``source`` to rank ``dest`` and receive them here."""
        if dest == source:
            return value.copy()
        if mpicomm.rank == source:
            value.send(dest=dest, tag=tag, mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == dest:
            toret = cls.recv(source=source, tag=tag, mpicomm=mpicomm)
        return toret


class MetaClass(type(BaseClass), type(UserDict)):

    pass


class ParameterContours(BaseClass, UserDict, metaclass=MetaClass):

    def __init__(self, data=None, **kwargs):
        self.data = dict(data or {})
        for name, value in self.items():
            self[name] = ParameterContour(value, **kwargs)

    @property
    def levels(self):
        return list(self.data)

    def deepcopy(self):
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)

    def update(self, other):
        """Update contours."""
        for cl, contour in other.items():
            contour = ParameterContour(contour)
            if cl in self.data:
                self[cl].update(contour)
            else:
                self[cl] = contour

    def clone(self, *args, **kwargs):
        """Clone, i.e. copy and optionally update."""
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {name: value.__getstate__() for name, value in self.items()}
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self.data = {name: ParameterContour.from_state(state[name]) for name in state}

    @classmethod
    @CurrentMPIComm.enable
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        """Broadcast input contours ``value`` from rank ``mpiroot`` to other processes."""
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            for name in state: state[name] = None
        state = mpicomm.bcast(state, root=mpiroot)
        for name in state:
            if mpicomm.bcast(name in value if mpicomm.rank == mpiroot else None, root=mpiroot):
                state[name] = ParameterContour.bcast(value.get(name) if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot).__getstate__()
            else:
                del state[name]
        return cls.from_state(state)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        if not isinstance(other, ParameterContours):
            return NotImplemented
        return self.levels == other.levels and all(other.get(name, None) == self.get(name, None) for name in self.data)


class ParameterProfiles(Samples):

    """
    Class holding parameter 1D profiles.

    Parameter arrays can be accessed (and updated) as for a dictionary:

    .. code-block:: python

        profile = ParameterProfiles([np.array([np.ones(10), np.linspace(0., 1., 10)]).T], params=['a'])
        profile['a']  # 2D array, profile['a'][:, 0] is 'a'-values, profile['a'][: 1] is the corresponding log-posterior

    """

    def choice(self, index='argmax', params=None, return_type='dict', **kwargs):
        """
        Return parameter best fit(s).

        Parameters
        ----------
        index : str, default='argmax'
            'argmax' to return best fit (as defined by the point with maximum log-posterior in the samples).

        params : list, ParameterCollection, default=None
            Parameters to compute best fit for. Defaults to all parameters.

        return_dict : default='dict'
            'dict' to return a dictionary mapping parameter names to best fit;
            'nparray' to return an array of parameter best fits;
            ``None`` to return a :class:`ParameterProfiles` instance with a single value.

        **kwargs : dict
            Optional arguments passed to :meth:`params` to select params to return, e.g. ``varied=True, derived=False``.

        Returns
        -------
        toret : dict, array, ParameterProfiles
        """
        if params is None:
            params = self.params(**kwargs)
        if isinstance(index, str) and index == 'argmax':
            index = [self[param][:, 1].argmax() for param in params]
        if len(index) != len(params):
            raise ValueError('Provide as many indices as params')
        di = {str(param): self[param][ii, 0] for param, ii in zip(params, index)}
        if return_type == 'dict':
            return di
        if return_type == 'nparray':
            return np.array(list(di.values()))
        toret = self.copy()
        toret.data = []
        for ii, (param, value) in zip(index, di.items()):
            value = np.asarray(value)
            isscalar = np.ndim(ii) == 0
            toret.data.append(self[param].clone(value=value[None, ...] if isscalar else value))
        return toret


class ParameterGrid(ParameterBestFit):

    pass


class Profiles(BaseClass):
    r"""
    Class holding results of posterior profiling.

    Attributes
    ----------
    start : Samples
        Initial parameter values.

    bestfit : ParameterBestFit
        Best fit parameters.

    error : Samples
        Parameter parabolic errors.

    covariance : ParameterCovariance
        Parameter covariance at best fit.

    interval : Samples
        Lower and upper errors corresponding to :math:`\Delta \chi^{2} = 1`.

    profile : Samples
        Parameter 1D profiles.

    contour : ParameterContours
        Parameter 2D contours.
    """
    _attrs = {'start': Samples, 'bestfit': ParameterBestFit, 'error': Samples, 'covariance': ParameterCovariance,
              'interval': Samples, 'profile': ParameterProfiles, 'contour': ParameterContours, 'grid': ParameterGrid}

    def __init__(self, attrs=None, **kwargs):
        """
        Initialize :class:`Profiles`.

        Parameters
        ----------
        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.

        **kwargs : dict
            Name and attributes; pass e.g. ``bestfit=...``.
        """
        self.attrs = attrs or {}
        self.set(**kwargs)

    def set(self, **kwargs):
        """Set attributes; pass e.g. ``bestfit=...``"""
        for name, cls in self._attrs.items():
            if name in kwargs:
                item = cls(kwargs[name])
                setattr(self, name, item)

    def get(self, *args, **kwargs):
        """Access attribute by name."""
        return getattr(self, *args, **kwargs)

    def params(self, *args, **kwargs):
        return self.start.params(*args, **kwargs)

    def __contains__(self, name):
        """Has this attribute?"""
        return hasattr(self, name)

    def __copy__(self):
        """Shallow copy."""
        new = super(Profiles, self).__copy__()
        import copy
        for name in ['attrs'] + list(self._attrs.keys()):
            if name in self:
                setattr(new, name, copy.copy(getattr(new, name)))
        return new

    def deepcopy(self):
        """Deep copy."""
        import copy
        return copy.deepcopy(self)

    def update(self, other):
        """Update ``self`` attributes with ``other`` :class:`Profiles` attributes (including :attr:`attrs`)."""
        self.attrs.update(other.attrs)
        for name in other._attrs:
            if name in other:
                if name in self and name != 'covariance':
                    self.get(name).update(other.get(name))
                else:
                    self.set(**{name: other.get(name)})

    def clone(self, *args, **kwargs):
        """Clone, i.e. copy and update."""
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def items(self):
        """Return list of tuples (name, attribute)."""
        toret = []
        for name in self._attrs.keys():
            try:
                toret.append((name, getattr(self, name)))
            except AttributeError:
                pass
        return toret

    def choice(self, index='argmax'):
        """
        Return profiles restricted to best fit(s).

        Parameters
        ----------
        index : str, default='argmax'
            'argmax' to return profiles for best fit (as defined by the point with maximum log-posterior in the samples).

        Returns
        -------
        toret : dict, array, ParameterBestFit
        """
        if isinstance(index, str) and index == 'argmax':
            index = self.bestfit.logposterior.argmax()
        toret = self.deepcopy()
        toret.bestfit = self.bestfit.choice(index=index, return_type=None)
        try:
            toret.error = self.error[index]
        except AttributeError:
            pass
        return toret

    @classmethod
    def concatenate(cls, *others, **kwargs):
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
                raise ValueError('Cannot concatenate two profiles if they do not have same attributes.')
        for name in attrs:
            setattr(new, name, new._attrs[name].concatenate(*[other.get(name) for other in others], **kwargs))
        notconcatenable_attrs = [attr for attr in new._attrs.keys() if attr not in concatenable_attrs]
        best = {'covariance': np.inf}
        for other in others[::-1]:
            for attr in notconcatenable_attrs:
                tmp = other.get(attr, None)
                if tmp is not None:
                    tmp = tmp.deepcopy()
                    if attr == 'covariance' and np.isfinite(tmp._value).all():
                        bestfit = other.get('bestfit', None)
                        if bestfit is not None and bestfit.chi2min < best[attr]:  # select the covariance at bestfit
                            setattr(new, attr, tmp)
                            best[attr] = bestfit.chi2min
                    else:
                        setattr(new, attr, tmp)
        return new

    def extend(self, other):
        """Extend profiles with ``other``."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        state['attrs'] = self.attrs
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self.attrs = state.get('attrs', {})
        for name, cls in self._attrs.items():
            if name in state:
                setattr(self, name, cls.from_state(state[name]))

    def to_fisher(self, params=None, **kwargs):
        """
        Return a Fisher, centered on :attr:`bestfit.choice`, with covariance matrix :attr:`covariance`.

        Parameters
        ----------
        params :  list, ParameterCollection, default=None
            Parameters to return Fisher for. Defaults to all parameters.

        ddof : int, default=1
            Number of degrees of freedom.

        **kwargs : dict
            Arguments for :meth:`bestfit.choice`, giving the mean of the output Fisher likelihood.

        Returns
        -------
        fisher : LikelihoodFisher
        """
        precision = self.covariance.to_precision(params=params, return_type=None)
        params = precision._params
        mean = self.bestfit.choice(params=params, return_type='nparray', **kwargs)
        return LikelihoodFisher(center=mean, params=params, offset=self.bestfit.logposterior.max(), hessian=-precision.view(params=params, return_type='nparray'), with_prior=True)

    def to_getdist(self, params=None, label=None, ignore_limits=True):
        """
        Return a GetDist Gaussian distribution, centered on :attr:`bestfit.choice`, with covariance matrix :attr:`covariance`.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to share to GetDist. Defaults to all parameters.

        label : str, default=None
            Name for GetDist to use for this distribution.

        ignore_limits : bool, default=True
            GetDist does not seem to be able to integrate over distribution if bounded;
            so drop parameter limits.

        Returns
        -------
        samples : getdist.gaussian_mixtures.MixtureND
        """
        return self.to_fisher(params=params).to_getdist(label=label, ignore_limits=ignore_limits)

    def to_stats(self, params=None, quantities=None, sigfigs=2, tablefmt='latex_raw', fn=None):
        """
        Export summary profiling quantities.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to export quantities for. Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['bestfit', 'error', 'interval']``.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : str, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.
            If 'list', return table as list of list of strings, and headers.
            If 'list_latex', return table as list of list of latex strings, and headers.

        fn : str, default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        tab : str
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
        is_latex = 'latex' in tablefmt
        argmax = self.bestfit.logposterior.argmax()

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs, positive_sign='u')[1:]
            if is_latex: return '${{}}_{{{}}}^{{{}}}$'.format(low, up)
            return '{}/{}'.format(low, up)

        for param in params:
            row = []
            if is_latex: row.append(param.latex(inline=True))
            else: row.append(str(param.name))
            row.append(str(param.varied))
            if param in getattr(self, 'error', []):
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
        chi2min = -2. * self.bestfit.logposterior[argmax]
        chi2min_str = '{:.2f}'.format(chi2min)
        ndof = self.bestfit.attrs.get('ndof', None)
        if ndof is not None:
            ndof_str = '{:d}'.format(ndof)
        size, nvaried = [self.bestfit.attrs.get(name, None) for name in ['size', 'nvaried']]
        if size is not None and nvaried is not None:
            if ndof is None: ndof = size - nvaried
            ndof_str = '({:d} - {:d})'.format(size, nvaried)

        chi2min_str = '/ {0} = {1} / {2:d} = {3:.2f}'.format(ndof_str, chi2min_str, ndof, chi2min / ndof) if ndof is not None else '= {}'.format(chi2min_str)
        headers.append((r'$\chi^{{2}} {}$' if is_latex else 'chi2 {}').format(chi2min_str))
        headers.append('varied')
        headers += [quantity.replace('_', ' ') for quantity in quantities]
        if 'list' in tablefmt:
            return data, quantities
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
        """Broadcast input profiles ``value`` from rank ``mpiroot`` to other processes."""
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            for name in cls._attrs: state[name] = None
        state = mpicomm.bcast(state, root=mpiroot)
        for name, acls in cls._attrs.items():
            if mpicomm.bcast(name in value if mpicomm.rank == mpiroot else None, root=mpiroot):
                state[name] = acls.bcast(value.get(name) if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot).__getstate__()
            else:
                del state[name]
        return cls.from_state(state)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return all(other.get(name, None) == self.get(name, None) for name in self._attrs)
