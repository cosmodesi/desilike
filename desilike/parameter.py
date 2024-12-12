"""Classes to handle parameters."""

import os
import re
import fnmatch
import copy
import numbers
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from itertools import chain as _chain
from itertools import repeat as _repeat
from itertools import starmap as _starmap
from operator import itemgetter as _itemgetter

import numpy as np
import scipy as sp
from .jax import numpy as jnp
from .jax import scipy as jsp
from .jax import rv_frozen, use_jax, register_pytree_node_class

from .io import BaseConfig
from . import mpi, utils
from .mpi import CurrentMPIComm
from .utils import BaseClass, NamespaceDict, deep_eq, is_path


def decode_name(name, default_start=0, default_stop=None, default_step=1):
    """
    Split ``name`` into strings and allowed index ranges.

    >>> decode_name('a_[-4:5:2]_b_[0:2]')
    ['a_', '_b_'], [range(-4, 5, 2), range(0, 2, 1)]

    Parameters
    ----------
    name : str
        Parameter name, e.g. ``a_[-4:5:2]``.

    default_start : int, default=0
        Range start to use as a default.

    default_stop : int, default=None
        Range stop to use as a default.

    default_step : int, default=1
        Range step to use as a default.

    Returns
    -------
    strings : list
        List of strings.

    ranges : list
        List of ranges.
    """
    name = str(name)
    #replaces = re.finditer(r'\[(-?\d*):(\d*):*(-?\d*)\]', name)
    replaces = re.finditer(r'\[([-+]?\d*):([-+]?\d*):*([-+]?\d*)\]', name)
    strings, ranges = [], []
    string_start = 0
    for ireplace, replace in enumerate(replaces):
        start, stop, step = replace.groups()
        if not start:
            start = default_start
            if start is None:
                raise ValueError('You must provide a lower limit to parameter index')
        else: start = int(start)
        if not stop:
            stop = default_stop
            if stop is None:
                raise ValueError('You must provide an upper limit to parameter index')
        else: stop = int(stop)
        if not step:
            step = default_step
            if step is None:
                raise ValueError('You must provide a step for parameter index')
        else: step = int(step)
        strings.append(name[string_start:replace.start()])
        string_start = replace.end()
        ranges.append(range(start, stop, step))

    strings += [name[string_start:]]

    return strings, ranges


def yield_names_latex(name, latex=None, **kwargs):
    r"""
    Yield parameter name and latex strings with template forms ``[::]`` replaced.

    >>> yield_names_latex('a_[-4:3:2]', latex='\alpha_[-4:5:2]')
    a_-4, \alpha_{-4}
    a_-2, \alpha_{-2}
    a_-0, \alpha_{-0}
    a_2, \alpha_{-2}

    Parameters
    ----------
    name : str
        Parameter name.

    latex : str, default=None
        Latex for parameter.

    kwargs : dict
        Arguments for :func:`decode_name`

    Returns
    -------
    name : str
        Parameter name with template forms ``[::]`` replaced.

    latex : str, None
        If input ``latex`` is ``None``, ``None``.
        Else latex string with template forms ``[::]`` replaced.
    """
    strings, ranges = decode_name(name, **kwargs)

    if not ranges:
        yield strings[0], latex

    else:
        import itertools

        template = '%d'.join(strings)
        if latex is not None:
            latex = latex.replace('[]', '%d')

        for nums in itertools.product(*ranges):
            yield template % nums, latex % nums if latex is not None else latex


def find_names(allnames, name, quiet=True):
    """
    Search parameter name ``name`` in list of names ``allnames``,
    matching template forms ``[::]``;
    return corresponding parameter names.
    Contrary to :func:`find_names_latex`, it does not handle latex strings,
    but can take a list of parameter names as ``name``
    (thus returning the concatenated list of matching names in ``allnames``).

    >>> find_names(['a_1', 'a_2', 'b_1', 'c_2'], ['a_[:]', 'b_[:]'])
    ['a_1', 'a_2', 'b_1']

    Parameters
    ----------
    allnames : list
        List of parameter names (strings).

    name : list, str
        List of parameter name(s) to match in ``allnames``.

    quiet : bool, default=True
        If ``False`` and no match for parameter name was found is ``allnames``, raise :class:`ParameterError`.

    Returns
    -------
    toret : list
        List of parameter names (strings).
    """
    if not utils.is_sequence(allnames):
        allnames = [allnames]

    if utils.is_sequence(name):
        toret = []
        for nn in name: toret += find_names(allnames, nn, quiet=quiet)
        return toret

    error = ParameterError('No match found for {}'.format(name))

    if isinstance(name, re.Pattern):
        pattern = name
        ranges = []
    else:
        #name = fnmatch.translate(name)  # does weird things to -
        name = name.replace('*', '.*?') + '$'  # ? for non-greedy, $ to match end of string
        strings, ranges = decode_name(name)
        pattern = re.compile(r'([-+]?\d*)'.join(strings))
    toret = []
    for paramname in allnames:
        match = re.match(pattern, paramname)
        if match:
            add = True
            nums = []
            for s, ra in zip(match.groups(), ranges):
                idx = int(s)
                nums.append(idx)
                add = idx in ra  # ra not in memory
                if not add: break
            if add:
                toret.append(paramname)
    if not toret and not quiet:
        raise error
    return toret


class ParameterError(Exception):

    """Exception raised when issue with :class:`ParameterError`."""


class Deriv(dict):
    """
    This class encodes derivative orders.
    It is a modification of- :class:`Counter` in https://github.com/python/cpython/blob/main/Lib/collections/__init__.py,
    restricting to positive elements.
    """
    # References:
    #   http://en.wikipedia.org/wiki/Multiset
    #   http://www.gnu.org/software/smalltalk/manual-base/html_node/Bag.html
    #   http://www.demo2s.com/Tutorial/Cpp/0380__set-multiset/Catalog0380__set-multiset.htm
    #   http://code.activestate.com/recipes/259174/
    #   Knuth, TAOCP Vol. II section 4.6.3

    def __init__(self, iterable=None, /, **kwds):
        r"""
        Create a new, empty :class:`Deriv` object.

        >>> c = Deriv()                           # a new, empty derivative object, i.e. zero lag
        >>> c = Deriv(['x', Parameter('x'), 'y']) # a new derivative from the list of parameters w.r.t. derivatives are taken
        >>> c = Deriv({'x': 2, 'y': 1})           # a new derivative from a mapping: :math:`\partial_{x}^{2} \partial y`
        >>> c = Deriv(x=2, y=1)                   # a new derivative from keyword args
        """
        super().__init__()
        if isinstance(iterable, Deriv):  # shortcut, saves ~1e-6 s
            super().update(iterable)
            return
        if iterable is None or isinstance(iterable, (Mapping,)):
            self.update(iterable, **kwds)
        else:
            iterable = (iterable,) if not utils.is_sequence(iterable) else iterable
            if all(isinstance(param, (Parameter, str)) for param in iterable):
                self.update((str(param) for param in iterable), **kwds)
            else:
                raise ValueError('Unable to make Deriv from {}'.format(iterable))

    # ADM changes
    def __setitem__(self, name, item):
        if item > 0:
            super(Deriv, self).__setitem__(name, item)

    def setdefault(self, name, item):
        if item > 0:
            super(Deriv, self).setdefault(name, item)

    def __missing__(self, key):
        """The order of a derivative w.r.t. a parameter not in the :class:`Deriv` is zero."""
        # Needed so that self[missing_item] does not raise KeyError
        return 0

    def total(self):
        """Total derivative order."""
        return sum(self.values())

    def most_common(self, n=None):
        """
        List the ``n`` most common derivatives and their orders from the most
        common to the least.  If n is ``None``, then list all derivative orders.

        >>> Deriv(['x', 'x', 'x', 'y', 'y', 'z', 'z', 't']).most_common(3)
        [('x', 3), ('y', 2), ('z', 2)]
        """
        # Emulate Bag.sortedByCount from Smalltalk
        if n is None:
            return sorted(self.items(), key=_itemgetter(1), reverse=True)

        # Lazy import to speedup Python startup time
        import heapq
        return heapq.nlargest(n, self.items(), key=_itemgetter(1))

    def elements(self):
        """
        Iterator over derivatives repeating each as many times as its order.

        >>> c = Deriv('xxyyzz')
        >>> sorted(c.elements())
        ['x', 'x', 'y', 'y', 'z', 'z']
        """
        # Emulate Bag.do from Smalltalk and Multiset.begin from C++.
        return _chain.from_iterable(_starmap(_repeat, self.items()))

    def update(self, iterable=None, /, **kwds):
        """
        Like :meth:`dict.update` but add derivative orders instead of replacing them.
        Source can be an iterable, a dictionary, or another :class:`Deriv` instance.

        >>> c = Deriv('xy')
        >>> c.update('x')               # add derivatives from another iterable
        >>> d = Deriv('xy')
        >>> c.update(d)
        >>> c['x']                      # 3 'x' in 'xy', 'x', 'xy'
        3
        """
        if iterable is not None:
            if isinstance(iterable, Mapping):
                if self:
                    self_get = self.get
                    for elem, count in iterable.items():
                        self[elem] = count + self_get(elem, 0)
                else:
                    # fast path when counter is empty
                    super().update(iterable)
            else:
                for elem in iterable:
                    self[elem] = self.get(elem, 0) + 1
        # ADM changes
        self._keep_positive()
        if kwds:
            self.update(kwds)

    def __reduce__(self):
        return self.__class__, (dict(self),)

    def __delitem__(self, elem):
        """Like :meth:`dict.__delitem__` but does not raise :class:`KeyError` for missing values."""
        if elem in self:
            super().__delitem__(elem)

    def __repr__(self):
        if not self:
            return f'{self.__class__.__name__}()'
        try:
            # dict() preserves the ordering returned by most_common()
            d = dict(self.most_common())
        except TypeError:
            # handle case where values are not orderable
            d = dict(self)
        return f'{self.__class__.__name__}({d!r})'

    def __eq__(self, other):
        """``True`` if all derivative orders agree. Missing derivatives are treated as zero-order derivatives."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return all(self[e] == other[e] for c in (self, other) for e in c)

    def __le__(self, other):
        """``True`` if all derivative orders in ``self`` are less than those in ``other``."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return all(self[e] <= other[e] for c in (self, other) for e in c)

    def __lt__(self, other):
        """``True`` if all derivative orders in ``self`` are strictly less than those in ``other``."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other):
        """``True`` if all derivative orders in ``self`` are greater than those in ``other``."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return all(self[e] >= other[e] for c in (self, other) for e in c)

    def __gt__(self, other):
        """``True`` if all derivative orders in ``self`` are strictly greater than those in ``other``."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return self >= other and self != other

    def __add__(self, other):
        """
        Add derivative orders.

        >>> Deriv('xxy') + Deriv('xyy')
        Deriv({'x': 3, 'y': 3})
        """
        if not isinstance(other, Deriv):
            return NotImplemented
        result = Deriv()
        for elem, count in self.items():
            newcount = count + other[elem]
            if newcount > 0:
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and count > 0:
                result[elem] = count
        return result

    def _keep_positive(self):
        """Internal method to strip derivatives with a negative or zero order"""
        nonpositive = [elem for elem, count in self.items() if not count > 0]
        for elem in nonpositive:
            del self[elem]
        return self

    def __iadd__(self, other):
        """
        Inplace add from another derivative.

        >>> c = Deriv('xxy')
        >>> c += Deriv('xyy')
        >>> c
        Deriv({'x': 3, 'y': 3})
        """
        for elem, count in other.items():
            self[elem] += count
        return self._keep_positive()


import numpy.lib.mixins

@register_pytree_node_class
class ParameterArray(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, value, param=None, derivs=None, copy=False, dtype=None, **kwargs):
        """
        Initalize :class:`ParameterArray`.

        Parameters
        ----------
        value : array
            Array value.

        param : Parameter, str, default=None
            Parameter.

        derivs : list
            List of derivatives (:class:`Deriv` instances).

        copy : bool, default=False
            Whether to copy input array.

        dtype : dtype, default=None
            If provided, enforce this dtype.

        **kwargs : dict
            Optional arguments for :func:`np.array`.
        """
        if isinstance(value, ParameterArray):
            value = value.value
        if value is not None and (copy or dtype or (not use_jax(value) and not isinstance(value, np.ndarray))):
            value = np.array(value, copy=copy, dtype=dtype, **kwargs)
        self._value = value
        self.param = None if param is None else Parameter(param)
        self._derivs = None if derivs is None else tuple(Deriv(deriv) for deriv in derivs)

    @property
    def value(self):
        return self._value

    @property
    def derivs(self):
        return self._derivs

    @property
    def shape(self):
        return self.value.shape

    def __float__(self):
        return float(self.value)

    def __len__(self):
        return len(self.value)

    def __bool__(self):
        return self.value.__bool__()

    def __iter__(self):
        values = self.value.__iter__()  # to raise TypeError in case of 0d array
        # yield would not raise an error in case of 0d array

        def get(value):
            new = self.__class__(value)
            new.__array_finalize__(self, copy=True)
            return new

        return (get(value) for value in values)

    @property
    def zero(self):
        """Return zero-order derivative."""
        if self.derivs is not None:
            return self[()]
        return self

    @property
    def pndim(self):
        """Number of dimensions of stored parameter, plus 1 if derivatives."""
        return int(self.derivs is not None) + (self.param.ndim if self.param is not None else 0)

    @property
    def andim(self):
        """Number of dimensions of array, minus parameter dimensions and derivatives (if any)."""
        return self.value.ndim - self.pndim

    @property
    def pshape(self):
        """Parameter shape, including derivatives along first dimension (if any)."""
        return self.value.shape[self.andim:]

    @property
    def ashape(self):
        """Array shape, removing parameter shape and derivatives (if any)."""
        return self.value.shape[:self.andim]

    @ashape.setter
    def ashape(self, shape):
        self.value.shape = self.ashape + tuple(shape)

    def __array_finalize__(self, obj, copy=False):
        if obj.derivs is not None and (self.shape[-obj.pndim:] == obj.shape[-obj.pndim:]):
            self._derivs = tuple(Deriv(deriv) for deriv in obj.derivs) if copy else obj.derivs
        if obj.param is not None:
            self.param = Parameter(obj.param) if copy else obj.param

    def __copy__(self):
        return self.clone(value=self.value.copy())

    def copy(self, *args, **kwargs):
        return self.__copy__(*args, **kwargs)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.param, self.derivs, self.value)

    def __array__(self, *args, **kwargs):
        return np.asarray(self._value, *args, **kwargs)

    def __jax_array__(self, *args, **kwargs):
        return jnp.asarray(self._value, *args, **kwargs)

    def __format__(self, *args, **kwargs):
        return self._value.__format__(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # Only authorise operations between arrays of same parameter / derivs
        input_param_derivs = [(input.param, input.derivs) for input in inputs if isinstance(input, self.__class__)]
        input_values = [input.value if isinstance(input, self.__class__) else input for input in inputs]
        if isinstance(out, self.__class__):
            new = self.__class__(getattr(ufunc, method)(*input_values, out=out, **kwargs))
            new.__array_finalize__(self)
            return new
        #if use_jax(self._value, *inputs):
        #    ufunc = getattr(jnp, ufunc.__name__, ufunc)

        new = getattr(ufunc, method)(*input_values, **kwargs)
        if input_param_derivs:
            param, derivs = input_param_derivs[0]
            if any(param_derivs[0] != param for param_derivs in input_param_derivs[1:]):
                param = None
            if any(param_derivs[1] != derivs for param_derivs in input_param_derivs[1:]):
                derivs = None
            if param is not None:
                param = Parameter(param)
            if derivs is not None:
                if (new.shape[-self.pndim:] == self.shape[-self.pndim:]):
                    derivs = tuple(Deriv(deriv) for deriv in derivs)
                else:
                    derivs = None
            new = self.__class__(new, param=param, derivs=derivs)
        return new

    def _isderiv(self, deriv):
        try:
            deriv = Deriv(deriv)
            return deriv, True
        except ValueError:
            return deriv, False

    def isin(self, deriv):
        """Test if input deriv in array."""
        deriv, isderiv = self._isderiv(deriv)
        if isderiv:
            return (self.derivs is not None) and (deriv in self.derivs)
        return np.isin(deriv, self)

    def _index(self, index):
        toret = index
        deriv, isderiv = self._isderiv(index)
        if isderiv:
            if self.derivs is not None:
                try:
                    ideriv = self.derivs.index(deriv)
                except ValueError as exc:
                    raise KeyError('{} is not in computed derivatives: {}'.format(deriv, self.derivs)) from exc
                else:
                    toret = (Ellipsis, ideriv)
                    if self.param is not None:
                        toret += (slice(None),) * self.param.ndim
            elif deriv:
                raise KeyError('Array has no derivatives')
            else:
                toret = Ellipsis
        return toret

    def __getitem__(self, deriv):
        """Derivative w.r.t. parameter 'a' can be obtained (if exists) as array[('a',)]."""
        deriv, isderiv = self._isderiv(deriv)
        #return self.__class__(self.value.__getitem__(self._index(deriv)), param=self.param, derivs=None if isderiv else self.derivs)
        if isderiv:
            return self.__class__(self.value.__getitem__(self._index(deriv)), param=self.param, derivs=None)
        new = self.__class__(self.value.__getitem__(deriv))
        new.__array_finalize__(self, copy=True)
        return new

    def __setitem__(self, deriv, item):
        """Derivative w.r.t. parameter 'a' can be set (if exists) as array[('a',)] = deriv."""
        return self.value.__setitem__(self._index(deriv), item)

    def __setstate__(self, state):
        self._value = state['value']
        self.param = state['param']
        if self.param is not None: self.param = Parameter.from_state(self.param)  # Set the info attribute
        self._derivs = state['derivs']
        if self._derivs is not None:
            self._derivs = tuple(Deriv(deriv) for deriv in self._derivs)

    def __getstate__(self):
        state = {name: getattr(self, name) for name in ['value', 'param', 'derivs']}
        if self.param is not None: state['param'] = self.param.__getstate__()
        if self.derivs is not None: state['derivs'] = [dict(deriv) for deriv in self.derivs]
        return state

    def __getattr__(self, name):
        return object.__getattribute__(self._value, name)

    @classmethod
    def from_state(cls, state):
        """Create :class:`ParameterArray` for state (dictionary)."""
        return cls(state['value'], None if state.get('param', None) is None else Parameter.from_state(state['param']), state.get('derivs', None))

    def clone(self, **kwargs):
        """Clone :class:`ParameterArray`, optionally updating :attr:`value`, :attr:`param` or :attr:`derivs`."""
        state = {name: getattr(self, name) for name in ['value', 'param', 'derivs']}
        state.update(**kwargs)
        return self.__class__(**state)

    def tree_flatten(self):
        return (self.value,), {name: getattr(self, name) for name in ['param', 'derivs']}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


def get_wrapper(func):

    def wrapper(self, *args, **kwargs):
        new = self.__class__(getattr(self.value, func)(*args, **kwargs))
        new.__array_finalize__(self, copy=True)
        return new

    return wrapper


for name in ['ravel', 'reshape']:
    setattr(ParameterArray, name, get_wrapper(name))


class Parameter(BaseClass):

    """Class that represents a parameter."""

    _attrs = ['basename', 'namespace', 'value', 'fixed', 'derived', 'prior', 'ref', 'proposal', 'delta', 'latex', 'depends', 'shape', 'drop']
    _allowed_solved = ['.best', '.marg', '.auto', '.best_not_derived', '.marg_not_derived', '.auto_not_derived', '.prec']
    #_allowed_solved += ['best', 'marg', 'auto', 'best_not_derived', 'marg_not_derived', 'auto_not_derived', 'prec']

    def __init__(self, basename, namespace='', value=None, fixed=None, derived=False, prior=None, ref=None, proposal=None, delta=None, latex=None, shape=(), drop=False):
        """
        Initialize :class:`Parameter`.

        Parameters
        ----------
        basename : str
            Parameter base name (which defines parameter meaning).
            If :class:`Parameter`, update ``self`` attributes.

        namespace : str, default=''
            Parameter namespace (to differentiate several occurences of the same parameter in the same pipeline).

        value : float, default=None
            Default value for parameter.

        fixed : bool, default=None
            Whether parameter is fixed.
            If ``None``, defaults to ``True`` if ``prior`` or ``ref`` is not ``None``, else ``False``.

        derived : bool, str, default=False
            ``True`` if parameter is taken from a calulator's attributes (or :meth:`BaseCalculator.__getstate__` at run time).
            '.best', '.marg', or '.auto' to solve for this parameter (given a Gaussian likelihood),
            respectively taking the best-fit solution, performing analytic marginalization, or choosing
            between these two options depending on whether a profiler ('.best') or a sampler ('.marg') is used.
            When using analytic marginalization, the hessian of the loglikelihood and of the prior is stored (as 'derived' parameters);
            potentially yielding large (in terms of memory space) chains. To circumvent this, you can provide e.g. '.auto_not_derived'.
            '.prec' can be used for linear parameters for which the gradient does not depend on the value of other parameters;
            in this case, the likelihood's precision matrix is marginalized over this parameter at initialization, and the parameter
            is ignored in the profiling / sampling.
            One can also define the value of this parameter as a function of others (e.g. 'a', 'b'), by providing
            e.g. the string '{a} + {b}' (or any other operation; numpy is available with 'np', scipy with 'sp',
            and their jax version with 'jnp' and 'jsp').

        prior : ParameterPrior, dict, default=None
            Prior distribution for parameter, arguments for :class:`ParameterPrior`.

        ref : Prior, dict, default=None
            Reference distribution for parameter, arguments for :class:`ParameterPrior`.
            This is supposed to represent the expected posterior for this parameter.
            If ``None``, defaults to ``prior``.

        proposal : float, default=None
            Proposal uncertainty for parameter.
            If ``None``, defaults to ``ref.std()``.

        delta : float, tuple, detault=None
            Variation for finite-differentiation, w.r.t. ``value``.
            If tuple, (variation below value, variation above value),
            e.g.: ``(0.1, 0.2)``, with ``value = 1``, means a variation range ``(0.9, 1.2)``.

        latex : str, default=None
            Latex string for parameter.

        shape : tuple, default=()
            Parameter shape; typically non-trivial when ``derived`` is ``True``.

        drop : bool, default=False
            If ``True``, this parameter will not be provided to the calculator.
        """
        from . import base
        if isinstance(basename, Parameter):
            self.__dict__.update(basename.__dict__)
            return
        if isinstance(basename, ParameterConfig):
            self.__dict__.update(basename.init().__dict__)
            return
        try:
            basename = dict(basename)
        except (ValueError, TypeError):
            pass
        else:
            if 'name' in basename:
                basename['basename'] = basename.pop('name')
            self.__init__(**basename)
            return
        if namespace is None: self._namespace = ''
        else: self._namespace = str(namespace)
        names = str(basename).split(base.namespace_delimiter)
        self._basename, namespace = names[-1], base.namespace_delimiter.join(names[:-1])
        if namespace:
            if self._namespace: self._namespace = base.namespace_delimiter.join([self._namespace, namespace])
            else: self._namespace = namespace
        self._value = float(value) if value is not None else None
        self._prior = prior if isinstance(prior, ParameterPrior) else ParameterPrior(**(prior or {}))
        if ref is not None:
            self._ref = ref if isinstance(ref, ParameterPrior) else ParameterPrior(**(ref or {}))
        else:
            self._ref = self._prior.copy()
        self._latex = latex
        self._proposal = proposal
        self._delta = delta
        if delta is not None:
            if np.ndim(delta) == 0:
                delta = (delta,) * 2
            self._delta = tuple(delta)
        self._derived = derived
        self._depends = {}
        if isinstance(derived, str):
            if self._derived in self._allowed_solved:
                allowed_dists = ['norm', 'uniform']
                if self._prior.dist not in allowed_dists or self._prior.is_limited():
                    raise ParameterError('Prior must be one of {}, with no limits, to use analytic marginalisation for {}'.format(allowed_dists, self))
            else:
                placeholders = re.finditer(r'\{.*?\}', derived)
                nderived = len(derived)
                for placeholder in placeholders:
                    placeholder = placeholder.group()
                    if placeholder not in derived: continue  # already replaced
                    key = '_' * nderived + '{:d}_'.format(len(self._depends) + 1)
                    assert key not in derived
                    derived = derived.replace(placeholder, key)
                    self._depends[key] = placeholder[1:-1]
                self._derived = derived
        else:
            self._derived = bool(self._derived)
        if fixed is None:
            fixed = prior is None and ref is None and not self.depends
        self._fixed = bool(fixed)
        self._shape = tuple(int(s) for s in (shape if utils.is_sequence(shape) else (shape,)))
        self._drop = bool(drop)
        self.updated = True

    @property
    def size(self):
        """Parameter size, typically non-zero when :attr:`derived` is ``True``."""
        return np.prod(self._shape, dtype='i')

    @property
    def ndim(self):
        """Parameter dimension, typically non-trivial when :attr:`derived` is ``True``."""
        return len(self._shape)

    def eval(self, **values):
        """
        Return parameter value, given all parameter values, e.g. if :attr:`derived` is '{a} + {b}',

        >>> param.eval(a=2., b=3.)
        5.
        """
        if isinstance(self._derived, str) and self._derived not in self._allowed_solved:
            try:
                values = {k: values[n] for k, n in self._depends.items()}
            except KeyError:
                raise ParameterError('Parameter {} is to be derived from parameters {}, as {}, but they are not provided'.format(self, list(self._depends.values()), self.derived))
            return utils.evaluate(self._derived, locals=values)
        return values[self.name]

    @property
    def value(self):
        """Default value for parameter; if not specified, defaults ``ref.center()``."""
        value = self._value
        if value is None:
            try:
                value = self._ref.center()
            except AttributeError as exc:
                raise AttributeError('reference distribution has no center(), probably because it is not proper... provide value argument or proper reference distribution') from exc
        return value

    @property
    def proposal(self):
        """Proposal uncertainty for parameter; if not specified, defaults to ``ref.std()``."""
        proposal = self._proposal
        if proposal is None:
            try:
                proposal = self._ref.std()
            except AttributeError as exc:
                raise AttributeError('reference distribution has no std(), probably because it is not proper... provide proposal argument or proper reference distribution') from exc
        return proposal

    @property
    def delta(self):
        """
        Variation for finite-differentiation;
        e.g.: ``(1., 0.1, 0.2)``, means a variation range ``(0.9, 1.2)``.
        If not specified, defaults to ``(value, 0.1 * proposal, 0.1 * proposal)`` (further limited by prior bounds if any).
        """
        delta = self._delta
        proposal_scale = 1e-1
        if delta is None:
            try:
                proposal = self.proposal
            except AttributeError as exc:
                raise AttributeError('reference distribution has no std(), probably because it is not proper... provide delta argument, or proposal, or proper reference distribution') from exc
            delta = (proposal_scale * proposal, proposal_scale * proposal)
            #center = self.value
            #delta = (min(delta[0], center - self.prior.limits[0]), min(delta[1], self.prior.limits[1] - center))
        if len(delta) == 2:
            delta = (self.value,) + tuple(delta)
        return delta

    @property
    def derived(self):
        """If parameter is derived from others 'a', 'b', return e.g. '{a} * {b}'."""
        if isinstance(self._derived, str) and self._derived not in self._allowed_solved:
            toret = self._derived
            for k, v in self._depends.items():
                toret = toret.replace(k, '{{{}}}'.format(v))
            return toret
        return self._derived

    @property
    def solved(self):
        """Whether parameter is solved, i.e. fixed at best fit or marginalized over."""
        return (not self._fixed) and self._derived in self._allowed_solved

    @property
    def input(self):
        """Whether parameter should be fed as input to calculator."""
        return ((self._derived is False) or isinstance(self._derived, str)) and not self.depends

    @property
    def name(self):
        """Return parameter name, as namespace.basename if :attr:`namespace` is not ``None``, else basename."""
        from . import base
        if self._namespace:
            return base.namespace_delimiter.join([self._namespace, self._basename])
        return self._basename

    def update(self, *args, **kwargs):
        """Update parameter attributes with new arguments ``kwargs``."""
        state = self.__getstate__()
        if len(args) == 1 and isinstance(args[0], self.__class__):
            state.update(args[0].__getstate__())
        elif len(args) == 1 and isinstance(args[0], ParameterConfig):
            state = ParameterConfig(self).clone(args[0]).init().__getstate__()
        elif len(args):
            raise ValueError('Unrecognized arguments {}'.format(args))
        if 'name' in kwargs:
            kwargs['basename'] = kwargs.pop('name')
            kwargs['namespace'] = None
        state.update(kwargs)
        state.pop('updated', None)
        self.__init__(**state)

    def clone(self, *args, **kwargs):
        """Clone parameter, i.e. copy and update."""
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    @property
    def varied(self):
        """Whether parameter is varied (i.e. not fixed)."""
        return (not self._fixed)

    @property
    def limits(self):
        """Parameter limits."""
        return self._prior.limits

    def __copy__(self):
        """Shallow copy."""
        new = super(Parameter, self).__copy__()
        new._depends = copy.copy(new._depends)
        return new

    def deepcopy(self):
        """Deep copy."""
        return copy.deepcopy(self)

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        for key in self._attrs:
            state[key] = getattr(self, '_' + key)
            if key in ['prior', 'ref']:
                state[key] = state[key].__getstate__()
        state['derived'] = self.derived
        state.pop('depends')
        state['updated'] = self.updated
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        state = state.copy()
        updated = state.pop('updated', True)
        # For backward-compatibility
        state.pop('saved', None)
        self.__init__(**state)
        self.updated = updated

    def __repr__(self):
        """Represent parameter as string (name and fixed or varied)."""
        return '{}({}, {})'.format(self.__class__.__name__, self.name, 'fixed' if self._fixed else 'varied')

    def __str__(self):
        """Return parameter as string (name)."""
        return str(self.name)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(deep_eq(getattr(other, '_' + name), getattr(self, '_' + name)) for name in self._attrs)

    def __diff__(self, other):
        toret = {}
        for name in self._attrs:
            self_value = getattr(self, '_' + name)
            other_value = getattr(other, '_' + name)
            if not deep_eq(self_value, other_value):
                toret[name] = (self_value, other_value)
        return toret

    def __hash__(self):
        return hash(str(self))

    def latex(self, namespace=None, inline=False):
        """
        Return latex string for parameter if :attr:`latex` is specified (i.e. not ``None``), else :attr:`name`.

        Parameters
        ----------
        namespace : bool, str, default=None
            If ``False``, no namespace is added to the latex string.
            If ``True``, :attr:`namespace` is turned into a latex string, and added as a subscript.
            If string, add this subscript to the latex string.
            If ``None``, and none of :attr:`namespace` "words" (defined as group of characters separated by ',', ' ', '_', '-')
            are in the current latex string, then same as ``True``; else, same as ``False``.

        inline : bool, default=False
            If ``True``, add '$' around the latex string.

        Returns
        -------
        latex : str
            Latex string.
        """
        auto_namespace = namespace is None
        force_namespace = namespace is True
        provided_namespace = False
        if force_namespace or auto_namespace:
            namespace = str(self._namespace)
        elif namespace is not False:
            namespace = str(namespace)
            provided_namespace = force_namespace = True

        if self._latex is not None:

            def add_namespace(group):
                words = re.split(', |_|-', namespace)  # parse namespace
                for word in words:
                    if word in self._latex and word not in self.basename:
                        return False
                return True

            latex = self._latex
            if namespace and (force_namespace or auto_namespace):
                match1 = re.match('(.*)_(.)$', self._latex)
                match2 = re.match('(.*)_{(.*)}$', self._latex)
                latex_namespace = namespace if provided_namespace else ('\mathrm{%s}' % namespace.replace('\_', '_').replace('_', '\_'))
                for match in [match1, match2, None]:
                    if match is not None:
                        if force_namespace or (auto_namespace and add_namespace(match.group(2))):  # check namespace is not in latex str already
                            latex = r'%s_{%s, %s}' % (match.group(1), match.group(2), latex_namespace)
                        break
                    elif force_namespace or (auto_namespace and add_namespace(namespace)):
                        latex = r'%s_{%s}' % (self._latex, latex_namespace)
            if inline:
                latex = '${}$'.format(latex)
            return latex
        return str(self.name)


def _make_property(name):

    def getter(self):
        return getattr(self, '_' + name)

    return getter


for name in Parameter._attrs:
    if name not in ['value', 'proposal', 'delta', 'derived', 'latex']:
        setattr(Parameter, name, property(_make_property(name)))


class BaseParameterCollection(BaseClass):

    """Base class holding a collection of items identified by parameter."""

    _type = Parameter
    _attrs = ['attrs']

    @classmethod
    def _get_name(cls, item):
        if isinstance(item, str):
            return item
        if isinstance(item, Parameter):
            param = item
        else:
            param = cls._get_param(item)
            if param is None:
                return None
        return str(param.name)

    @classmethod
    def _get_param(cls, item):
        return item

    def __init__(self, data=None, attrs=None):
        """
        Initialize :class:`BaseParameterCollection`.

        Parameters
        ----------
        data : list, tuple, str, dict, ParameterCollection
            Can be:

            - list (or tuple) of items
            - dictionary mapping name to item
            - :class:`BaseParameterCollection` instance

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return

        self.attrs = dict(attrs or {})
        self.data = []
        if data is None:
            return

        if utils.is_sequence(data):
            dd = data
            for item in dd:
                self[self._get_name(item)] = item  # only name is provided

        else:
            for name, item in data.items():
                self[name] = item

    def __setitem__(self, name, item):
        """
        Update parameter in collection.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        item : Parameter
            Parameter.
        """
        if not isinstance(item, self._type):
            raise TypeError('{} is not a {} instance.'.format(item, self._type))
        try:
            self.data[name] = item  # list index
        except TypeError:
            item_name = self._get_name(item)
            if self._get_name(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

    def __getitem__(self, name):
        """
        Return item corresponding to parameter ``name``.

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
            return self.data[self.index(name)]

    def __delitem__(self, name):
        """
        Delete parameter ``name``.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.
        """
        try:
            del self.data[name]
        except TypeError:
            del self.data[self.index(name)]

    def sort(self, key=None):
        """
        Sort (in-place) collection, such that if follows the list of parameter names ``key``.
        If ``None``, no sorting is performed.
        """
        if key is not None:
            self.data = [self[kk] for kk in key]
        else:
            self.data = self.data.copy()
        return self

    def pop(self, name, *args, **kwargs):
        """Remove and return item indexed by ``name``."""
        toret = self.get(name, *args, **kwargs)
        try:
            del self[name]
        except (IndexError, KeyError):
            pass
        return toret

    def get(self, name, *args, **kwargs):
        """
        Return item of parameter name ``name`` in collection.

        Parameters
        ----------
        name : Parameter, str
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
        """
        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = kwargs['default']
        try:
            return self[name]
        except KeyError:
            if has_default:
                return default
            raise KeyError('Parameter {} not found'.format(name))

    def set(self, item):
        """
        Set item in collection.
        If there is already a parameter with same name in collection, replace this stored item by the input one.
        Else, append item to collection.
        """
        try:
            self.data[self.index(item)] = item
        except KeyError:
            self.data.append(item)

    def setdefault(self, item):
        """Set item in collection if not already in it."""
        if not isinstance(item, self._type):
            raise TypeError('{} is not a {} instance.'.format(item, self._type))
        if item not in self:
            self.set(item)

    def index(self, name):
        """
        Return index of parameter ``name``.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        Returns
        -------
        index : int
        """
        return self._index_name(self._get_name(name))

    def _index_name(self, name):
        # get index of parameter name ``name``
        for ii, item in enumerate(self.data):
            if self._get_name(item) == name:
                return ii
        raise KeyError('Parameter {} not found'.format(name))

    def __contains__(self, name):
        """Whether collection contains parameter ``name``."""
        try:
            self._index_name(self._get_name(name))
            return True
        except KeyError:
            return False

    def _select(self, **kwargs):
        toret = self.copy()
        if not kwargs:
            return toret
        toret.clear()
        for item in self:
            param = self._get_param(item)
            match = True
            for key, value in kwargs.items():
                param_value = getattr(param, key)
                if key in ['name', 'basename', 'namespace']:
                    key_match = value is None or bool(find_names([param_value], value))
                else:
                    key_match = deep_eq(value, param_value)
                    if not key_match:
                        try:
                            key_match |= any(deep_eq(v, param_value) for v in value)
                        except TypeError:
                            pass
                match &= key_match
                if not key_match: break
            if match:
                toret.data.append(item)
        return toret

    def select(self, **kwargs):
        """
        Return new collection, after selection of parameters whose attribute match input values::

            collection.select(fixed=True)

        returns collection of fixed parameters.
        If 'name' is provided, consider all matching parameters, e.g.::

            collection.select(varied=True, name='a_[0:2]')

        returns a collection of varied parameters, with name in ``['a_0', 'a_1']``.
        """
        return self._select(**kwargs)

    def params(self, **kwargs):
        """Return :class:`ParameterCollection`, collection of parameters corresponding to items stored in this collection."""
        return ParameterCollection([self._get_param(item) for item in self._select(**kwargs)])

    def names(self, **kwargs):
        """Return parameter names in collection."""
        params = self.params(**kwargs)
        return [param.name for param in params]

    def basenames(self, **kwargs):
        """Return base parameter names in collection."""
        params = self.params(**kwargs)
        return [param.basename for param in params]

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input collections.
        Unique items only are kept.
        """
        if not others: return cls()
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = cls(others[0])
        for other in others[1:]:
            other = cls(other)
            for item in other.data:
                new.set(item)
        return new

    def extend(self, other):
        """
        Extend collection with ``other``.
        Unique items only are kept.
        """
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.names())

    def __len__(self):
        """Collection length, i.e. number of items."""
        return len(self.data)

    def __iter__(self):
        """Iterator on collection."""
        return iter(self.data)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {'data': [item.__getstate__() for item in self.data]}
        for name in self._attrs:
            # if hasattr(self, name):
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(BaseParameterCollection, self).__setstate__(state)
        self.data = [self._type.from_state(item) for item in state['data']]

    def __copy__(self):
        new = super(BaseParameterCollection, self).__copy__()
        for name in ['data'] + self._attrs:
            # if hasattr(self, name):
            setattr(new, name, copy.copy(getattr(new, name)))
        return new

    def clear(self):
        """Empty collection."""
        self.data.clear()
        return self

    def update(self, *args, **kwargs):
        """
        Update collection with new one; arguments can be a :class:`BaseParameterCollection`
        or arguments to instantiate such a class (see :meth:`__init__`).
        """
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
        else:
            other = self.__class__(*args, **kwargs)
        for item in other:
            self.set(item)

    def clone(self, *args, **kwargs):
        """Clone collection, i.e. (shallow) copy and update."""
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def keys(self, **kwargs):
        """Return parameter names."""
        return [self._get_name(item) for item in self._select(**kwargs)]

    def values(self, **kwargs):
        """Return items."""
        return [item for item in self._select(**kwargs)]

    def items(self, **kwargs):
        """Return list of tuples (parameter name, item)."""
        return [(self._get_name(item), item) for item in self._select(**kwargs)]

    def deepcopy(self):
        """Deep copy."""
        return copy.deepcopy(self)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and list(other.params()) == list(self.params()) and all(deep_eq(other_value, self_value) for other_value, self_value in zip(other, self))


class ParameterConfig(NamespaceDict):

    """A convenient object, used internally by the code, to store configuration for a given parameter."""

    def __init__(self, conf=None, **kwargs):

        if isinstance(conf, Parameter):
            conf = conf.__getstate__()
            if conf['namespace'] is None:
                conf.pop('namespace')
            conf.pop('updated', None)

        super(ParameterConfig, self).__init__(conf, **kwargs)
        for name in ['prior', 'ref']:
            if name in self:
                if isinstance(self[name], ParameterPrior):
                    self[name] = self[name].__getstate__()

    def init(self):
        state = self.__getstate__()
        if not isinstance(self.get('namespace', None), str):
            state['namespace'] = None
        for name in ['prior', 'ref']:
            if state.get(name, None) is not None and 'rescale' in state[name]:
                value = copy.copy(state[name])
                rescale = value.pop('rescale')
                if 'scale' in value:
                    value['scale'] *= rescale
                if 'limits' in value:
                    limits = np.array(value['limits'])
                    if np.isfinite(limits).all():
                        center = np.mean(limits)
                        value['limits'] = (limits - center) * rescale + center
                state[name] = value
        return Parameter(**state)

    @property
    def param(self):
        return self.init()

    def update(self, *args, exclude=(), **kwargs):
        other = self.__class__(*args, **kwargs)
        for name, value in other.items():
            if name not in exclude:
                if name in ['prior', 'ref'] and name in self and 'rescale' in value and len(value) == 1:
                    value = {**self[name], 'rescale': value['rescale']}
                self[name] = copy.copy(value)

    def update_derived(self, oldname, newname):
        if self.get('derived', False) and isinstance(self.derived, str) and self.derived not in Parameter._allowed_solved:
            self.derived = self.derived.replace('{{{}}}'.format(str(oldname)), '{{{}}}'.format(str(newname)))

    @property
    def name(self):
        from . import base
        namespace = self.get('namespace', None)
        if isinstance(namespace, str) and namespace:
            return base.namespace_delimiter.join([namespace, self.basename])
        return self.basename

    @name.setter
    def name(self, name):
        from . import base
        names = str(name).split(base.namespace_delimiter)
        if len(names) >= 2:
            self.basename, self.namespace = names[-1], base.namespace_delimiter.join(names[:-1])
        else:
            self.basename = names[0]


class ParameterCollectionConfig(BaseParameterCollection):
    """
    A collection of :class:`ParameterConfig` objects, used internally by the code.
    TODO: As :class:`ParameterConfig`, should be either documented and/or simplified.
    """
    _type = ParameterConfig
    _attrs = ['fixed', 'derived', 'namespace', 'delete', 'wildcard', 'identifier']

    def _get_name(self, item):
        from . import base
        if isinstance(item, str):
            return item.split(base.namespace_delimiter)[-1]
        return getattr(item, self.identifier)

    @classmethod
    def _get_param(cls, item):
        return item.param

    def __init__(self, data=None, identifier='basename', **kwargs):
        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            self.identifier = identifier
            return
        self.identifier = identifier
        if isinstance(data, ParameterCollection):
            dd = data
            data = {getattr(param, self.identifier): ParameterConfig(param) for param in data}
            if len(data) != len(dd):
                raise ValueError('Found different parameters with same {} in {}'.format(self.identifier, dd))
        else:
            data = BaseConfig(data, **kwargs).data

        self.fixed, self.derived, self.namespace, self.delete, self.wildcard = {}, {}, {}, {}, []
        for meta_name in ['fixed', 'varied', 'derived', 'namespace', 'delete']:
            meta = data.pop('.{}'.format(meta_name), {})
            if utils.is_sequence(meta):
                meta = {name: True for name in meta}
            elif not isinstance(meta, dict):
                meta = {meta: True}
            if meta_name in ['fixed', 'varied']:
                self.fixed.update({name: (meta_name == 'fixed' and value) or (meta_name == 'varied' and not value) for name, value in meta.items()})
            else:
                getattr(self, meta_name).update({name: bool(value) for name, value in meta.items()})
        self.data = []
        for name, conf in data.items():
            if isinstance(conf, numbers.Number):
                conf = {'value': conf}
            conf = dict(conf or {})
            latex = conf.pop('latex', None)
            for name, latex in yield_names_latex(name, latex=latex):
                tmp = conf.__getstate__() if isinstance(conf, ParameterConfig) else conf
                tmp = ParameterConfig(name=name, **tmp)
                if latex is not None: tmp.latex = latex
                if '*' in tmp[self.identifier]:
                    self.wildcard.append(tmp)
                    continue
                self.set(tmp)
                #for meta_name in ['fixed', 'derived', 'namespace']:
                #    meta = getattr(self, meta_name)
                #    if meta_name in tmp:
                #        meta.pop(name, None)
                #        meta[name] = tmp[meta_name]
        self._set_meta()

    def _set_meta(self):
        for meta_name in ['fixed', 'derived', 'namespace']:
            meta = getattr(self, meta_name)
            for name in reversed(meta):
                for tmpconf in self.select(**{self.identifier: name}):
                    if meta_name not in tmpconf:
                        tmpconf[meta_name] = meta[name]
        # Wildcard
        for conf in reversed(self.wildcard):
            for tmpconf in self.select(**{self.identifier: conf[self.identifier]}):
                tmpconf.update(conf.clone(tmpconf))

        for conf in self:
            if 'namespace' not in conf:
                conf.namespace = True

    def updated(self, param):
        # Updated with meta?
        paramname = self._get_name(param)
        for meta_name in ['fixed', 'derived', 'namespace']:
            meta = getattr(self, meta_name)
            for name in meta:
                if find_names([paramname], name):
                    return True
        for conf in self.wildcard:
            if find_names([paramname], conf[self.identifier]):
                return True
        return False

    def select(self, **kwargs):
        toret = self.copy()
        if not kwargs:
            return toret
        toret.clear()
        for item in self:
            param = item
            match = True
            for key, value in kwargs.items():
                param_value = getattr(param, key)
                if key in ['name', 'basename', 'namespace']:
                    key_match = value is None or bool(find_names([param_value], value))
                else:
                    key_match = deep_eq(value, param_value)
                    if not key_match:
                        try:
                            key_match |= any(deep_eq(v, param_value) for v in value)
                        except TypeError:
                            pass
                match &= key_match
                if not key_match: break
            if match:
                toret.data.append(item)
        return toret

    def update(self, *args, **kwargs):
        other = self.__class__(*args, **kwargs)

        for name, b in other.delete.items():
            if b:
                for param in self.select(**{self.identifier: name}):
                    del self[param[self.identifier]]

        for meta_name in ['fixed', 'derived', 'namespace']:
            meta = getattr(other, meta_name)
            for name in meta:
                for tmpconf in self.select(**{self.identifier: name}):
                    tmpconf[meta_name] = meta[name]

        for conf in other.wildcard:
            for tmpconf in self.select(**{self.identifier: conf[self.identifier]}):
                self[tmpconf[self.identifier]] = tmpconf.clone(conf, exclude=('basename',))

        for conf in other:
            new = self.pop(conf[self.identifier], ParameterConfig()).clone(conf)
            self.set(new)

        def update_order(d1, d2):
            toret = {name: value for name, value in d1.items() if name not in d2}
            for name, value in d2.items():
                toret[name] = value
            return toret

        for meta_name in ['fixed', 'derived', 'namespace', 'delete']:
            setattr(self, meta_name, update_order(getattr(self, meta_name), getattr(other, meta_name)))
        self.wildcard = self.wildcard + other.wildcard
        self._set_meta()

    def with_namespace(self, namespace=None):
        new = self.deepcopy()
        new._set_meta()
        for name, param in new.items():
            if not isinstance(param.namespace, str) and param.namespace:
                param.namespace = namespace
                for dparam in new:
                    dparam.update_derived(param.basename, param.name)
        return new

    def init(self, namespace=None):
        return ParameterCollection([conf.param for conf in self.with_namespace(namespace=namespace)])

    def set(self, item):
        if not isinstance(item, ParameterConfig):
            item = ParameterConfig(item)
        try:
            self.data[self.index(item)] = item
        except KeyError:
            self.data.append(item)

    def __setitem__(self, name, item):
        if not isinstance(item, ParameterConfig):
            item = ParameterConfig(item)
            item.setdefault(self.identifier, name)
        try:
            self.data[name] = item
        except TypeError:
            item_name = self._get_name(item)
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect name {})'.format(item_name, name))
            self.data[self._index_name(name)] = item


class ParameterCollection(BaseParameterCollection):
    """
    Class holding a collection of parameters.
    It additionally keeps track whether the collection has been updated,
    as used in :class:`BasePipeline`.
    """
    _attrs = ['_updated']

    def __init__(self, data=None, attrs=None):
        """
        Initialize :class:`ParameterCollection`.

        Parameters
        ----------
        data : list, tuple, str, Path, dict, ParameterCollection
            Can be:

            - list (or tuple) of parameters (:class:`Parameter` or dictionary to initialize :class:`Parameter`)
            - path to *yaml* defining a list of parameters
            - dictionary mapping name to parameter
            - :class:`ParameterCollection` instance

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        if is_path(data):
            data = ParameterCollectionConfig(data)

        if isinstance(data, ParameterCollectionConfig):
            data = data.init()

        if isinstance(data, self.__class__):
            self.__dict__.update(data.copy().__dict__)
            return

        self.attrs = dict(attrs or {})
        self.data = []
        self._updated = True
        if data is None:
            return

        if utils.is_sequence(data):
            dd = data
            data = {}
            for name in dd:
                if isinstance(name, Parameter):
                    data[name.name] = name
                elif isinstance(name, dict):
                    data[name['name']] = name
                else:
                    data[name] = {}  # only name is provided

        for name, conf in data.items():
            if isinstance(conf, Parameter):
                self.set(conf)
            else:
                if not isinstance(conf, dict):  # parameter value
                    conf = {'value': conf}
                else:
                    conf = conf.copy()
                latex = conf.pop('latex', None)
                for name, latex in yield_names_latex(name, latex=latex):
                    param = Parameter(basename=name, latex=latex, **conf)
                    self.set(param)

    @property
    def updated(self):
        """Whether the collection (the list of parameters itself of any of these parameters) has just been updated."""
        return self._updated or any(param.updated for param in self.data)

    @updated.setter
    def updated(self, updated):
        """Set the 'updated' status."""
        updated = bool(updated)
        self._updated = updated
        for param in self.data: param.updated = updated

    def __delitem__(self, name):
        """
        Delete parameter ``name``.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.
        """
        self._updated = True
        return super(ParameterCollection, self).__delitem__(name)

    def update(self, *args, name=None, basename=None, **kwargs):
        """
        Update collection with new one.
        To e.g. fix parameters whose name matches the pattern 'a*':

        >>> params.update(name='a*', fixed=True)
        """
        self._updated = True
        if len(args) == 1 and (isinstance(args[0], self.__class__) or utils.is_sequence(args[0])):
            other = self.__class__(args[0])
            self_basenames = self.basenames()
            for item in other:
                if item in self:
                    self[item] = self[item].clone(item)
                elif basename is True and item.basename in self_basenames:
                    index = self_basenames.index(item.basename)
                    self[index] = self[index].clone(item)
                else:
                    self.set(item.copy())
        elif len(args) <= 1:
            list_update = self.names(name=name, basename=basename)
            for meta_name, fixed in zip(['fixed', 'varied'], [True, False]):
                if meta_name in kwargs:
                    meta = kwargs[meta_name]
                    if isinstance(meta, bool):
                        if meta:
                            meta = list_update
                        else:
                            meta = []
                    for name in meta:
                        for name in self.names(name=name):
                            self[name] = self[name].clone(fixed=fixed)
            if 'namespace' in kwargs:
                namespace = kwargs['namespace']
                indices = [self.index(name) for name in list_update]
                oldnames = self.names()
                for index in indices:
                    self.data[index] = self.data[index].clone(namespace=namespace)
                newnames = self.names()
                for index in indices:
                    dparam = self.data[index]
                    for k, v in dparam.depends.items():
                        if v in oldnames: dparam.depends[k] = newnames[oldnames.index(v)]
                names = {}
                for param in self.data: names[param.name] = names.get(param.name, 0) + 1
                duplicates = {basename: multiplicity for basename, multiplicity in names.items() if multiplicity > 1}
                if duplicates:
                    raise ValueError('Cannot update namespace, as following duplicates found: {} in {}'.format(duplicates, self))
        else:
            raise ValueError('Unrecognized arguments {}'.format(args))

    def __add__(self, other):
        """Concatenate two parameter collections."""
        return self.concatenate(self, self.__class__(other))

    def __radd__(self, other):
        if other == 0: return self.copy()
        return self.__class__(other).__add__(self)

    def __sub__(self, other):
        """Subtract two parameter collections."""
        other = self.__class__(other)
        return self.__class__([param for param in self if param not in other])

    def __rsub__(self, other):
        if other == 0: return self.copy()
        return self.__class__(other).__sub__(self)

    def __and__(self, other):
        """Intersection of two parameter collections."""
        return self.__class__([param for param in self if param in other])

    def __rand__(self, other):
        if other == 0: return self.copy()
        return self.__class__(other).__and__(self)

    def params(self, **kwargs):
        """Return a collection of parameters :class:`ParameterCollection`, with optional selection. See :meth:`select`."""
        return self.select(**kwargs)

    def set(self, item):
        """
        Set parameter in collection.
        If there is already a parameter with same name in collection, replace this stored parameter by the input one.
        Else, append parameter to collection.
        """
        self._updated = True
        if not isinstance(item, Parameter):
            item = Parameter(item)
        try:
            self.data[self.index(item)] = item
        except KeyError:
            self.data.append(item)

    def __setitem__(self, name, item):
        """
        Update parameter in collection.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        item : Parameter
            Parameter.
        """
        self._updated = True
        if not isinstance(item, Parameter):
            if not isinstance(item, ParameterConfig):
                try:
                    item = {'basename': name, **item}
                except TypeError:
                    pass
            item = Parameter(item)
        try:
            self.data[name] = item
        except TypeError:
            item_name = self._get_name(item)
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

    def eval(self, **params):
        """
        Return parameter values, given all parameter values, e.g. if ``c.derived`` is '{a} + {b}',

        >>> params.eval(a=2., b=3.)
        {'a': 2., 'b': 3, 'c': 5}

        See :meth:`Parameter.eval`.
        """
        toret = {}
        for param in self:
            try:
                toret[param.name] = param.eval(**params)
            except (ParameterError, KeyError):
                pass
        return toret

    def prior(self, **params):
        """Compute total (log-)prior for input parameter values (except parameters that are solved)."""
        eval_params = self.eval(**params)
        toret = 0.
        for param in self.data:
            if param.varied and (param.depends or (not param.derived)) and param.name in eval_params:

                toret += param.prior(eval_params[param.name])
        return toret


class ParameterPriorError(Exception):

    """Exception raised when issue with prior."""


class ParameterPrior(BaseClass):
    """
    Class that describes a 1D prior distribution.
    TODO: make immutability explicit.

    Parameters
    ----------
    dist : str
        Distribution name.

    rv : rv_continuous
        Random variate.

    attrs : dict
        Arguments used to initialize :attr:`rv`.
    """

    def __init__(self, dist='uniform', limits=None, **kwargs):
        r"""
        Initialize :class:`ParameterPrior`.

        Parameters
        ----------
        dist : str
            Distribution name in :mod:`jax.scipy.stats`, and as a fallback, :mod:`scipy.stats`.

        limits : tuple, default=None
            Tuple corresponding to lower, upper limits.
            ``None`` means :math:`-\infty` for lower bound and :math:`\infty` for upper bound.
            Defaults to :math:`-\infty, +\infty`.

        kwargs : dict
            Arguments for distribution, typically ``loc``, ``scale``
            (mean and standard deviation in case of a normal distribution ``'dist' == 'norm'``).
        """
        if isinstance(dist, ParameterPrior):
            self.__dict__.update(dist.__dict__)
            return

        if limits is None:
            limits = (-np.inf, np.inf)
        limits = list(limits)
        if limits[0] is None: limits[0] = -np.inf
        if limits[1] is None: limits[1] = np.inf
        limits = tuple(limits)
        if limits[1] <= limits[0]:
            raise ParameterPriorError('ParameterPrior range {} has min greater than max'.format(limits))
        self.limits = limits
        self.attrs = dict(kwargs)
        for name, value in self.attrs.items():
            if name in ['loc', 'scale', 'a', 'b']: self.attrs[name] = float(value)

        self.dist = str(dist)
        if self.dist.startswith('trunc'): self.dist = self.dist[5:]
        if self.is_limited():
            dist = dist if dist == 'uniform' else 'trunc{}'.format(dist)
        try:
            dist = getattr(jsp.stats, dist)
        except AttributeError:
            try:
                dist = getattr(sp.stats, dist)
            except AttributeError:
                raise AttributeError('Neither jax.scipy.stats nor scipy.stats have {} for attribute'.format(dist))

        if self.is_limited():
            if self.dist == 'uniform':
                args = (self.limits[0], self.limits[1] - self.limits[0])
                kwargs = {}
            else:
                loc, scale = self.attrs.get('loc', 0.), self.attrs.get('scale', 1.)
                # See notes of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
                limits = tuple((lim - loc) / scale for lim in limits)
                args = limits
        elif self.dist == 'uniform':  # improper prior
            return
        else:
            args = ()
        self.rv = rv_frozen(dist, *args, **kwargs)
        # self.limits = self.rv.support()

    def isin(self, x):
        """Whether ``x`` is within prior, i.e. within limits - strictly positive probability."""
        x = jnp.asarray(x)
        return (self.limits[0] < x) & (x < self.limits[1])

    def __hash__(self):
        return super().__hash__()

    #@jit(static_argnums=[0, 2])
    def logpdf(self, x, remove_zerolag=True):
        """
        Return log-probability density at ``x``.
        If ``remove_zerolag`` is ``True``, remove the maximum log-probability density.
        """
        _jnp = jnp if use_jax(x) else np  # Worth testing if input is jax, as jax incurs huge overheads
        x = _jnp.asarray(x)
        isin = (self.limits[0] <= x) & (x <= self.limits[1])
        # Fast version
        if remove_zerolag:
            if self.dist == 'uniform':
                return _jnp.where(isin, 0, -np.inf)
            if self.dist == 'norm':
                return _jnp.where(isin, - 0.5 * (x - self.attrs['loc'])**2 / self.attrs['scale']**2, -np.inf)

        if not self.is_proper():
            return _jnp.where(isin, 0, -np.inf)

        toret = self.rv.logpdf(x)
        if remove_zerolag:
            loc = self.attrs.get('loc', None)
            if loc is None: loc = np.mean(self.limits)
            toret -= self.rv.logpdf(loc)
        return toret

    def __call__(self, x, remove_zerolag=True):
        return self.logpdf(x, remove_zerolag=remove_zerolag)

    def sample(self, size=None, random_state=None):
        """
        Draw ``size`` samples from prior. Possible only if prior is proper.

        Parameters
        ---------
        size : int, default=None
            Number of samples to draw.
            If ``None``, return one sample (float).

        random_state : int, numpy.random.Generator, numpy.random.RandomState, default=None
            If integer, a new :class:`numpy.random.RandomState` instance is used, seeded with ``random_state``.
            If ``random_state`` is a :class:`numpy.random.Generator` or :class:`numpy.random.RandomState` instance then that instance is used.
            If ``None``, the :class:`numpy.random.RandomState` singleton is used.

        Returns
        -------
        samples : float, array
            Samples drawn from prior.
        """
        if not self.is_proper():
            raise ParameterPriorError('Cannot sample from improper prior')
        return self.rv.rvs(size=size, random_state=random_state)

    def __repr__(self):
        """String representation with distribution name, limits, and attributes (e.g. ``loc`` and ``scale``)."""
        base = self.dist
        if self.is_limited():
            base = '{}[{}, {}]'.format(base, *self.limits)
        return '{}({})'.format(base, self.attrs)

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self.__init__(**state)

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {'dist': self.dist, 'limits': self.limits}
        state.update(self.attrs)
        return state

    def is_proper(self):
        """Whether distribution is proper, i.e. has finite integral."""
        return self.dist != 'uniform' or not np.isinf(self.limits).any()

    def is_limited(self):
        """Whether distribution has (at least one) finite limit."""
        return not np.isinf(self.limits).all()

    def center(self):
        try:
            center = self.loc
        except AttributeError:
            if self.is_limited():
                center = np.mean([lim for lim in self.limits if not np.isinf(lim)])
            else:
                center = 0.
        return center

    def affine_transform(self, loc=0., scale=1.):
        """
        Apply affine transform to the distribution: shifted by ``loc``,
        and dispersion multiplied by ``scale``.
        Useful to e.g. normalize a parameter (together with its prior).
        """
        state = self.__getstate__()
        center = self.center()
        for name, value in state.items():
            if name in ['loc']:
                state[name] = center + loc
            elif name in ['limits']:
                state[name] = tuple((lim - center) * scale + center + loc for lim in value)
            elif name in ['scale']:
                state[name] = value * scale
        return self.from_state(state)

    def __getattr__(self, name):
        """Make :attr:`rv` attributes directly available in :class:`ParameterPrior`."""
        try:
            return getattr(object.__getattribute__(self, 'rv'), name)
        except AttributeError as exc:
            if object.__getattribute__(self, 'dist') == 'uniform':
                raise AttributeError('uniform distribution has no {}'.format(name)) from exc
            attrs = object.__getattribute__(self, 'attrs')
            if name in attrs:
                return attrs[name]
            raise exc

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other, key) == getattr(self, key) for key in ['dist', 'limits', 'attrs'])



def _reshape(array, shape):
    if np.ndim(shape) == 0:
        shape = (shape,)
    shape = tuple(shape)
    try:
        return array.reshape(shape + array.pshape)
    except ValueError as exc:
        raise ValueError('Error with array {}'.format(repr(array))) from exc


@register_pytree_node_class
class Samples(BaseParameterCollection):

    """Class that holds samples, as a collection of :class:`ParameterArray`."""

    _type = ParameterArray
    _attrs = BaseParameterCollection._attrs + ['_derived']
    _derived = []

    def __init__(self, data=None, params=None, attrs=None):
        """
        Initialize :class:`Samples`.

        Parameters
        ----------
        data : list, dict, Samples
            Can be:

            - list of :class:`ParameterArray`, or :class:`np.ndarray` if list of parameters
              (or :class:`ParameterCollection`) is provided in ``params``
            - dictionary mapping parameter to array

        params : list, ParameterCollection
            Optionally, list of parameters.

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
            super(Samples, self).__init__(data=data, attrs=attrs)

    def save(self, filename):
        """Save samples to disk."""
        filename = str(filename)
        state = self.__getstate__()
        for array in state['data']: array['value'] = np.asarray(array['value'])  # could be jax
        state = {'__class__': utils.serialize_class(self.__class__), **state}
        self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        if filename.endswith('.npz'):
            statez = {'others': dict(state)}
            statez['others'].pop('data', None)
            statez['__class__'] = statez['others'].pop('__class__')
            statez['params'] = []
            for iarray, array in enumerate(state['data']):
                statez['data.{:d}'.format(iarray)] = array['value']
                statez['params'].append({key: value for key, value in array.items() if key not in ['value']})
            np.savez(filename, **statez)
        else:
            np.save(filename, state, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """Load samples from disk."""
        filename = str(filename)
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)
        if filename.endswith('.npz'):
            state = dict(state)
            data = [{**param, 'value': state.pop('data.{:d}'.format(iarray))} for iarray, param in enumerate(state.pop('params')[()])]
            if 'others' in state:  # better as does not change type, e.g. _derived remains a list
                others = {name: value for name, value in state['others'][()].items()}
            else:  # backward-compatibility
                others = {name: value[()] for name, value in state.items()}
            state = {**others, 'data': data}
        else:
            state = state[()]
        state.pop('__class__', None)
        new = cls.from_state(state)
        return new

    @staticmethod
    def _get_param(item):
        return item.param

    @property
    def shape(self):
        """Shape of samples."""
        toret = ()
        for array in self.data:
            toret = array.ashape
            break
        return toret

    @shape.setter
    def shape(self, shape):
        """Set samples shape."""
        self._reshape(shape)

    def _reshape(self, shape):
        for array in self.data:
            super(Samples, self).set(_reshape(array, shape))  # this is to circumvent automatic reshaping of :meth:`Samples.set`

    def reshape(self, *args):
        """Reshape samples (with shallow copy)."""
        new = self.copy()
        if len(args) == 1:
            shape = args[0]
        else:
            shape = args
        new._reshape(shape)
        return new

    def ravel(self):
        """Flatten samples."""
        return self.reshape(self.size)

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Total number of samples."""
        return np.prod(self.shape, dtype='intp')

    def __len__(self):
        """Length of samples."""
        if self.shape:
            return self.shape[0]
        return 0

    @classmethod
    def concatenate(cls, *others, intersection=False):
        """
        Concatenate input samples, which requires all samples to hold same parameters,
        except if ``intersection == True``, in which case common parameters are selected.
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        if not others: return cls()
        new = others[0].copy()
        new.data = []
        new_params = others[0].params()
        others = list(others[:1]) + [other for other in others[1:] if other.params() and other.size]
        if intersection:
            for other in others:
                new_params &= other.params()
        else:
            new_names = new_params.names()
            for other in others:
                other_names = other.names()
                if set(other_names) != set(new_names):
                    raise ValueError('cannot concatenate values as parameters do not match: {} != {}.'.format(new_names, other_names))

        def atleast_1d(item):
            shape = item.ashape
            if not shape: shape = (1,)
            item = _reshape(item, shape)
            return item

        for param in new_params:
            try:
                value = np.concatenate([atleast_1d(other[param]) for other in others], axis=0)
            except ValueError as exc:
                raise ValueError('error while concatenating array for parameter {}'.format(param)) from exc
            new[param] = others[0][param].clone(value=value)
        return new

    def update(self, *args, **kwargs):
        """
        Update samples with new one; arguments can be a :class:`Samples`
        or arguments to instantiate such a class (see :meth:`__init__`).
        """
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
        else:
            other = self.__class__(*args, **kwargs)
        for item in other:
            self.set(item)
        self.attrs.update(other.attrs)

    def set(self, item):
        """Add new :class:`ParameterArray` to samples."""
        if self.data:
            shape = self.shape
        else:
            shape = item.ashape
            #if not shape: shape = (1,)
        item = _reshape(item, shape)
        super(Samples, self).set(item)

    def __setitem__(self, name, item):
        """
        Update array in samples.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        item : ParameterArray, array
            Array.
        """
        if not isinstance(item, self._type):
            try:
                name = self.data[name].param  # list index
            except TypeError:
                pass
            is_derived = str(name) in self._derived
            if isinstance(name, Parameter):
                param = name
            else:
                param = Parameter(name, latex=utils.outputs_to_latex(str(name)) if is_derived else None, derived=is_derived)
                if param in self:
                    param = param.clone(self[param].param)
            item = ParameterArray(item, param)
        try:
            self.data[name] = item  # list index
        except TypeError:
            item_name = self._get_name(item)
            if str(name) != item_name:
                item = item.copy()
                if isinstance(name, Parameter):
                    item.param = name
                else:
                    if item.param is None:
                        item.param = Parameter(name)
                    item.param = item.param.clone(name=name)
                #raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

    def __getitem__(self, name):
        """
        Get samples parameter ``name`` if :class:`Parameter` or string,
        else return copy with local slice of samples.
        """
        if isinstance(name, (Parameter, str)):
            return super().__getitem__(name)
        new = self.copy()
        try:
            index = [name] if not isinstance(name, slice) and np.ndim(name) == 0 else name
            new.data = [column[index] for column in self.data]
        except IndexError as exc:
            raise IndexError('Unrecognized indices {}'.format(name)) from exc
        return new

    def __repr__(self):
        """Return string representation, including shape and parameters."""
        return '{}(shape={}, params={})'.format(self.__class__.__name__, self.shape, self.params())

    def to_array(self, params=None, struct=True, derivs=None):
        """
        Return samples as numpy array.

        Parameters
        ----------
        params : ParameterCollection, list, default=None
            Parameters to use. Defaults to all parameters.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['x']``.
            If ``False``, numpy will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
        if params is None: params = self.params()
        names = [str(param) for param in params]
        values = []
        for name in names:
            value = self[name]
            if derivs is not None:
                value = value[derivs]
            values.append(value)
        if struct:
            toret = np.empty(self.shape, dtype=[(name, value.dtype, value.shape[len(self.shape):]) for value in values])
            for name, value in zip(names, values): toret[name] = value
            return toret
        return np.array(values)

    def to_dict(self, params=None):
        """
        Return samples as a dictionary.

        Parameters
        ----------
        params : ParameterCollection, list, default=None
            Parameters to use. Defaults to all parameters.

        Returns
        -------
        dict : dict
            Dictionary mapping parameter name to array.
        """
        if params is None: params = self.params()
        return {str(param): self[param] for param in params}

    def match(self, other, eps=1e-7, params=None):
        """
        Match other :class:`Samples` against ``self``, for parameters ``params``.

        Parameters
        ----------
        other : Samples
            Samples to match.

        eps : float, default=1e-7
            Distance upper bound above which samples are not considered equal.
            1e-7 to handle float32/float64 conversions.

        params : ParameterCollection, list, default=None
            Parameters to use. Defaults to all parameters that are not derived.

        Returns
        -------
        index_in_other, index_in_self
        """
        if params is None:
            params = set(self.names(derived=False)) & set(other.names(derived=False))
        from scipy import spatial
        kdtree = spatial.cKDTree(np.column_stack([self[name].ravel() for name in params]), leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
        array = np.column_stack([other[name].ravel() for name in params])
        dist, indices = kdtree.query(array, k=1, eps=0, p=2, distance_upper_bound=eps)
        mask = indices < self.size
        return np.unravel_index(np.flatnonzero(mask), shape=other.shape), np.unravel_index(indices[mask], shape=self.shape)

    @classmethod
    @CurrentMPIComm.enable
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        """Broadcast input samples ``value`` from rank ``mpiroot`` to other processes."""
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['data'] = [(array['param'], array['derivs']) for array in state['data']]
        state = mpicomm.bcast(state, root=mpiroot)
        for ivalue, (param, derivs) in enumerate(state['data']):
            state['data'][ivalue] = {'value': mpi.bcast(value.data[ivalue] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot), 'param': param, 'derivs': derivs}
        return cls.from_state(state)

    @CurrentMPIComm.enable
    def send(self, dest, tag=0, mpicomm=None):
        """Send ``self`` to rank ``dest``."""
        state = self.__getstate__()
        state['data'] = [(array['param'], array['derivs']) for array in state['data']]
        mpicomm.send(state, dest=dest, tag=tag)
        for array in self:
            mpi.send(array, dest=dest, tag=tag, mpicomm=mpicomm)

    @classmethod
    @CurrentMPIComm.enable
    def recv(cls, source=mpi.ANY_SOURCE, tag=mpi.ANY_TAG, mpicomm=None):
        """Receive samples from rank ``source``."""
        state = mpicomm.recv(source=source, tag=tag)
        for ivalue, (param, derivs) in enumerate(state['data']):
            state['data'][ivalue] = {'value': mpi.recv(source, tag=tag, mpicomm=mpicomm), 'param': param, 'derivs': derivs}
        return cls.from_state(state)

    @classmethod
    @CurrentMPIComm.enable
    def sendrecv(cls, value, source=0, dest=0, tag=0, mpicomm=None):
        """Send samples from rank ``source`` to rank ``dest`` and receive them here."""
        if dest == source == mpicomm.rank:
            return value.deepcopy()
        if mpicomm.rank == source:
            value.send(dest=dest, tag=tag, mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == dest:
            toret = cls.recv(source=source, tag=tag, mpicomm=mpicomm)
        return toret

    #def tree_flatten(self):
    #    data = [array.value for array in self.data]
    #    param_derivs = [(array.param, array.derivs) for array in self.data]
    #    return tuple(data), (param_derivs, {name: getattr(self, name) for name in self._attrs})

    #@classmethod
    #def tree_unflatten(cls, aux_data, children):
    #    new = cls([ParameterArray(value, *param_derivs) for value, param_derivs in zip(children, aux_data[0])])
    #    for name, value in aux_data[1].items():
    #        setattr(new, name, value)
    #    return new

    def tree_flatten(self):
        return self.data, {name: getattr(self, name) for name in self._attrs}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls()
        new.data = children
        for name, value in aux_data.items():
            setattr(new, name, value)
        return new


def is_parameter_sequence(params):
    # ``True`` if ``params`` is a sequence of parameters."""
    return isinstance(params, ParameterCollection) or utils.is_sequence(params)


class BaseParameterMatrix(BaseClass):

    _fill_value = np.nan

    """Base class representing a parameter matrix."""

    def __init__(self, value, params=None, attrs=None):
        """
        Initialize :class:`BaseParameterMatrix`.

        Parameters
        ----------
        value : array
            2D array representing matrix.

        params : list, ParameterCollection
            Parameters corresponding to input ``value``.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        if isinstance(value, self.__class__):
            self.__dict__.update(value.__dict__)
            return
        if params is None:
            raise ValueError('Provide matrix parameters')
        self._params = ParameterCollection(params)
        self._params = ParameterCollection([param.clone(fixed=False) for param in self._params])
        if not self._params:
            raise ValueError('Got no parameters')
        if getattr(value, 'derivs', None) is not None:
            value = np.array([[value[param1, param2] for param2 in self._params] for param1 in self._params])
        self._value = np.atleast_2d(np.array(value))
        if self._value.ndim != 2:
            raise ValueError('Input matrix must be 2D')
        shape = self._value.shape
        if shape[1] != shape[0]:
            raise ValueError('Input matrix must be square')
        if shape[0] != len(self._params):
            raise ValueError('Number of parameters and matrix size are different: {:d} vs {:d}'.format(len(self._params), shape[0]))
        self._sizes
        self.attrs = dict(attrs or {})

    def params(self, *args, **kwargs):
        """Return parameters in matrix."""
        return self._params.params(*args, **kwargs)

    def names(self, *args, **kwargs):
        """Return names of parameters in matrix."""
        return self._params.names(*args, **kwargs)

    def select(self, params=None, **kwargs):
        """
        Return a sub-matrix.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Optionally, parameters to limit to.

        **kwargs : dict
            If ``params`` is ``None``, optional arguments passed to :meth:`ParameterCollection.select`
            to select parameters (e.g. ``varied=True``).

        Returns
        -------
        new : BaseParameterMatrix
            A sub-matrix.
        """
        if params is None: params = self._params.select(**kwargs)
        return self.view(params=params, return_type=None)

    def det(self, params=None):
        """Return matrix determinant, limiting to input parameters ``params`` if not ``None``."""
        return np.linalg.det(self.view(params=params, return_type='nparray'))

    @property
    def _sizes(self):
        # Return parameter sizes
        toret = [max(param.size, 1) for param in self._params]
        if sum(toret) != self._value.shape[0]:
            raise ValueError('number * size of input params must match input matrix shape')
        return toret

    def clone(self, value=None, params=None, attrs=None):
        """
        Clone this matrix, i.e. copy and optionally update ``value``, ``params`` and ``attrs``.

        Parameters
        ----------
        value : array, default=None
            2D array to replace matrix value with.

        params : list, ParameterCollection, default=None
            New parameters.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.

        Returns
        -------
        new : BaseParameterMatrix
            A new matrix, optionally with ``value`` and ``params`` updated.
        """
        new = self.view(params=params, return_type=None)
        if value is not None:
            new._value[...] = value
        if attrs is not None:
            new.attrs = dict(attrs)
        return new

    def view(self, params=None, return_type=None):
        """
        Return matrix for input parameters ``params``.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.
            If a single parameter is provided, and this parameter is a scalar, return a scalar.
            If a parameter in ``params`` is not in matrix, add it, filling in the returned matrix with zeros,
            except on the diagonal, which is filled with :attr:`_fill_value`.

        return_type : str, default=None
            If 'nparray', return a numpy array.
            Else, return a new :class:`BaseParameterMatrix`, restricting to ``params``.

        Returns
        -------
        new : array, float, BaseParameterMatrix
        """
        if params is None:
            params = self._params
        isscalar = not is_parameter_sequence(params)
        if isscalar:
            params = [params]
        params = [self._params[param] if param in self._params else Parameter(param) for param in params]
        params_in_self = [param for param in params if param in self._params]
        params_not_in_self = [param for param in params if param not in params_in_self]
        sizes = [max(param.size, 1) for param in params]
        new = self.__class__(np.zeros((sum(sizes),) * 2, dtype='f8'), params=params, attrs=self.attrs)
        if params_in_self:
            index_new, index_self = new._index(params_in_self), self._index(params_in_self)
            new._value[np.ix_(index_new, index_new)] = self._value[np.ix_(index_self, index_self)]
        if params_not_in_self:
            index_new = new._index(params_not_in_self)
            new._value[np.ix_(index_new, index_new)] = self._fill_value
        if return_type == 'nparray':
            new = new._value
            if isscalar:
                new.shape = params[0].shape
            return new
        return new

    def __array__(self, *args, **kwargs):
        return np.asarray(self._value, *args, **kwargs)

    def _index(self, params):
        # Internal method to return indices in matrix array corresponding to input params.""""
        cumsizes = np.cumsum([0] + self._sizes)
        idx = [self._params.index(param) for param in params]
        if idx:
            return np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx], dtype='i4')
        return np.array(idx, dtype='i4')

    def __contains__(self, name):
        """Has this parameter?"""
        return name in self._params

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        state['value'] = self._value
        state['params'] = self._params.__getstate__()
        state['attrs'] = self.attrs
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self._params = ParameterCollection.from_state(state['params'])
        self._value = state['value']
        self.attrs = state.get('attrs', {})

    def __repr__(self):
        """Return string representation of parameter matrix, including parameters."""
        return '{}({})'.format(self.__class__.__name__, self._params)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(deep_eq(getattr(other, name), getattr(self, name)) for name in ['_params', '_value'])

    @classmethod
    @CurrentMPIComm.enable
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        """Broadcast input samples ``value`` from rank ``mpiroot`` to other processes."""
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['value'] = None
        state = mpicomm.bcast(state, root=mpiroot)
        state['value'] = mpi.bcast(value._value if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return cls.from_state(state)

    def deepcopy(self):
        """Deep copy"""
        return copy.deepcopy(self)

    def __mul__(self, other):
        """Multiply matrix by ``other`` (typically, a float)."""
        new = self.deepcopy()
        new._value *= other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide matrix by ``other`` (typically, a float)."""
        new = self.deepcopy()
        new._value /= other
        return new

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    @property
    def shape(self):
        """Return matrix shape."""
        return self._value.shape


class ParameterCovariance(BaseParameterMatrix):

    """Class that represents a parameter covariance matrix."""

    def view(self, params=None, return_type='nparray', fill=None):
        """
        Return matrix for input parameters ``params``.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.
            If a single parameter is provided, this parameter is a scalar, and ``return_type`` is 'nparray', return a scalar.
            If a parameter in ``params`` is not in matrix, add it, filling in the returned matrix with zeros,
            except on the diagonal, which is filled with :attr:`Parameter.proposal`.

        return_type : str, default='nparray'
            If 'nparray', return a numpy array.
            Else, return a new :class:`ParameterCovariance`, restricting to ``params``.

        Returns
        -------
        new : array, float, ParameterCovariance
        """
        new = super(ParameterCovariance, self).view(params=params, return_type=None)
        if fill == 'proposal':
            params_not_in_self = [param for param in new._params if param not in self._params and param.proposal is not None]
            index = new._index(params_not_in_self)
            new._value[index, index] = [param.proposal**2 for param in params_not_in_self]
        toret = super(ParameterCovariance, new).view(return_type=return_type)
        if return_type == 'nparray' and params is not None and not is_parameter_sequence(params):
            toret.shape = new._params[params].shape
        return toret

    def fom(self, **params):
        """Figure-of-Merit, as inverse square root of the matrix determinant (optionally restricted to input parameters)."""
        return self.det(**params)**(-0.5)

    def corrcoef(self, params=None):
        """Return correlation matrix array (optionally restricted to input parameters)."""
        return utils.cov_to_corrcoef(self.view(params=params, return_type='nparray'))

    def var(self, params=None):
        """
        Return variance (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        """
        cov = self.view(params=params, return_type='nparray')
        if np.ndim(cov) == 0: return cov  # single param
        return np.diag(cov)

    def std(self, params=None):
        """
        Return standard deviation (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        """
        return self.var(params=params)**0.5

    def to_precision(self, params=None, return_type=None):
        """
        Return inverse covariance matrix (precision matrix) for input parameters ``params``.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.
            If a single parameter is provided, this parameter is a scalar, and ``return_type`` is 'nparray', return a scalar.

        return_type : str, default=None
            If 'nparray', return a numpy array.
            Else, return a new :class:`ParameterPrecision`.

        Returns
        -------
        new : array, float, ParameterPrecision
        """
        if params is None: params = self._params
        view = self.view(params, return_type=None)
        invcov = utils.inv(view._value)
        if return_type == 'nparray':
            return invcov
        return ParameterPrecision(invcov, params=params, attrs=view.attrs)

    def to_stats(self, params=None, sigfigs=2, tablefmt='latex_raw', fn=None):
        """
        Export covariance matrix to string.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : str, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.

        fn : str, default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        txt : str
            Summary table.
        """
        import tabulate
        is_latex = 'latex_raw' in tablefmt

        view = self.view(params, return_type=None)
        headers = [param.latex(inline=True) if is_latex else str(param) for param in view._params]

        data = [[str(param)] + [utils.round_measurement(value, value, sigfigs=sigfigs)[0] for value in row] for param, row in zip(view._params, view._value)]
        txt = tabulate.tabulate(data, headers=headers, tablefmt=tablefmt)
        if fn is not None:
            utils.mkdir(os.path.dirname(fn))
            self.log_info('Saving to {}.'.format(fn))
            with open(fn, 'w') as file:
                file.write(txt)
        return txt

    def to_getdist(self, params=None, label=None, center=None, ignore_limits=True):
        """
        Return a GetDist Gaussian distribution, with covariance matrix :meth:`cov`.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to share to GetDist. Defaults to all parameters.

        label : str, default=None
            Name for GetDist to use for this distribution.

        center : list, array, default=None
            Optionally, override :attr:`Parameter.value`.

        ignore_limits : bool, default=True
            GetDist does not seem to be able to integrate over distribution if bounded;
            so drop parameter limits.

        Returns
        -------
        samples : getdist.gaussian_mixtures.MixtureND
        """
        from getdist.gaussian_mixtures import MixtureND
        cov = self.view(params=params, return_type=None)
        labels = [param.latex() for param in cov._params]
        names = [str(param) for param in cov._params]
        # ignore_limits to avoid issue in GetDist with analytic marginalization
        ranges = None
        if not ignore_limits:
            ranges = [tuple(None if limit is None or not np.isfinite(limit) else limit for limit in param.prior.limits) for param in cov._params]
        center = np.asarray([param.value for param in cov._params]) if center is None else np.asarray(center)
        return MixtureND([center], [cov._value], lims=ranges, names=names, labels=labels, label=label)

    @classmethod
    def read_getdist(cls, base_fn):
        """
        Read covariance matrix from GetDist format.

        Parameters
        ----------
        base_fn : str
            Base *CosmoMC* file name. Will be appended by '.margestats' for marginalized parameter mean,
            '.covmat' for parameter covariance matrix.

        Returns
        -------
        cov : CovarianceMatrix
        """
        covmat_fn = '{}.covmat'.format(base_fn)
        cls.log_info('Loading covariance file: {}.'.format(covmat_fn))
        covariance = []
        with open(covmat_fn, 'r') as file:
            for line in file:
                line = [item.strip() for item in line.split()]
                if line:
                    if line[0] == '#':
                        params = line[1:]
                    else:
                        covariance.append([float(value) for value in line])
        return cls(covariance, params=[Parameter(param, fixed=False) for param in params])


class ParameterPrecision(BaseParameterMatrix):

    _fill_value = 0.

    def fom(self, **params):
        """Figure-of-Merit, as square root of the precision (optionally restricted to input parameters)."""
        return self.to_covariance().fom(params=params)

    def to_covariance(self, params=None, return_type=None):
        """
        Return inverse precision matrix (covariance matrix) for input parameters ``params``.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.
            If a single parameter is provided, this parameter is a scalar, and ``return_type`` is 'nparray', return a scalar.

        return_type : str, default=None
            If 'nparray', return a numpy array.
            Else, return a new :class:`ParameterCovariance`.

        Returns
        -------
        new : array, float, ParameterCovariance
        """
        cov = utils.inv(self._value)
        return ParameterCovariance(cov, params=self._params, attrs=self.attrs).view(params=params, return_type=return_type)

    @classmethod
    def sum(cls, *others):
        """Add precision matrices."""
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        params = ParameterCollection.concatenate([other._params for other in others])
        new = others[0].view(params, return_type=None)
        for other in others[1:]:
            other = other.view(new._params, return_type=None)
            new._value += other._value
            new.attrs.update(other.attrs)
        return new

    def __add__(self, other):
        """Sum of `self`` + ``other`` precision matrices."""
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)