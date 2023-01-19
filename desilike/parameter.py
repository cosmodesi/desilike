"""Classes to handle parameters."""

import re
import fnmatch
import copy
import numbers
from collections import Counter

import numpy as np
import scipy as sp
from .jax import numpy as jnp
from .jax import scipy as jsp
from .jax import rv_frozen

from .io import BaseConfig
from . import mpi, utils
from .mpi import CurrentMPIComm
from .utils import BaseClass, NamespaceDict, deep_eq, path_types


def decode_name(name, default_start=0, default_stop=None, default_step=1):
    """
    Split ``name`` into strings and allowed index ranges.

    >>> decode_name('a_[-4:5:2]_b_[0:2]')
    ['a_', '_b_'], [range(-4, 5, 2), range(0, 2, 1)]

    Parameters
    ----------
    name : string
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
    replaces = re.finditer(r'\[(-?\d*):(\d*):*(-?\d*)\]', name)
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
    name : string
        Parameter name.

    latex : string, default=None
        Latex for parameter.

    kwargs : dict
        Arguments for :func:`decode_name`

    Returns
    -------
    name : string
        Parameter name with template forms ``[::]`` replaced.

    latex : string, None
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

    name : list, string
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
        name = fnmatch.translate(name)
        strings, ranges = decode_name(name)
        pattern = re.compile(r'(-?\d*)'.join(strings))
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


from collections.abc import Mapping

from itertools import chain as _chain
from itertools import repeat as _repeat
from itertools import starmap as _starmap
from operator import itemgetter as _itemgetter


class Deriv(dict):
    """
    This class is a modification of https://github.com/python/cpython/blob/main/Lib/collections/__init__.py,
    restricting to positive elements.
    """
    # References:
    #   http://en.wikipedia.org/wiki/Multiset
    #   http://www.gnu.org/software/smalltalk/manual-base/html_node/Bag.html
    #   http://www.demo2s.com/Tutorial/Cpp/0380__set-multiset/Catalog0380__set-multiset.htm
    #   http://code.activestate.com/recipes/259174/
    #   Knuth, TAOCP Vol. II section 4.6.3

    def __init__(self, iterable=None, /, **kwds):
        """Create a new, empty Deriv object.  And if given, count elements
        from an input iterable.  Or, initialize the count from another mapping
        of elements to their counts.
        >>> c = Deriv()                           # a new, empty counter
        >>> c = Deriv('gallahad')                 # a new counter from an iterable
        >>> c = Deriv({'a': 4, 'b': 2})           # a new counter from a mapping
        >>> c = Deriv(a=4, b=2)                   # a new counter from keyword args
        """
        super().__init__()
        self.update(iterable, **kwds)

    # ADM changes
    def __setitem__(self, name, item):
        if item > 0:
            super(Deriv, self).__setitem__(name, item)

    def setdefault(self, name, item):
        if item > 0:
            super(Deriv, self).setdefault(name, item)

    def __missing__(self, key):
        """The count of elements not in the Deriv is zero."""
        # Needed so that self[missing_item] does not raise KeyError
        return 0

    def total(self):
        """Sum of the counts."""
        return sum(self.values())

    def most_common(self, n=None):
        """List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.
        >>> Deriv('abracadabra').most_common(3)
        [('a', 5), ('b', 2), ('r', 2)]
        """
        # Emulate Bag.sortedByCount from Smalltalk
        if n is None:
            return sorted(self.items(), key=_itemgetter(1), reverse=True)

        # Lazy import to speedup Python startup time
        import heapq
        return heapq.nlargest(n, self.items(), key=_itemgetter(1))

    def elements(self):
        """Iterator over elements repeating each as many times as its count.
        >>> c = Deriv('ABCABC')
        >>> sorted(c.elements())
        ['A', 'A', 'B', 'B', 'C', 'C']
        # Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1
        >>> import math
        >>> prime_factors = Deriv({2: 2, 3: 3, 17: 1})
        >>> math.prod(prime_factors.elements())
        1836
        Note, if an element's count has been set to zero or is a negative
        number, elements() will ignore it.
        """
        # Emulate Bag.do from Smalltalk and Multiset.begin from C++.
        return _chain.from_iterable(_starmap(_repeat, self.items()))

    def update(self, iterable=None, /, **kwds):
        """Like dict.update() but add counts instead of replacing them.
        Source can be an iterable, a dictionary, or another Deriv instance.
        >>> c = Deriv('which')
        >>> c.update('witch')           # add elements from another iterable
        >>> d = Deriv('watch')
        >>> c.update(d)                 # add elements from another counter
        >>> c['h']                      # four 'h' in which, witch, and watch
        4
        """
        # The regular dict.update() operation makes no sense here because the
        # replace behavior results in the some of original untouched counts
        # being mixed-in with all of the other counts for a mismash that
        # doesn't have a straight-forward interpretation in most counting
        # contexts.  Instead, we implement straight-addition.  Both the inputs
        # and outputs are allowed to contain zero and negative counts.

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

    def copy(self):
        """Return a shallow copy."""
        return self.__class__(self)

    def __reduce__(self):
        return self.__class__, (dict(self),)

    def __delitem__(self, elem):
        """Like dict.__delitem__() but does not raise KeyError for missing values."""
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

    # Multiset-style mathematical operations discussed in:
    #       Knuth TAOCP Volume II section 4.6.3 exercise 19
    #       and at http://en.wikipedia.org/wiki/Multiset
    #
    # Outputs guaranteed to only include positive counts.
    #
    # To strip negative and zero counts, add-in an empty counter:
    #       c += Deriv()
    #
    # Results are ordered according to when an element is first
    # encountered in the left operand and then by the order
    # encountered in the right operand.
    #
    # When the multiplicities are all zero or one, multiset operations
    # are guaranteed to be equivalent to the corresponding operations
    # for regular sets.
    #     Given counter multisets such as:
    #         cp = Deriv(a=1, b=0, c=1)
    #         cq = Deriv(c=1, d=0, e=1)
    #     The corresponding regular sets would be:
    #         sp = {'a', 'c'}
    #         sq = {'c', 'e'}
    #     All of the following relations would hold:
    #         set(cp + cq) == sp | sq
    #         set(cp - cq) == sp - sq
    #         set(cp | cq) == sp | sq
    #         set(cp & cq) == sp & sq
    #         (cp == cq) == (sp == sq)
    #         (cp != cq) == (sp != sq)
    #         (cp <= cq) == (sp <= sq)
    #         (cp < cq) == (sp < sq)
    #         (cp >= cq) == (sp >= sq)
    #         (cp > cq) == (sp > sq)

    def __eq__(self, other):
        """True if all counts agree. Missing counts are treated as zero."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return all(self[e] == other[e] for c in (self, other) for e in c)

    def __ne__(self, other):
        """True if any counts disagree. Missing counts are treated as zero."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return not self == other

    def __le__(self, other):
        """True if all counts in self are a subset of those in other."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return all(self[e] <= other[e] for c in (self, other) for e in c)

    def __lt__(self, other):
        """True if all counts in self are a proper subset of those in other."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other):
        """True if all counts in self are a superset of those in other."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return all(self[e] >= other[e] for c in (self, other) for e in c)

    def __gt__(self, other):
        """True if all counts in self are a proper superset of those in other."""
        if not isinstance(other, Deriv):
            return NotImplemented
        return self >= other and self != other

    def __add__(self, other):
        """Add counts from two counters.
        >>> Deriv('abbb') + Deriv('bcc')
        Deriv({'b': 4, 'c': 2, 'a': 1})
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
        """Internal method to strip elements with a negative or zero count"""
        nonpositive = [elem for elem, count in self.items() if not count > 0]
        for elem in nonpositive:
            del self[elem]
        return self

    def __iadd__(self, other):
        """Inplace add from another counter, keeping only positive counts.
        >>> c = Deriv('abbb')
        >>> c += Deriv('bcc')
        >>> c
        Deriv({'b': 4, 'c': 2, 'a': 1})
        """
        for elem, count in other.items():
            self[elem] += count
        return self._keep_positive()


def _make_deriv(deriv):
    if isinstance(deriv, Deriv):
        return deriv
    deriv_types = (Parameter, str)
    deriv = (deriv,) if not utils.is_sequence(deriv) else deriv
    if all(isinstance(param, deriv_types) for param in deriv):
        return Deriv(str(param) for param in deriv)
    raise ValueError('Unable to make deriv from {}'.format(deriv))


class ParameterArray(np.ndarray):

    def __new__(cls, value, param=None, derivs=None, copy=False, dtype=None, **kwargs):
        """
        Initalize :class:`array`.

        Parameters
        ----------
        value : array
            Local array value.

        copy : bool, default=False
            Whether to copy input array.

        dtype : dtype, default=None
            If provided, enforce this dtype.
        """
        value = np.array(value, copy=copy, dtype=dtype, **kwargs)
        obj = value.view(cls)
        obj.param = None if param is None else Parameter(param)
        obj.derivs = None if derivs is None else [_make_deriv(deriv) for deriv in derivs]
        return obj

    def __array_finalize__(self, obj):
        self.param = getattr(obj, 'param', None)
        self.derivs = getattr(obj, 'derivs', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, ParameterArray):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, ParameterArray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], ParameterArray):
                inputs[0].param = self.param
                inputs[0].derivs = self.derivs
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(ParameterArray)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        for result in results:
            if isinstance(result, ParameterArray):
                result.param = self.param
                result.derivs = self.derivs

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.param, self.derivs, self)

    def __reduce__(self):
        # See https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        # Get the parent's __reduce__ tuple
        pickled_state = super(ParameterArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (None if self.param is None else self.param.__getstate__(), self.derivs)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.param = None if state[-2] is None else Parameter.from_state(state[-2])  # Set the info attribute
        self.derivs = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(ParameterArray, self).__setstate__(state[:-2])

    def __getstate__(self):
        return {'value': self.view(np.ndarray), 'param': None if self.param is None else self.param.__getstate__(), 'derivs': self.derivs}

    def _index(self, deriv):
        ideriv = deriv
        if self.derivs is not None:
            try:
                deriv = _make_deriv(deriv)
            except ValueError:
                pass
            else:
                try:
                    ideriv = self.derivs.index(deriv)
                except ValueError as exc:
                    raise KeyError('{} is not in computed derivatives: {}'.format(deriv, self.derivs)) from exc
                else:
                    ideriv = (Ellipsis, ideriv)
                    if self.param is not None:
                        ideriv += (slice(None),) * self.param.ndim
        return ideriv

    @property
    def zero(self):
        if self.derivs is not None:
            return self[()]
        return self

    @property
    def pndim(self):
        return (1 if self.derivs is not None else 0) + (self.param.ndim if self.param is not None else 0)

    @property
    def andim(self):
        return self.ndim - self.pndim

    @property
    def ashape(self):
        return self.shape[:self.andim]

    @property
    def pshape(self):
        return self.shape[self.andim:]

    def __getitem__(self, deriv):
        return super(ParameterArray, self).__getitem__(self._index(deriv))

    def __setitem__(self, deriv, item):
        return super(ParameterArray, self).__setitem__(self._index(deriv), item)

    @classmethod
    def from_state(cls, state):
        return cls(state['value'], None if state.get('param', None) is None else Parameter.from_state(state['param']), state.get('derivs', None))


class Parameter(BaseClass):
    """
    Class that represents a parameter.

    Attributes
    ----------
    name : string
        Parameter name.

    value : float
        Default value for parameter.

    fixed : bool
        Whether parameter is fixed.

    prior : ParameterPrior
        Prior distribution.

    ref : ParameterPrior
        Reference distribution.
        This is supposed to represent the expected posterior for this parameter.

    proposal : float
        Proposal uncertainty.

    latex : string, default=None
        Latex for parameter.
    """
    _attrs = ['basename', 'namespace', 'value', 'fixed', 'derived', 'prior', 'ref', 'proposal', 'latex', 'depends', 'shape']
    _allowed_solved = ['.best', '.marg', '.auto']

    def __init__(self, basename, namespace='', value=None, fixed=None, derived=False, prior=None, ref=None, proposal=None, latex=None, shape=()):
        """
        Initialize :class:`Parameter`.

        Parameters
        ----------
        name : string, Parameter
            If :class:`Parameter`, update ``self`` attributes.

        value : float, default=False
            Default value for parameter.

        fixed : bool, default=None
            Whether parameter is fixed.
            If ``None``, defaults to ``True`` if ``prior`` or ``ref`` is not ``None``, else ``False``.

        prior : ParameterPrior, dict, default=None
            Prior distribution for parameter, arguments for :class:`ParameterPrior`.

        ref : Prior, dict, default=None
            Reference distribution for parameter, arguments for :class:`ParameterPrior`.
            This is supposed to represent the expected posterior for this parameter.
            If ``None``, defaults to ``prior``.

        proposal : float, default=None
            Proposal uncertainty for parameter.
            If ``None``, defaults to scale (or half of limiting range) of ``ref``.

        latex : string, default=None
            Latex for parameter.
        """
        from . import base
        if isinstance(basename, Parameter):
            self.__dict__.update(basename.__dict__)
            return
        if isinstance(basename, ParameterConfig):
            self.__dict__.update(basename.init().__dict__)
            return
        try:
            self.__init__(**basename)
        except TypeError:
            pass
        else:
            return
        self._namespace = namespace
        names = str(basename).split(base.namespace_delimiter)
        self._basename, namespace = names[-1], base.namespace_delimiter.join(names[:-1])
        if self._namespace:
            if namespace:
                self._namespace = base.namespace_delimiter.join([self._namespace, namespace])
        else:
            if namespace:
                self._namespace = namespace
        self._value = float(value) if value is not None else None
        self._prior = prior if isinstance(prior, ParameterPrior) else ParameterPrior(**(prior or {}))
        if ref is not None:
            self._ref = ref if isinstance(ref, ParameterPrior) else ParameterPrior(**(ref or {}))
        else:
            self._ref = self._prior.copy()
        self._latex = latex
        self._proposal = proposal
        if proposal is None:
            if (ref is not None or prior is not None) and self._ref.is_proper():
                self._proposal = self._ref.std()
        self._derived = derived
        self._depends = {}
        if isinstance(derived, str):
            if self.solved:
                allowed_dists = ['norm', 'uniform']
                if self._prior.dist not in allowed_dists or self._prior.is_limited():
                    raise ParameterError('Prior must be one of {}, with no limits, to use analytic marginalisation for {}'.format(allowed_dists, self))
            else:
                placeholders = re.finditer(r'\{.*?\}', derived)
                for placeholder in placeholders:
                    placeholder = placeholder.group()
                    key = '_' * len(derived) + '{:d}_'.format(len(self._depends) + 1)
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
        self.updated = True

    @property
    def size(self):
        return np.prod(self._shape, dtype='i')

    @property
    def ndim(self):
        return len(self._shape)

    def eval(self, **values):
        if isinstance(self._derived, str) and not self.solved:
            try:
                values = {k: values[n] for k, n in self._depends.items()}
            except KeyError:
                raise ParameterError('Parameter {} is to be derived from parameters {}, as {}, but they are not provided'.format(self, list(self._depends.values()), self.derived))
            return utils.evaluate(self._derived, locals=values)
        return values[self.name]

    @property
    def value(self):
        value = self._value
        if value is None:
            if hasattr(self._ref, 'loc'):
                value = self._ref.loc
            elif self._ref.is_proper():
                value = np.mean(self._ref.limits)
        return value

    @property
    def derived(self):
        if isinstance(self._derived, str) and not self.solved:
            toret = self._derived
            for k, v in self._depends.items():
                toret = toret.replace(k, '{{{}}}'.format(v))
            return toret
        return self._derived

    @property
    def solved(self):
        return self._derived in self._allowed_solved

    @property
    def name(self):
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
        state.update(kwargs)
        state.pop('updated', None)
        self.__init__(**state)

    def clone(self, *args, **kwargs):
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
        new = super(Parameter, self).__copy__()
        new._depends = copy.copy(new._depends)
        return new

    def deepcopy(self):
        return copy.deepcopy(self)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in self._attrs:
            state[key] = getattr(self, '_' + key)
            if hasattr(state[key], '__getstate__'):
                state[key] = state[key].__getstate__()
        state['derived'] = self.derived
        state.pop('depends')
        state['updated'] = self.updated
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        state = state.copy()
        updated = state.pop('updated', True)
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

    def __hash__(self):
        return hash(str(self))

    def latex(self, namespace=False, inline=False):
        """If :attr:`latex` is specified (i.e. not ``None``), return :attr:`latex` surrounded by '$' signs, else :attr:`name`."""
        if namespace:
            namespace = self._namespace
        if self._latex is not None:
            if namespace:
                match1 = re.match('(.*)_(.)$', self._latex)
                match2 = re.match('(.*)_{(.*)}$', self._latex)
                if match1 is not None:
                    latex = r'%s_{%s,\mathrm{%s}}' % (match1.group(1), match1.group(2), namespace)
                elif match2 is not None:
                    latex = r'%s_{%s,\mathrm{%s}}' % (match2.group(1), match2.group(2), namespace)
                else:
                    latex = r'%s_{\mathrm{%s}}' % (self._latex, namespace)
            else:
                latex = self._latex
            if inline:
                latex = '${}$'.format(latex)
            return latex
        return str(self)


def _make_property(name):

    def getter(self):
        return getattr(self, '_' + name)

    return getter


for name in Parameter._attrs:
    if name not in ['value', 'derived', 'latex']:
        setattr(Parameter, name, property(_make_property(name)))


class BaseParameterCollection(BaseClass):

    """Class holding a collection of parameters."""

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
        return param.name

    @classmethod
    def _get_param(cls, item):
        return item

    def __init__(self, data=None, attrs=None):
        """
        Initialize :class:`BaseParameterCollection`.

        Parameters
        ----------
        data : list, tuple, string, dict, ParameterCollection
            Can be:

            - list (or tuple) of parameters (:class:`Parameter` or dictionary to initialize :class:`Parameter`).
            - dictionary of name: parameter
            - :class:`ParameterCollection` instance

        string : string
            If not ``None``, *yaml* format string to decode.
            Added on top of ``data``.
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
            data = {}
            for item in dd:
                data[self._get_name(item)] = item  # only name is provided

        for name, item in data.items():
            self[name] = item

    def __setitem__(self, name, item):
        """
        Update parameter in collection.
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
        Return parameter ``name``.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        Returns
        -------
        param : Parameter
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
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.
        """
        try:
            del self.data[name]
        except TypeError:
            del self.data[self.index(name)]

    def sort(self, key=None):
        if key is not None:
            self.data = [self[kk] for kk in key]
        else:
            self.data = self.data.copy()
        return self

    def pop(self, name, *args, **kwargs):
        toret = self.get(name, *args, **kwargs)
        try:
            del self[name]
        except (IndexError, KeyError):
            pass
        return toret

    def get(self, name, *args, **kwargs):
        """
        Return parameter of name ``name`` in collection.

        Parameters
        ----------
        name : Parameter, string
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.

        Returns
        -------
        param : Parameter
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
            return self.data[self.index(name)]
        except KeyError:
            if has_default:
                return default
            raise KeyError('Parameter {} not found'.format(name))

    def set(self, item):
        """
        Set parameter ``param`` in collection.
        If there is already a parameter with same name in collection, replace this stored parameter by the input one.
        Else, append parameter to collection.
        """
        try:
            self.data[self.index(item)] = item
        except KeyError:
            self.data.append(item)

    def setdefault(self, item):
        """Set parameter ``param`` in collection if not already in it."""
        if not isinstance(item, self._type):
            raise TypeError('{} is not a {} instance.'.format(item, self._type))
        if item not in self:
            self.set(item)

    def index(self, name):
        """
        Return index of parameter ``name``.

        Parameters
        ----------
        name : Parameter, string, int
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
        return self._get_name(name) in (self._get_name(item) for item in self.data)

    def select(self, **kwargs):
        """
        Return new collection, after selection of parameters whose attribute match input values::

            collection.select(fixed=True)

        returns collection of fixed parameters.
        If 'name' is provided, consider all matching parameters, e.g.::

            collection.select(varied=True, name='a_[0:2]')

        returns a collection of varied parameters, with name in ``['a_0', 'a_1']``.
        """
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

    def params(self, **kwargs):
        return ParameterCollection([self._get_param(item) for item in self.select(**kwargs)])

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
        """Update collection with new one."""
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
        else:
            other = self.__class__(*args, **kwargs)
        for item in other:
            self.set(item)

    def clone(self, *args, **kwargs):
        new = self.copy()
        new.update(*args, **kwargs)
        return new

    def keys(self, **kwargs):
        return [self._get_name(item) for item in self.select(**kwargs)]

    def values(self, **kwargs):
        return [item for item in self.select(**kwargs)]

    def items(self, **kwargs):
        return [(self._get_name(item), item) for item in self.select(**kwargs)]

    def deepcopy(self):
        return copy.deepcopy(self)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and list(other.params()) == list(self.params()) and all(deep_eq(other_value, self_value) for other_value, self_value in zip(other, self))


class ParameterConfig(NamespaceDict):

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
            if name in state and 'rescale' in state[name]:
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
        """
        Return new collection, after selection of parameters whose attribute match input values::

            collection.select(fixed=True)

        returns collection of fixed parameters.
        If 'name' is provided, consider all matching parameters, e.g.::

            collection.select(varied=True, name='a_[0:2]')

        returns a collection of varied parameters, with name in ``['a_0', 'a_1']``.
        """
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

        #self.delete = {}
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
                #self.namespace.pop(name, None)
                #self.namespace[name] = namespace
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
        if not isinstance(item, ParameterConfig):
            item = ParameterConfig(item)
            item.setdefault(self.identifier, name)
        try:
            self.data[name] = item
        except TypeError:
            item_name = str(self._get_name(item))
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect name {})'.format(item_name, name))
            self.data[self._index_name(name)] = item


class ParameterCollection(BaseParameterCollection):

    """Class holding a collection of parameters."""
    _attrs = ['_updated']

    def __init__(self, data=None, attrs=None):
        """
        Initialize :class:`ParameterCollection`.

        Parameters
        ----------
        data : list, tuple, string, dict, ParameterCollection
            Can be:

            - list (or tuple) of parameters (:class:`Parameter` or dictionary to initialize :class:`Parameter`).
            - dictionary of name: parameter
            - :class:`ParameterCollection` instance

        string : string
            If not ``None``, *yaml* format string to decode.
            Added on top of ``data``.
        """
        if isinstance(data, path_types):
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
        return self._updated or any(param.updated for param in self.data)

    @updated.setter
    def updated(self, updated):
        updated = bool(updated)
        self._updated = updated
        for param in self.data: param.updated = updated

    def __delitem__(self, name):
        """
        Delete parameter ``name``.

        Parameters
        ----------
        name : Parameter, string, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.
        """
        self._updated = True
        return super(ParameterCollection, self).__delitem__(name)

    def update(self, *args, name=None, basename=None, **kwargs):
        """Update collection with new one."""
        self._updated = True
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
            for item in other:
                if item in self:
                    tmp = self[item].clone(item)
                else:
                    tmp = item.copy()
                self.set(tmp)
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
                duplicates = {name: multiplicity for basename, multiplicity in names.items() if multiplicity > 1}
                if duplicates:
                    raise ValueError('Cannot update namespace, as following duplicates found: {}'.format(duplicates))
        else:
            raise ValueError('Unrecognized arguments {}'.format(args))

    def __add__(self, other):
        return self.concatenate(self, self.__class__(other))

    def __radd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self
        self.__dict__.update(self.__add__(other).__dict__)
        return self

    def params(self, **kwargs):
        return self.select(**kwargs)

    def set(self, item):
        self._updated = True
        if not isinstance(item, Parameter):
            item = Parameter(item)
        try:
            self.data[self.index(item)] = item
        except KeyError:
            self.data.append(item)

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
            item_name = str(self._get_name(item))
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

    def eval(self, **params):
        toret = {}
        for param in params:
            try:
                toret[param] = self[param].eval(**params)
            except KeyError:
                pass
        return toret

    def prior(self, **params):
        eval_params = self.eval(**params)
        toret = 0.
        for param in self.data:
            if param.name in eval_params and param.varied and (param.depends or (not param.derived)):
                toret += param.prior(eval_params[param.name])
        return toret


class ParameterPriorError(Exception):

    """Exception raised when issue with prior."""


class ParameterPrior(BaseClass):
    """
    Class that describes a 1D prior distribution.

    Parameters
    ----------
    dist : string
        Distribution name.

    rv : scipy.stats.rv_continuous
        Random variate.

    attrs : dict
        Arguments used to initialize :attr:`rv`.
    """

    def __init__(self, dist='uniform', limits=None, **kwargs):
        r"""
        Initialize :class:`ParameterPrior`.

        Parameters
        ----------
        dist : string
            Distribution name in :mod:`scipy.stats`

        limits : tuple, default=None
            Tuple corresponding to lower, upper limits.
            ``None`` means :math:`-\infty` for lower bound and :math:`\infty` for upper bound.
            Defaults to :math:`-\infty, \infty`.

        kwargs : dict
            Arguments for :func:`scipy.stats.dist`, typically ``loc``, ``scale``
            (mean and standard deviation in case of a normal distribution ``'dist' == 'norm'``)
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
        if self.is_limited():
            dist = dist if dist.startswith('trunc') or dist == 'uniform' else 'trunc{}'.format(dist)
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
        #self.limits = self.rv.support()

    def isin(self, x):
        """Whether ``x`` is within prior, i.e. within limits - strictly positive probability."""
        x = jnp.asarray(x)
        return (self.limits[0] < x) & (x < self.limits[1])

    def __call__(self, x, remove_zerolag=True):
        """Return probability density at ``x``."""
        if not self.is_proper():
            return jnp.where(self.isin(x), 0, -np.inf)
        toret = self.rv.logpdf(x)
        if remove_zerolag:
            loc = self.attrs.get('loc', None)
            if loc is None: loc = np.mean(self.limits)
            toret -= self.rv.logpdf(loc)
        return toret

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

    def __str__(self):
        """Return string with distribution name, limits, and attributes (e.g. ``loc`` and ``scale``)."""
        base = self.dist
        if self.is_limited():
            base = '{}[{}, {}]'.format(base, *self.limits)
        return '{}({})'.format(base, self.attrs)

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.__init__(**state)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {'dist': self.dist, 'limits': self.limits}
        state.update(self.attrs)
        return state

    def is_proper(self):
        """Whether distribution is proper, i.e. has finite integral."""
        return self.dist != 'uniform' or not np.isinf(self.limits).any()

    def is_limited(self):
        """Whether distribution has (at least one) finite limit."""
        return not np.isinf(self.limits).all()

    def affine_transform(self, loc=0., scale=1.):
        state = self.__getstate__()
        try:
            center = self.loc
        except AttributeError:
            if self.is_limited():
                center = np.mean([lim for lim in self.limits if not np.isinf(lim)])
            else:
                center = 0.
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
    return array.reshape(shape + array.pshape)


class Samples(BaseParameterCollection):

    """Class that holds samples drawn from likelihood."""

    _type = ParameterArray
    _attrs = BaseParameterCollection._attrs + ['_derived']
    _derived = []

    def __init__(self, data=None, params=None, attrs=None):
        self.attrs = dict(attrs or {})
        self.data = []
        if params is not None:
            if len(params) != len(data):
                raise ValueError('Provide as many parameters as arrays')
            for param, value in zip(params, data):
                self[param] = value
        else:
            super(Samples, self).__init__(data=data, attrs=attrs)

    @staticmethod
    def _get_param(item):
        return item.param

    @property
    def shape(self):
        toret = ()
        for array in self.data:
            toret = array.ashape
            break
        return toret

    @shape.setter
    def shape(self, shape):
        self._reshape(shape)

    def _reshape(self, shape):
        for array in self:
            self.set(_reshape(array, shape))

    def reshape(self, *args):
        new = self.copy()
        if len(args) == 1:
            shape = args[0]
        else:
            shape = args
        new._reshape(shape)
        return new

    def ravel(self):
        # Flatten along iteration axis
        return self.reshape(self.size)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape, dtype='intp')

    def __len__(self):
        if self.shape:
            return self.shape[0]
        return 0

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input collections.
        Unique items only are kept.
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        if not others: return cls()
        new = cls()
        new_params = others[0].params()
        new_names = new_params.names()
        for other in others:
            other_names = other.names()
            if new_names and other_names and set(other_names) != set(new_names):
                raise ValueError('Cannot concatenate values as parameters do not match: {} != {}.'.format(new_names, other_names))
        for param in new_params:
            new[param] = np.concatenate([np.atleast_1d(other[param]) for other in others], axis=0)
        return new

    def update(self, *args, **kwargs):
        """Update collection with new one."""
        if len(args) == 1 and isinstance(args[0], self.__class__):
            other = args[0]
        else:
            other = self.__class__(*args, **kwargs)
        for item in other:
            self.set(item)

    def set(self, item):
        if self.data:
            shape = self.shape
        else:
            try:
                shape = item.ashape
            except AttributeError:
                shape = item.shape
            if not shape: shape = (1,)
        item = _reshape(item, shape)
        super(Samples, self).set(item)

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
            item_name = str(self._get_name(item))
            if str(name) != item_name:
                raise KeyError('Parameter {} must be indexed by name (incorrect {})'.format(item_name, name))
            self.set(item)

    def __getitem__(self, name):
        """
        Get samples parameter ``name`` if :class:`Parameter` or string,
        else return copy with local slice of samples.
        """
        if isinstance(name, (Parameter, str)):
            return self.get(name)
        new = self.copy()
        try:
            new.data = [column[name] for column in self.data]
        except IndexError as exc:
            raise IndexError('Unrecognized indices {}'.format(name)) from exc
        return new

    def __repr__(self):
        """Return string representation, including shape and columns."""
        return '{}(shape={}, params={})'.format(self.__class__.__name__, self.shape, self.params())

    def to_array(self, params=None, struct=True):
        """
        Return samples as numpy array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, numpy will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
        if params is None: params = self.params()
        names = [str(param) for param in params]
        if struct:
            toret = np.empty(len(self), dtype=[(name, self[name].dtype, self.shape[1:]) for name in names])
            for name in names: toret[name] = self[name]
            return toret
        return np.array([self[name] for name in names])

    def to_dict(self, params=None):
        if params is None: params = self.params()
        return {str(param): self[param] for param in params}

    def match(self, other, eps=1e-8, params=None):
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
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['data'] = [array['param'] for array in state['data']]
        state = mpicomm.bcast(state, root=mpiroot)
        for ivalue, param in enumerate(state['data']):
            state['data'][ivalue] = {'value': mpi.bcast(value.data[ivalue] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot), 'param': param}
        return cls.from_state(state)

    @CurrentMPIComm.enable
    def send(self, dest, tag=0, mpicomm=None):
        state = self.__getstate__()
        state['data'] = [array['param'] for array in state['data']]
        mpicomm.send(state, dest=dest, tag=tag)
        for array in self:
            mpi.send(array, dest=dest, tag=tag, mpicomm=mpicomm)

    @classmethod
    @CurrentMPIComm.enable
    def recv(cls, source=mpi.ANY_SOURCE, tag=mpi.ANY_TAG, mpicomm=None):
        state = mpicomm.recv(source=source, tag=tag)
        for ivalue, param in enumerate(state['data']):
            state['data'][ivalue] = {'value': mpi.recv(source, tag=tag, mpicomm=mpicomm), 'param': param}
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


def is_parameter_sequence(params):
    return isinstance(params, ParameterCollection) or utils.is_sequence(params)


class BaseParameterMatrix(BaseClass):

    _fill_value = np.nan

    """Class that represents a parameter matrix."""

    def __init__(self, value, params=None, center=None):
        """
        Initialize :class:`BaseParameterMatrix`.

        Parameters
        ----------
        value : array
            2D array representing matrix.

        params : list, ParameterCollection
            Parameters corresponding to input ``value``.
        """
        if isinstance(value, self.__class__):
            self.__dict__.update(value.__dict__)
            return
        if params is None:
            raise ValueError('Provide matrix parameters')
        self._params = ParameterCollection(params)
        if not self._params:
            raise ValueError('Got no parameters')
        if isinstance(value, ParameterArray):
            value = np.array([[value[param1, param2] for param2 in self._params] for param1 in self._params])
        self._value = np.atleast_2d(np.array(value))
        if self._value.ndim != 2:
            raise ValueError('Input matrix must be 2D')
        shape = self._value.shape
        if shape[1] != shape[0]:
            raise ValueError('Input matrix must be square')
        if center is None:
            center = [param.value if param.value is not None else np.nan for param in self._params]
        self._center = np.concatenate([np.ravel(c) for c in center])
        if self._center.size != shape[0]:
            raise ValueError('Input center and matrix have different sizes: {:d} vs {:d}'.format(self._center.size, shape[0]))
        self._sizes

    def params(self, *args, **kwargs):
        return self._params.params(*args, **kwargs)

    def names(self, *args, **kwargs):
        return self._params.names(*args, **kwargs)

    def select(self, params=None, **kwargs):
        if params is None: params = self._params.select(**kwargs)
        return self.view(params=params, return_type=None)

    def det(self, params=None):
        return np.linalg.det(self.view(params=params, return_type='nparray'))

    @property
    def _sizes(self):
        toret = [max(param.size, 1) for param in self._params]
        if sum(toret) != self._value.shape[0]:
            raise ValueError('number * size of input params must match input matrix shape')
        return toret

    def clone(self, value=None, params=None, center=None):
        new = self.view(params=params, return_type=None)
        if value is not None:
            new._value[...] = value
        if center is not None:
            new._center[...] = center
        return new

    def center(self, params=None, return_type='nparray'):
        if params is None:
            params = self._params
        isscalar = not is_parameter_sequence(params)
        if isscalar:
            params = [params]
        center = self._center[self._index(params)]
        if return_type == 'nparray':
            if isscalar:
                return center.item()
            return center
        return {str(param): value for param, value in zip(params, center)}

    def view(self, params=None, return_type=None):
        """Return matrix for input parameters ``params``."""
        if params is None:
            params = self._params
        isscalar = not is_parameter_sequence(params)
        if isscalar:
            params = [params]
        params = [self._params[param] if param in self._params else Parameter(param) for param in params]
        params_in_self = [param for param in params if param in self._params]
        params_not_in_self = [param for param in params if param not in params_in_self]
        sizes = [max(param.size, 1) for param in params]
        new = self.__class__(np.zeros((sum(sizes),) * 2, dtype='f8'), params=params)
        if params_in_self:
            index_new, index_self = new._index(params_in_self), self._index(params_in_self)
            new._value[np.ix_(index_new, index_new)] = self._value[np.ix_(index_self, index_self)]
            new._center[index_new] = self._center[index_self]
        if params_not_in_self:
            index_new = new._index(params_not_in_self)
            new._value[np.ix_(index_new, index_new)] = self._fill_value
            new._center[index_new] = np.nan
        if return_type == 'nparray':
            new = new._value
            if isscalar and not params[0].size:
                new = new[0, 0]
            return new
        return new

    def _index(self, params):
        cumsizes = np.cumsum([0] + self._sizes)
        idx = [self._params.index(param) for param in params]
        return np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx])

    def __contains__(self, name):
        """Has this parameter?"""
        return name in self._params

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in ['value', 'center']: state[name] = getattr(self, '_' + name)
        state['params'] = self._params.__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self._params = ParameterCollection.from_state(state['params'])
        for name in ['value', 'center']: setattr(self, '_' + name, state[name])

    def __repr__(self):
        """Return string representation of parameter matrix, including parameters."""
        return '{}({})'.format(self.__class__.__name__, self._params)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(np.all(getattr(other, name) == getattr(self, name)) for name in ['_params', '_value', '_center'])

    @classmethod
    @CurrentMPIComm.enable
    def bcast(cls, value, mpicomm=None, mpiroot=0):
        state = None
        if mpicomm.rank == mpiroot:
            state = value.__getstate__()
            state['value'] = None
        state = mpicomm.bcast(state, root=mpiroot)
        state['value'] = mpi.bcast(value._value if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, mpiroot=mpiroot)
        return cls.from_state(state)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __mul__(self, other):
        new = self.deepcopy()
        new._value *= other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        new = self.deepcopy()
        new._value /= other
        return new

    def __rtruediv__(self, other):
        return self.__truediv__(other)


class ParameterCovariance(BaseParameterMatrix):

    """Class that represents a parameter covariance."""

    def view(self, params=None, return_type=None, fill=None):
        """Return matrix for input parameters ``params``."""
        new = super(ParameterCovariance, self).view(params=params, return_type=None)
        if fill == 'proposal':
            params_not_in_self = [param for param in new._params if param not in self._params and param.proposal is not None]
            index = new._index(params_not_in_self)
            new._value[index, index] = [param.proposal**2 for param in params_not_in_source]
            new._center[index] = [param.value if param.value is not None else np.nan for param in params_not_in_source]
        toret = super(ParameterCovariance, new).view(return_type=return_type)
        if return_type == 'nparray' and params is not None and not is_parameter_sequence(params):
            return toret[0, 0]
        return toret

    cov = view

    def fom(self, **params):
        return self.det(**params)**(-0.5)

    def rescale(self):
        """Divide by center values."""
        new = self.deepcopy()
        new._value = self._value / (self._center[:, None] *  self._center)
        new._center[:] = 1.
        return new

    def corrcoef(self, params=None):
        """Return correlation matrix for input parameters ``params``."""
        return utils.cov_to_corrcoef(self.cov(params=params, return_type='nparray'))

    def std(self, params=None):
        cov = self.cov(params=params, return_type='nparray')
        if np.ndim(cov) == 0: return cov**0.5  # single param
        return np.diag(cov)**0.5

    def invcov(self, params=None, return_type='nparray'):
        """Return inverse covariance (precision) matrix for input parameters ``params``."""
        return self.to_precision(params=params, return_type=return_type)

    def to_precision(self, params=None, return_type=None):
        if params is None: params = self._params
        view = self.view(params, return_type=None)
        invcov = utils.inv(view._value)
        if return_type == 'nparray':
            return invcov
        return ParameterPrecision(invcov, params=params, center=view._center)

    def to_stats(self, params=None, sigfigs=2, tablefmt='latex_raw', fn=None):
        import tabulate
        is_latex = 'latex_raw' in tablefmt

        cov = self.view(params, return_type=None)
        headers = [param.latex(inline=True) if is_latex else str(param) for param in cov._params]

        txt = tabulate.tabulate([['FoM', '{:.2f}'.format(self.fom())]], tablefmt=tablefmt) + '\n'
        errors = np.diag(cov._value)**0.5
        data = [('center', 'std')] + [utils.round_measurement(value, error, sigfigs=sigfigs)[:2] for value, error in zip(cov._center, errors)]
        data = list(zip(*data))
        txt += tabulate.tabulate(data, headers=headers, tablefmt=tablefmt) + '\n'

        data = [[str(param)] + [utils.round_measurement(value, value, sigfigs=sigfigs)[0] for value in row] for param, row in zip(cov._params, cov._value)]
        txt += tabulate.tabulate(data, headers=headers, tablefmt=tablefmt)
        if fn is not None:
            utils.mkdir(os.path.dirname(fn))
            self.log_info('Saving to {}.'.format(fn))
            with open(fn, 'w') as file:
                file.write(txt)
        return txt

    def to_getdist(self, params=None, label=None, center=None, ignore_limits=True, **kwargs):
        from getdist.gaussian_mixtures import MixtureND
        toret = None
        cov = self.view(params, return_type=None)
        labels = [param.latex() for param in cov._params]
        names = [str(param) for param in cov._params]
        # ignore_limits to avoid issue in GetDist with analytic marginalization
        ranges = None
        if not ignore_limits:
            ranges = [tuple(None if limit is None or not np.isfinite(limit) else limit for limit in param.prior.limits) for param in cov._params]
        return MixtureND([cov._center if center is None else np.asarray(center)], [cov._value], lims=ranges, names=names, labels=labels, label=label)

    @classmethod
    def read_getdist(cls, base_fn):
        mean = {}
        col = None
        stats_fn = '{}.margestats'.format(base_fn)
        cls.log_info('Loading stats file: {}.'.format(stats_fn))
        with open(stats_fn, 'r') as file:
            for line in file:
                line = [item.strip() for item in line.split()]
                if line:
                    if col is not None:
                        name, value = line[0], float(line[col])
                        if not name.endswith('*'):  # covmat is not provided for derived parameters
                            param = Parameter(name, value=value, fixed=False)
                            mean[param] = value
                    if line[0] == 'parameter':
                        # Let's get the column col where to find the mean
                        for col, item in enumerate(line):
                            if item.strip() == 'mean': break

        params = list(mean.keys())
        center = list(mean.values())
        iline, col, covariance = 0, None, [None for p in params]
        covmat_fn = '{}.covmat'.format(base_fn)
        with open(covmat_fn, 'r') as file:
            for line in file:
                line = [item.strip() for item in line.split()]
                if line:
                    if col is not None and iline in col:
                        covariance[col.index(iline)] = [float(line[i]) for i in col]
                        iline += 1
                    if line[0] == '#':
                        iline, col = 0, [line.index(str(param)) - 1 for param in params]
        return cls(covariance, params=params, center=center)


class ParameterPrecision(BaseParameterMatrix):

    _fill_value = 0.

    def fom(self, **params):
        return self.to_covariance().fom(params=params)

    def cov(self, params=None, return_type='nparray'):
        return self.to_covariance(params=params, return_type=return_type)

    def to_covariance(self, params=None, return_type=None):
        cov = utils.inv(self._value)
        return ParameterCovariance(cov, params=self._params, center=self._center).view(params=params, return_type=return_type)

    @classmethod
    def sum(cls, *others):
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        params = ParameterCollection.concatenate([other._params for other in others])
        new = others[0].view(params, return_type=None)
        centers = []
        for other in others:
            view = other.view(new._params, return_type=None)
            new._value += view._value
            centers.append(view._center)
        new._center = np.nanmean(centers, axis=0)
        return new

    def __add__(self, other):
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)
