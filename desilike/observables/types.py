import copy

import numpy as np

from desilike.theories.primordial_cosmology import get_cosmo
from desilike import plotting
from desilike.utils import BaseClass
from desilike import utils


def _format_slice(sl, size):
    if sl is None: sl = slice(None)
    start, stop, step = sl.start, sl.stop, sl.step
    # To handle slice(0, None, 1)
    if start is None: start = 0
    if step is None: step = 1
    if stop is None: stop = (size - start) // step * step
    #start, stop, step = sl.indices(len(self._x[iproj]))
    if step < 0:
        raise IndexError('positive slicing step only supported')
    return slice(start, stop, step)


class ObservableArray(BaseClass):
    """
    Class representing observable data.

    Example
    -------
    >>> observable = ObservableArray(x=[np.linspace(0.01, 0.2, 10), np.linspace(0.01, 0.2, 10)], projs=[0, 2])
    >>> observable = observable.select(projs=2, xlim=(0., 0.15))

    Attributes
    ----------
    x : list, array, ObservableArray
        Coordinates.

    projs : list, default=None
        Projections.

    value : list, array
        Data vector value.

    weights : list, array
        Weights for rebinning.

    name : str
        Name.

    attrs : dict
        Other attributes.
    """

    def __init__(self, x=None, edges=None, projs=None, value=None, weights=None, name=None, attrs=None):
        """
        Initialize observable array.
        'x' is an axis along which it may makes sense to rebin.

        Example
        -------
        >>> observable = ObservableArray(x=[np.linspace(0.01, 0.2, 10), np.linspace(0.01, 0.2, 10)], projs=[0, 2])
        >>> observable = observable.select(projs=2, xlim=(0., 0.15))

        Parameters
        ----------
        x : list, array, ObservableArray
            Coordinates.

        edges : list, array, default=None
            Edges.

        projs : list, default=None
            Projections.

        value : list, array
            Data vector value.

        weights : list, array
            Weights for rebinning.

        name : str, default=None
            Optionally, name.

        attrs : dict, default=None
            Optionally, attributes.
        """
        if isinstance(x, self.__class__):
            self.__dict__.update(x.__dict__)
            return
        self.name = str(name or '')
        self.attrs = dict(attrs or {})
        self._projs = list(projs) if projs is not None else [None]
        if projs is None:
            x = [x]
            weights = [weights]
            if value is not None: value = [value]
        nprojs = len(self._projs)
        if x is None:
            if edges is not None:
                edges = [np.atleast_1d(xx) for xx in edges]
                self._x = [(edges[:-1] + edges[1:]) / 2. for edges in edges]
            else:
                self._x = [np.full(1, np.nan) for xx in range(nprojs)]
        else:
            self._x = [np.atleast_1d(xx) for xx in x]
        if len(self._x) != nprojs:
            raise ValueError('x should be of same length as the number of projs = {:d}, found {:d}'.format(nprojs, len(self._x)))
        if edges is None:
            self._edges = []
            for xx in self._x:
                if len(xx) >= 2 and not np.isnan(xx).all():
                    tmp = (xx[:-1] + xx[1:]) / 2.
                    tmp = np.concatenate([[tmp[0] - (xx[1] - xx[0])], tmp, [tmp[-1] + (xx[-1] - xx[-2])]])
                else:
                    tmp = np.full(len(xx) + 1, np.nan)
                self._edges.append(tmp)
        else:
            self._edges = [np.atleast_1d(xx) for xx in edges]
        shape = tuple(len(xx) for xx in self._x)
        eshape = tuple(len(edges) - 1 for edges in self._edges)
        if eshape != shape:
            raise ValueError('edges should be of length(x) - 1, found = {}'.format(shape, eshape))
        if weights is None:
            weights = [None] * len(self._x)
        self._weights = [np.atleast_1d(ww) if ww is not None else np.ones(len(xx), dtype='f8') for xx, ww in zip(self._x, weights)]
        wshape = tuple(len(ww) for ww in self._weights)
        if wshape != shape:
            raise ValueError('weights should be of same length as x = {}, found = {}'.format(shape, wshape))
        self._value = value
        if value is None:
            self._value = [np.full(len(xx), np.nan) for xx in self._x]
        else:
            self._value = [np.atleast_1d(vv) for vv in value]
            vshape = tuple(len(vv) for vv in self._value)
            if vshape != shape:
                raise ValueError('value should be of same length as x = {}, found = {}'.format(shape, vshape))

    @property
    def projs(self):
        """Projections."""
        return self._projs

    @property
    def flatx(self):
        """Flat x-coordinate array."""
        return np.concatenate(self._x, axis=0)

    @property
    def flatvalue(self):
        #self.value  # for error
        """Flat value array."""
        return np.concatenate(self._value, axis=0)

    @property
    def size(self):
        """Size of the data vector."""
        return sum(len(v) for v in self._value)

    def _slice_xmatch(self, x, projs=Ellipsis, method='mid'):
        """Return list of (proj, slice1, slice2) to apply to obtain the same x-coordinates."""
        if projs is Ellipsis: projs = self.projs
        if not isinstance(x, list):
            x = [x] * len(projs)
        toret = []
        for xx, proj in zip(x, projs):
            xx = np.asarray(xx)
            iproj = self._index_projs(proj)
            if method == 'mid': sx = (self._edges[iproj][:-1] + self._edges[iproj][1:]) / 2.
            else: sx = self._x[iproj]
            found = False
            for step in range(1, len(sx) // len(xx) + 1):
                sl1 = slice(0, len(sx) // step * step, step)
                if method == 'mid':
                    edges = self._edges[iproj][::sl1.step]
                    x1 = (edges[:-1] + edges[1:]) / 2.
                else:
                    nmatrix1 = self._slice_matrix(sl1, projs=proj, normalize=True)
                    x1 = nmatrix1.dot(sx)
                index = np.flatnonzero(np.isclose(xx[0], x1, equal_nan=True))
                if index.size:
                    if len(xx) > 1:
                        if np.allclose(xx, x1[index[0]:index[0] + len(xx)], equal_nan=True):
                            found = True
                            break
                    else:
                        found = True
                        break
            if not found:
                raise ValueError('could not find slice to match {} to {} (proj {})'.format(xx, sx, proj))
            sl2 = slice(index[0], index[0] + len(xx), 1)
            toret.append((proj, sl1, sl2))
        return toret

    def xmatch(self, x, projs=Ellipsis, select_projs=False, method='mid'):
        """
        Apply selection to match input x-coordinates.

        Parameters
        ----------
        x : array, list
            Coordinates. Can be a list of ``x`` for the list of ``projs``.

        projs : list, default=None
            List of projections.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableArray
        """
        new = self.slice()
        for proj, sl1, sl2 in self._slice_xmatch(x=x, projs=projs, method=method):
            new = new.slice(sl1, projs=proj)
            new = new.slice(sl2, projs=proj)
        if select_projs:
            new = new.slice(projs=projs, select_projs=True)
        return new

    def _index(self, xlim, projs=Ellipsis, method='mid', concatenate=True):
        """
        Return indices for given input x-limits and projs.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projs : list, default=None
            List of projections to return indices for.
            Defaults to :attr:`projs`.

        concatenate : bool, default=True
            If ``False``, return list of indices, for each input projection.

        Returns
        -------
        indices : array, list
        """
        iprojs = self._index_projs(projs)
        if not isinstance(iprojs, list): iprojs = [iprojs]
        toret = []
        for iproj in iprojs:
            if method == 'mid': xx = (self._edges[iproj][:-1] + self._edges[iproj][1:]) / 2.
            else: xx = self._x[iproj]
            if xlim is not None:
                tmp = (xx >= xlim[0]) & (xx <= xlim[1])
                tmp = np.all(tmp, axis=tuple(range(1, tmp.ndim)))
            else:
                tmp = np.ones(xx.shape[0], dtype='?')
                #print(xlim, self._x[iproj].shape, self._edges[iproj].shape)
            tmp = np.flatnonzero(tmp)
            if concatenate: tmp += sum(len(xx) for xx in self._x[:iproj])
            toret.append(tmp)
        if concatenate:
            return np.concatenate(toret, axis=0)
        #if isscalar:
        #    return toret[0]
        return toret

    def _index_projs(self, projs=Ellipsis):
        """Return projs indices."""
        if projs is Ellipsis:
            return list(range(len(self._projs)))
        isscalar = not isinstance(projs, list)
        if isscalar: projs = [projs]
        toret = []
        for proj in projs:
            try: iproj = self.projs.index(proj)
            except (ValueError, IndexError):
                raise ValueError('{} could not be found in {}'.format(proj, self.projs))
            toret.append(iproj)
        if isscalar:
            return toret[0]
        return toret

    def select(self, xlim=None, rebin=1, projs=Ellipsis, select_projs=False, method='mid'):
        """
        Apply x-cuts for given projections.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        rebin : int, default=1
            Optionally, rebinning factor (after ``xlim`` cut).

        projs : list, default=None
            List of projections to apply ``xlim`` and ``rebin`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableArray
        """
        iprojs = self._index_projs(projs)
        if not isinstance(iprojs, list): iprojs = [iprojs]
        self = self.slice(slice(0, None, rebin), projs=projs)
        x, edges, weights, value = list(self._x), list(self._edges), list(self._weights), list(self._value)
        for iproj in iprojs:
            index = self._index(xlim=xlim, projs=[self._projs[iproj]], method=method, concatenate=False)[0]
            x[iproj] = self._x[iproj][index]
            edges[iproj] = self._edges[iproj][np.append(index, index[-1] + 1)]
            weights[iproj] = self._weights[iproj][index]
            value[iproj] = self._value[iproj][index]
        projs = self.projs
        if select_projs:
            x, edges, projs, value, weights = ([v[iproj] for iproj in iprojs] for v in [x, edges, projs, value, weights])
        return self.__class__(x=x, edges=edges, projs=projs, value=value, weights=weights, name=self.name, attrs=self.attrs)

    def _slice_matrix(self, sl=None, projs=Ellipsis, weighted=True, normalize=True):
        # Return, for a given slice, the corresponding matrix to apply to the data arrays.
        toret = []
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: iprojs = [iprojs]
        if sl is None: sl = slice(None)
        for iproj in iprojs:
            sl = _format_slice(sl, len(self._x[iproj]))
            start, stop, step = sl.start, sl.stop, sl.step
            oneslice = slice(start, stop, 1)
            ww = self._weights[iproj][oneslice]
            if not weighted:
                ww = np.ones_like(ww)
            if len(ww) % step != 0:
                raise IndexError('slicing step = {:d} does not divide length {:d}'.format(step, len(ww)))
            tmp_lim = np.zeros((len(ww), len(self._weights[iproj])), dtype='f8')
            tmp_lim[np.arange(tmp_lim.shape[0]), start + np.arange(tmp_lim.shape[0])] = 1.
            tmp_bin = np.zeros((len(ww) // step, len(ww)), dtype='f8')
            #print(np.repeat(np.arange(tmp_bin.shape[0]), step).shape, np.arange(tmp_bin.shape[-1]).shape, ww.shape)
            tmp_bin[np.repeat(np.arange(tmp_bin.shape[0]), step), np.arange(tmp_bin.shape[-1])] = ww
            #print(step, self._projs[iproj], ww, np.sum(tmp_bin, axis=-1))
            if normalize: tmp_bin /= np.sum(tmp_bin, axis=-1)[:, None]
            toret.append(tmp_bin.dot(tmp_lim))
        if isscalar:
            return toret[0]
        return toret

    def slice(self, slice=None, projs=Ellipsis, select_projs=False):
        """
        Apply selections to the data, slicing for given projections.

        Parameters
        ----------
        slice : slice, default=None
            Slicing to apply, defaults to ``slice(None)``.

        projs : list, default=None
            List of projections to apply ``slice`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableArray
        """
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: iprojs = [iprojs]
        x, edges, weights, value = list(self._x), list(self._edges), list(self._weights), list(self._value)
        for iproj in iprojs:
            proj = self._projs[iproj]
            xx, ww, vv = self._x[iproj], self._weights[iproj], self._value[iproj]
            sl = _format_slice(slice, len(self._x[iproj]))
            start, stop, step = sl.start, sl.stop, sl.step
            matrix = self._slice_matrix(slice, projs=proj, weighted=False, normalize=False)
            nwmatrix = self._slice_matrix(slice, projs=proj, weighted=True, normalize=True)
            x[iproj] = nwmatrix.dot(xx)
            edges[iproj] = self._edges[iproj][start::step][:len(x[iproj]) + 1]
            weights[iproj] = matrix.dot(ww)
            #print(proj, weights[iproj], slice, ww)
            value[iproj] = nwmatrix.dot(vv)
        #if isscalar:
        #    x, projs, weights, value = x[0], None, weights[0], value[0]
        projs = self.projs
        if select_projs:
            x, edges, projs, value, weights = ([v[iproj] for iproj in iprojs] for v in [x, edges, projs, value, weights])
        return self.__class__(x=x, edges=edges, projs=projs, value=value, weights=weights, name=self.name, attrs=self.attrs)

    def view(self, xlim=None, projs=Ellipsis, method='mid', return_type='nparray'):
        """
        Return observable array for input x-limits and projections.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projs : list, default=None
            Restrict to these projections.
            Defaults to :attr:`projs`.

        return_type : str, default='nparray'
            If 'nparray', return numpy array :attr:`flatvalue`.
            Else, return a new :class:`ObservableArray`, restricting to ``xlim`` and ``projs``.

        Returns
        -------
        new : array, ObservableArray
        """
        toret = self.select(xlim=xlim, projs=projs, select_projs=True, method=method)
        if return_type is None:
            return toret
        return toret.flatvalue

    def x(self, projs=Ellipsis):
        """x-coordinates (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._x[iprojs]
        return [self._x[iproj] for iproj in iprojs]

    def xavg(self, projs=Ellipsis, method='mid'):
        """x-coordinates (optionally restricted to input projs)."""
        def get_x(iproj):
            if method == 'mid':
                return (self._edges[iproj][:-1] + self._edges[iproj][1:]) / 2.
            return self._x[iproj]
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return get_x(iprojs)
        return [get_x(iproj) for iproj in iprojs]

    def edges(self, projs=Ellipsis):
        """edges (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._edges[iprojs]
        return [self._edges[iproj] for iproj in iprojs]

    def weights(self, projs=Ellipsis):
        """Weights (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._weights[iprojs]
        return [self._weights[iproj] for iproj in iprojs]

    def __repr__(self):
        """Return string representation of observable data."""
        return '{}(name={}, projs={}, size={:d})'.format(self.__class__.__name__, self.name, self._projs, self.size)

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        for name in ['x', 'edges', 'projs', 'value', 'weights']:
            state[name] = getattr(self, '_' + name)
        for name in ['name', 'attrs']:
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        for name in ['x', 'edges', 'projs', 'value', 'weights']:
            setattr(self, '_' + name, state[name])
        for name in ['name', 'attrs']:
            setattr(self, name, state[name])

    def __array__(self, *args, **kwargs):
        return np.asarray(self.flatvalue, *args, **kwargs)

    def deepcopy(self):
        """Deep copy"""
        return copy.deepcopy(self)

    @plotting.plotter
    def plot(self, xlabel=None, ylabel=None, fig=None):
        """
        Plot data.

        Parameters
        ----------
        xlabel : str, default=None
            Optionally, label for the x-axis.

        ylabel : str, default=None
            Optionally, label for the y-axis.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for iproj, proj in enumerate(self.projs):
            ax.plot(self._x[iproj], self._value[iproj], color='C{:d}'.format(iproj), linestyle='-', label=str(proj))
        ax.grid(True)
        if self.projs: ax.legend()
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        return fig


class ObservableCovariance(BaseClass):
    """
    Class representing the covariance of observable data.

    Example
    -------
    >>> cov = ObservableCovariance(np.eye(30), observables=[{'name': 'PowerSpectrumMultipoles', 'x': [np.linspace(0.01, 0.2, 10)] * 3, 'projs': [0, 2, 4]}])

    Attributes
    ----------
    value : array
        Covariance 2D array.

    observables : list
        List of observables correspondonding to the covariance.

    attrs : dict
        Other attributes.
    """

    def __init__(self, value, observables, nobs=None, attrs=None):
        """
        Initialize :class:`ObservableCovariance`.

        Parameters
        ----------
        value : array
            Covariance 2D array.

        observables : list
            List of observables, :class:`ObservableArray` or dict to initialize such array.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        if isinstance(value, self.__class__):
            self.__dict__.update(value.__dict__)
            return
        if not isinstance(observables, list): observables = [observables]
        self._value = np.array(value, dtype='f8')
        shape = self._value.shape
        if shape[1] != shape[0]:
            raise ValueError('Input matrix must be square')
        self._observables = [observable if isinstance(observable, ObservableArray) else ObservableArray(**observable) for observable in observables]
        sizes = [observable.size for observable in self._observables]
        size = sum(sizes)
        if size != self._value.shape[0]:
            raise ValueError('size = {:d} = sum({}) of input observables must match input matrix shape = {}'.format(size, sizes, self._value.shape[0]))
        self.nobs = int(nobs) if nobs is not None else None
        self.attrs = dict(attrs or {})

    @classmethod
    def from_observations(cls, observations):
        """
        Construct covariance matrix from list of observations.

        Parameters
        ----------
        observations : dict, list
            Either a dictionary, with keys corresponding to the observable names, e.g.
            ``{'name1': [{'x': x, 'value': value}, ...], 'name2': [{'x': x, 'value': value}, ...]}``
            or a list of observations, optionally with several observables: ``[{'name': 'name1', 'x': x, 'value': value}, ...]``
            or ``[[{'name': 'name1', 'x': x, 'value': value}, {'name': 'name2', 'x': x, 'value': value}], ...]``.

        Returns
        -------
        new : ObservableCovariance
        """
        if hasattr(observations, 'items'):
            nobs = 0
            for name, observation in observations.items():
                nobs = len(observation)
                break
            observations = [[{'name': name, **observation[iobs]} for name, observation in observations.items()] for iobs in range(nobs)]
        values, x, edges, weights, projs, nobservables = [], [], [], [], [], 0
        nobs = len(observations)
        if not nobs: raise ValueError('no observations found, cannot compute covariance matrix')
        for observation in observations:
            if not isinstance(observation, list): observation = [observation]
            observation = [observable if isinstance(observable, ObservableArray) else ObservableArray(**observable) for observable in observation]
            nobservables = len(observation)
            values.append([observable._value for observable in observation])
            x.append([observable._x for observable in observation])
            edges.append([observable._edges for observable in observation])
            weights.append([observable._weights for observable in observation])
            projs = [observable.projs for observable in observation]
            name = [observable.name for observable in observation]
        observables = []
        for iobs in range(nobservables):
            vv = [np.mean([vv[iobs][iproj] for vv in values], axis=0) for iproj in range(len(projs[iobs]))]
            xx = [np.mean([xx[iobs][iproj] for xx in x], axis=0) for iproj in range(len(projs[iobs]))]
            ee = [np.mean([ee[iobs][iproj] for ee in edges], axis=0) for iproj in range(len(projs[iobs]))]
            ww = [np.mean([ww[iobs][iproj] for ww in weights], axis=0) for iproj in range(len(projs[iobs]))]
            observables.append(ObservableArray(x=xx, edges=ee, value=vv, weights=ww, projs=projs[iobs], name=name[iobs]))
        values = [np.concatenate([np.concatenate(vv, axis=0) for vv in value]) for value in values]
        cov = np.cov(values, rowvar=False, ddof=1)
        return cls(cov, observables=observables, nobs=nobs)

    def hartlap2017_factor(self):
        """Return Hartlap factor (:math:`< 1`), to apply to the precision matrix."""
        if self.nobs is None: return 1.
        nbins = self.shape[0]
        return (self.nobs - nbins - 2.) / (self.nobs - 1.)

    def percival2014_factor(self, nparams):
        """Return Percival 2014 factor, to apply to the parameter covariance matrix."""
        if self.nobs is None: return 1.
        nbins = self.shape[0]
        A = 2. / (self.nobs - nbins - 1.) / (self.nobs - nbins - 4.)
        B = (self.nobs - nbins - 2.) / (self.nobs - nbins - 1.) / (self.nobs - nbins - 4.)
        return (1 + B * (nbins - nparams)) / (1 + A + B * (nparams + 1))

    def _observable_index(self, observables=None):
        # Return the indices corresponding to the given observables (:class:`ObservableArray`, :attr:`ObservableArray.name` or index integer).
        if observables is None: observables = list(range(len(self._observables)))
        isscalar = not isinstance(observables, list)
        if isscalar: observables = [observables]
        from desilike.parameter import find_names
        names = [obs.name for obs in self._observables]
        observable_indices = []
        for iobs in observables:
            if isinstance(iobs, ObservableArray):
                iobs = self._observables.index(iobs)
                observable_indices.append(iobs)
            elif isinstance(iobs, str):
                tmp = find_names(names, iobs, quiet=True)
                if tmp == [iobs]:
                    observable_indices.append(names.index(iobs))
                else:
                    tmp = [names.index(iobs) for iobs in tmp]
                    observable_indices += [iobs for iobs in tmp if iobs not in observable_indices]
            else:
                iobs = int(iobs)
                observable_indices.append(iobs)
        if isscalar and len(observable_indices) == 1:
            observable_indices = observable_indices[0]
        return observable_indices

    def _slice_matrix(self, slice, observables=None, projs=Ellipsis):
        # Return, for a given slice, the corresponding matrix to apply to the data arrays.
        import scipy
        observable_indices = self._observable_index(observables=observables)
        if not isinstance(observable_indices, list): observable_indices = [observable_indices]
        if projs is not Ellipsis and not isinstance(projs, list): projs = [projs]
        matrix = []
        for iobs, observable in enumerate(self._observables):
            #all_projs = observable.projs# or [None]
            proj_indices = observable._index_projs(projs) if iobs in observable_indices else []  #if projs is not Ellipsis else list(range(len(observable.projs)))
            for iproj, proj in enumerate(observable.projs):
                matrix.append(observable._slice_matrix(slice if iproj in proj_indices else None, projs=proj))
        return scipy.linalg.block_diag(*matrix)

    def slice(self, slice=None, observables=None, projs=Ellipsis, select_observables=False, select_projs=False):
        """
        Apply selections to the covariance, slicing for given observables and projections.

        Parameters
        ----------
        slice : slice, default=None
            Slicing to apply, defaults to ``slice(None)``.

        observables : list
            List of observables (:class:`ObservableArray`, :attr:`ObservableArray.name` or index integer) to apply ``slice`` to.

        projs : list, default=None
            List of projections to apply ``slice`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableCovariance
        """
        observable_indices = self._observable_index(observables=observables)
        if not isinstance(observable_indices, list): observable_indices = [observable_indices]
        observables = []
        for iobs, observable in enumerate(self._observables):
            observable = observable.slice(slice if iobs in observable_indices else None, projs=projs if iobs in observable_indices else Ellipsis)
            observables.append(observable)
        matrix = self._slice_matrix(slice, observables=observable_indices, projs=projs)
        value = matrix.dot(self._value).dot(matrix.T)
        self = self.__class__(value=value, observables=observables, nobs=self.nobs, attrs=self.attrs)
        if select_observables or select_projs:
            index, observables = [], []
            for iobs, observable in enumerate(self._observables):
                if select_observables and iobs not in observable_indices: continue
                sprojs = select_projs and iobs in observable_indices
                index.append(self._index(observables=iobs, projs=projs if sprojs else Ellipsis, concatenate=True))
                observable = observable.select(projs=projs if sprojs else Ellipsis, select_projs=sprojs)
                observables.append(observable)
            index = np.concatenate(index, axis=0)
            value = self._value[np.ix_(index, index)]
            self = self.__class__(value=value, observables=observables, nobs=self.nobs, attrs=self.attrs)
        return self

    def xmatch(self, x, observables=None, projs=Ellipsis, select_observables=False, select_projs=False, method='mid'):
        """
        Apply selection to match input x-coordinates.

        Parameters
        ----------
        x : array, list
            Coordinates. Can be a list of ``x`` for the list of ``projs``.

        observables : list
            List of observables (:class:`ObservableArray`, :attr:`ObservableArray.name` or index integer) to match.

        projs : list, default=None
            List of projections.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableCovariance
        """
        new = self.slice()
        observable_indices = self._observable_index(observables=observables)
        if not isinstance(observable_indices, list): observable_indices = [observable_indices]
        for iobs in observable_indices:
            for proj, sl1, sl2 in self._observables[iobs]._slice_xmatch(x=x, projs=projs, method=method):
                new = new.slice(sl1, observables=iobs, projs=proj)
                new = new.slice(sl2, observables=iobs, projs=proj)
        if select_observables or select_projs:
            new = new.slice(observables=observable_indices, projs=projs, select_observables=select_observables, select_projs=select_projs)
        return new

    def _index(self, observables=None, xlim=None, projs=Ellipsis, method='mid', concatenate=True):
        """
        Return indices for given input observables, x-limits and projs.

        Parameters
        ----------
        observables : list
            List of observables (:class:`ObservableArray`, :attr:`ObservableArray.name` or index integer) to return index for.

        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projs : list, default=None
            List of projections to return indices for.
            Defaults to :attr:`projs`.

        concatenate : bool, default=True
            If ``False``, return list of indices, for each input observable and projection.

        Returns
        -------
        indices : array, list
        """
        observable_indices = self._observable_index(observables=observables)
        if not isinstance(observable_indices, list): observable_indices = [observable_indices]
        indices = []
        for iobs in observable_indices:
            observable = self._observables[iobs]
            #print(observable.projs, observables, iobs, projs)
            index = observable._index(xlim=xlim, projs=projs, method=method, concatenate=concatenate)
            if concatenate:
                index += sum(observable.size for observable in self._observables[:iobs])
            indices.append(index)
        if concatenate:
            indices = np.concatenate(indices, axis=0)
        return indices

    def select(self, xlim=None, rebin=1, observables=None, projs=Ellipsis, select_observables=False, select_projs=False, method='mid'):
        """
        Apply selections for given observables and projections.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        rebin : int, default=1
            Optionally, rebinning factor (after ``xlim`` cut).

        observables : list
            List of observables (:class:`ObservableArray`, :attr:`ObservableArray.name` or index integer) to apply ``xlim`` and ``rebin`` to.

        projs : list, default=None
            List of projections to apply ``xlim`` and ``rebin`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableArray
        """
        observable_indices = self._observable_index(observables=observables)
        if not isinstance(observable_indices, list): observable_indices = [observable_indices]
        if projs is not Ellipsis and not isinstance(projs, list): projs = [projs]
        #print(observable_indices, projs)
        self = self.slice(slice(0, None, rebin), observables=observable_indices, projs=projs)
        observables, indices = [], []
        for iobs, observable in enumerate(self._observables):
            observable = self._observables[iobs]
            if iobs in observable_indices:
                #all_projs = observable.projs# or [None]
                proj_indices = observable._index_projs(projs)
                index = np.concatenate([observable._index(xlim=xlim if iproj in proj_indices else None, projs=proj, method=method, concatenate=True) for iproj, proj in enumerate(observable.projs)])
                observable = observable.select(xlim=xlim, projs=projs, method=method)
            else:
                index = np.arange(observable.size)
            index += sum(observable.size for observable in self._observables[:iobs])
            observables.append(observable)
            indices.append(index)
        index = np.concatenate(indices, axis=0)
        self = self.__class__(value=self._value[np.ix_(index, index)], observables=observables, nobs=self.nobs, attrs=self.attrs)
        if select_observables or select_projs:
            index, observables = [], []
            for iobs, observable in enumerate(self._observables):
                if select_observables and iobs not in observable_indices: continue
                sprojs = select_projs and iobs in observable_indices
                index.append(self._index(observables=iobs, projs=projs if sprojs else Ellipsis, concatenate=True))
                observable = observable.select(projs=projs if sprojs else Ellipsis, select_projs=sprojs)
                observables.append(observable)
            index = np.concatenate(index, axis=0)
            value = self._value[np.ix_(index, index)]
            self = self.__class__(value=value, observables=observables, nobs=self.nobs, attrs=self.attrs)
        return self

    def view(self, observables=None, xlim=None, projs=Ellipsis, method='mid', return_type='nparray'):
        """
        Return observable covariance for input x-limits and projections.

        Parameters
        ----------
        observables : list
            List of observables (:class:`ObservableArray`, :attr:`ObservableArray.name` or index integer) to return covariance for.

        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projs : list, default=None
            Restrict to these projections.
            Defaults to :attr:`projs`.

        return_type : str, default='nparray'
            If 'nparray', return numpy array :attr:`value`.
            Else, return a new :class:`ObservableCovariance`, restricting to ``xlim`` and ``projs``.

        Returns
        -------
        new : array, ObservableCovariance
        """
        toret = self.select(observables=observables, xlim=xlim, projs=projs, select_observables=True, select_projs=True, method=method)
        if return_type is None:
            return toret
        return toret._value

    def corrcoef(self, **kwargs):
        """Return correlation matrix array (optionally restricted to input observables, xlim and projs)."""
        return utils.cov_to_corrcoef(self.view(**kwargs, return_type='nparray'))

    def var(self, **kwargs):
        """Return variance (optionally restricted input observables / xlim / projs)."""
        cov = self.view(**kwargs, return_type='nparray')
        if np.ndim(cov) == 0: return cov  # single param
        return np.diag(cov)

    def std(self, **kwargs):
        """Return standard deviation (optionally restricted to input observables, xlim and projs)."""
        return self.var(**kwargs)**0.5

    def inv(self):
        """Return the inverse of the covariance."""
        indices = [self._index(observables=observable, concatenate=True) for observable in self._observables]
        # blockinv to help with numerical errors
        return utils.blockinv([[self._value[np.ix_(index1, index2)] for index2 in indices] for index1 in indices])

    def marginalize(self, templates, prior=1., **kwargs):
        """
        Marginalize over input templates.

        Parameters
        ----------
        templates : list, array
            (List of) templates to marginalize over

        prior : float, array
            Prior covariance for input ``templates``.

        **kwargs : dict
            Optionally, :meth:`_index` arguments, specifying for which part of the data input templates correspond to:
            ``observables``, ``xlim`` and ``projs``.

        Returns
        -------
        new : ObservableCovariance
        """
        index = self._index(**kwargs, concatenate=True)
        templates = np.atleast_2d(np.asarray(templates, dtype='f8'))  # adds first dimension
        deriv = np.zeros(templates.shape[:1] + self.shape[:1], dtype='f8')
        deriv[..., index] = templates
        invcov = self.inv()
        fisher = deriv.dot(invcov).dot(deriv.T)
        derivp = deriv.dot(invcov)
        prior = np.array(prior)
        if prior.ndim == 2:
            iprior = utils.inv(prior)
        else:
            iprior = np.ones(templates.shape[:1], dtype='f8')
            iprior[...] = prior
            iprior = np.diag(1. / iprior)
        fisher += iprior
        invcov = invcov - derivp.T.dot(np.linalg.solve(fisher, derivp))
        indices = [self._index(observables=observable, concatenate=True) for observable in self._observables]
        value = utils.blockinv([[invcov[np.ix_(index1, index2)] for index2 in indices] for index1 in indices])
        return self.clone(value=value)

    def clone(self, value=None, observables=None, attrs=None):
        """Clone observable covariance, with input ``value``."""
        new = self.view(observables=observables, return_type=None)
        if value is not None:
            new._value[...] = value
        if attrs is not None:
            new.attrs = dict(attrs)
        return new

    def observables(self, *args, **kwargs):
        """List of observables."""
        index = self._observable_index(*args, **kwargs)
        if isinstance(index, list):
            return [self._observables[i] for i in index]
        return self._observables[index]

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {'nobs': self.nobs, 'attrs': self.attrs}
        state['value'] = self._value
        state['observables'] = [obs.__getstate__() for obs in self._observables]
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        self._observables = [ObservableArray.from_state(obs) for obs in state['observables']]
        self._value = state['value']
        self.nobs = state.get('nobs', None)
        self.attrs = state.get('attrs', {})

    def __repr__(self):
        """Return string representation of observable covariance."""
        return '{}({}, shape={})'.format(self.__class__.__name__, self._observables, self.shape)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(utils.deep_eq(getattr(other, name), getattr(self, name)) for name in ['observables', '_value'])

    def __array__(self, *args, **kwargs):
        return np.asarray(self._value, *args, **kwargs)

    def deepcopy(self):
        """Deep copy"""
        return copy.deepcopy(self)

    @property
    def shape(self):
        """Return covariance matrix shape."""
        return self._value.shape

    @plotting.plotter
    def plot(self, corrcoef=True, split_projs=True, **kwargs):
        """
        Plot covariance matrix.

        Parameters
        ----------
        corrcoef : bool, default=True
            If ``True``, plot the correlation matrix; else the covariance.

        barlabel : str, default=None
            Optionally, label for the color bar.

        label1 : str, list of str, default=None
            Optionally, label(s) for the observable(s).

        figsize : int, tuple, default=None
            Optionally, figure size.

        norm : matplotlib.colors.Normalize, default=None
            Scales the covariance / correlation to the canonical colormap range [0, 1] for mapping to colors.
            By default, the covariance / correlation range is mapped to the color bar range using linear scaling.

        labelsize : int, default=None
            Optionally, size for labels.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``len(self._observables) * len(self._observables)`` axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from desilike.observables.plotting import plot_covariance_matrix
        indices = []
        for observable in self._observables:
            if split_projs:
                all_projs = observable.projs# or [None]
                for proj in all_projs:
                    indices.append(self._index(observables=observable, projs=proj, concatenate=True))
            else:
                indices.append(self._index(observables=observable, concatenate=True))
        mat = [[self._value[np.ix_(index1, index2)] for index2 in indices] for index1 in indices]
        return plot_covariance_matrix(mat, corrcoef=corrcoef, **kwargs)