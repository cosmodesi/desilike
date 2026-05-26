"""
Observables for galaxy clustering.

Classes
-------
Spectrum2PolesObservable
    Power spectrum multipoles observable: applies window matrix to theory, compares to data.
Correlation2PolesObservable
    Correlation function multipoles observable: applies window matrix to theory, compares to data.
Spectrum3PolesObservable
    Bispectrum multipoles observable: applies window matrix to theory, compares to data.
"""

import numpy as np
import scipy as sp
import jax.numpy as jnp
import lsstypes as types

from ..base import Calculator


def _make_list(array, nitems=1):
    if not isinstance(array, (list, tuple)):
        array = [array] * nitems
    return list(array)


def _format_clustering_data_window_covariance(data, window, covariance,
                                               coords, ells, coordin, ellsin,
                                               coord_name='k'):
    """Convert data/window/covariance from any input format to lsstypes objects.

    Returns (data, window, covariance) all as lsstypes objects (covariance may be None).
    """
    custom_data = not isinstance(data, types.ObservableLike)
    custom_window = not isinstance(window, types.WindowMatrix)
    custom_covariance = not isinstance(covariance, types.CovarianceMatrix)

    # ── data ──────────────────────────────────────────────────────────────
    if custom_data:
        for name, value in {coord_name: coords, 'ells': ells}.items():
            if value is None:
                raise ValueError(f'when input data is an array or None, provide {name}')
        coords = _make_list(coords, len(ells))
        csizes = tuple(len(coord) for coord in coords)
        if data is None:
            data = np.zeros(sum(csizes), dtype='f8')
        else:
            data = np.ravel(data)
        if sum(csizes) != data.size:
            raise ValueError(f'total {coord_name}-size should match data, but got {csizes} and {data.size}')
        value = data
        leaves = []
        start = 0
        for ell, coord in zip(ells, coords):
            stop = start + len(coord)
            leaf = types.ObservableLeaf(**{coord_name: coord}, value=value[start:stop],
                                        coords=[coord_name], meta={'ells': ell})
            start = stop
            leaves.append(leaf)
        data = types.ObservableTree(leaves, ells=ells)

    # ── window ────────────────────────────────────────────────────────────
    if window is None:
        coordin = np.unique(np.concatenate([pole.coords(coord_name) for pole in data], axis=0), axis=0)
        win_blocks, ellsin = [], []
        for label, pole in data.items():
            mask = pole.coords(coord_name)[:, None, ...] == coordin[None, :, ...]
            if mask.ndim > 2:
                mask = mask.all(axis=-1)
            win_blocks.append(1. * mask)
            ellsin.append(label['ells'])
        window = sp.linalg.block_diag(*win_blocks)

    if custom_window:
        window = np.asarray(window)
        assert window.ndim == 2
        if coordin is None:
            raise ValueError(f'when input window is an array, provide {coord_name}in')
        if ellsin is None:
            # Default theory ells to data ells when not specified.
            ellsin = [label['ells'] for label, _ in data.items()]
        theory_leaves = []
        for ell in ellsin:
            leaf = types.ObservableLeaf(**{coord_name: coordin},
                                        value=np.zeros(coordin.shape[:1], dtype='f8'),
                                        coords=[coord_name])
            theory_leaves.append(leaf)
        theory = types.ObservableTree(theory_leaves, ells=ellsin)
        window = types.WindowMatrix(value=window, theory=theory,
                                    observable=data.clone(value=np.zeros_like(data.value())))
    elif not custom_data:
        window = window.at.observable.match(data)

    assert window.shape[0] == data.size, (
        f'output window dimension must match data size, but got {window.shape[0]} != {data.size}')

    # ── covariance ────────────────────────────────────────────────────────
    if covariance is not None:
        if custom_covariance:
            covariance = np.asarray(covariance, dtype='f8')
            if covariance.ndim == 1:
                covariance = np.diag(covariance)
            assert covariance.ndim == 2
            covariance = types.CovarianceMatrix(
                value=covariance,
                observable=data.clone(value=np.zeros_like(data.value())))
        elif not custom_data:
            covariance = covariance.at.observable.match(data)
        assert covariance.shape[0] == data.size, 'covariance shape must match data size'

    return data, window, covariance


class Spectrum2PolesObservable(Calculator):
    r"""
    Power spectrum multipoles observable.

    Computes ``flattheory = window_matrix @ theory.spectrum.ravel()`` and stores
    ``flatdata`` for comparison by a likelihood.

    Parameters
    ----------
    data : array, lsstypes.Mesh2SpectrumPoles, or None
        Flat data vector or lsstypes observable (``flatdata``, ``k``, ``ells`` are
        extracted automatically). If None, set to zeros (requires ``k``).
    theory : Calculator
        Theory calculator that exposes a ``spectrum`` attribute of shape
        ``(n_ells, n_k)`` after calling.
    k : array or list of arrays, default=None
        Wavenumbers [h/Mpc], required when ``data`` is a plain array and ``window``
        is None. A single array is shared across all multipoles.
    ells : tuple of int, default=None
        Multipole orders. Extracted from ``data`` when it is a lsstypes object;
        defaults to ``(0, 2)`` otherwise.
    window : 2-D array, lsstypes.WindowMatrix, or None, default=None
        Window matrix. If None a trivial identity selection is used.
    kin : array, default=None
        Theory wavenumbers (required when ``window`` is a plain 2-D array).
    ellsin : tuple of int, default=None
        Theory multipoles (required when ``window`` is a plain 2-D array).
    covariance : 2-D array, lsstypes.CovarianceMatrix, or None, default=None
        Covariance matrix stored for use by :class:`ObservablesGaussianLikelihood`.
    name : str, default='spectrum2poles'
        Observable name.

    Attributes set by ``__call__``
    --------------------------------
    flattheory : ndarray, shape (n_data,)
        Window-convolved theory prediction.
    """

    def __init__(self, data, theory, k=None, ells=None,
                 window=None, kin=None, ellsin=None,
                 covariance=None, name='spectrum2poles'):
        self.name = str(name)
        if not isinstance(data, types.ObservableLike) and ells is not None:
            ells = [int(ell) for ell in (ells if hasattr(ells, '__iter__') else [ells])]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(
            data=data, window=window, covariance=covariance,
            coords=k, ells=ells, coordin=kin, ellsin=ellsin, coord_name='k')
        self.flatdata = self.data.value()
        self._window_matrix = self.window.value()

    def __post_init__(self, data, theory, k=None, ells=None,
                      window=None, kin=None, ellsin=None,
                      covariance=None, name='spectrum2poles'):
        self.theory = theory
        self.theory.update(k=next(iter(self.window.theory)).coords('k'),
                           ells=self.window.theory.ells)

    def __call__(self):
        self.flattheory = jnp.dot(self._window_matrix, jnp.ravel(self.theory.spectrum))
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


class Correlation2PolesObservable(Calculator):
    r"""
    Correlation function multipoles observable.

    Computes ``flattheory = window_matrix @ theory.correlation.ravel()`` and
    stores ``flatdata`` for comparison by a likelihood.

    Parameters
    ----------
    data : array, lsstypes observable, or None
        Flat data vector or lsstypes observable (``flatdata``, ``s``, ``ells``
        are extracted automatically). If None, set to zeros (requires ``s``).
    theory : Calculator
        Theory calculator that exposes a ``correlation`` attribute of shape
        ``(n_ells, n_s)`` after calling.
    s : array or list of arrays, default=None
        Separations [Mpc/h], required when ``data`` is a plain array and
        ``window`` is None.
    ells : tuple of int, default=None
        Multipole orders. Extracted from ``data`` when it is a lsstypes object;
        defaults to ``(0, 2)`` otherwise.
    window : 2-D array, lsstypes.WindowMatrix, or None, default=None
        Window matrix. If None a trivial identity selection is used.
    sin : array, default=None
        Theory separations (required when ``window`` is a plain 2-D array).
    ellsin : tuple of int, default=None
        Theory multipoles (required when ``window`` is a plain 2-D array).
    covariance : 2-D array, lsstypes.CovarianceMatrix, or None, default=None
        Covariance matrix stored for use by :class:`ObservablesGaussianLikelihood`.
    name : str, default='correlation2poles'
        Observable name.

    Attributes set by ``__call__``
    --------------------------------
    flattheory : ndarray, shape (n_data,)
        Window-convolved theory prediction.
    """

    def __init__(self, data, theory, s=None, ells=None,
                 window=None, sin=None, ellsin=None,
                 covariance=None, name='correlation2poles'):
        self.name = str(name)
        if not isinstance(data, types.ObservableLike) and ells is not None:
            ells = [int(ell) for ell in (ells if hasattr(ells, '__iter__') else [ells])]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(
            data=data, window=window, covariance=covariance,
            coords=s, ells=ells, coordin=sin, ellsin=ellsin, coord_name='s')
        self.flatdata = self.data.value()
        self._window_matrix = self.window.value()

    def __post_init__(self, data, theory, s=None, ells=None,
                      window=None, sin=None, ellsin=None,
                      covariance=None, name='correlation2poles'):
        self.theory = theory
        self.theory.update(s=next(iter(self.window.theory)).coords('s'),
                           ells=self.window.theory.ells)

    def __call__(self):
        self.flattheory = jnp.dot(self._window_matrix, jnp.ravel(self.theory.correlation))
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj


class Spectrum3PolesObservable(Calculator):
    r"""
    Bispectrum multipoles observable.

    Computes ``flattheory = window_matrix @ theory.spectrum.ravel()`` and
    stores ``flatdata`` for comparison by a likelihood.

    Parameters
    ----------
    data : array, lsstypes.Mesh3SpectrumPoles, or None
        Flat data vector or lsstypes observable. If None, set to zeros
        (requires ``k``).
    theory : Calculator
        Theory calculator that exposes a ``spectrum`` attribute after calling.
    k : list of arrays, default=None
        Wavenumbers per multipole (required when ``data`` is a plain array and
        ``window`` is None).
    ells : tuple, default=None
        Multipole orders. Extracted from ``data`` when it is a lsstypes object;
        defaults to ``((0, 0, 0), (2, 0, 2))`` otherwise.
    window : 2-D array, lsstypes.WindowMatrix, or None, default=None
        Window matrix. If None a trivial identity selection is used.
    kin : array, default=None
        Theory wavenumbers (required when ``window`` is a plain 2-D array).
    ellsin : tuple, default=None
        Theory multipoles (required when ``window`` is a plain 2-D array).
    covariance : 2-D array, lsstypes.CovarianceMatrix, or None, default=None
        Covariance matrix stored for use by :class:`ObservablesGaussianLikelihood`.
    name : str, default='spectrum3poles'
        Observable name.

    Attributes set by ``__call__``
    --------------------------------
    flattheory : ndarray, shape (n_data,)
        Window-convolved theory prediction.
    """

    def __init__(self, data, theory, k=None, ells=None,
                 window=None, kin=None, ellsin=None,
                 covariance=None, name='spectrum3poles'):
        self.name = str(name)
        if not isinstance(data, types.ObservableLike) and ells is not None:
            if hasattr(ells[0], '__iter__'):
                ells = [tuple(ell) for ell in ells]
            else:
                ells = [int(ell) for ell in ells]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(
            data=data, window=window, covariance=covariance,
            coords=k, ells=ells, coordin=kin, ellsin=ellsin, coord_name='k')
        self.flatdata = self.data.value()
        self._window_matrix = self.window.value()

    def __post_init__(self, data, theory, k=None, ells=None,
                      window=None, kin=None, ellsin=None,
                      covariance=None, name='spectrum3poles'):
        self.theory = theory
        self.theory.update(k=next(iter(self.window.theory)).coords('k'),
                           ells=self.window.theory.ells)

    def __call__(self):
        self.flattheory = jnp.dot(self._window_matrix, jnp.ravel(self.theory.spectrum))
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj
