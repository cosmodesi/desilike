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
from matplotlib import pyplot as plt

from ...base import Calculator, Parameter, compile, copy, replace, get_params as _get_params
from ...base import _iter_calculators
from ...theories.galaxy_clustering.template import Spectrum2Template
from ... import plotting


def _compute_flattheory_nobao(observable):
    """Return *observable*'s flat theory vector computed with only the smooth (no-wiggle) power spectrum.

    Locates the :class:`Spectrum2Template` in *observable*'s theory dependency tree, builds
    an independent copy of *observable* (so the original and its compiled pipeline are not
    mutated), replaces the template in the copy with ``template.clone(only_now=True)``,
    then compiles and runs the copy at the current parameter values and returns its
    ``flattheory``.

    Raises :class:`ValueError` if no template is found (i.e. the theory does not support
    BAO wiggle removal).
    """
    if not any(isinstance(calc, Spectrum2Template) for calc in _iter_calculators(observable.theory)):
        raise ValueError(
            'Cannot compute no-BAO theory: no Spectrum2Template instance found in theory dependency tree.')
    # Copy the whole observable (its theory tree included) so that replacing the template
    # below does not mutate the original observable or its theory.
    nobao_observable = copy(observable, level=None)
    template_node = next(calc for calc in _iter_calculators(nobao_observable.theory)
                         if isinstance(calc, Spectrum2Template))
    replace(nobao_observable, template_node, template_node.clone(only_now=True))
    nobao_graph = compile(nobao_observable)
    current_params = {param.name: param._value for param in _get_params(nobao_graph)
                      if param._value is not None}
    nobao_graph(current_params)
    return np.asarray(nobao_observable.flattheory)


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


def _parse_templates(templates, n_data):
    """Normalise the *templates* constructor argument into a list of (Parameter, ndarray) pairs.

    Each entry may be ``(param, array)`` where *param* is either a :class:`Parameter` instance
    or a dict of keyword arguments for ``Parameter(**param)``.  *array* must be a numpy array
    of shape ``(n_data,) + param.shape``.
    """
    if templates is None:
        return []
    result = []
    for param, template_array in templates:
        if isinstance(param, dict):
            param = Parameter(**param)
        template_array = np.asarray(template_array, dtype='f8')
        expected_shape = (n_data,) + param.shape
        if template_array.shape != expected_shape:
            raise ValueError(
                f"template array for parameter '{param.name}' has shape {template_array.shape}, "
                f"expected {expected_shape} = (n_data={n_data},) + param.shape={param.shape}")
        result.append((param, template_array))
    return result


def _apply_templates(flattheory, templates):
    """Add template contributions to *flattheory* and return the result."""
    for param, template_array in templates:
        flattheory = flattheory + jnp.dot(template_array.reshape(template_array.shape[0], -1),
            jnp.atleast_1d(param.value).ravel())
    return flattheory


class Spectrum2PolesObservable(Calculator):
    r"""
    Power spectrum multipoles observable.

    Computes ``flattheory = window_matrix @ theory.poles.ravel()`` and stores
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
    templates : list of (Parameter or dict, array) pairs, or None, default=None
        Extra linear templates added to the theory prediction.  Each entry is
        ``(param, array)`` where *param* is a :class:`Parameter` instance (or a dict
        of keyword arguments for ``Parameter(**param)``) and *array* has shape
        ``(n_data,) + param.shape``.  The contribution ``array.reshape(n_data, -1) @
        ravel(param.value)`` is added to ``flattheory`` after window convolution.
    name : str, default='spectrum2poles'
        Observable name.

    Attributes set by ``__call__``
    --------------------------------
    flattheory : ndarray, shape (n_data,)
        Window-convolved theory prediction plus any template contributions.
    """

    def __init__(self, data, theory, k=None, ells=None,
                 window=None, kin=None, ellsin=None,
                 covariance=None, templates=None,
                 name='spectrum2poles'):
        self.name = str(name)
        if not isinstance(data, types.ObservableLike) and ells is not None:
            ells = [int(ell) for ell in (ells if hasattr(ells, '__iter__') else [ells])]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(
            data=data, window=window, covariance=covariance,
            coords=k, ells=ells, coordin=kin, ellsin=ellsin, coord_name='k')
        self.flatdata = self.data.value()
        self._window_matrix = self.window.value()
        # Node dep (theory) and its update() live in __init__.
        self.theory = theory
        self.theory.update(k=next(iter(self.window.theory)).coords('k'),
                           ells=self.window.theory.ells)
        self.templates = _parse_templates(templates, n_data=self.flatdata.size)

    def __call__(self):
        self.flattheory = jnp.dot(self._window_matrix, jnp.ravel(self.theory.poles))
        self.flattheory = _apply_templates(self.flattheory, self.templates)
        return self.flattheory

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, kw_theory=None, scaling='kpk', kpower=None, fig=None, figsize=None):
        """
        Plot data and theory power spectrum multipoles.

        Parameters
        ----------
        kw_theory : dict or list of dict, default=None
            Line style overrides for the theory curve, one dict per multipole.
        scaling : str, default='kpk'
            ``'kpk'``: plot k * P_ell(k).  ``'loglog'``: log-log plot of P_ell(k).
        kpower : int or None, default=None
            When not None, overrides the k exponent implied by *scaling*.
        fig : matplotlib.figure.Figure, default=None
            Existing figure with at least ``1 + len(ells)`` axes.
        figsize : tuple, default=None
            Figure size passed to :func:`matplotlib.pyplot.subplots`.
        fn : str or Path, default=None
            Path where to save the figure.
        kw_save : dict, default=None
            Extra arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            Call :func:`matplotlib.pyplot.show` after returning.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if kw_theory is None:
            kw_theory = {}
        if isinstance(kw_theory, dict):
            kw_theory = [kw_theory]
        labels = self.data.labels()
        if len(kw_theory) != len(labels):
            kw_theory = [{key: value for key, value in kw_theory[0].items()
                          if key != 'label' or ill == 0}
                         for ill in range(len(labels))]
        kw_theory = [{'color': f'C{ill:d}', **kw} for ill, kw in enumerate(kw_theory)]

        if fig is None:
            height_ratios = [max(len(labels), 3)] + [1] * len(labels)
            figsize = (6, 1.5 * sum(height_ratios)) if figsize is None else figsize
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
                                    gridspec_kw={'height_ratios': height_ratios},
                                    figsize=figsize, squeeze=True)
            fig.subplots_adjust(hspace=0.1)
            show_legend = True
        else:
            lax = fig.axes
            show_legend = False

        wtheory = self.data.clone(value=np.asarray(self.flattheory))
        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            x = data_pole.coords('k')
            xlabel = r'$k$ [$h/\mathrm{Mpc}$]'
            if scaling == 'kpk':
                k_exp = 1 if kpower is None else kpower
                scale = x ** k_exp
                ylabel = r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]'
            elif scaling == 'loglog':
                scale = 1.
                ylabel = r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]'
                lax[0].set_yscale('log')
                lax[0].set_xscale('log')
            std = self.covariance.at.observable.get(**label).std()
            lax[0].errorbar(x, scale * data_pole.value(), yerr=scale * std,
                            color=kw_theory[ill]['color'], linestyle='none', marker='o',
                            label=rf'$\ell = {ell}$')
            lax[0].plot(x, scale * wtheory_pole.value(), **kw_theory[ill])
        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            std = self.covariance.at.observable.get(**label).std()
            lax[ill + 1].plot(x, (data_pole.value() - wtheory_pole.value()) / std, **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]:
                lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(rf'$\Delta P_{{{ell}}} / \sigma_{{P_{{{ell}}}}}$')

        for ax in lax:
            ax.grid(True)
        if show_legend:
            lax[0].legend()
        lax[0].set_ylabel(ylabel)
        lax[-1].set_xlabel(xlabel)
        return fig

    @plotting.plotter
    def plot_bao(self, fig=None):
        """
        Plot BAO wiggles: data and theory minus the smooth (no-BAO) baseline.

        The smooth (no-wiggle) baseline is computed automatically by re-running
        the theory with the template's BAO wiggles switched off (``only_now=True``).

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Existing figure with at least ``len(ells)`` axes.
        fn : str or Path, default=None
            Path where to save the figure.
        kw_save : dict, default=None
            Extra arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            Call :func:`matplotlib.pyplot.show` after returning.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        labels = self.data.labels()
        if fig is None:
            figsize = (6, 2 * len(labels))
            fig, lax = plt.subplots(len(labels), sharex=True, sharey=False,
                                    figsize=figsize, squeeze=False)
            lax = lax.ravel()
            fig.subplots_adjust(hspace=0)
        else:
            lax = fig.axes

        wtheory = self.data.clone(value=np.asarray(self.flattheory))
        wtheory_nobao = self.data.clone(value=_compute_flattheory_nobao(self))

        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            wtheory_nobao_pole = wtheory_nobao.get(**label)
            std = self.covariance.at.observable.get(**label).std()
            x = data_pole.coords('k')
            color = f'C{ill:d}'
            lax[ill].errorbar(x, x * (data_pole.value() - wtheory_nobao_pole.value()),
                                  yerr=x * std, color=color, linestyle='none', marker='o')
            lax[ill].plot(x, x * (wtheory_pole.value() - wtheory_nobao_pole.value()), color=color)
            lax[ill].set_ylabel(rf'$k \Delta P_{{{ell:d}}}(k)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
            lax[ill].grid(True)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig

    def tree_flatten(self):
        return [self.flattheory, self.flatdata], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory, obj.flatdata = children
        return obj


class Correlation2PolesObservable(Calculator):
    r"""
    Correlation function multipoles observable.

    Computes ``flattheory = window_matrix @ theory.poles.ravel()`` and
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
    templates : list of (Parameter or dict, array) pairs, or None, default=None
        Extra linear templates added to the theory prediction.  Each entry is
        ``(param, array)`` where *param* is a :class:`Parameter` instance (or a dict
        of keyword arguments for ``Parameter(**param)``) and *array* has shape
        ``(n_data,) + param.shape``.  The contribution ``array.reshape(n_data, -1) @
        ravel(param.value)`` is added to ``flattheory`` after window convolution.
    name : str, default='correlation2poles'
        Observable name.

    Attributes set by ``__call__``
    --------------------------------
    flattheory : ndarray, shape (n_data,)
        Window-convolved theory prediction plus any template contributions.
    """

    def __init__(self, data, theory, s=None, ells=None,
                 window=None, sin=None, ellsin=None,
                 covariance=None, templates=None, name='correlation2poles'):
        self.name = str(name)
        if not isinstance(data, types.ObservableLike) and ells is not None:
            ells = [int(ell) for ell in (ells if hasattr(ells, '__iter__') else [ells])]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(
            data=data, window=window, covariance=covariance,
            coords=s, ells=ells, coordin=sin, ellsin=ellsin, coord_name='s')
        self.flatdata = self.data.value()
        self._window_matrix = self.window.value()
        self.theory = theory
        self.theory.update(s=next(iter(self.window.theory)).coords('s'),
                           ells=self.window.theory.ells)
        self.templates = _parse_templates(templates, n_data=self.flatdata.size)

    def __call__(self):
        self.flattheory = jnp.dot(self._window_matrix, jnp.ravel(self.theory.poles))
        self.flattheory = _apply_templates(self.flattheory, self.templates)
        return self.flattheory

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, kw_theory=None, fig=None):
        """
        Plot data and theory correlation function multipoles.

        Parameters
        ----------
        kw_theory : dict or list of dict, default=None
            Line style overrides for the theory curve, one dict per multipole.
        fig : matplotlib.figure.Figure, default=None
            Existing figure with at least ``1 + len(ells)`` axes.
        fn : str or Path, default=None
            Path where to save the figure.
        kw_save : dict, default=None
            Extra arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            Call :func:`matplotlib.pyplot.show` after returning.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if kw_theory is None:
            kw_theory = {}
        if isinstance(kw_theory, dict):
            kw_theory = [kw_theory]
        labels = self.data.labels()
        if len(kw_theory) != len(labels):
            kw_theory = [{key: value for key, value in kw_theory[0].items()
                          if key != 'label' or ill == 0}
                         for ill in range(len(labels))]
        kw_theory = [{'color': f'C{ill:d}', **kw} for ill, kw in enumerate(kw_theory)]

        if fig is None:
            height_ratios = [max(len(labels), 3)] + [1] * len(labels)
            figsize = (6, 1.5 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
                                    gridspec_kw={'height_ratios': height_ratios},
                                    figsize=figsize, squeeze=True)
            fig.subplots_adjust(hspace=0.1)
            show_legend = True
        else:
            lax = fig.axes
            show_legend = False

        wtheory = self.data.clone(value=np.asarray(self.flattheory))
        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            x = data_pole.coords('s')
            std = self.covariance.at.observable.get(**label).std()
            scale = x ** 2
            lax[0].errorbar(x, scale * data_pole.value(), yerr=scale * std,
                            color=f'C{ill:d}', linestyle='none', marker='o',
                            label=rf'$\ell = {ell}$')
            lax[0].plot(x, scale * wtheory_pole.value(), **kw_theory[ill])
        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            std = self.covariance.at.observable.get(**label).std()
            lax[ill + 1].plot(x, (data_pole.value() - wtheory_pole.value()) / std, **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]:
                lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(rf'$\Delta \xi_{{{ell}}} / \sigma_{{\xi_{{{ell}}}}}$')

        for ax in lax:
            ax.grid(True)
        if show_legend:
            lax[0].legend()
        lax[0].set_ylabel(r'$s^2 \xi_\ell(s)$ [$(\mathrm{Mpc}/h)^2$]')
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig

    @plotting.plotter
    def plot_bao(self, fig=None):
        """
        Plot BAO wiggles: data and theory minus the smooth (no-BAO) baseline.

        The smooth (no-wiggle) baseline is computed automatically by re-running
        the theory with the template's BAO wiggles switched off (``only_now=True``).

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Existing figure with at least ``len(ells)`` axes.
        fn : str or Path, default=None
            Path where to save the figure.
        kw_save : dict, default=None
            Extra arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            Call :func:`matplotlib.pyplot.show` after returning.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        labels = self.data.labels()
        if fig is None:
            figsize = (6, 2 * len(labels))
            fig, lax = plt.subplots(len(labels), sharex=True, sharey=False,
                                    figsize=figsize, squeeze=False)
            lax = lax.ravel()
            fig.subplots_adjust(hspace=0)
        else:
            lax = fig.axes

        wtheory = self.data.clone(value=np.asarray(self.flattheory))
        wtheory_nobao = self.data.clone(value=_compute_flattheory_nobao(self))

        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            wtheory_nobao_pole = wtheory_nobao.get(**label)
            std = self.covariance.at.observable.get(**label).std()
            x = data_pole.coords('s')
            scale = x ** 2
            color = f'C{ill:d}'
            lax[ill].errorbar(x, scale * (data_pole.value() - wtheory_nobao_pole.value()),
                                  yerr=scale * std, color=color, linestyle='none', marker='o')
            lax[ill].plot(x, scale * (wtheory_pole.value() - wtheory_nobao_pole.value()), color=color)
            lax[ill].set_ylabel(rf'$s^2 \Delta \xi_{{{ell:d}}}(s)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
            lax[ill].grid(True)
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig

    def tree_flatten(self):
        return [self.flattheory, self.flatdata], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory, obj.flatdata = children
        return obj


class Spectrum3PolesObservable(Calculator):
    r"""
    Bispectrum multipoles observable.

    Computes ``flattheory = window_matrix @ theory.poles.ravel()`` and
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
    templates : list of (Parameter or dict, array) pairs, or None, default=None
        Extra linear templates added to the theory prediction.  Each entry is
        ``(param, array)`` where *param* is a :class:`Parameter` instance (or a dict
        of keyword arguments for ``Parameter(**param)``) and *array* has shape
        ``(n_data,) + param.shape``.  The contribution ``array.reshape(n_data, -1) @
        ravel(param.value)`` is added to ``flattheory`` after window convolution.
    name : str, default='spectrum3poles'
        Observable name.

    Attributes set by ``__call__``
    --------------------------------
    flattheory : ndarray, shape (n_data,)
        Window-convolved theory prediction plus any template contributions.
    """

    def __init__(self, data, theory, k=None, ells=None,
                 window=None, kin=None, ellsin=None,
                 covariance=None, templates=None, name='spectrum3poles'):
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
        self.theory = theory
        self.theory.update(k=next(iter(self.window.theory)).coords('k'),
                           ells=self.window.theory.ells)
        self.templates = _parse_templates(templates, n_data=self.flatdata.size)

    def __call__(self):
        self.flattheory = jnp.dot(self._window_matrix, jnp.ravel(self.theory.poles))
        self.flattheory = _apply_templates(self.flattheory, self.templates)
        return self.flattheory

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, kw_theory=None, fig=None):
        """
        Plot data and theory bispectrum multipoles.

        Parameters
        ----------
        kw_theory : dict or list of dict, default=None
            Line style overrides for the theory curve, one dict per multipole.
        fig : matplotlib.figure.Figure, default=None
            Existing figure with at least ``1 + len(ells)`` axes.
        fn : str or Path, default=None
            Path where to save the figure.
        kw_save : dict, default=None
            Extra arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            Call :func:`matplotlib.pyplot.show` after returning.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if kw_theory is None:
            kw_theory = {}
        if isinstance(kw_theory, dict):
            kw_theory = [kw_theory]
        labels = self.data.labels()
        if len(kw_theory) != len(labels):
            kw_theory = [{key: value for key, value in kw_theory[0].items()
                          if key != 'label' or ill == 0}
                         for ill in range(len(labels))]
        kw_theory = [{'color': f'C{ill:d}', **kw} for ill, kw in enumerate(kw_theory)]

        if fig is None:
            height_ratios = [max(len(labels), 3)] + [1] * len(labels)
            figsize = (6, 1.5 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False,
                                    gridspec_kw={'height_ratios': height_ratios},
                                    figsize=figsize, squeeze=True)
            fig.subplots_adjust(hspace=0.1)
            show_legend = True
        else:
            lax = fig.axes
            show_legend = False

        wtheory = self.data.clone(value=np.asarray(self.flattheory))
        xlabel = ylabel = None
        xx = []
        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            k_array = data_pole.coords('k')
            if 'scoccimarro' in data_pole.basis:
                x = np.arange(data_pole.size)
                scale = k_array.prod(axis=-1)
                xlabel = r'triangle index'
                ylabel = r'$k_1 k_2 k_3 B_{\ell}(k_1, k_2, k_3)$ [$(\mathrm{Mpc}/h)^3$]'
            else:
                scale = k_array.prod(axis=-1)
                if np.allclose(k_array[..., 1], k_array[..., 0]):
                    x = k_array[..., 0]
                    xlabel = r'$k$ [$h/\mathrm{Mpc}$]'
                else:
                    x = np.arange(data_pole.size)
                    xlabel = r'triangle index'
                ylabel = r'$k^2 B_{\ell}(k, k)$ [$(\mathrm{Mpc}/h)^4$]'
            xx.append(x)
            std = self.covariance.at.observable.get(**label).std()
            lax[0].errorbar(x, scale * data_pole.value(), yerr=scale * std,
                            color=kw_theory[ill]['color'], linestyle='none', marker='o',
                            label=rf'$\ell = {ell}$')
            lax[0].plot(x, scale * wtheory_pole.value(), **kw_theory[ill])
        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            std = self.covariance.at.observable.get(**label).std()
            lax[ill + 1].plot(xx[ill], (data_pole.value() - wtheory_pole.value()) / std, **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]:
                lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(rf'$\Delta B_{{{ell}}} / \sigma_{{B_{{{ell}}}}}$')

        for ax in lax:
            ax.grid(True)
        if show_legend:
            lax[0].legend()
        if ylabel is not None:
            lax[0].set_ylabel(ylabel)
        if xlabel is not None:
            lax[-1].set_xlabel(xlabel)
        return fig

    def tree_flatten(self):
        return [self.flattheory, self.flatdata], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory, obj.flatdata = children
        return obj
