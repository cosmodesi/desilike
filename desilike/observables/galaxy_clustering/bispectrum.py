import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import lsstypes as types

from desilike import plotting
from desilike.base import BaseCalculator
from desilike.jax import numpy as jnp


def _make_list(array, nitems=1):
    if not isinstance(array, (tuple, list)):
        array = [array] * nitems
    return list(array)


class TracerBispectrumMultipolesObservable(BaseCalculator):
    """
    Tracer bispectrum multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : array, lsstypes.Mesh3SpectrumPoles, default=None
        Data bispectrum measurement. Either:
        - flat array (of all multipoles). Additionally provide list of multipoles ``ells`` and wavenumbers ``k`` and optionally ``shotnoise``;
        - :class:`lsstypes.Mesh3SpectrumPoles` (contains all necessary information).
    window : array, lsstypes.WindowMatrix, default=None
        Window matrix. Either:
        - ``None``. A trivial window matrix is assumed.
        - 2D array. Additionally provide input array ``kin`` and input multipoles ``ellsin``.
        - :class:`lsstypes.WindowMatrix` (contains all necessary information).
    covariance : array, list, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``.
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information).
    theory: BaseCalculator
        Theory for the bispectrum.
    k : tuple of arrays, default=None
        If ``data`` is an array. List of wavenumbers (for each multipole) of shape (nk, 3) (Scoccimarro basis)
        or (nk, 2) (Sugiyama basis). ``sum(len(kk) for kk in k)`` should match ``len(data)``.
    ells : tuple
        Data bispectrum multipole orders. Typically:
        - [(0, 0, 0), (2, 0, 2)] for the Sugiyama basis;
        - [0, 2] for the Scoccimarro basis.
    kin : array, optional
        If ``window`` is a 2D array. Theory wavenumbers (assumed same for each multipole).
    ellsin : tuple
        If ``window`` is a 2D array. Theory bispectrum multipole orders. Typically:
        - [(0, 0, 0), (2, 0, 2)] for the Sugiyama basis;
        - [0, 2] for the Scoccimarro basis.
    shotnoise : float, optional
        Shot noise to scale theory stochastic parameters.
    """
    name = 'spectrum3poles'

    def initialize(self, data=None, window=None, covariance=None, theory=None,
                   k=None, ells=None, basis='sugiyama',
                   kin=None, ellsin=None, shotnoise=None):
        assert data is not None
        assert theory is not None
        self.theory = theory
        custom_data = not isinstance(data, types.ObservableLike)
        custom_window = not isinstance(window, types.WindowMatrix)
        custom_covariance = not isinstance(covariance, types.CovarianceMatrix)
        if custom_data:
            data = np.ravel(data)
            for name, value in {'k': k, 'ells': ells}.items():
                if value is None:
                    raise ValueError(f'when input data is an array, provide {name}')
            if basis == 'sugiyama':
                if np.ndim(ells) == 0:
                    raise ValueError('input ell should be a list of tuples, e.g. [(0, 0, 0), (2, 0, 2)]')
                if np.ndim(ells[0]) == 0:
                    ells = [ells]
                ells = [tuple(ell) for ell in ells]
            k = _make_list(k, len(ells))
            ksizes = tuple(len(kk) for kk in k)
            assert sum(ksizes) == data.size, f'total k-size should be the same as data, but got {ksizes} and {data.size}'
            value = data
            data = []
            start = 0
            for ell, kk in zip(ells, k):
                stop = start + len(kk)
                leaf = types.ObservableLeaf(k=kk, value=value[start:stop], coords=['k'], meta={'basis': basis})
                start = stop
                data.append(leaf)
            data = types.ObservableTree(data, ells=ells)
        if window is None:
            kin = np.unique(np.concatenate([pole.coords('k') for pole in data], axis=0), axis=0)
            window, ellsin = [], []
            for label, pole in data.items():
                tmp = np.all(pole.coords('k')[:, None, :] == kin[None, :, :], axis=-1)
                window.append(1. * tmp)
                ellsin.append(label['ells'])
            window = sp.linalg.block_diag(*window)
        if custom_window:
            window = np.array(window)
            assert window.ndim == 2
            theory = []
            start = 0
            for ell in ellsin:
                leaf = types.ObservableLeaf(k=kin, value=np.zeros_like(kin[..., 0]), coords=['k'])
                theory.append(leaf)
            theory = types.ObservableTree(theory, ells=ellsin)
            window = types.WindowMatrix(value=window, theory=theory, observable=data.clone(value=np.zeros_like(data.value())))
        elif not custom_data:  # match
            window = window.at.observable.match(data)
        assert window.shape[0] == data.size, f'output window dimension should match data size, but got {window.shape[0]} != {data.size}'
        if covariance is not None:
            if custom_covariance:
                covariance = np.array(covariance)
                assert covariance.ndim == 2
                covariance = types.CovarianceMatrix(value=covariance, observable=data.clone(value=np.zeros_like(data.value())))
            elif not custom_data:  # match
                covariance = covariance.at.observable.match(data)
            assert covariance.shape[0] == data.size, 'covariance shape should match data size'
        self.data, self.window, self.cov = data, window, covariance
        self.flatdata = self.data.value()
        self.covariance = self.cov.value()
        self.theory.init.update(k=next(iter(self.window.theory)).coords('k'), ells=self.window.theory.ells)
        if shotnoise is not None:
            self.theory.init.update(shotnoise=shotnoise)

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, kw_theory=None, fig=None):
        """
        Plot data and theory bispectrum multipoles.

        Parameters
        ----------
        kw_theory : list of dict, default=None
            Change the default line parametrization of the theory, one dictionary for each ell or duplicate it.
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1 + len(self.ells)`` axes.
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.
        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.
        interactive : bool, default=False
            If ``True``, use interactive interface provided by ipywidgets.

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
            kw_theory = [{key: value for key, value in kw_theory[0].items() if (key != 'label') or (ill == 0)} for ill in range(len(labels))]
        kw_theory = [{'color': f'C{ill:d}', **kw} for ill, kw in enumerate(kw_theory)]

        if fig is None:
            height_ratios = [max(len(labels), 3)] + [1] * len(labels)
            figsize = (6, 1.5 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
            fig.subplots_adjust(hspace=0.1)
            show_legend = True
        else:
            lax = fig.axes
            show_legend = False

        wtheory = self.data.clone(value=self.flattheory)
        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            if 'scoccimarro' in data_pole.basis:
                x = np.arange(data_pole.size)
                scale = data_pole.coords('k').prod(axis=-1)
                xlabel = r'$k$ triangle index'
                ylabel = r'$k_1 k_2 k_3 B_{\ell}(k_1, k_2, k_3)$ [$(\mathrm{Mpc}/h)^3$]'
            elif 'sugiyama' in data_pole.basis:
                k = data_pole.coords('k')
                scale = k.prod(axis=-1)
                if np.allclose(k[..., 1], k[..., 0]):
                    x = k[..., 0]
                    xlabel = r'$k$ [h/\mathrm{Mpc}]'
                    ylabel = r'$k^2 B_{\ell}(k, k)$ [$(\mathrm{Mpc}/h)^4$]'
                else:
                    x = np.arange(data_pole.size)
                    xlabel = r'$k$ triangle index'
                ylabel = r'$k^2 B_{\ell}(k, k)$ [$(\mathrm{Mpc}/h)^4$]'
            std = self.cov.at.observable.get(**label).std()
            lax[0].errorbar(x, scale * data_pole.value(), yerr=scale * std, color=f'C{ill:d}', linestyle='none', marker='o', label=rf'$\ell = {ell}$')
            lax[0].plot(x, scale * wtheory_pole.value(), **kw_theory[ill])
        for ill, label in enumerate(labels):
            ell = label['ells']
            lax[ill + 1].plot(x, (data_pole.value() - wtheory_pole.value()) / std, **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(rf'$\Delta B_{{{ell}}} / \sigma_{{ B_{{{ell}}} }}$')

        for ax in lax: ax.grid(True)
        if show_legend: lax[0].legend()
        lax[0].set_ylabel(ylabel)
        lax[-1].set_xlabel(xlabel)
        return fig

    def calculate(self):
        # Set flattheory with window-convolved bispectrum prediction
        self.flattheory = jnp.dot(self.window.value(), self.theory.power.ravel())

    def __getstate__(self):
        # Optional
        state = {}
        for name in ['flatdata', 'flattheory']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state