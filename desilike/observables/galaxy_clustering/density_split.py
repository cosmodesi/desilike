"""Density-split galaxy clustering observables."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import lsstypes as types

from desilike import plotting
from desilike.base import BaseCalculator
from desilike.jax import numpy as jnp

from desilike.theories.galaxy_clustering import DensitySplitTracerPowerSpectrumMultipoles


_DEFAULT_QUANTILES = (1, 2, 3, 4, 5)
_DEFAULT_ELLS = (0, 2, 4)


def _normalize_quantiles(quantiles=None):
    if quantiles is None:
        quantiles = _DEFAULT_QUANTILES
    if np.ndim(quantiles) == 0:
        quantiles = (quantiles,)
    quantiles = tuple(int(quantile) for quantile in quantiles)
    if not quantiles:
        raise ValueError('quantiles must not be empty')
    invalid = [quantile for quantile in quantiles if quantile not in _DEFAULT_QUANTILES]
    if invalid:
        raise ValueError('quantiles must be drawn from {}; found {}'.format(_DEFAULT_QUANTILES, invalid))
    if len(set(quantiles)) != len(quantiles):
        raise ValueError('quantiles must be unique')
    return quantiles


def _normalize_ells(ells=None):
    if ells is None:
        ells = _DEFAULT_ELLS
    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(int(ell) for ell in ells)
    if not ells:
        raise ValueError('ells must not be empty')
    if len(set(ells)) != len(ells):
        raise ValueError('ells must be unique')
    return ells


def _as_density_split_tree(data):
    if isinstance(data, (str, Path)):
        return types.read(str(data))
    if isinstance(data, types.ObservableTree):
        return data
    raise TypeError('data must be an lsstypes.ObservableTree or path to a raw HDF5 observable')


def _subset_density_split_tree(data, quantiles=None, ells=None):
    quantiles = _normalize_quantiles(quantiles)
    ells = _normalize_ells(ells)
    branches = []
    for quantile in quantiles:
        branch = data.get(quantiles=quantile - 1)
        leaves = [branch.get(ells=ell) for ell in ells]
        try:
            branch = branch.__class__(leaves, ells=ells, attrs=branch.attrs)
        except TypeError:
            branch = types.ObservableTree(leaves, ells=ells, attrs=branch.attrs)
        branches.append(branch)
    return types.ObservableTree(branches, quantiles=[quantile - 1 for quantile in quantiles], attrs=data.attrs, meta=data.meta)


def infer_density_split_quantiles(data):
    """Return 1-based density-split quantile labels from an observable tree."""
    return tuple(int(label['quantiles']) + 1 for label in data.labels())


def infer_density_split_ells(data, quantile=None):
    """Return multipole labels for a density-split observable tree."""
    if quantile is None:
        quantile = infer_density_split_quantiles(data)[0]
    branch = data.get(quantiles=int(quantile) - 1)
    return tuple(int(label['ells']) for label in branch.labels())


def get_density_split_k(data, quantile=None, ell=None):
    """Return the k-grid for a density-split observable tree."""
    if quantile is None:
        quantile = infer_density_split_quantiles(data)[0]
    if ell is None:
        ell = infer_density_split_ells(data, quantile=quantile)[0]
    return np.asarray(data.get(quantiles=int(quantile) - 1).get(ells=int(ell)).coords('k'), dtype='f8')


def load_density_split_power_spectrum_multipoles(data, quantiles=None, ells=None, rebin=13, kmin=0.01, kmax=0.2):
    """
    Load raw density-split quantile-galaxy cross-power multipoles.

    The returned observable tree is ordered in quantile-major, ell-major order,
    matching the theory array layout ``(n_quantiles, n_ells, n_k)``.
    """
    data = _as_density_split_tree(data)
    if rebin is not None:
        rebin = int(rebin)
        if rebin <= 0:
            raise ValueError('rebin must be positive')
        data = data.select(k=slice(0, None, rebin))
    if kmin is not None or kmax is not None:
        kmin = -np.inf if kmin is None else float(kmin)
        kmax = np.inf if kmax is None else float(kmax)
        data = data.select(k=(kmin, kmax))
    return _subset_density_split_tree(data, quantiles=quantiles, ells=ells)


def flatten_density_split_power_spectrum_multipoles(data, quantiles=None, ells=None, k=None):
    """Flatten density-split multipoles in quantile-major, ell-major order."""
    quantiles = infer_density_split_quantiles(data) if quantiles is None else _normalize_quantiles(quantiles)
    ells = infer_density_split_ells(data, quantile=quantiles[0]) if ells is None else _normalize_ells(ells)
    if k is not None:
        k = np.asarray(k, dtype='f8')
    values = []
    for quantile in quantiles:
        branch = data.get(quantiles=quantile - 1)
        for ell in ells:
            leaf = branch.get(ells=ell)
            value = np.asarray(leaf.value(), dtype='f8')
            if k is not None:
                kin = np.asarray(leaf.coords('k'), dtype='f8')
                same_grid = kin.shape == k.shape and np.allclose(kin, k, rtol=1e-10, atol=1e-12)
                if not same_grid:
                    value = np.interp(k, kin, value)
            values.append(value)
    return np.concatenate(values, axis=0)


def _density_split_mock_paths(directory, pattern='dsc_pkqg_poles_ph*.h5', max_mocks=None):
    paths = sorted(Path(directory).glob(pattern))
    paths = [path for path in paths if not path.name.startswith('._')]
    if max_mocks is not None:
        paths = paths[:int(max_mocks)]
    if not paths:
        raise ValueError("No density-split mock files matched pattern '{}' in {}".format(pattern, directory))
    return paths


def load_density_split_mock_matrix(directory, quantiles=None, ells=None, rebin=13, kmin=0.01, kmax=0.2,
                                   k=None, pattern='dsc_pkqg_poles_ph*.h5', max_mocks=None):
    """Load raw density-split mock files into a row-stacked matrix."""
    quantiles = _normalize_quantiles(quantiles)
    ells = _normalize_ells(ells)
    target_k = None if k is None else np.asarray(k, dtype='f8')
    rows = []
    for path in _density_split_mock_paths(directory, pattern=pattern, max_mocks=max_mocks):
        data = load_density_split_power_spectrum_multipoles(path, quantiles=quantiles, ells=ells, rebin=rebin, kmin=kmin, kmax=kmax)
        if target_k is None:
            target_k = get_density_split_k(data)
        rows.append(flatten_density_split_power_spectrum_multipoles(data, quantiles=quantiles, ells=ells, k=target_k))
    return np.asarray(target_k, dtype='f8'), np.asarray(rows, dtype='f8')


def density_split_sample_covariance(directory, quantiles=None, ells=None, rebin=13, kmin=0.01, kmax=0.2,
                                    k=None, pattern='dsc_pkqg_poles_ph*.h5', max_mocks=None, covariance_rescale=64.):
    """Build the raw-mock sample covariance for density-split multipoles."""
    _, mock_matrix = load_density_split_mock_matrix(directory, quantiles=quantiles, ells=ells, rebin=rebin, kmin=kmin, kmax=kmax,
                                                    k=k, pattern=pattern, max_mocks=max_mocks)
    if mock_matrix.shape[0] < 2:
        raise ValueError('at least two mocks are required to estimate a covariance')
    covariance = np.cov(mock_matrix.T)
    if covariance_rescale is not None:
        covariance = covariance / float(covariance_rescale)
    return np.asarray(covariance, dtype='f8')


class DensitySplitPowerSpectrumMultipolesObservable(BaseCalculator):
    """Density-split quantile-galaxy power-spectrum multipoles observable."""

    def initialize(self, data=None, covariance=None, theory=None, quantiles=None, ells=None,
                   rebin=13, kmin=0.01, kmax=0.2, name='density_split'):
        if data is None:
            raise ValueError('provide density-split data')
        self.name = str(name)
        self.data = load_density_split_power_spectrum_multipoles(data, quantiles=quantiles, ells=ells, rebin=rebin, kmin=kmin, kmax=kmax)
        self.quantiles = infer_density_split_quantiles(self.data)
        self.ells = infer_density_split_ells(self.data, quantile=self.quantiles[0])
        self.k = get_density_split_k(self.data, quantile=self.quantiles[0], ell=self.ells[0])
        self.flatdata = flatten_density_split_power_spectrum_multipoles(self.data, quantiles=self.quantiles, ells=self.ells)

        if theory is None:
            theory = DensitySplitTracerPowerSpectrumMultipoles(k=self.k, quantiles=self.quantiles, ells=self.ells)
        self.theory = theory
        self.theory.init.update(k=self.k, quantiles=self.quantiles, ells=self.ells)

        self.covariance = None
        if covariance is not None:
            if isinstance(covariance, types.CovarianceMatrix):
                self.covariance = covariance.at.observable.match(self.data)
            else:
                covariance = np.asarray(covariance, dtype='f8')
                if covariance.shape != (self.flatdata.size, self.flatdata.size):
                    raise ValueError('covariance shape must be ({0}, {0}); got {1}'.format(self.flatdata.size, covariance.shape))
                observable = self.data.clone(value=np.zeros_like(self.flatdata))
                self.covariance = types.CovarianceMatrix(value=covariance, observable=observable)

    def calculate(self):
        self.flattheory = jnp.ravel(self.theory.power)

    def get(self):
        return self.flattheory

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, kw_theory=None, scaling='kpk', kpower=None, fig=None, figsize=None):
        """
        Plot density-split data and theory power-spectrum multipoles.

        Parameters
        ----------
        kw_theory : dict, list of dict, default=None
            Optional keyword arguments passed to theory lines, one dictionary for each quantile or duplicate it.
        scaling : str, default='kpk'
            Either 'kpk' or 'loglog'.
        kpower : int or None, default=None
            If not ``None``, overwrite the power of k suggested by ``scaling``.
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``2 * len(self.ells)`` axes.
        figsize : (width, height), default=None
            If no figure is passed, fix the size of the created figure.
        fn : str, Path, default=None
            Optionally, path where to save figure.
        kw_save : dict, default=None
            Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.
        interactive : bool, default=False
            If ``True``, use interactive interface provided by ipywidgets.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.covariance is None:
            raise ValueError('plot requires a covariance to compute data errors')
        if kw_theory is None:
            kw_theory = {}
        if isinstance(kw_theory, dict):
            kw_theory = [kw_theory]
        if len(kw_theory) != len(self.quantiles):
            kw_theory = list(kw_theory[:1]) * len(self.quantiles)

        nells = len(self.ells)
        if fig is None:
            figsize = (4. * nells, 5.) if figsize is None else figsize
            fig, lax = plt.subplots(2, nells, sharex='col', sharey=False, squeeze=False,
                                    gridspec_kw={'height_ratios': [3, 1]}, figsize=figsize)
            fig.subplots_adjust(hspace=0.08, wspace=0.25)
            show_legend = True
        else:
            lax = np.asarray(fig.axes[:2 * nells], dtype=object).reshape(2, nells)
            show_legend = False

        colors = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, len(self.quantiles)))
        wtheory = self.data.clone(value=np.asarray(self.flattheory))

        xlabel = r'$k$ [$h/\mathrm{Mpc}$]'
        for iell, ell in enumerate(self.ells):
            ax, rax = lax[:, iell]
            for iquantile, quantile in enumerate(self.quantiles):
                label = {'quantiles': quantile - 1, 'ells': ell}
                data_pole = self.data.get(quantiles=label['quantiles']).get(ells=ell)
                wtheory_pole = wtheory.get(quantiles=label['quantiles']).get(ells=ell)
                x = np.asarray(data_pole.coords('k'), dtype='f8')
                if scaling == 'kpk':
                    k_exp = 1 if kpower is None else kpower
                    scale = x**k_exp
                    ylabel = r'$k P^{qg}_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]'
                elif scaling == 'loglog':
                    scale = 1.
                    ylabel = r'$P^{qg}_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]'
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                else:
                    raise ValueError("scaling must be either 'kpk' or 'loglog'")

                std = self.covariance.at.observable.get(**label).std()
                color = colors[iquantile]
                ax.errorbar(x, scale * data_pole.value(), yerr=scale * std, color=color,
                            linestyle='none', marker='o', markersize=3, label=rf'$Q_{{{quantile:d}}}$')
                theory_kw = {'color': color, **kw_theory[iquantile]}
                if 'label' in theory_kw and (iell != 0 or iquantile != 0):
                    theory_kw = {key: value for key, value in theory_kw.items() if key != 'label'}
                ax.plot(x, scale * wtheory_pole.value(), **theory_kw)
                rax.plot(x, (data_pole.value() - wtheory_pole.value()) / std, color=color)

            ax.set_title(rf'$\ell = {ell:d}$')
            rax.set_ylim(-4, 4)
            rax.axhline(0., color='k', linewidth=0.8)
            for offset in [-2., 2.]:
                rax.axhline(offset, color='k', linestyle='--', linewidth=0.8)
            rax.set_xlabel(xlabel)

        for ax in lax.ravel():
            ax.grid(True)
        lax[0, 0].set_ylabel(ylabel)
        lax[1, 0].set_ylabel(r'$\Delta P / \sigma$')
        if show_legend:
            lax[0, 0].legend()
        return fig

    def __getstate__(self, varied=True, fixed=True):
        state = {}
        for name in (['data', 'flatdata', 'covariance', 'k', 'quantiles', 'ells', 'name'] if fixed else []) + (['flattheory'] if varied else []):
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
