import glob

import numpy as np

from desilike import plotting, jax, utils
from desilike.base import BaseCalculator
from desilike.jax import numpy as jnp
from desilike.observables.types import ObservableArray, ObservableCovariance


def _is_array(data):
    return isinstance(data, (np.ndarray,) + jax.array_types)


class TracerBispectrumMultipolesObservable(BaseCalculator):
    """
    Tracer bispectrum multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : array, dict, default=None
        Data power spectrum measurement: flat array (of all multipoles).
        If dict, parameters to be passed to theory to generate mock measurement.
        Additionally provide list of multipoles ``ells`` and wavenumbers ``k`` and optionally ``shotnoise``.

    covariance : array, list, default=None
        2D array, of shape ``(len(data), len(data))``.

    k : tuple of arrays, default=None
        Triangles of wavenumbers of shape (nk, 3) (scoccimarro basis) where to evaluate multipoles.
        ``sum(len(kk) for kk in k)`` should match ``len(data)``.

    ells : tuple, default=((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2))
        Bispectrum multipoles to compute.

    shotnoise : array, default=1e4
        Shot noise for each of the multipoles. Same length as ``k``.

    theory: BaseCalculator
        Theory for the bispectrum.

    """
    def initialize(self, data=None, covariance=None, k=None, ells=None, shotnoise=None, theory=None):
        assert data is not None
        assert theory is not None
        self.wmatrix = theory
        self.wmatrix.init.update(k=k, ells=ells, shotnoise=shotnoise)
        self.wmatrix.runtime_info.initialize()
        for name in ['k', 'ells', 'shotnoise']:
            setattr(self, name, getattr(self.wmatrix, name))
        self.flatdata = None
        if not isinstance(data, dict):
            self.flatdata = np.concatenate(data)
        if self.flatdata is None:
            self.wmatrix(**data)
            self.flatdata = np.concatenate(self.wmatrix.power)
        self.covariance = None
        if self.mpicomm.bcast(_is_array(covariance) or isinstance(covariance, ObservableCovariance), root=0):
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        elif covariance is not None:
            raise ValueError('Provide covariance array')

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
        from matplotlib import pyplot as plt

        if kw_theory is None:
            kw_theory = {}
        if isinstance(kw_theory, dict):
            kw_theory = [kw_theory]
        if len(kw_theory) != len(self.ells):
            kw_theory = [{key: value for key, value in kw_theory[0].items() if (key != 'label') or (ill == 0)} for ill in range(len(self.ells))]
        kw_theory = [{'color': 'C{:d}'.format(ill), **kw} for ill, kw in enumerate(kw_theory)]

        if fig is None:
            height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
            figsize = (6, 1.5 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
            fig.subplots_adjust(hspace=0.1)
            show_legend = True
        else:
            lax = fig.axes
            show_legend = False

        data, theory, std = self.data, self.theory, self.std
        ik = [np.arange(len(kk)) for kk in self.k]

        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(ik[ill], data[ill], yerr=std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o', label=r'$\ell = {}$'.format(ell))
            lax[0].plot(ik[ill], theory[ill], **kw_theory[ill])
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(ik[ill], (data[ill] - theory[ill]) / std[ill], **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta B_{{{0}}} / \sigma_{{ B_{{{0}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        if show_legend: lax[0].legend()
        lax[0].set_ylabel(r'$B_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{6}$]')
        scaling = 'log'
        if scaling == 'loglog':
            lax[0].set_yscale('log')
        lax[-1].set_xlabel(r'$k$ triangle index')
        return fig

    @plotting.plotter
    def plot_bao(self, fig=None):
        """
        Plot data and theory BAO bispectrum wiggles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``len(self.ells)`` axes.

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
            height_ratios = [1] * len(self.ells)
            figsize = (6, 2 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=False)
            lax = lax.ravel()  # in case only one ell
            fig.subplots_adjust(hspace=0)
        else:
            lax = fig.axes
        data, theory, std = self.data, self.theory, self.std
        nobao = self.theory_nobao
        ik = [np.arange(len(kk)) for kk in self.k]

        for ill, ell in enumerate(self.ells):
            lax[ill].errorbar(ik[ill], data[ill] - nobao[ill], yerr=std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o')
            lax[ill].plot(ik[ill], theory[ill] - nobao[ill], color='C{:d}'.format(ill))
            lax[ill].set_ylabel(r'$k \Delta B_{{{:d}}}(k)$ [$(\mathrm{{Mpc}}/h)^{{6}}$]'.format(ell))
        for ax in lax: ax.grid(True)
        lax[-1].set_xlabel(r'$k$ triangle index')
        return fig

    @plotting.plotter
    def plot_covariance_matrix(self, corrcoef=True, **kwargs):
        """
        Plot covariance matrix.

        Parameters
        ----------
        corrcoef : bool, default=True
            If ``True``, plot the correlation matrix; else the covariance.

        barlabel : str, default=None
            Optionally, label for the color bar.

        figsize : int, tuple, default=None
            Optionally, figure size.

        norm : matplotlib.colors.Normalize, default=None
            Scales the covariance / correlation to the canonical colormap range [0, 1] for mapping to colors.
            By default, the covariance / correlation range is mapped to the color bar range using linear scaling.

        labelsize : int, default=None
            Optionally, size for labels.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``len(self.ells) * len(self.ells)`` axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from desilike.observables.plotting import plot_covariance_matrix
        cumsize = np.insert(np.cumsum([len(k) for k in self.k]), 0, 0)
        mat = [[self.covariance[start1:stop1, start2:stop2] for start2, stop2 in zip(cumsize[:-1], cumsize[1:])] for start1, stop1 in zip(cumsize[:-1], cumsize[1:])]
        ik = [np.arange(len(kk)) for kk in self.k]
        return plot_covariance_matrix(mat, x1=ik, xlabel1=r'$k$ [$h/\mathrm{Mpc}$]', label1=[r'$\ell = {}$'.format(ell) for ell in self.ells], corrcoef=corrcoef, **kwargs)

    def calculate(self):
        # Set flattheory with bispectrum prediction
        self.flattheory = jnp.concatenate(self.wmatrix.power)

    @property
    def theory(self):
        return self.wmatrix.power

    @property
    def theory_nobao(self):
        # Remove BAO wiggles, a bit hacky
        template = self.wmatrix.template
        only_now = template.only_now
        template.only_now = True

        def callback(calculator):
            all_requires.append(calculator)
            for require in calculator.runtime_info.requires:
                if require in all_requires:
                    del all_requires[all_requires.index(require)]  # we want first dependencies at the end
                callback(require)

        all_requires = []
        callback(self)
        all_requires = all_requires[::-1]

        for calculator in all_requires:
            calculator.runtime_info.tocalculate = True
            calculator.runtime_info.calculate(calculator.runtime_info.input_values)

        nobao = self.theory

        template.only_now = only_now
        for calculator in all_requires:
            calculator.runtime_info.tocalculate = True
            calculator.runtime_info.calculate(calculator.runtime_info.input_values)

        return nobao

    @property
    def data(self):
        # data, split by multipoles
        cumsize = np.insert(np.cumsum([len(k) for k in self.k]), 0, 0)
        return [self.flatdata[start:stop] for start, stop in zip(cumsize[:-1], cumsize[1:])]

    @property
    def std(self):
        # Standard deviation, split by multipoles
        cumsize = np.insert(np.cumsum([len(k) for k in self.k]), 0, 0)
        diag = np.diag(self.covariance)**0.5
        return [diag[start:stop] for start, stop in zip(cumsize[:-1], cumsize[1:])]

    def __getstate__(self):
        # Optional
        state = {}
        for name in ['k', 'ells', 'flatdata', 'flattheory', 'shotnoise']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def to_array(self):
        from desilike.observables import ObservableArray
        return ObservableArray(x=self.k, value=self.data, projs=self.ells, attrs={'shotnoise': self.shotnoise}, name=self.__class__.__name__)