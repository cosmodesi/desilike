import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import lsstypes as types

from desilike import plotting
from desilike.base import BaseCalculator
from .spectrum import BaseClusteringObservable, _get_templates, _format_clustering_data_window_covariance


class TracerCorrelation2PolesObservable(BaseClusteringObservable):
    """
    Tracer 2pt correlation function multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : array, lsstypes.Correlation2Poles, default=None
        Data correlation function measurement. Either:
        - flat array (of all multipoles). Additionally provide list of multipoles ``ells`` and separation ``s`` and optionally ``shotnoise``;
        - :class:`lsstypes.Correlation2Poles` (contains all necessary information);
        - ``None``. Data vector set to 0.
    window : array, lsstypes.WindowMatrix, default=None
        Window matrix. Either:
        - ``None``. A trivial window matrix is assumed;
        - 2D array, of shape ``(len(data), len(ellsin) * len(sin))``.
        Additionally provide input array ``sin`` and input multipoles ``ellsin``.
        - :class:`lsstypes.WindowMatrix` (contains all necessary information).
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    theory : BaseCalculator
        Theory for the correlation function.
    s : tuple of arrays, default=None
        If ``data`` is an array. List of separations (for each multipole). ``sum(len(ss) for ss in s)`` must match ``len(data)``.
    ells : tuple, list
        Correlation function multipoles, e.g. [0, 2, 4].
    sin : array, optional
        If ``window`` is a 2D array. Theory separations (assumed same for each multipole).
    ellsin : tuple
        If ``window`` is a 2D array. Theory correlation function multipole orders, e.g. [0, 2, 4].
    shotnoise : float, optional
        Shot noise to scale theory stochastic parameters.
    templates : dict, optional
        Dictionary of template name: template array (of size ``data.size``) to add to theory with free amplitude;
        to marginalize over systematic effects.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    """
    def initialize(self, data: None | types.ObservableLike | np.ndarray=None,
                   window: None | types.WindowMatrix | np.ndarray=None,
                   covariance: None | types.CovarianceMatrix | np.ndarray=None,
                   theory: BaseCalculator=None,
                   s: None | list=None, ells: None | list=None,
                   sin: None | np.ndarray=None, ellsin: None | list=None,
                   shotnoise: None | float=None, templates: None | dict=None,
                   name: str='correlation2poles'):
        assert data is not None
        assert theory is not None
        self.theory = theory
        self.name = str(name)
        custom_data = not isinstance(data, types.ObservableLike)
        if custom_data:
            if np.ndim(ells) == 0:
                ells = [ells]
            ells = [int(ell) for ell in ells]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(data=data, window=window, covariance=covariance,
                                                                          coords=s, ells=ells,
                                                                          coordin=sin, ellsin=ellsin, coord_name='s')
        self.flatdata = self.data.value()
        self.theory.init.update(s=next(iter(self.window.theory)).coords('s'), ells=self.window.theory.ells)
        if shotnoise is not None:
            self.theory.init.update(shotnoise=shotnoise)
        self.templates = _get_templates(templates=templates)

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, kw_theory=None, fig=None):
        """
        Plot data and theory correlation function multipoles.

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
            x = data_pole.coords('s')
            xlabel = r'$s$ [\mathrm{Mpc}/h]'
            scale = x**2
            ylabel = r'$s^2 \xi_\ell(s)$ [$(\mathrm{Mpc}/h)^2$]'
            std = self.covariance.at.observable.get(**label).std()
            lax[0].errorbar(x, scale * data_pole.value(), yerr=scale * std, color=f'C{ill:d}', linestyle='none', marker='o', label=rf'$\ell = {ell}$')
            lax[0].plot(x, scale * wtheory_pole.value(), **kw_theory[ill])
        for ill, label in enumerate(labels):
            ell = label['ells']
            lax[ill + 1].plot(x, (data_pole.value() - wtheory_pole.value()) / std, **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(rf'$\Delta \xi_{{{ell}}} / \sigma_{{ P_{{{ell}}} }}$')

        for ax in lax: ax.grid(True)
        if show_legend: lax[0].legend()
        lax[0].set_ylabel(ylabel)
        lax[-1].set_xlabel(xlabel)
        return fig

    @plotting.plotter
    def plot_bao(self, fig=None):
        """
        Plot data and theory BAO power spectrum wiggles.

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
        labels = self.data.labels()
        if fig is None:
            height_ratios = [1] * len(labels)
            figsize = (6, 2 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=False)
            lax = lax.ravel()  # in case only one ell
            fig.subplots_adjust(hspace=0)
        else:
            lax = fig.axes
        wtheory = self.data.clone(value=self.flattheory)
        wtheorynobao = self.data.clone(value=self.flattheory_nobao)

        for ill, label in enumerate(labels):
            ell = label['ells']
            data_pole = self.data.get(**label)
            wtheory_pole = wtheory.get(**label)
            wtheorynobao_pole = wtheorynobao.get(**label)
            std = self.covariance.at.observable.get(**label).std()
            ax = lax[ill]
            x = data_pole.coords('s')
            scale = x**2
            xlabel = r'$s$ [\mathrm{Mpc}/h]'
            color = f'C{ill:d}'
            ax.errorbar(x, scale * (data_pole.value() - wtheorynobao_pole.value()), yerr=scale * std, color=color, linestyle='none', marker='o')
            ax.plot(x, scale * (wtheory_pole.value() - wtheorynobao_pole.value()), color=color)
            ax.set_ylabel(rf'$s^2 \Delta \xi_{{{ell:d}}}(s)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
            ax.grid(True)
        lax[-1].set_xlabel(xlabel)
        return fig


TracerCorrelationFunctionMultipolesObservable = TracerCorrelation2PolesObservable