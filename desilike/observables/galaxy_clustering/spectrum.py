from collections.abc import Mapping
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import lsstypes as types

from desilike import plotting, utils
from desilike.base import BaseCalculator
from desilike.jax import numpy as jnp


def _get_templates(templates=None):
    """Return templates in a standard format, i.e. a dict of name: template (numpy array)."""
    if templates is None:
        templates = {}
    if not isinstance(templates, Mapping):
        if not utils.is_sequence(templates):
            templates = [templates]
        templates = {'syst_{:d}'.format(i): v for i, v in enumerate(templates)}
    toret = {}
    for name, template in templates.items():
        toret[name] = template
    return toret


def _make_list(array, nitems=1):
    if not utils.is_sequence(array):
        array = [array] * nitems
    return list(array)


def _format_clustering_data_window_covariance(data: None | types.ObservableLike | np.ndarray,
                                   window: None | types.WindowMatrix | np.ndarray,
                                   covariance: None | types.CovarianceMatrix | np.ndarray,
                                   coords: None | list, ells: None | list,
                                   coordin: None | np.ndarray, ellsin: None | list,
                                   coord_name: str='k', meta: dict=None) -> tuple[types.ObservableLike, types.WindowMatrix, types.CovarianceMatrix]:
    """Create data, window and covariance objects in the expected format from various input formats."""
    custom_data = not isinstance(data, types.ObservableLike)
    custom_window = not isinstance(window, types.WindowMatrix)
    custom_covariance = not isinstance(covariance, types.CovarianceMatrix)
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
            raise ValueError(f'total k-size should be the same as data, but got {csizes} and {data.size}')
        value = data
        data = []
        start = 0
        for ell, coord in zip(ells, coords):
            stop = start + len(coord)
            leaf = types.ObservableLeaf(**{coord_name: coord}, value=value[start:stop], coords=[coord_name], meta={'ells': ell, **(meta or {})})
            start = stop
            data.append(leaf)
        data = types.ObservableTree(data, ells=ells)
    if window is None:
        coordin = np.unique(np.concatenate([pole.coords(coord_name) for pole in data], axis=0), axis=0)
        window, ellsin = [], []
        for label, pole in data.items():
            mask = pole.coords(coord_name)[:, None, ...] == coordin[None, :, ...]
            if mask.ndim > 2:  # e.g. bispectrum
                mask = mask.all(axis=-1)
            window.append(1. * mask)
            ellsin.append(label['ells'])
        window = sp.linalg.block_diag(*window)
    if custom_window:
        window = np.array(window)
        assert window.ndim == 2
        for name, value in {coord_name + 'in': coordin, 'ellsin': ellsin}.items():
            if value is None:
                raise ValueError(f'when input window is an array, provide {name}')
        theory = []
        start = 0
        for ell in ellsin:
            leaf = types.ObservableLeaf(**{coord_name: coordin}, value=np.zeros(coordin.shape[:1], dtype='f8'), coords=[coord_name])
            theory.append(leaf)
        theory = types.ObservableTree(theory, ells=ellsin)
        window = types.WindowMatrix(value=window, theory=theory, observable=data.clone(value=np.zeros_like(data.value())))
    elif not custom_data:  # match
        window = window.at.observable.match(data)
    assert window.shape[0] == data.size, f'output window dimension must match data size, but got {window.shape[0]} != {data.size}'
    if covariance is not None:
        if custom_covariance:
            covariance = np.array(covariance)
            assert covariance.ndim == 2
            covariance = types.CovarianceMatrix(value=covariance, observable=data.clone(value=np.zeros_like(data.value())))
        elif not custom_data:  # match
            covariance = covariance.at.observable.match(data)
        assert covariance.shape[0] == data.size, 'covariance shape must match data size'
    return data, window, covariance


class BaseClusteringObservable(BaseCalculator):
    """Base class for observables. Not to be used directly."""

    @staticmethod
    def _params(params, templates=None):
        """Return parameters for systematic templates."""
        names = list(_get_templates(templates=templates).keys())
        for iname, name in enumerate(names):
            params[name] = dict(value=0., ref=dict(limits=[-1e-3, 1e-3]), delta=0.005, latex=f's_{iname:d}')
        return params

    def calculate(self, **params):
        """Set flattheory with window-convolved theory prediction."""
        self.flattheory = jnp.dot(self.window.value(), self.theory.get().ravel())
        if self.templates:
            self.flattheory += jnp.array([params[name] for name in self.templates]).dot(jnp.array(list(self.templates.values())))

    def get(self):
        """Returned value when calling observable()."""
        return self.flattheory

    @property
    def flattheory_nobao(self):
        # Return theory without BAO wiggles, for plotting
        # TODO: make simpler
        template = self.theory.template
        only_now = template.only_now
        template.only_now = True
        if not template.with_now:
            raise ValueError(f'pass with_now to power spectrum template {template}')

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

        nobao = self.flattheory

        template.only_now = only_now
        for calculator in all_requires:
            calculator.runtime_info.tocalculate = True
            calculator.runtime_info.calculate(calculator.runtime_info.input_values)

        return nobao


class TracerSpectrum2PolesObservable(BaseClusteringObservable):
    """
    Tracer power spectrum multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : array, lsstypes.Mesh2SpectrumPoles, default=None
        Data power spectrum measurement. Either:
        - flat array (of all multipoles). Additionally provide list of multipoles ``ells`` and wavenumbers ``k`` and optionally ``shotnoise``;
        - :class:`lsstypes.Mesh2SpectrumPoles` (contains all necessary information);
        - ``None``. Data vector set to 0.
    window : array, lsstypes.WindowMatrix, default=None
        Window matrix. Either:
        - ``None``. A trivial (diagonal) window matrix is assumed;
        - 2D array, of shape ``(len(data), len(ellsin) * len(kin))``.
        Additionally provide input array ``kin`` and input multipoles ``ellsin``.
        - :class:`lsstypes.WindowMatrix` (contains all necessary information).
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    theory : BaseCalculator
        Theory for the power spectrum.
    k : tuple of arrays, default=None
        If ``data`` is an array. List of wavenumbers (for each multipole). ``sum(len(kk) for kk in k)`` must match ``len(data)``.
    ells : tuple, list
        Power spectrum multipoles, e.g. [0, 2, 4].
    kin : array, optional
        If ``window`` is a 2D array. Theory wavenumbers (assumed same for each multipole).
    ellsin : tuple
        If ``window`` is a 2D array. Theory power spectrum multipole orders, e.g. [0, 2, 4].
    shotnoise : float, optional
        Shot noise to scale theory stochastic parameters.
    templates : dict, optional
        Dictionary of template name: template array (of size ``data.size``) to add to theory with free amplitude;
        to marginalize over systematic effects.
    transform : str, default=None
        Transform to gaussianize the likelihood of the power spectrum.
        For 'cubic', see eq. 16 of https://arxiv.org/pdf/2302.07484.pdf.
        If ``None``, no transform is applied.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    """
    def initialize(self, data: None | types.ObservableLike | np.ndarray=None,
                   window: None | types.WindowMatrix | np.ndarray=None,
                   covariance: None | types.CovarianceMatrix | np.ndarray=None,
                   theory: BaseCalculator=None,
                   k: None | list=None, ells: None | list=None,
                   kin: None | np.ndarray=None, ellsin: None | list=None,
                   shotnoise: None | float=None, templates: None | dict=None,
                   transform: str=None, name: str='spectrum2poles'):
        assert theory is not None, 'provide theory'
        self.theory = theory
        self.name = str(name)
        custom_data = not isinstance(data, types.ObservableLike)
        if custom_data:
            if np.ndim(ells) == 0:
                ells = [ells]
            ells = [int(ell) for ell in ells]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(data=data, window=window, covariance=covariance,
                                                                          coords=k, ells=ells,
                                                                          coordin=kin, ellsin=ellsin, coord_name='k')
        self.flatdata = self.data.value()
        self.theory.init.update(k=next(iter(self.window.theory)).coords('k'), ells=self.window.theory.ells)
        if shotnoise is not None:
            self.theory.init.update(shotnoise=shotnoise)
        self.templates = _get_templates(templates=templates)
        self.transform = transform
        allowed_transform = [None, 'cubic']
        if self.transform not in allowed_transform:
            raise ValueError('transform must be one of {}'.format(allowed_transform))

    def calculate(self, **params):
        """Set flattheory with window-convolved theory prediction."""
        super().calculate(**params)
        if self.transform == 'cubic':
            # See Eq. 16 of https://arxiv.org/pdf/2302.07484.pdf
            self.flattheory = (3. * (self.flattheory / self.flatdata)**(1. / 3.) - 2.) * self.flatdata

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, kw_theory=None, scaling='kpk', fig=None):
        """
        Plot data and theory power spectrum multipoles.

        Parameters
        ----------
        scaling : str, default='kpk'
            Either 'kpk' or 'loglog'.
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
            x = data_pole.coords('k')
            xlabel = r'$k$ [$h/\mathrm{Mpc}$]'
            if scaling == 'kpk':
                scale = x
                ylabel = r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]'
            elif scaling == 'loglog':
                scale = 1.
                ylabel = r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]'
                lax[0].set_yscale('log')
                lax[0].set_xscale('log')
            std = self.covariance.at.observable.get(**label).std()
            lax[0].errorbar(x, scale * data_pole.value(), yerr=scale * std, color=kw_theory[ill]['color'], linestyle='none', marker='o', label=rf'$\ell = {ell}$')
            lax[0].plot(x, scale * wtheory_pole.value(), **kw_theory[ill])
        for ill, label in enumerate(labels):
            ell = label['ells']
            lax[ill + 1].plot(x, (data_pole.value() - wtheory_pole.value()) / std, **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(rf'$\Delta P_{{{ell}}} / \sigma_{{ P_{{{ell}}} }}$')

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
            x = data_pole.coords('k')
            scale = x
            xlabel = r'$k$ [$h/\mathrm{Mpc}$]'
            color = f'C{ill:d}'
            ax.errorbar(x, scale * (data_pole.value() - wtheorynobao_pole.value()), yerr=scale * std, color=color, linestyle='none', marker='o')
            ax.plot(x, scale * (wtheory_pole.value() - wtheorynobao_pole.value()), color=color)
            ax.set_ylabel(rf'$k \Delta P_{{{ell:d}}}(k)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
            ax.grid(True)
        lax[-1].set_xlabel(xlabel)
        return fig


class TracerSpectrum3PolesObservable(BaseClusteringObservable):
    """
    Tracer bispectrum multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : array, lsstypes.Mesh3SpectrumPoles, default=None
        Data bispectrum measurement. Either:
        - flat array (of all multipoles). Additionally provide list of multipoles ``ells`` and wavenumbers ``k`` and optionally ``shotnoise``;
        - :class:`lsstypes.Mesh3SpectrumPoles` (contains all necessary information);
        - ``None``. Data vector set to 0.
    window : array, lsstypes.WindowMatrix, default=None
        Window matrix. Either:
        - ``None``. A trivial (diagonal) window matrix is assumed;
        - 2D array, of shape ``(len(data), len(ellsin) * len(kin))``.
        Additionally provide input array ``kin`` and input multipoles ``ellsin``.
        - :class:`lsstypes.WindowMatrix` (contains all necessary information).
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    theory : BaseCalculator
        Theory for the bispectrum.
    k : tuple of arrays, default=None
        If ``data`` is an array. List of wavenumbers (for each multipole) of shape (nk, 3) (Scoccimarro basis)
        or (nk, 2) (Sugiyama basis). ``sum(len(kk) for kk in k)`` must match ``len(data)``.
    ells : tuple, list
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
                   k: None | list=None, ells: None | list=None, basis='sugiyama',
                   kin: None | np.ndarray=None, ellsin: None | list=None,
                   shotnoise: None | float=None, templates: None | dict=None,
                   name: str='spectrum3poles'):
        assert theory is not None, 'provide theory'
        self.theory = theory
        self.name = str(name)
        custom_data = not isinstance(data, types.ObservableLike)
        if custom_data:
            if basis == 'sugiyama':
                if np.ndim(ells) == 0:
                    raise ValueError('input ell should be a list of tuples, e.g. [(0, 0, 0), (2, 0, 2)]')
                if np.ndim(ells[0]) == 0:
                    ells = [ells]
                ells = [tuple(ell) for ell in ells]
            elif basis == 'scoccimarro':
                if np.ndim(ells) == 0:
                    ells = [ells]
                ells = [int(ell) for ell in ells]
        self.data, self.window, self.covariance = _format_clustering_data_window_covariance(data=data, window=window, covariance=covariance,
                                                                          coords=k, ells=ells,
                                                                          coordin=kin, ellsin=ellsin, coord_name='k',
                                                                          meta={'basis': basis})
        self.flatdata = self.data.value()
        self.theory.init.update(k=next(iter(self.window.theory)).coords('k'), ells=self.window.theory.ells)
        if shotnoise is not None:
            self.theory.init.update(shotnoise=shotnoise)
        self.templates = _get_templates(templates=templates)

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
                    xlabel = r'$k$ [$h/\mathrm{Mpc}$]'
                    ylabel = r'$k^2 B_{\ell}(k, k)$ [$(\mathrm{Mpc}/h)^4$]'
                else:
                    x = np.arange(data_pole.size)
                    xlabel = r'$k$ triangle index'
                ylabel = r'$k^2 B_{\ell}(k, k)$ [$(\mathrm{Mpc}/h)^4$]'
            std = self.covariance.at.observable.get(**label).std()
            lax[0].errorbar(x, scale * data_pole.value(), yerr=scale * std, color=kw_theory[ill]['color'], linestyle='none', marker='o', label=rf'$\ell = {ell}$')
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



TracerPowerSpectrumMultipolesObservable = TracerSpectrum2PolesObservable
TracerBispectrumMultipolesObservable = TracerSpectrum3PolesObservable