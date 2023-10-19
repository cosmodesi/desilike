import glob

import numpy as np

from desilike import plotting, jax, utils
from desilike.base import BaseCalculator
from .window import WindowedCorrelationFunctionMultipoles


def _is_array(data):
    return isinstance(data, (np.ndarray,) + jax.array_types)


def _is_from_pycorr(data):
    return utils.is_path(data) or not _is_array(data)


class TracerCorrelationFunctionMultipolesObservable(BaseCalculator):
    """
    Tracer correlation function multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : str, Path, list, pycorr.BaseTwoPointEstimator, dict, default=None
        Data correlation function measurement: flat array (of all multipoles), :class:`pycorr.BaseTwoPointEstimator` instance,
        or path to such instances, or list of such objects (in which case the average of them is taken).
        If dict, parameters to be passed to theory to generate mock measurement.
        If a (list of) flat array, additionally provide list of multipoles ``ells`` and separations ``s`` (see **kwargs).

    covariance : list, default=None
        2D array, list of :class:`pycorr.BaseTwoPointEstimator` instances, or paths to such instances;
        these are used to compute the covariance matrix.

    slim : dict, default=None
        Separation limits: a dictionary mapping multipoles to (min separation, max separation, (optionally) step (float)),
        e.g. ``{0: (30., 160., 5.), 2: (30., 160., 5.)}``. If ``None``, no selection is applied for the given multipole.

    **kwargs : dict
        Optional arguments for :class:`WindowedCorrelationFunctionMultipoles`, e.g.:

        - theory: defaults to :class:`KaiserTracerCorrelationFunctionMultipoles`.
        - fiber_collisions
        - if one only provided simple arrays for ``data`` and ``covariance``,
          one can provide the list of multipoles ``ells`` and the corresponding (list of) :math:`s` separations as a (list of) array ``s``.
    """
    def initialize(self, data=None, covariance=None, slim=None, wmatrix=None, ignore_nan=False, **kwargs):
        self.s, self.sedges, self.ells = None, None, None
        self.flatdata, self.mocks, self.covariance = None, None, None
        if not isinstance(data, dict):
            self.flatdata = self.load_data(data=data, slim=slim, ignore_nan=ignore_nan)[0]
        if self.mpicomm.bcast(_is_array(covariance), root=0):
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        else:
            self.mocks = self.load_data(data=covariance, slim=slim, ignore_nan=ignore_nan)[-1]
        if self.mpicomm.bcast(self.mocks is not None, root=0):
            covariance = None
            if self.mpicomm.rank == 0:
                covariance = np.cov(self.mocks, rowvar=False, ddof=1)
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        self.wmatrix = wmatrix
        if not isinstance(wmatrix, WindowedCorrelationFunctionMultipoles):
            self.wmatrix = WindowedCorrelationFunctionMultipoles()
            if wmatrix is not None:
                self.wmatrix.init.update(wmatrix=wmatrix)
        if self.sedges is not None:  # set by data
            slim = {ell: (edges[0], edges[-1], np.mean(np.diff(edges))) for ell, edges in zip(self.ells, self.sedges)}
            self.wmatrix.init.update(s=self.s)
        if slim is not None:
            self.wmatrix.init.update(slim=slim)
        self.wmatrix.init.update(kwargs)
        if self.flatdata is None:
            self.wmatrix(**data)
            self.flatdata = self.wmatrix.flatcorr.copy()
        else:
            self.wmatrix.runtime_info.initialize()
        for name in ['s', 'ells', 'sedges']:
            setattr(self, name, getattr(self.wmatrix, name))

    def load_data(self, data=None, slim=None, ignore_nan=False):

        def load_data(fn):
            with utils.LoggingContext(level='warning'):
                try:
                    from pycorr import TwoPointCorrelationFunction
                    toret = TwoPointCorrelationFunction.load(fn)
                except:
                    from pypower import MeshFFTCorr, CorrelationFunctionMultipoles
                    with utils.LoggingContext(level='warning'):
                        toret = MeshFFTCorr.load(fn)
                        if hasattr(toret, 'poles'):
                            toret = toret.poles
                        else:
                            toret = CorrelationFunctionMultipoles.load(fn)
            return toret

        def lim_data(corr, slim=slim):
            if slim is None:
                slim = {ell: (0, np.inf) for ell in (0, 2, 4)}
            ells, list_s, list_sedges, list_data = [], [], [], []
            for ell, lim in slim.items():
                corr_slice = corr.copy().select(lim)
                ells.append(ell)
                list_s.append(corr_slice.sepavg())
                list_sedges.append(corr_slice.edges[0])
                try:
                    d = corr_slice(ell=ell, return_std=False, ignore_nan=ignore_nan)  # pycorr
                except:
                    d = corr_slice(ell=ell)  # pypower
                list_data.append(d)
            return list_s, list_sedges, ells, list_data

        def load_all(lmocks):

            list_mocks = []
            for mocks in lmocks:
                if utils.is_path(mocks):
                    list_mocks += sorted(glob.glob(mocks))
                else:
                    list_mocks.append(mocks)

            fns = [mock for mock in list_mocks if utils.is_path(mock)]
            if len(fns):
                nfns = 5
                if len(fns) < nfns:
                    msg = 'Loading {}.'.format(fns)
                else:
                    msg = 'Loading [{}].'.format(', ..., '.join(fns[::len(fns) // nfns]))
                self.log_info(msg)

            list_y = []
            for mock in list_mocks:
                if utils.is_path(mock):
                    mock = load_data(mock)
                mock_s, mock_sedges, mock_ells, mock_y = lim_data(mock)
                if self.s is None:
                    self.s, self.sedges, self.ells = mock_s, mock_sedges, mock_ells
                if not all(np.allclose(ss, ms, atol=0., rtol=1e-3) for ss, ms in zip(self.sedges, mock_sedges)):
                    raise ValueError('{} does not have expected s-edges (based on previous data)'.format(mock))
                if mock_ells != self.ells:
                    raise ValueError('{} does not have expected poles (based on previous data)'.format(mock))
                list_y.append(np.concatenate(mock_y))
            return list_y

        flatdata, list_y = None, None
        if self.mpicomm.rank == 0 and data is not None:
            if not utils.is_sequence(data):
                data = [data]
            if any(_is_from_pycorr(dd) for dd in data):
                list_y = load_all(data)
                if not list_y: raise ValueError('no data/mocks could be obtained from {}'.format(data))
            else:
                list_y = list(data)
            flatdata = np.mean(list_y, axis=0)
        self.s, self.sedges, self.ells, flatdata = self.mpicomm.bcast((self.s, self.sedges, self.ells, flatdata) if self.mpicomm.rank == 0 else None, root=0)
        return flatdata, list_y

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
        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(self.s[ill], self.s[ill]**2 * data[ill], yerr=self.s[ill]**2 * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.s[ill], self.s[ill]**2 * theory[ill], **kw_theory[ill])
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.s[ill], (data[ill] - theory[ill]) / std[ill], **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta \xi_{{{0:d}}} / \sigma_{{ \xi_{{{0:d}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        if show_legend: lax[0].legend()
        lax[0].set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig

    @plotting.plotter
    def plot_bao(self, fig=None):
        """
        Plot data and theory BAO correlation function peak.

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
            figsize = (4, 3 * sum(height_ratios))
            fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=False)
            lax = lax.ravel()  # in case only one ell
            fig.subplots_adjust(hspace=0)
        else:
            lax = fig.axes
        data, theory, std = self.data, self.theory, self.std
        only_now = self.wmatrix.theory.template.only_now
        self.wmatrix.theory.template.only_now = True

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
            calculator.runtime_info.calculate()
        nowiggle = self.theory

        for ill, ell in enumerate(self.ells):
            lax[ill].errorbar(self.s[ill], self.s[ill]**2 * (data[ill] - nowiggle[ill]), yerr=self.s[ill]**2 * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o')
            lax[ill].plot(self.s[ill], self.s[ill]**2 * (theory[ill] - nowiggle[ill]), color='C{:d}'.format(ill))
            lax[ill].set_ylabel(r'$s^{{2}} \Delta \xi_{{{:d}}}(s)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        for ax in lax: ax.grid(True)
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')

        self.wmatrix.theory.template.only_now = only_now
        for calculator in all_requires:
            calculator.runtime_info.tocalculate = True
            calculator.runtime_info.calculate()

        return fig

    def plot_wiggles(self, *args, **kwargs):
        import warnings
        warnings.warn('plot_wiggles is deprecated, use plot_bao instead')
        self.plot_bao(*args, **kwargs)

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
        cumsize = np.insert(np.cumsum([len(s) for s in self.s]), 0, 0)
        mat = [[self.covariance[start1:stop1, start2:stop2] for start2, stop2 in zip(cumsize[:-1], cumsize[1:])] for start1, stop1 in zip(cumsize[:-1], cumsize[1:])]
        return plot_covariance_matrix(mat, x1=self.s, xlabel1=r'$s$ [$\mathrm{Mpc}/h$]', label1=[r'$\ell = {:d}$'.format(ell) for ell in self.ells], corrcoef=corrcoef, **kwargs)

    def calculate(self):
        self.flattheory = self.wmatrix.flatcorr

    @property
    def theory(self):
        return self.wmatrix.corr

    @property
    def data(self):
        cumsize = np.insert(np.cumsum([len(s) for s in self.s]), 0, 0)
        return [self.flatdata[start:stop] for start, stop in zip(cumsize[:-1], cumsize[1:])]

    @property
    def std(self):
        cumsize = np.insert(np.cumsum([len(s) for s in self.s]), 0, 0)
        diag = np.diag(self.covariance)**0.5
        return [diag[start:stop] for start, stop in zip(cumsize[:-1], cumsize[1:])]

    def __getstate__(self):
        state = {}
        for name in ['s', 'sedges', 'ells', 'flatdata', 'shotnoise', 'flattheory']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @classmethod
    def install(cls, config):
        # TODO: remove this dependency
        #config.pip('git+https://github.com/cosmodesi/pycorr')
        pass
