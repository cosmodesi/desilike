import glob

import numpy as np

from desilike import plotting, jax, utils
from desilike.base import BaseCalculator
from .window import WindowedPowerSpectrumMultipoles
from desilike.observables.types import ObservableArray, ObservableCovariance


def _is_array(data):
    return isinstance(data, (np.ndarray,) + jax.array_types)


def _is_from_pypower(data):
    return utils.is_path(data) or not _is_array(data)


class TracerPowerSpectrumMultipolesObservable(BaseCalculator):
    """
    Tracer power spectrum multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : array, str, Path, list, pypower.PowerSpectrumMultipoles, dict, default=None
        Data power spectrum measurement: flat array (of all multipoles), :class:`pypower.PowerSpectrumMultipoles` instance,
        or path to such instances, or list of such objects (in which case the average of them is taken).
        If dict, parameters to be passed to theory to generate mock measurement.
        If a (list of) flat array, additionally provide list of multipoles ``ells`` and wavenumbers ``k``, and optionally ``shotnoise`` (see ``kwargs``).

    covariance : array, list, default=None
        2D array, list of :class:`pypower.PowerSpectrumMultipoles` instance` instances, or paths to such instances;
        these are used to compute the covariance matrix.

    klim : dict, default=None
        Wavenumber limits: a dictionary mapping multipoles to (min separation, max separation, (optionally) step (float)),
        e.g. ``{0: (0.01, 0.2, 0.01), 2: (0.01, 0.15, 0.01)}``. If ``None``, no selection is applied for the given multipole.

    wmatrix : str, Path, pypower.BaseMatrix, WindowedPowerSpectrumMultipoles, default=None
        Optionally, window matrix.

    transform : array, default=None
        Transform to gaussianize the likelihood of the power spectrum.
        For 'cubic', see eq. 16 of https://arxiv.org/pdf/2302.07484.pdf.
        If ``None``, no transform is applied.

    **kwargs : dict
        Optional arguments for :class:`WindowedPowerSpectrumMultipoles`, e.g.:

        - theory: defaults to :class:`KaiserTracerPowerSpectrumMultipoles`.
        - shotnoise: take shot noise from ``data``, or ``covariance`` (mocks) if provided.
        - fiber_collisions
        - systematic_templates
        - if one only provided simple arrays for ``data`` and ``covariance``,
          one can provide the list of multipoles ``ells`` and the corresponding (list of) :math:`k` wavenumbers as a (list of) array ``k``,
          and optionally ``shotnoise``.

    """
    def initialize(self, data=None, covariance=None, klim=None, wmatrix=None, transform=None, kedges=None, **kwargs):
        self.k, self.kedges, self.ells, self.shotnoise = None, kedges, None, None
        self.flatdata, self.mocks, self.covariance = None, None, None
        if not isinstance(data, dict):
            self.flatdata = self.load_data(data=data, klim=klim)[0]
        if self.mpicomm.bcast(_is_array(covariance) or isinstance(covariance, ObservableCovariance), root=0):
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        else:
            self.mocks = self.load_data(data=covariance, klim=klim)[-1]
        if self.mpicomm.bcast(self.mocks is not None, root=0):
            covariance = None
            if self.mpicomm.rank == 0:
                covariance = np.cov(self.mocks, rowvar=False, ddof=1)
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        self.wmatrix = wmatrix
        if not isinstance(wmatrix, WindowedPowerSpectrumMultipoles):
            self.wmatrix = WindowedPowerSpectrumMultipoles()
            if wmatrix is not None:
                self.wmatrix.init.update(wmatrix=wmatrix)
        if self.ells is not None:  # set by data
            self.wmatrix.init.update(ells=self.ells)
        if self.kedges is not None:  # set by data
            self.wmatrix.init.update(kedges=self.kedges)
        if klim is not None:
            self.wmatrix.init.update(klim=klim)
        self.wmatrix.init.update(kwargs)
        self.wmatrix.init.setdefault('shotnoise', self.shotnoise)
        #if self.shotnoise is None: self.shotnoise = 0.
        if self.flatdata is None:
            self.wmatrix(**data)
            self.flatdata = self.wmatrix.flatpower.copy()
        else:
            self.wmatrix.runtime_info.initialize()
        input_kedges = self.kedges is not None
        for name in ['k', 'ells', 'kedges']:
            setattr(self, name, getattr(self.wmatrix, name))
        kmasklim = self.wmatrix.kmasklim
        if kmasklim is not None:  # cut has been applied to input k
            cumsize = np.insert(np.cumsum([len(kk) for kk in kmasklim.values()]), 0, 0)
            data = [self.flatdata[start:stop] for start, stop in zip(cumsize[:-1], cumsize[1:])]
            ells = list(kmasklim)
            self.flatdata = np.concatenate([data[ells.index(ell)][kmasklim[ell]] for ell in self.ells])
        if isinstance(self.covariance, ObservableCovariance):
            if input_kedges: x, method = [(edges[:-1] + edges[1:]) / 2. for edges in self.kedges], 'mid'
            else: x, method = list(self.k),'mean'
            self.nobs = self.covariance.nobs
            self.covariance = self.covariance.xmatch(x=x, projs=list(self.ells), method=method).view(projs=list(self.ells))
        self.transform = transform
        allowed_transform = [None, 'cubic']
        if self.transform not in allowed_transform:
            raise ValueError('transform must be one of {}'.format(allowed_transform))

    def load_data(self, data=None, klim=None):

        def load_data(fn):
            from pypower import MeshFFTPower, PowerSpectrumMultipoles
            #with utils.LoggingContext(level='warning'):
            state = np.load(fn, allow_pickle=True)[()]
            if '_projs' in state:
                toret = ObservableArray.from_state(state)
            else:
                toret = MeshFFTPower.from_state(state)
                if hasattr(toret, 'poles'):
                    toret = toret.poles
                else:
                    toret = PowerSpectrumMultipoles.from_state(state)
            return toret

        def lim_data(power, klim=klim):
            ells, list_k, list_kedges, list_data = [], [], [], []

            if isinstance(power, ObservableArray):
                shotnoise = power.attrs.get('shotnoise', None)
                if klim is None:
                    klim = {ell: (0, np.inf) for ell in power.projs}
                for ell, lim in klim.items():
                    start, stop, *step = lim
                    rebin = 1
                    if step and step[0] != 1: rebin = np.rint(step[0] / np.diff(power.edges(projs=ell)).mean()).astype(int)
                    power_slice = power.copy().select(xlim=(start, stop), rebin=rebin, projs=ell)
                    ells.append(ell)
                    list_k.append(power_slice.x(projs=ell))
                    list_kedges.append(power_slice.edges(projs=ell))
                    list_data.append(power_slice.view(projs=ell))
            else:
                if hasattr(power, 'poles'):
                    power = power.poles
                shotnoise = power.shotnoise
                if klim is None:
                    klim = {ell: (0, np.inf) for ell in power.ells}
                for ell, lim in klim.items():
                    power_slice = power.copy().select(lim)
                    ells.append(ell)
                    list_k.append(power_slice.modeavg())
                    list_kedges.append(power_slice.edges[0])
                    list_data.append(power_slice(ell=ell, complex=False))
            return list_k, list_kedges, tuple(ells), list_data, shotnoise

        def load_all(lmocks):
            list_mocks = []
            for mocks in lmocks:
                if utils.is_path(mocks):
                    gmocks = glob.glob(str(mocks))  # glob takes str
                    if not gmocks:
                        import warnings
                        warnings.warn('file {} is not found'.format(mocks))
                    list_mocks += sorted(gmocks)
                else:
                    list_mocks.append(mocks)

            fns = [mock for mock in list_mocks if utils.is_path(mock)]
            if len(fns):
                nfns = 5
                if len(fns) < nfns:
                    msg = 'Loading 1 file {}.'.format(fns)
                else:
                    msg = 'Loading {:d} files [{}].'.format(len(fns), ', ..., '.join(fns[::len(fns) // nfns]))
                self.log_info(msg)

            list_y, list_shotnoise = [], []
            for mock in list_mocks:
                if utils.is_path(mock):
                    mock = load_data(mock)
                mock_k, mock_kedges, mock_ells, mock_y, mock_shotnoise = lim_data(mock)
                if self.k is None or self.kedges is None:
                    self.k, self.kedges, self.ells = mock_k, mock_kedges, mock_ells
                if self.kedges is not None and mock_kedges is not None and not all(np.allclose(sk, mk, atol=0., rtol=1e-3) for sk, mk in zip(self.kedges, mock_kedges)):
                    raise ValueError('{} does not have expected k-edges (based on previous data)'.format(mock))
                if mock_ells != self.ells:
                    raise ValueError('{} does not have expected poles (based on previous data)'.format(mock))
                list_y.append(np.concatenate(mock_y))
                if mock_shotnoise is not None:
                    list_shotnoise.append(mock_shotnoise)
            return list_y, list_shotnoise

        flatdata, shotnoise, list_shotnoise, list_y = None, None, None, None
        if self.mpicomm.rank == 0 and data is not None:
            if not utils.is_sequence(data):
                data = [data]
            if any(_is_from_pypower(dd) or isinstance(dd, ObservableArray) for dd in data):
                list_y, list_shotnoise = load_all(data)
                if not list_y: raise ValueError('no data/mocks could be obtained from {}'.format(data))
            else:
                list_y = list(data)
            flatdata = np.mean(list_y, axis=0)
            if list_shotnoise:
                shotnoise = np.mean(list_shotnoise, axis=0)

        self.k, self.kedges, self.ells, flatdata, shotnoise = self.mpicomm.bcast((self.k, self.kedges, self.ells, flatdata, shotnoise) if self.mpicomm.rank == 0 else None, root=0)
        if self.shotnoise is None: self.shotnoise = shotnoise
        return flatdata, list_y

    @plotting.plotter(interactive={'kw_theory': {'color': 'black', 'label': 'reference'}})
    def plot(self, scaling='kpk', kw_theory=None, fig=None):
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
        k_exp = 1 if scaling == 'kpk' else 0

        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(self.k[ill], self.k[ill]**k_exp * data[ill], yerr=self.k[ill]**k_exp * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.k[ill], self.k[ill]**k_exp * theory[ill], **kw_theory[ill])
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.k[ill], (data[ill] - theory[ill]) / std[ill], **kw_theory[ill])
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        if show_legend: lax[0].legend()
        if scaling == 'kpk':
            lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if scaling == 'loglog':
            lax[0].set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
            lax[0].set_yscale('log')
            lax[0].set_xscale('log')
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
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

        for ill, ell in enumerate(self.ells):
            lax[ill].errorbar(self.k[ill], self.k[ill] * (data[ill] - nobao[ill]), yerr=self.k[ill] * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o')
            lax[ill].plot(self.k[ill], self.k[ill] * (theory[ill] - nobao[ill]), color='C{:d}'.format(ill))
            lax[ill].set_ylabel(r'$k \Delta P_{{{:d}}}(k)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        for ax in lax: ax.grid(True)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

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
        cumsize = np.insert(np.cumsum([len(k) for k in self.k]), 0, 0)
        mat = [[self.covariance[start1:stop1, start2:stop2] for start2, stop2 in zip(cumsize[:-1], cumsize[1:])] for start1, stop1 in zip(cumsize[:-1], cumsize[1:])]
        return plot_covariance_matrix(mat, x1=self.k, xlabel1=r'$k$ [$h/\mathrm{Mpc}$]', label1=[r'$\ell = {:d}$'.format(ell) for ell in self.ells], corrcoef=corrcoef, **kwargs)

    def calculate(self):
        self.flattheory = self.wmatrix.flatpower
        if self.transform == 'cubic':
            # See eq. 16 of https://arxiv.org/pdf/2302.07484.pdf
            self.flattheory = (3. * (self.flattheory / self.flatdata)**(1. / 3.) - 2.) * self.flatdata

    @property
    def theory(self):
        return self.wmatrix.power

    @property
    def theory_nobao(self):
        template = self.wmatrix.theory.template
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
        cumsize = np.insert(np.cumsum([len(k) for k in self.k]), 0, 0)
        return [self.flatdata[start:stop] for start, stop in zip(cumsize[:-1], cumsize[1:])]

    @property
    def std(self):
        cumsize = np.insert(np.cumsum([len(k) for k in self.k]), 0, 0)
        diag = np.diag(self.covariance)**0.5
        return [diag[start:stop] for start, stop in zip(cumsize[:-1], cumsize[1:])]

    def __getstate__(self):
        state = {}
        for name in ['k', 'kedges', 'ells', 'flatdata', 'flattheory', 'shotnoise']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @classmethod
    def install(cls, config):
        # TODO: remove this dependency
        #config.pip('git+https://github.com/cosmodesi/pypower')
        pass

    def to_array(self):
        from desilike.observables import ObservableArray
        return ObservableArray(x=self.k, edges=self.kedges, value=self.data, projs=self.ells, attrs={'shotnoise': self.shotnoise}, name=self.__class__.__name__)