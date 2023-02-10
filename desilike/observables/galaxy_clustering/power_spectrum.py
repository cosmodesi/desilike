import glob

import numpy as np

from desilike import plotting, jax, utils
from desilike.utils import path_types
from desilike.base import BaseCalculator
from .window import WindowedPowerSpectrumMultipoles


def _is_array(data):
    return isinstance(data, (np.ndarray,) + jax.array_types)


def _is_from_pypower(data):
    return isinstance(data, path_types) or not _is_array(data)


class TracerPowerSpectrumMultipolesObservable(BaseCalculator):
    """
    Tracer power spectrum multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : array, str, Path, list, pypower.PowerSpectrumMultipoles, dict, default=None
        Data power spectrum measurement: array, :class:`pypower.PowerSpectrumMultipoles` instance,
        or path to such instances, or list of such objects (in which case the average of them is taken).
        If dict, parameters to be passed to theory to generate mock measurement.

    covariance : array, list, default=None
        2D array, list of :class:`pypower.PowerSpectrumMultipoles` instance` instances, or paths to such instances;
        these are used to compute the covariance matrix.

    klim : dict, default=None
        Wavenumber limits: a dictionary mapping multipoles to (min separation, max separation, step (float)),
        e.g. ``{0: (0.01, 0.2, 0.01), 2: (0.01, 0.15, 0.01)}``.

    wmatrix : str, Path, pypower.BaseMatrix, WindowedPowerSpectrumMultipoles, default=None
        Optionally, window matrix.

    **kwargs : dict
        Optional arguments for :class:`WindowedPowerSpectrumMultipoles`, e.g.:

        - theory: defaults to :class:`KaiserTracerPowerSpectrumMultipoles`.
        - shotnoise: take shot noise from ``data``, or ``covariance`` (mocks) if provided.
        - fiber_collisions
    """
    def initialize(self, data=None, covariance=None, klim=None, wmatrix=None, **kwargs):
        self.k, self.kedges, self.ells, self.shotnoise = None, None, None, 0.
        self.flatdata, self.mocks, self.covariance = None, None, None
        if not isinstance(data, dict):
            self.flatdata = self.load_data(data=data, klim=klim)[0]
        if self.mpicomm.bcast(_is_array(covariance), root=0):
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
        if self.kedges is not None:  # set by data
            klim = {ell: (edges[0], edges[-1], np.mean(np.diff(edges))) for ell, edges in zip(self.ells, self.kedges)}
            self.wmatrix.init.update(k=self.k)
        if klim is not None:
            self.wmatrix.init.update(klim=klim)
        self.wmatrix.init.update(kwargs)
        self.wmatrix.init.setdefault('shotnoise', self.shotnoise)
        if self.flatdata is None:
            self.wmatrix(**data)
            self.flatdata = self.flattheory.copy()
        else:
            self.wmatrix.runtime_info.initialize()
        for name in ['k', 'ells', 'kedges', 'shotnoise']:
            setattr(self, name, getattr(self.wmatrix, name))

    def load_data(self, data=None, klim=None):

        def load_data(fn):
            from pypower import MeshFFTPower, PowerSpectrumMultipoles
            toret = MeshFFTPower.load(fn)
            if hasattr(toret, 'poles'):
                return toret.poles
            return PowerSpectrumMultipoles.load(fn)

        def lim_data(power, klim=klim):
            if hasattr(power, 'poles'):
                power = power.poles
            shotnoise = power.shotnoise
            if klim is None:
                klim = {ell: (0, np.inf) for ell in power.ells}
            ells, list_k, list_kedges, list_data = [], [], [], []
            for ell, (lo, hi, *step) in klim.items():
                power_slice = power
                if step:
                    rebin = int(step / np.diff(power.kedges).mean() + 0.5)  # nearest integer
                    power_slice = power[:(power.shape[0] // rebin) * rebin:rebin]
                ells.append(ell)
                mask = (power_slice.k >= lo) & (power_slice.k < hi)
                index = np.flatnonzero(mask)
                list_k.append(power_slice.k[index])
                list_kedges.append(power_slice.kedges[np.append(index, index[-1] + 1)])
                list_data.append(power_slice(ell=ell, complex=False)[index])
            return list_k, list_kedges, tuple(ells), list_data, shotnoise

        def load_all(list_mocks):
            list_y, list_shotnoise = [], []
            for mocks in list_mocks:
                if isinstance(mocks, path_types):
                    mocks = [load_data(mock) for mock in sorted(glob.glob(mocks))]
                else:
                    mocks = [mocks]
                for mock in mocks:
                    mock_k, mock_kedges, mock_ells, mock_y, mock_shotnoise = lim_data(mock)
                    if self.k is None:
                        self.k, self.kedges, self.ells = mock_k, mock_kedges, mock_ells
                    if not all(np.allclose(sk, mk, atol=0., rtol=1e-3) for sk, mk in zip(self.k, mock_k)):
                        raise ValueError('{} does not have expected k-binning (based on previous data)'.format(mock))
                    if mock_ells != self.ells:
                        raise ValueError('{} does not have expected poles (based on previous data)'.format(mock))
                    list_y.append(np.concatenate(mock_y))
                    list_shotnoise.append(mock_shotnoise)
            return list_y, list_shotnoise

        flatdata, shotnoise, list_y = None, None, None
        if self.mpicomm.rank == 0 and data is not None:
            if not utils.is_sequence(data):
                data = [data]
            if any(_is_from_pypower(dd) for dd in data):
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

    @plotting.plotter
    def plot(self, scaling='kpk'):
        """
        Plot data and theory power spectrum multipoles.

        Parameters
        ----------
        scaling : str, default='kpk'
            Either 'kpk' or 'loglog'.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        data, theory, std = self.data, self.theory, self.std
        k_exp = 1 if scaling == 'kpk' else 0
        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(self.k[ill], self.k[ill]**k_exp * data[ill], yerr=self.k[ill]**k_exp * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.k[ill], self.k[ill]**k_exp * theory[ill], color='C{:d}'.format(ill))
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.k[ill], (data[ill] - theory[ill]) / std[ill], color='C{:d}'.format(ill))
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        if scaling == 'kpk':
            lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if scaling == 'loglog':
            lax[0].set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
            lax[0].set_yscale('log')
            lax[0].set_xscale('log')
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return lax

    @plotting.plotter
    def plot_wiggles(self):
        """
        Plot data and theory BAO power spectrum wiggles.

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        height_ratios = [1] * len(self.ells)
        figsize = (6, 2 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        data, theory, std = self.data, self.theory, self.std
        try:
            mode = self.wmatrix.theory.wiggle
        except AttributeError as exc:
            raise ValueError('Theory {} has no mode wiggle'.format(self.theory.theory.__class__)) from exc
        self.wmatrix.theory.wiggle = False
        for calculator in self.runtime_info.pipeline.calculators:
            calculator.runtime_info.tocalculate = True
            calculator.runtime_info.calculate()
        nowiggle = self.theory
        self.wmatrix.theory.wiggle = mode
        for ill, ell in enumerate(self.ells):
            lax[ill].errorbar(self.k[ill], self.k[ill] * (data[ill] - nowiggle[ill]), yerr=self.k[ill] * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o')
            lax[ill].plot(self.k[ill], self.k[ill] * (theory[ill] - nowiggle[ill]), color='C{:d}'.format(ill))
            lax[ill].set_ylabel(r'$k \Delta P_{{{:d}}}(k)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        for ax in lax: ax.grid(True)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return lax

    @plotting.plotter
    def plot_covariance_matrix(self, corrcoef=True):
        from desilike.observables.plotting import plot_covariance_matrix
        cumsize = np.insert(np.cumsum([len(k) for k in self.k]), 0, 0)
        mat = [[self.covariance[start1:stop1, start2:stop2] for start2, stop2 in zip(cumsize[:-1], cumsize[1:])] for start1, stop1 in zip(cumsize[:-1], cumsize[1:])]
        return plot_covariance_matrix(mat, x1=self.k, xlabel1=r'$k$ [$h/\mathrm{Mpc}$]', label1=[r'$\ell = {:d}$'.format(ell) for ell in self.ells], corrcoef=corrcoef)

    @property
    def flattheory(self):
        return self.wmatrix.flatpower

    @property
    def theory(self):
        return self.wmatrix.power

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
        state = super(TracerPowerSpectrumMultipolesObservable, self).__getstate__()
        for name in ['k', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @classmethod
    def install(cls, config):
        # TODO: remove this dependency
        #config.pip('git+https://github.com/cosmodesi/pypower')
        pass
