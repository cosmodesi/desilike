import glob

import numpy as np

from desilike import plotting, jax, utils
from desilike.utils import path_types
from desilike.base import BaseCalculator
from .window import WindowedCorrelationFunctionMultipoles


def _is_array(data):
    return isinstance(data, (np.ndarray,) + jax.array_types)


def _is_from_pycorr(data):
    return isinstance(data, path_types) or not _is_array(data)


class TracerCorrelationFunctionMultipolesObservable(BaseCalculator):
    """
    Tracer correlation function multipoles observable: compare measurement to theory.

    Parameters
    ----------
    data : str, Path, list, pycorr.BaseTwoPointEstimator, dict, default=None
        Data correlation function measurement: array, :class:`pycorr.BaseTwoPointEstimator` instance,
        or path to such instances, or list of such objects (in which case the average of them is taken).
        If dict, parameters to be passed to theory to generate mock measurement.
    
    covariance : list, default=None
        2D array, list of :class:`pycorr.BaseTwoPointEstimator` instances, or paths to such instances;
        these are used to compute the covariance matrix.
    
    slim : dict, default=None
        Separation limits: a dictionary mapping multipoles to (min separation, max separation, step (float)),
        e.g. ``{0: (30., 160., 5.), 2: (30., 160., 5.)}``.
    
    **kwargs : dict
        Optional arguments for :class:`WindowedCorrelationFunctionMultipoles`, e.g.:
        
        - theory: defaults to :class:`KaiserTracerCorrelationFunctionMultipoles`.
        - fiber_collisions
    """
    def initialize(self, data=None, covariance=None, slim=None, wmatrix=None, **kwargs):
        self.s, self.sedges, self.ells = None, None, None
        self.flatdata, self.mocks, self.covariance = None, None, None
        if not isinstance(data, dict):
            self.flatdata = self.load_data(data=data, slim=slim)[0]
        if self.mpicomm.bcast(_is_array(covariance), root=0):
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        else:
            self.mocks = self.load_data(data=covariance, slim=slim)[-1]
        if self.mpicomm.bcast(self.mocks is not None, root=0):
            covariance = None
            if self.mpicomm.rank == 0:
                covariance = np.cov(self.mocks, rowvar=False, ddof=1)
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        self.wmatrix = wmatrix
        if not isinstance(wmatrix, WindowedCorrelationFunctionMultipoles):
            self.wmatrix = WindowedCorrelationFunctionMultipoles()
        if self.sedges is not None:  # set by data
            slim = {ell: (edges[0], edges[-1], np.mean(np.diff(edges))) for ell, edges in zip(self.ells, self.sedges)}
            self.wmatrix.init.update(s=self.s)
        if slim is not None:
            self.wmatrix.init.update(slim=slim)
        self.wmatrix.init.update(kwargs)
        if self.flatdata is None:
            self.wmatrix(**data)
            self.flatdata = self.flattheory.copy()
        else:
            self.wmatrix.runtime_info.initialize()
        for name in ['s', 'ells', 'sedges']:
            setattr(self, name, getattr(self.wmatrix, name))

    def load_data(self, data=None, slim=None):

        def load_data(fn):
            from pycorr import TwoPointCorrelationFunction
            return TwoPointCorrelationFunction.load(fn)

        def lim_data(corr, slim=slim):
            if slim is None:
                slim = {ell: (0, np.inf) for ell in (0, 2, 4)}
            ells, list_s, list_sedges, list_data = [], [], [], []
            for ell, (lo, hi, *step) in slim.items():
                corr_slice = corr
                if step:
                    rebin = int(step / np.diff(corr.edges[0]).mean() + 0.5)  # nearest integer
                    corr_slice = corr[:(corr.shape[0] // rebin) * rebin:rebin]
                s, data = corr_slice(ells=ell, return_sep=True, return_std=False)
                ells.append(ell)
                mask = (s >= lo) & (s < hi)
                index = np.flatnonzero(mask)
                list_s.append(s[mask])
                list_sedges.append(corr.edges[0][np.append(index, index[-1] + 1)])
                list_data.append(data[index])
            return list_s, list_sedges, ells, list_data

        def load_all(list_mocks):
            list_y = []
            for mocks in list_mocks:
                if isinstance(mocks, path_types):
                    mocks = [load_data(mock) for mock in glob.glob(mocks)]
                else:
                    mocks = [mocks]
                for mock in mocks:
                    mock_s, mock_sedges, mock_ells, mock_y = lim_data(mock)
                    if self.s is None:
                        self.s, self.sedges, self.ells = mock_s, mock_sedges, mock_ells
                    if not all(np.allclose(ss, ms, atol=0., rtol=1e-3) for ss, ms in zip(self.s, mock_s)):
                        raise ValueError('{} does not have expected s-binning (based on previous data)'.format(mock))
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

    @plotting.plotter
    def plot(self):
        """
        Plot data and theory correlation function multipoles.

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
        height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        data, model, std = self.data, self.model, self.std
        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(self.s[ill], self.s[ill]**2 * data[ill], yerr=self.s[ill]**2 * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.s[ill], self.s[ill]**2 * model[ill], color='C{:d}'.format(ill))
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.s[ill], (data[ill] - model[ill]) / std[ill], color='C{:d}'.format(ill))
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta \xi_{{{0:d}}} / \sigma_{{ \xi_{{{0:d}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        lax[0].set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return lax

    @plotting.plotter
    def plot_covariance_matrix(self, corrcoef=True):
        from desilike.observables.plotting import plot_covariance_matrix
        cumsize = np.insert(np.cumsum([len(s) for s in self.s]), 0, 0)
        mat = [[self.covariance[start1:stop1, start2:stop2] for start2, stop2 in zip(cumsize[:-1], cumsize[1:])] for start1, stop1 in zip(cumsize[:-1], cumsize[1:])]
        return plot_covariance_matrix(mat, x1=self.s, xlabel1=r'$s$ [$\mathrm{Mpc}/h$]', label1=[r'$\ell = {:d}$'.format(ell) for ell in self.ells], corrcoef=corrcoef)

    @property
    def flattheory(self):
        return self.wmatrix.flatcorr

    @property
    def model(self):
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
        state = super(TracerCorrelationFunctionMultipolesObservable, self).__getstate__()
        for name in ['s', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @classmethod
    def install(cls, config):
        # TODO: remove this dependency
        #config.pip('git+https://github.com/cosmodesi/pycorr')
        pass
