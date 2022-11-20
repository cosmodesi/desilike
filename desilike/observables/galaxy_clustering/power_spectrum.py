import glob

import numpy as np

from desilike import plotting, utils
from desilike.base import BaseCalculator
from desilike.theories.galaxy_clustering.base import WindowedPowerSpectrumMultipoles


class ObservedTracerPowerSpectrum(BaseCalculator):

    def initialize(self, data=None, mocks=None, wmatrix=None, theory=None, **kwargs):
        self.k, self.ells = None, None
        self.flatdata, shotnoise = self.load_data(data=data, **kwargs)[:2]
        self.mocks = self.load_data(data=mocks, **kwargs)[-1]
        if self.mpicomm.bcast(self.mocks is not None, root=0):
            covariance = None
            if self.mpicomm.rank == 0:
                covariance = np.cov(self.mocks, rowvar=False, ddof=1)
            self.covariance = self.mpicomm.bcast(covariance, root=0)
        self.wmatrix = WindowedPowerSpectrumMultipoles(k=self.k, ells=self.ells, wmatrix=wmatrix, theory=theory, shotnoise=shotnoise)

    def load_data(self, data=None, klim=None, kstep=None, krebin=None):

        def load_data(fn):
            from pypower import MeshFFTPower, PowerSpectrumMultipoles
            toret = MeshFFTPower.load(fn)
            if hasattr(toret, 'poles'):
                return toret.poles
            return PowerSpectrumMultipoles.load(fn)

        def lim_data(power, klim=klim, kstep=kstep, krebin=krebin):
            if hasattr(power, 'poles'):
                power = power.poles
            shotnoise = power.shotnoise
            if krebin is None:
                krebin = 1
                if kstep is not None:
                    krebin = int(np.rint(kstep / np.diff(power.kedges).mean()))
            power = power[:(power.shape[0] // krebin) * krebin:krebin]
            data = power.get_power(complex=False)
            nells = len(power.ells)
            if klim is None:
                klim = {ell: [0, np.inf] for ell in power.ells}
            elif utils.is_sequence(klim):
                if not utils.is_sequence(klim[0]):
                    klim = [klim] * nells
                if len(klim) > nells:
                    raise ValueError('{:d} limits provided but only {:d} poles computed'.format(len(klim), nells))
                klim = {ell: klim[ill] for ill, ell in enumerate(power.ells)}
            elif not isinstance(klim, dict):
                raise ValueError('Unknown klim format; provide e.g. {0: (0.01, 0.2), 2: (0.01, 0.15)}')
            list_k, list_data, ells = [], [], []
            for ell, lim in klim.items():
                mask = (power.k >= lim[0]) & (power.k < lim[1])
                list_k.append(power.k[mask])
                list_data.append(data[power.ells.index(ell)][mask])
                ells.append(ell)
            return list_k, tuple(ells), list_data, shotnoise

        def load_all(list_mocks):
            list_y, list_shotnoise = [], []
            for mocks in list_mocks:
                if isinstance(mocks, str):
                    mocks = [load_data(mock) for mock in sorted(glob.glob(mocks))]
                else:
                    mocks = [mocks]
                for mock in mocks:
                    mock_k, mock_ells, mock_y, mock_shotnoise = lim_data(mock)
                    if self.k is None:
                        self.k, self.ells = mock_k, mock_ells
                    if not all(np.allclose(sk, mk, atol=0., rtol=1e-3) for sk, mk in zip(self.k, mock_k)):
                        raise ValueError('{} does not have expected k-binning (based on previous data)'.format(mock))
                    if mock_ells != self.ells:
                        raise ValueError('{} does not have expected poles (based on previous data)'.format(mock))
                    list_y.append(np.ravel(mock_y))
                    list_shotnoise.append(mock_shotnoise)
            return list_y, list_shotnoise

        flatdata, shotnoise, list_y = None, None, None
        if self.mpicomm.rank == 0 and data is not None:
            if not utils.is_sequence(data):
                data = [data]
            list_y, list_shotnoise = load_all(data)
            flatdata = np.mean(list_y, axis=0)
            shotnoise = np.mean(list_shotnoise, axis=0)

        self.k, self.ells, flatdata, shotnoise = self.mpicomm.bcast((self.k, self.ells, flatdata, shotnoise) if self.mpicomm.rank == 0 else None, root=0)
        return flatdata, shotnoise, list_y

    def plot(self, fn=None, kw_save=None, show=False):
        from matplotlib import pyplot as plt
        height_ratios = [max(len(self.ells), 3)] + [1] * len(self.ells)
        figsize = (6, 1.5 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        data, model, std = self.data, self.model, self.std
        for ill, ell in enumerate(self.ells):
            lax[0].errorbar(self.k[ill], self.k[ill] * data[ill], yerr=self.k[ill] * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o', label=r'$\ell = {:d}$'.format(ell))
            lax[0].plot(self.k[ill], self.k[ill] * model[ill], color='C{:d}'.format(ill))
        for ill, ell in enumerate(self.ells):
            lax[ill + 1].plot(self.k[ill], (data[ill] - model[ill]) / std[ill], color='C{:d}'.format(ill))
            lax[ill + 1].set_ylim(-4, 4)
            for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
            lax[ill + 1].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell))
        for ax in lax: ax.grid(True)
        lax[0].legend()
        lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        if show: plt.show()
        return lax

    def plot_bao(self, fn=None, kw_save=None, show=False):
        from matplotlib import pyplot as plt
        height_ratios = [1] * len(self.ells)
        figsize = (6, 2 * sum(height_ratios))
        fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0)
        data, model, std = self.data, self.model, self.std
        try:
            mode = self.theory.theory.nowiggle
        except AttributeError as exc:
            raise ValueError('Theory {} has no mode nowiggle'.format(self.theory.theory.__class__)) from exc
        self.theory.theory.nowiggle = True
        for calc in self.runtime_info.pipeline.calculators: calc.runtime_info.torun = True
        self.runtime_info.run()
        nowiggle = self.model
        self.theory.theory.nowiggle = mode
        for ill, ell in enumerate(self.ells):
            lax[ill].errorbar(self.k[ill], self.k[ill] * (data[ill] - nowiggle[ill]), yerr=self.k[ill] * std[ill], color='C{:d}'.format(ill), linestyle='none', marker='o')
            lax[ill].plot(self.k[ill], self.k[ill] * (model[ill] - nowiggle[ill]), color='C{:d}'.format(ill))
            lax[ill].set_ylabel(r'$k \Delta P_{{{:d}}}(k)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        for ax in lax: ax.grid(True)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        if show: plt.show()
        return lax

    def unpack(self, array):
        toret = []
        nout = 0
        for k in self.k:
            sl = slice(nout, nout + len(k))
            toret.append(array[sl])
            nout = sl.stop
        return toret

    @property
    def flatmodel(self):
        return self.wmatrix.flatpower

    @property
    def model(self):
        return self.wmatrix.power

    @property
    def data(self):
        return self.unpack(self.flatdata)

    @property
    def std(self):
        return self.unpack(np.diag(self.covariance)**0.5)

    def __getstate__(self):
        state = super(ObservedTracerPowerSpectrum, self).__getstate__()
        for name in ['k', 'ells']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
