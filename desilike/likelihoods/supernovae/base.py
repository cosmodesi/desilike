"""Base class for type Ia supernovae (SN) likelihoods."""

import os

import numpy as np

from desilike.base import GaussianLikelihood
from desilike.parameter import Parameter, VariableCollection
from desilike.plotting import plotter


class BaseSNLikelihood(GaussianLikelihood):
    r"""Base likelihood for type Ia supernovae Hubble-diagram samples.

    Computes the distance-modulus residual between observed (corrected) apparent
    magnitudes and the theoretical distance modulus :math:`5 \log_{10}(d_L / h) + 25`
    from ``cosmo``, marginalizing over a nuisance absolute-magnitude offset.

    Subclasses must set ``data_file``/``covariance_file`` (filenames within
    ``data_dir``), implement :meth:`read_light_curve_params` if the file layout
    differs from the default, and override :meth:`__call__` to set
    ``self.flattheory`` before calling ``super().__call__()``.

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory. Defaults to the path saved by :class:`~desilike.install.Installer`
        once the likelihood has been installed (``installer(likelihood_instance)``).
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator. If ``None``, defaults to ``CosmoprimoCosmology(fiducial='DESI')``.
    params : Parameter, VariableCollection, dict, default=None
        Override the default nuisance parameter(s) (e.g. ``Mb``).
    """
    installer_section = None
    data_file = None
    covariance_file = None
    _zname = 'zcmb'  # name of the redshift column in light_curve_params; override per subclass

    @classmethod
    def propose_params(cls):
        """Return the default nuisance :class:`~desilike.parameter.VariableCollection` (``Mb``)."""
        return VariableCollection([Parameter('Mb', value=-19.263, prior=dict(limits=[-20., -18.]), latex='M_b')])

    def __init__(self, data_dir=None, cosmo=None, params=None, zrange=(None, None)):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer().data_dir(self.installer_section)
        self.light_curve_params = self.read_light_curve_params(os.path.join(data_dir, self.data_file))
        self.covariance = self.read_covariance(os.path.join(data_dir, self.covariance_file))
        zmin, zmax = zrange
        if zmin is not None or zmax is not None:
            z = self.light_curve_params[self._zname]
            mask = np.ones(len(z), dtype=bool)
            if zmin is not None:
                mask &= (z >= zmin)
            if zmax is not None:
                mask &= (z <= zmax)
            if not np.all(mask):
                self.light_curve_params = {key: arr[mask] for key, arr in self.light_curve_params.items()}
                self.covariance = self.covariance[np.ix_(mask, mask)]
        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(fiducial='DESI')
        self.cosmo = cosmo
        vc = self.propose_params()
        if params is not None:
            vc = vc + VariableCollection(params)
        for param in vc:
            setattr(self, param.basename, param)

    def read_covariance(self, fn):
        """Read a CosmoMC-format covariance file: leading line is the matrix size, then the flattened values."""
        with open(fn, 'r') as file:
            size = int(file.readline())
        return np.loadtxt(fn, skiprows=1).reshape(size, size)

    def read_light_curve_params(self, fn, header='#', sep=' ', skip=None):
        """Read a whitespace/comma-separated light-curve parameter table into a dict of arrays."""
        with open(fn, 'r') as file:
            names, values = None, None
            for iline, line in enumerate(file.readlines()):
                if skip is not None:
                    if isinstance(skip, str):
                        if line.strip().startswith(skip):
                            continue
                    elif iline <= skip:
                        continue
                if names is None:
                    names = [name.strip() for name in line[len(header):].split(sep) if name.strip()]
                    values = {name: [] for name in names}
                    continue
                row = [el for el in line.split(sep) if el.strip()]
                for name, value in zip(names, row):
                    try: value = float(value)
                    except ValueError: pass  # str
                    values[name].append(value)
        return {name: np.array(value) for name, value in values.items()}

    @property
    def std(self):
        """Per-point standard deviation, the sqrt of the covariance diagonal."""
        return np.diag(self.covariance) ** 0.5

    @plotter
    def plot(self, fig=None):
        """
        Plot Hubble diagram: Hubble residuals as a function of redshift.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 2 axes.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, lax = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, figsize=(6, 6), squeeze=True)
            fig.subplots_adjust(hspace=0)
        else:
            lax = fig.axes
        alpha = 0.3
        zname = self._zname
        argsort = np.argsort(self.light_curve_params[zname])
        zdata = self.light_curve_params[zname][argsort]
        flatdata, flattheory, std = np.asarray(self.flatdata)[argsort], np.asarray(self.flattheory)[argsort], self.std[argsort]
        lax[0].plot(zdata, flatdata, marker='o', markeredgewidth=0., linestyle='none', alpha=alpha, color='b')
        lax[0].plot(zdata, flattheory, linestyle='-', marker=None, color='k')
        lax[0].set_xscale('log')
        lax[1].errorbar(zdata, flatdata - flattheory, yerr=std, linestyle='none', marker='o', alpha=alpha, color='b')
        lax[0].set_ylabel(r'distance modulus [$\mathrm{mag}$]')
        lax[1].set_ylabel(r'Hubble res. [$\mathrm{mag}$]')
        lax[1].set_xlabel('$z$')
        return fig
