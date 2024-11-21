import os
import numpy as np
from desilike import plotting, utils
from desilike.cosmo import is_external_cosmo
from desilike.jax import numpy as jnp
from .base import BaseSNLikelihood


class DESY5SNLikelihood(BaseSNLikelihood):
    """
    Likelihood for DES-Y5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2401.02929

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    config_fn = 'des.yaml'
    installer_section = 'DESY5SNLikelihood'
    name = 'DESY5SN'

    def initialize(self, *args, cosmo=None, **kwargs):
        BaseSNLikelihood.initialize(self, *args, cosmo=cosmo, **kwargs)
        self.covariance = self.covariance + np.diag(self.light_curve_params['MUERR_FINAL'])**2
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'background': {'luminosity_distance': {'z': self.light_curve_params['zHD']}}}

    def calculate(self, Mb=0):
        z = self.light_curve_params['zHD']
        self.flattheory = 5 * jnp.log10(self.cosmo.luminosity_distance(z) / self.cosmo['h']) + 25
        self.flatdata = self.light_curve_params['MU'] - Mb - 5 * np.log10((1 + self.light_curve_params['zHEL']) / (1 + z))
        BaseSNLikelihood.calculate(self)

    def read_light_curve_params(self, fn):
        return BaseSNLikelihood.read_light_curve_params(self, fn, header='', sep=',', skip='#')

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot Hubble diagram: Hubble residuals as a function of distance.

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
        argsort = np.argsort(self.light_curve_params['zHD'])
        zdata = self.light_curve_params['zHD'][argsort]
        flatdata, flattheory, std = self.flatdata[argsort], self.flattheory[argsort], self.std[argsort]
        lax[0].plot(zdata, flatdata, marker='o', markeredgewidth=0., linestyle='none', alpha=alpha, color='b')
        lax[0].plot(zdata, flattheory, linestyle='-', marker=None, color='k')
        lax[0].set_xscale('log')
        lax[1].errorbar(zdata, flatdata - flattheory, yerr=std, linestyle='none', marker='o', alpha=alpha, color='b')
        lax[0].set_ylabel(r'distance modulus [$\mathrm{mag}$]')
        lax[1].set_ylabel(r'Hubble res. [$\mathrm{mag}$]')
        lax[1].set_xlabel('$z$')
        return lax

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        data_fn = os.path.join(data_dir, 'DES-SN5YR_HD.csv')
        cov_fn = os.path.join(data_dir, 'STAT+SYS.txt')

        if installer.reinstall or not exists_path(data_fn):
            github = 'https://raw.githubusercontent.com/des-science/DES-SN5YR/main/4_DISTANCES_COVMAT/'
            for fn in [data_fn, cov_fn]:
                fngz = fn.replace('.txt', '.txt.gz')
                download(os.path.join(github, os.path.basename(fngz)), fngz)
                if fngz.endswith('.gz'): extract(fngz, fn, remove=True)

            # Creates config file to ensure compatibility with base class
            config_fn = os.path.join(data_dir, 'config.dataset')
            with open(config_fn, 'w') as file:
                for text in ['name = DESY5', 'data_file = {}'.format(os.path.basename(data_fn)), 'mag_covmat_file = {}'.format(os.path.basename(cov_fn))]:
                    file.write(text + '\n')

        installer.write({cls.__name__: {'data_dir': data_dir}})