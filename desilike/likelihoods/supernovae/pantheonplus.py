import os

import numpy as np

from desilike import plotting, utils
from desilike.cosmo import is_external_cosmo
from desilike.jax import numpy as jnp
from .base import BaseSNLikelihood


class PantheonPlusSNLikelihood(BaseSNLikelihood):
    """
    Likelihood for Pantheon+ (without SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    config_fn = 'pantheonplus.yaml'
    installer_section = 'PantheonPlusSNLikelihood'
    name = 'PantheonPlusSN'

    def initialize(self, *args, cosmo=None, **kwargs):
        BaseSNLikelihood.initialize(self, *args, cosmo=cosmo, **kwargs)
        zmask = self.light_curve_params['zHD'] > 0.01  # Only those SNe at z > 0.01 are used for cosmology
        self.light_curve_params = {name: value[zmask] for name, value in self.light_curve_params.items()}
        self.covariance = self.covariance[np.ix_(zmask, zmask)]
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'background': {'luminosity_distance': {'z': self.light_curve_params['zHD']}}}

    def calculate(self, Mb=0):
        z = self.light_curve_params['zHD']
        self.flattheory = 5 * jnp.log10(self.cosmo.luminosity_distance(z) / self.cosmo['h']) + 25
        self.flatdata = self.light_curve_params['m_b_corr'] - Mb - 5 * np.log10((1 + self.light_curve_params['zHEL']) / (1 + z))
        BaseSNLikelihood.calculate(self)

    def read_light_curve_params(self, fn):
        return BaseSNLikelihood.read_light_curve_params(self, fn, header='', sep=' ')

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

        from desilike.install import exists_path, download

        data_fn = os.path.join(data_dir, 'Pantheon+SH0ES.dat')
        cov_fn = os.path.join(data_dir, 'Pantheon+SH0ES_STAT+SYS.cov')

        if installer.reinstall or not exists_path(data_fn):
            github = 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/'
            for fn in [data_fn, cov_fn]:
                download(os.path.join(github, os.path.basename(fn)), fn)

            # Creates config file to ensure compatibility with base class
            config_fn = os.path.join(data_dir, 'config.dataset')
            with open(config_fn, 'w') as file:
                for text in ['name = PantheonPlus', 'data_file = {}'.format(os.path.basename(data_fn)), 'mag_covmat_file = {}'.format(os.path.basename(cov_fn))]:
                    file.write(text + '\n')

        installer.write({cls.__name__: {'data_dir': data_dir}})
