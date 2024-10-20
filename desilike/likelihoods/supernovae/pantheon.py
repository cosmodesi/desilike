import os

import numpy as np

from desilike import plotting, utils
from desilike.cosmo import is_external_cosmo
from desilike.jax import numpy as jnp
from .base import BaseSNLikelihood


class PantheonSNLikelihood(BaseSNLikelihood):
    """
    Likelihood for Pantheon type Ia supernovae sample.
    TODO: organize as SNObservable / ObservablesGaussianLikelihood?

    Reference
    ---------
    https://arxiv.org/abs/1710.00845

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    config_fn = 'pantheon.yaml'
    installer_section = 'PantheonSNLikelihood'
    name = 'PantheonSN'

    def initialize(self, *args, cosmo=None, **kwargs):
        super(PantheonSNLikelihood, self).initialize(*args, cosmo=cosmo, **kwargs)
        # Add statistical error
        self.covariance += np.diag(self.light_curve_params['dmb']**2)
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'background': {'luminosity_distance': {'z': self.light_curve_params['zcmb']}}}

    def calculate(self, Mb=0):
        z = self.light_curve_params['zcmb']
        self.flattheory = 5 * jnp.log10(self.cosmo.luminosity_distance(z) / self.cosmo['h']) + 25
        self.flatdata = self.light_curve_params['mb'] - Mb - 5 * np.log10((1 + self.light_curve_params['zhel']) / (1 + z))
        super(PantheonSNLikelihood, self).calculate()

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
        argsort = np.argsort(self.light_curve_params['zcmb'])
        zdata = self.light_curve_params['zcmb'][argsort]
        flatdata, flattheory, std = self.flatdata[argsort], self.flattheory[argsort], self.std[argsort]
        lax[0].plot(zdata, flatdata, marker='o', markeredgewidth=0., linestyle='none', alpha=alpha, color='b')
        lax[0].plot(zdata, flattheory, linestyle='-', marker=None, color='k')
        lax[0].set_xscale('log')
        lax[1].errorbar(zdata, flatdata - flattheory, yerr=std, linestyle='none', marker='o', alpha=alpha, color='b')
        lax[0].set_ylabel(r'distance modulus [$\mathrm{mag}$]')
        lax[1].set_ylabel(r'Hubble res. [$\mathrm{mag}$]')
        lax[1].set_xlabel('$z$')
        return fig

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download

        config_fn = os.path.join(data_dir, 'full_long.dataset')

        if installer.reinstall or not exists_path(config_fn):
            github = 'https://raw.githubusercontent.com/dscolnic/Pantheon/master/'
            for fn in [os.path.basename(config_fn), 'lcparam_full_long.txt', 'lcparam_full_long_zhel.txt', 'sys_full_long.txt']:
                download(os.path.join(github, fn), os.path.join(data_dir, fn))
            with open(config_fn, 'r') as file:
                txt = file.read()
            txt = txt.replace('/your-path/', '')
            with open(config_fn, 'w') as file:
                file.write(txt)

        installer.write({cls.__name__: {'data_dir': data_dir}})
