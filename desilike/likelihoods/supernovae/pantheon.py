import os

import numpy as np
import scipy as sp

from desilike import plotting, utils
from .base import SNLikelihood


class PantheonSNLikelihood(SNLikelihood):

    """Pantheon type Ia supernova sample."""

    config_fn = 'pantheon.yaml'
    installer_section = 'PantheonSNLikelihood'

    def initialize(self, *args, **kwargs):
        super(PantheonSNLikelihood, self).initialize(*args, **kwargs)
        # Add statistical error
        self.covariance += np.diag(self.light_curve_params['dmb']**2)
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5

    def calculate(self, Mb=0):
        z = self.light_curve_params['zcmb']
        self.flattheory = 5 * np.log10(self.cosmo.luminosity_distance(z)) + 25
        self.flatdata = self.light_curve_params['mb'] - Mb - 5 * np.log10((1 + self.light_curve_params['zhel']) / (1 + z))
        super(PantheonSNLikelihood, self).calculate()

    def plot(self, fn, kw_save=None):
        from matplotlib import pyplot as plt
        fig, lax = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, figsize=(6, 6), squeeze=True)
        fig.subplots_adjust(hspace=0)
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
        if fn is not None:
            plotting.savefig(fn, fig=fig, **(kw_save or {}))
        return lax

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download

        config_fn = os.path.join(data_dir, 'full_long.dataset')

        if installer.force_reinstall or not exists_path(config_fn):
            github = 'https://raw.githubusercontent.com/dscolnic/Pantheon/master/'
            for fn in ['full_long.dataset', 'lcparam_full_long.txt', 'lcparam_full_long_zhel.txt', 'sys_full_long.txt']:
                download(os.path.join(github, fn), os.path.join(data_dir, fn))
            with open(config_fn, 'r') as file:
                txt = file.read()
            txt = txt.replace('/your-path/', '')
            with open(config_fn, 'w') as file:
                file.write(txt)
            installer.write({cls.__name__: {'data_dir': data_dir}})
