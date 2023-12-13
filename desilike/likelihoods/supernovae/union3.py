import os
import numpy as np
from desilike import plotting,utils
from desilike.cosmo import is_external_cosmo
from .base import BaseSNLikelihood

class Union3SNLikelihood(BaseSNLikelihood):
    """
    Likelihood for the Union3&UNITY1.5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/pdf/2311.12098.pdf

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    config_fn = 'union3.yaml'
    installer_section = 'Union3SNLikelihood'
    name = 'Union3'
    
    def initialize(self, *args, cosmo=None, **kwargs):
        BaseSNLikelihood.initialize(self, *args, cosmo=cosmo, **kwargs)
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'background': {'luminosity_distance': {'z': self.light_curve_params['zcmb']}}}
    
    def calculate(self, dM=0):
        z = self.light_curve_params['zcmb']
        # Dimensionless luminosity distance 
        # D_L = H0*d_L = 100*h * cosmoprimo.luminosity_distance(z) | Cosmoprimo returns distances in [Mpc/h]
        # Thus, the dependence on H0 is absorbed in dM
        self.flattheory = 5 * np.log10(100*self.cosmo.luminosity_distance(z)) + 25
        self.flatdata = self.light_curve_params['mu'] - dM 
        BaseSNLikelihood.calculate(self)
    
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

        data_fn = os.path.join(data_dir, 'union3_mu.dat')

        if installer.reinstall or not exists_path(data_fn):
            github = 'https://raw.githubusercontent.com/rodri981/tmp-data/main/'
            for fn in ['union3_mu.dat','union3.cov','union3.invcov']:
                download(os.path.join(github, fn), os.path.join(data_dir, fn))

            # Creates config file to ensure compatibility with base class
            config_fn = os.path.join(data_dir, 'config.dataset')
            with open(config_fn, 'w') as file:
                for text in ['name = Union3\n', 'data_file = union3_mu.dat\n', 'mag_covmat_file = union3.cov\n']: #,'inv_covmat = union3.invcov\n'
                    file.write(text)
            installer.write({cls.__name__: {'data_dir': data_dir}})