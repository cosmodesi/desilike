import os
import numpy as np

from desilike import plotting, utils
from desilike.cosmo import is_external_cosmo
from desilike.jax import numpy as jnp
from .base import BaseSNLikelihood


class Union3SNLikelihood(BaseSNLikelihood):
    """
    Likelihood for the Union3 & UNITY1.5 type Ia supernovae sample.

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
    name = 'Union3SN'

    def initialize(self, *args, cosmo=None, correct_prior=False, **kwargs):
        BaseSNLikelihood.initialize(self, *args, cosmo=cosmo, **kwargs)
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5
        self.correct_prior = correct_prior
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'background': {'luminosity_distance': {'z': self.light_curve_params['zcmb']}}}
        if self.correct_prior:
            from cosmoprimo import Cosmology
            self.fid = Cosmology(Omega_m=0.3, engine='eisenstein_hu').luminosity_distance(self.light_curve_params['zcmb'])

    def calculate(self, dM=0):
        z = self.light_curve_params['zcmb']
        # Dimensionless luminosity distance
        # D_L = H0*d_L = 100*h * cosmoprimo.luminosity_distance(z) | Cosmoprimo returns distances in [Mpc/h]
        # Thus, the dependence on H0 is absorbed in dM
        self.flattheory = 5 * jnp.log10(100 * self.cosmo.luminosity_distance(z)) + 25
        self.flatdata = self.light_curve_params['mb'] - dM
        BaseSNLikelihood.calculate(self)

        if self.correct_prior:
            from desilike.jax import jax
            varied_names = self.cosmo.runtime_info.params.names(varied=True)

            def func(values):
                cosmo = self.cosmo.clone(**dict(zip(varied_names, values)), engine='eisenstein_hu')
                return 5 * jnp.log10(cosmo.luminosity_distance(z) / self.fid)

            values = jnp.array([self.cosmo[name] for name in varied_names])
            jac = jax.jacfwd(func)(values)
            jj = jac.T.dot(jac)
            # fill with ones to compute det below
            #ivar = jnp.diag(1. / jnp.array([self.cosmo.runtime_info.params[name].proposal for name in varied_names])**2)
            #eigenvalues, eigenvectors = jnp.linalg.eigh(jj)
            #print(eigenvalues)
            #tmp = np.diag(eigenvectors.T.dot(ivar).dot(eigenvectors))
            #eigenvalues = jnp.where(eigenvalues < tmp * 1e-14, jnp.ones_like(eigenvalues), eigenvalues)
            diff = func(values)
            # loglikelihood - original priors - the priors we set
            #self.loglikelihood += - 0.5 * jnp.sum(diff**2) - 0.5 * jnp.sum(jnp.log(eigenvalues))
            #self.loglikelihood += 0.5 * jnp.sum(diff**2) #- 0.5 * jnp.linalg.slogdet(jj)[1]
            self.loglikelihood = - 0.5 * jnp.sum(diff**2) + 0.5 * jnp.linalg.slogdet(jj)[1]

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

        data_fn = os.path.join(data_dir, 'lcparam_full.txt')
        cov_fn = os.path.join(data_dir, 'mag_covmat.txt')

        if installer.reinstall or not exists_path(data_fn):
            github = 'https://raw.githubusercontent.com/CobayaSampler/sn_data/master/Union3/'
            for fn in [data_fn, cov_fn]:
                download(os.path.join(github, os.path.basename(fn)), fn)

            # Creates config file to ensure compatibility with base class
            config_fn = os.path.join(data_dir, 'config.dataset')
            with open(config_fn, 'w') as file:
                for text in ['name = Union3', 'data_file = {}'.format(os.path.basename(data_fn)), 'mag_covmat_file = {}'.format(os.path.basename(cov_fn))]:
                    file.write(text + '\n')

        installer.write({cls.__name__: {'data_dir': data_dir}})