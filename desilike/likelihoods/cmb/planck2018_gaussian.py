import os

from desilike.likelihoods import BaseGaussianLikelihood
from desilike.jax import numpy as jnp
from desilike import utils


class BasePlanck2018GaussianLikelihood(BaseGaussianLikelihood):
    r"""
    Gaussian approximation of "base" likelihoods of Planck's 2018 data release.

    Reference
    ---------
    https://arxiv.org/abs/1807.06209

    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    Parameters
    ----------
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.

    basename : str, default='base_plikHM_TTTEEE_lowl_lowE_lensing'
        Likelihood base name, e.g. 'base_plikHM_TT', 'base_plikHM_TTTEEE', 'base_plikHM_TTTEEE_lowl_lowE_lensing'.

    source : str, default='covmat'
        Source, either:

        - 'covmat': use '.margestats' for mean and '.covmat' file as covariance.
        - 'chains': compute mean and covariance from chains

        Both options are very close (within precision in provided file).
    """
    config_fn = 'planck2018_gaussian.yaml'
    installer_section = 'BasePlanck2018GaussianLikelihood'
    data_file_id = 'COM_CosmoParams_base-plikHM_R3.01.zip'

    def initialize(self, cosmo=None, data_dir=None, basename='base_plikHM_TTTEEE_lowl_lowE_lensing', source='covmat'):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer()[self.installer_section]['data_dir']
        try:
            base_dir, obs_dir = basename.split('_plikHM_')
        except ValueError as exc:
            raise ValueError('basename {} is expected to contain "_plikHM_"'.format(basename)) from exc
        self.base_chain_fn = os.path.join(data_dir, base_dir, 'plikHM_' + obs_dir, basename)
        self.base_dist_fn = os.path.join(data_dir, base_dir, 'plikHM_' + obs_dir, 'dist', basename)
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        convert_params = {'omegabh2': 'omega_b', 'omegach2': 'omega_cdm', 'omegak': 'Omega_k', 'w': 'w0_fld', 'wa': 'wa_fld', 'theta': 'theta_cosmomc', 'tau': 'tau_reio',
                          'mnu': 'm_ncdm_tot', 'logA': 'ln10^10A_s', 'ns': 'n_s', 'nrun': 'alpha_s', 'r': 'r'}
        basenames = list(convert_params.keys())
        if source == 'covmat':
            from desilike import LikelihoodFisher
            fisher = LikelihoodFisher.read_getdist(self.base_dist_fn)
        elif source == 'chains' or source[0] == 'chains':
            burnin = None
            if utils.is_sequence(source): burnin = source[1]
            from desilike.samples import Chain
            chains = Chain.read_getdist(self.base_chain_fn)
            chain = Chain.concatenate([chain.select(basename=basenames).remove_burnin(burnin) if burnin is not None else chain for chain in chains])
            fisher = chain.to_fisher()
        else:
            raise ValueError('source must be one of ["covmat", "chains"]')
        params = fisher.params(basename=basenames)
        super(BasePlanck2018GaussianLikelihood, self).initialize(data=fisher.mean(params=params), covariance=fisher.covariance(params=params))
        self.cosmo_quantities = [convert_params[param.basename] for param in params]

    @property
    def flattheory(self):
        toret = []
        for param in self.cosmo_quantities:
            if param == 'theta_cosmomc':
                cosmo = self.cosmo.clone(engine='camb')
                toret.append(100. * cosmo.theta_cosmomc)
            else:
                toret.append(self.cosmo[param])
        return jnp.array(toret)

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        if installer.force_reinstall or not exists_path(os.path.join(data_dir, 'base')):
            # Install data
            tar_base, size = cls.data_file_id, None
            if utils.is_sequence(tar_base):
                tar_base, size = cls.data_file_id
            url = 'http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID={}'.format(tar_base)
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn, size=size)
            extract(tar_fn, data_dir)
            installer.write({cls.installer_section: {'data_dir': data_dir}})


class FullGridPlanck2018GaussianLikelihood(BasePlanck2018GaussianLikelihood):
    r"""
    Gaussian approximation of the full grid of likelihoods of Planck's 2018 data release.

    Reference
    ---------
    https://arxiv.org/abs/1807.06209

    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    Parameters
    ----------
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.

    basename : str, default='base_plikHM_TTTEEE_lowl_lowE_lensing'
        Likelihood base name, e.g. 'base_plikHM_TT', 'base_plikHM_TTTEEE', 'base_plikHM_TTTEEE_lowl_lowE_lensing', 'base_mnu_plikHM_TTTEEE_lowl_lowE_lensing'.

    source : str, default='covmat'
        Source, either:

        - 'covmat': use '.margestats' for mean and '.covmat' file as covariance.
        - 'chains': compute mean and covariance from chains

        Both options are very close (within precision in provided file).
    """
    config_fn = 'planck2018_gaussian.yaml'
    installer_section = 'FullGridPlanck2018GaussianLikelihood'
    data_file_id = ('COM_CosmoParams_fullGrid_R3.01.zip', 11e9)
