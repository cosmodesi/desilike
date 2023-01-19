import os

from desilike.likelihoods import BaseGaussianLikelihood
from desilike.jax import numpy as jnp
from desilike import utils


class BasePlanck2018GaussianLikelihood(BaseGaussianLikelihood):# WARNING:
    """
    Approximate Planck likelihood from marginalized mean and covariance.

    Note
    ----
    *.covmat files are the covariance of the final chains, which are supposedly burnin-free.
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
        base_chain_fn = os.path.join(data_dir, base_dir, 'plikHM_' + obs_dir, basename)
        base_dist_fn = os.path.join(data_dir, base_dir, 'plikHM_' + obs_dir, 'dist', basename)
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        convert_params = {'omegabh2': 'omega_b', 'omegach2': 'omega_cdm', 'omegak': 'Omega_k', 'w': 'w0_fld', 'wa': 'wa_fld', 'theta': 'theta_cosmomc', 'tau': 'tau_reio',
                          'mnu': 'm_ncdm_tot', 'logA': 'ln10^10A_s', 'ns': 'n_s', 'nrun': 'alpha_s', 'r': 'r'}
        if source == 'covmat':
            from desilike.parameter import ParameterCovariance
            covariance = ParameterCovariance.read_getdist(base_dist_fn)
        elif source == 'chains' or source[0] == 'chains':
            burnin = None
            if utils.is_sequence(source): burnin = source[1]
            from desilike.samples import Chain
            chains = Chain.read_getdist(base_chain_fn)
            chain = Chain.concatenate([chain.remove_burnin(burnin) if burnin is not None else chain for chain in chains])
            covariance = chain.cov(return_type=None)
        else:
            raise ValueError('source must be one of ["covmat", "chains"]')
        self.covariance = covariance.select(basename=list(convert_params.keys()))
        for param in self.covariance._params: param.update(basename=convert_params[param.name])
        super(BasePlanck2018GaussianLikelihood, self).initialize(data=self.covariance.center(), precision=self.covariance.invcov(return_type='nparray'))
        self.cosmo_quantities = self.covariance.names()

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

        from desilike.install import exists_package, exists_path, download, extract, InstallError

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


class FullGridPlanck2018GaussianLikelihood(BaseGaussianLikelihood):

    config_fn = 'planck2018_gaussian.yaml'
    installer_section = 'FullGridPlanck2018GaussianLikelihood'
    data_file_id = ('COM_CosmoParams_fullGrid_R3.01.zip', 11e9)
