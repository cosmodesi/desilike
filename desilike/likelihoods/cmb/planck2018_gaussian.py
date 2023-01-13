import os

from desilike.likelihoods import BaseGaussianLikelihood
from desilike.jax import numpy as jnp
from desilike import utils


class BasePlanck2018GaussianLikelihood(BaseGaussianLikelihood):

    config_fn = 'planck2018_gaussian.yaml'
    installer_section = 'BasePlanck2018GaussianLikelihood'
    data_file_id = 'COM_CosmoParams_base-plikHM_R3.01.zip'

    def initialize(self, cosmo=None, data_dir=None, basename='base_plikHM_TTTEEE_lowl_lowE_lensing'):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer()[self.installer_section]['data_dir']
        try:
            base_dir, obs_dir = basename.split('_plikHM_')
        except ValueError as exc:
            raise ValueError('basename {} is expected to contain "_plikHM_"'.format(basename)) from exc
        base_data_fn = os.path.join(data_dir, base_dir, 'plikHM_' + obs_dir, 'dist', basename)
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        convert_params = {'omegabh2': 'omega_b', 'omegach2': 'omega_cdm', 'omegak': 'Omega_k', 'w': 'w0_fld', 'wa': 'wa_fld', 'theta': 'theta_cosmomc', 'tau': 'tau_reio',
                          'mnu': 'm_ncdm_tot', 'logA': 'ln10^10A_s', 'ns': 'n_s', 'nrun': 'alpha_s', 'r': 'r'}
        mean = {}
        col = None
        with open(base_data_fn + '.margestats', 'r') as file:
            for line in file:
                line = [item.strip() for item in line.split()]
                if line:
                    if col is not None:
                        if line[0] in convert_params: mean[line[0]] = float(line[col])
                        else: break
                    if line[0] == 'parameter':
                        for col, item in enumerate(line):
                            if item.strip() == 'mean': break

        params = list(mean.keys())
        iline, col, cov = 0, None, [None for p in params]
        with open(base_data_fn + '.covmat', 'r') as file:
            for line in file:
                line = [item.strip() for item in line.split()]
                if line:
                    if col is not None and iline in col:
                        cov[col.index(iline)] = [float(line[i]) for i in col]
                        iline += 1
                    if line[0] == '#':
                        iline, col = 0, [line.index(param) - 1 for param in params]
        super(BasePlanck2018GaussianLikelihood, self).initialize(data=list(mean.values()), covariance=cov)
        self.cosmo_quantities = [convert_params[param] for param in params]

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
