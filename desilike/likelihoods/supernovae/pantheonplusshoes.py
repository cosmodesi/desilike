import numpy as np

from desilike import utils
from desilike.cosmo import is_external_cosmo
from desilike.jax import numpy as jnp
from .base import BaseSNLikelihood
from .pantheonplus import PantheonPlusSNLikelihood


class PantheonPlusSHOESSNLikelihood(PantheonPlusSNLikelihood):
    """
    Likelihood for Pantheon+ (with SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077

    Parameters
    ----------
    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    config_fn = 'pantheonplusshoes.yaml'
    installer_section = 'PantheonPlusSNLikelihood'
    name = 'PantheonPlusSHOESSN'

    def initialize(self, *args, cosmo=None, **kwargs):
        BaseSNLikelihood.initialize(self, *args, cosmo=cosmo, **kwargs)
        # Select only those SNe at z > 0.01 or the ones used as calibrators
        zmask = (self.light_curve_params['zcmb'] > 0.01) | self.light_curve_params['is_calibrator']
        self.light_curve_params = {name: value[zmask] for name, value in self.light_curve_params.items()}
        self.covariance = self.covariance[np.ix_(zmask, zmask)]
        self.precision = utils.inv(self.covariance)
        self.std = np.diag(self.covariance)**0.5
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'background': {'luminosity_distance': {'z': self.light_curve_params['zcmb']}}}

    def calculate(self, Mb=0):
        self.flattheory = self.light_curve_params['zcmb'] * np.nan

        # Use Cepheids host distances as theory
        is_calibrator = self.light_curve_params['is_calibrator']
        self.flattheory[is_calibrator] = self.light_curve_params['cepheid_distance'][is_calibrator]

        # Compute predictions at those redshifts that are not used as calibrators
        zcmb = self.light_curve_params['zcmb'][~is_calibrator]
        zhel = self.light_curve_params['zhel'][~is_calibrator]
        self.flattheory[~is_calibrator] = 5 * jnp.log10(self.cosmo.luminosity_distance(zcmb) / self.cosmo['h']) + 25 + 5 * np.log10((1 + zhel) / (1 + zcmb))

        self.flatdata = self.light_curve_params['mb'] - Mb
        BaseSNLikelihood.calculate(self)

    def read_light_curve_params(self, fn):
        data = BaseSNLikelihood.read_light_curve_params(self, fn, header='', sep=' ')
        return {'zcmb': data['zHD'], 'zhel': data['zHEL'], 'mb': data['m_b_corr'], 'is_calibrator': data['IS_CALIBRATOR'].astype('?'), 'cepheid_distance': data['CEPH_DIST']}