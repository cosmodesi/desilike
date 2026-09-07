"""Pantheon+ (with SH0ES Cepheid calibration) type Ia supernovae likelihood."""

import os

import numpy as np
import jax.numpy as jnp

from .base import BaseSNLikelihood


class PantheonPlusSHOESSNLikelihood(BaseSNLikelihood):
    """
    Likelihood for the Pantheon+ (with SH0ES) type Ia supernovae sample.

    SNe hosted by a Cepheid calibrator use the Cepheid host distance as theory
    (constraining ``h`` through the calibrator subsample) instead of the
    cosmological distance modulus.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077
    """
    installer_section = 'PantheonPlusSNLikelihood'
    data_file = 'Pantheon+SH0ES.dat'
    covariance_file = 'Pantheon+SH0ES_STAT+SYS.cov'
    _zname = 'zcmb'

    def read_light_curve_params(self, fn):
        data = super().read_light_curve_params(fn, header='', sep=' ')
        return {'zcmb': data['zHD'], 'zhel': data['zHEL'], 'mb': data['m_b_corr'],
                'is_calibrator': data['IS_CALIBRATOR'].astype('?'), 'cepheid_distance': data['CEPH_DIST']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Select SNe at z > 0.01, plus those used as Cepheid calibrators.
        zmask = (self.light_curve_params['zcmb'] > 0.01) | self.light_curve_params['is_calibrator']
        self.light_curve_params = {name: value[zmask] for name, value in self.light_curve_params.items()}
        self.covariance = self.covariance[np.ix_(zmask, zmask)]
        self.flatdata = jnp.asarray(self.light_curve_params['mb'])
        self.precision = jnp.linalg.inv(jnp.asarray(self.covariance))

    def __post_init__(self, *args, **kwargs):
        self.cosmo.add_requirements({'background.luminosity_distance': [{'z': self.light_curve_params['zcmb']}]})

    def __call__(self):
        is_calibrator = self.light_curve_params['is_calibrator']
        zcmb = self.light_curve_params['zcmb']
        zhel = self.light_curve_params['zhel']
        dL = self.cosmo.get_background().luminosity_distance(z=zcmb)
        distance_modulus = 5 * jnp.log10(dL / self.cosmo['h']) + 25 + 5 * jnp.log10((1 + zhel) / (1 + zcmb))
        # Cepheid host distances replace the cosmological prediction for calibrator SNe.
        self.flattheory = jnp.where(is_calibrator, self.light_curve_params['cepheid_distance'], distance_modulus) + self.Mb.value
        return super().__call__()

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download

        data_fn = os.path.join(data_dir, cls.data_file)

        if installer.reinstall or not exists_path(data_fn):
            github = 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/'
            for fn in [cls.data_file, cls.covariance_file]:
                download(os.path.join(github, fn), os.path.join(data_dir, fn))

        installer.write({cls.installer_section: {'data_dir': data_dir}})
