"""Pantheon type Ia supernovae likelihood."""

import os

import numpy as np
import jax.numpy as jnp

from .base import BaseSNLikelihood


class PantheonSNLikelihood(BaseSNLikelihood):
    """
    Likelihood for the Pantheon type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/1710.00845
    """
    installer_section = 'PantheonSNLikelihood'
    data_file = 'lcparam_full_long.txt'
    covariance_file = 'sys_full_long.txt'
    _zname = 'zcmb'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add statistical error (diagonal) on top of the systematics covariance.
        self.covariance = self.covariance + np.diag(self.light_curve_params['dmb'] ** 2)
        flatdata = self.light_curve_params['mb'] - 5 * np.log10((1 + self.light_curve_params['zhel']) / (1 + self.light_curve_params['zcmb']))
        self.flatdata = jnp.asarray(flatdata)
        self.precision = jnp.linalg.inv(jnp.asarray(self.covariance))
        self.cosmo.add_requirements({'background.luminosity_distance': [{'z': self.light_curve_params['zcmb']}]})

    def __call__(self):
        z = self.light_curve_params['zcmb']
        dL = self.cosmo.get_background().luminosity_distance(z=z)
        self.flattheory = 5 * jnp.log10(dL / self.cosmo['h']) + 25 + self.Mb.value
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
            github = 'https://raw.githubusercontent.com/dscolnic/Pantheon/master/'
            for fn in [cls.data_file, 'lcparam_full_long_zhel.txt', cls.covariance_file]:
                download(os.path.join(github, fn), os.path.join(data_dir, fn))

        installer.write({cls.installer_section: {'data_dir': data_dir}})
