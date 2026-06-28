"""Pantheon+ (without SH0ES) type Ia supernovae likelihood."""

import os

import numpy as np
import jax.numpy as jnp

from .base import BaseSNLikelihood


class PantheonPlusSNLikelihood(BaseSNLikelihood):
    """
    Likelihood for the Pantheon+ (without SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077
    """
    installer_section = 'PantheonPlusSNLikelihood'
    data_file = 'Pantheon+SH0ES.dat'
    covariance_file = 'Pantheon+SH0ES_STAT+SYS.cov'
    _zname = 'zHD'

    def read_light_curve_params(self, fn):
        return super().read_light_curve_params(fn, header='', sep=' ')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only those SNe at z > 0.01 are used for cosmology.
        zmask = self.light_curve_params['zHD'] > 0.01
        self.light_curve_params = {name: value[zmask] for name, value in self.light_curve_params.items()}
        self.covariance = self.covariance[np.ix_(zmask, zmask)]
        self.flatdata = jnp.asarray(self.light_curve_params['m_b_corr']
                                     - 5 * np.log10((1 + self.light_curve_params['zHEL']) / (1 + self.light_curve_params['zHD'])))
        self.precision = jnp.linalg.inv(jnp.asarray(self.covariance))

    def __post_init__(self, *args, **kwargs):
        self.cosmo.add_requirements({'background.luminosity_distance': [{'z': self.light_curve_params['zHD']}]})

    def __call__(self):
        z = self.light_curve_params['zHD']
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
            github = 'https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/'
            for fn in [cls.data_file, cls.covariance_file]:
                download(os.path.join(github, fn), os.path.join(data_dir, fn))

        installer.write({cls.installer_section: {'data_dir': data_dir}})
