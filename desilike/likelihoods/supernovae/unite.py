"""Unite type Ia supernovae likelihood (Pantheon+ and DES-SN5YR combined)."""

import numpy as np
import jax.numpy as jnp

from .des import _BaseDESY5SNLikelihood, DESY5DovekieSNLikelihood


class UniteSNLikelihood(_BaseDESY5SNLikelihood):
    """
    Likelihood for the Unite type Ia supernovae sample (Pantheon+ and DES-SN5YR combined),
    nominal (unanchored) Hubble diagram.

    The light-curve file uses the same 'VARNAMES:'/'SN:'-prefixed layout as
    :class:`DESY5DovekieSNLikelihood`, and the distance-modulus theory is the same:
    ``MU`` is a distance modulus, so ``Mb`` is an offset relative to the fiducial absolute
    magnitude used to build it (of order 0.1, not :math:`M_B \\sim -19.3`).
    Set ``Mb`` to ``derived='marg'`` to marginalize over it analytically, as in the reference
    CosmoSIS module.

    The covariance file stores the upper triangle of the *precision* (inverse stat+sys
    covariance) matrix in a .npz archive, with the matrix size under ``arr_0`` and the packed
    values under ``arr_1``. It is inverted back to the covariance on reading, so that ``zrange``
    cuts mask the covariance, not its inverse. No redshift cut is applied by default:
    the reference analysis fits all supernovae.

    The data files are not downloaded by desilike, hence there is nothing to install:
    pass ``data_dir``, the directory containing ``Unite_COSMO_HD.txt`` and
    ``Unite_COSMO_covtot_inv_000.npz``.

    Reference
    ---------
    https://arxiv.org/abs/2609.05053
    """
    installer_section = 'UniteSNLikelihood'
    data_file = 'Unite_COSMO_HD.txt'
    covariance_file = 'Unite_COSMO_covtot_inv_000.npz'

    read_light_curve_params = DESY5DovekieSNLikelihood.read_light_curve_params

    def read_covariance(self, fn):
        """Unpack the .npz precision matrix (upper-triangle-packed, symmetrized) and return its inverse, the covariance."""
        data = np.load(fn)
        nsn = int(data['arr_0'])
        precision = np.zeros((nsn, nsn))
        precision[np.triu_indices(nsn)] = data['arr_1']
        idx_lower = np.tril_indices(nsn, -1)
        precision[idx_lower] = precision.T[idx_lower]
        return np.linalg.inv(precision)

    def __init__(self, data_dir=None, **kwargs):
        if data_dir is None:
            from desilike.install import Installer
            if self.installer_section not in Installer():
                raise ValueError(f'{type(self).__name__} requires data_dir, '
                                 f'the directory containing {self.data_file} and {self.covariance_file}')
        super().__init__(data_dir=data_dir, **kwargs)
        self.precision = jnp.linalg.inv(jnp.asarray(self.covariance))

    @classmethod
    def install(cls, installer):
        """Nothing to download: the data files are provided through ``data_dir``."""
