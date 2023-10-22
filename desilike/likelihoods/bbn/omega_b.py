import numpy as np

from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseGaussianLikelihood


class BBNOmegaBLikelihood(BaseGaussianLikelihood):
    r"""
    BBN-inspired :math:`omega_{b}` likelihood.

    Parameters
    ----------
    mean : float
        :math:`omega_{b}` value.

    std : float
        :math:`omega_{b}` uncertainty.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.
    """
    config_fn = 'omega_b.yaml'
    name = 'BBNOmegaB'

    def initialize(self, mean, std, cosmo=None):
        self.cosmo = cosmo
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'params': {'Omega_b': None, 'h': None}}
        elif self.cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            self.cosmo = Cosmoprimo()
        super(BBNOmegaBLikelihood, self).initialize(data=mean, covariance=std**2)

    @property
    def flattheory(self):
        return np.array([self.cosmo.Omega0_b * self.cosmo.h**2])