import numpy as np

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
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        super(BBNOmegaBLikelihood, self).initialize(data=mean, covariance=std**2)

    @property
    def flattheory(self):
        return np.asarray(self.cosmo.Omega0_b * self.cosmo.h**2)