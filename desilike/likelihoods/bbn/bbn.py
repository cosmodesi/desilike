import numpy as np

from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseGaussianLikelihood


class BaseBBNLikelihood(BaseGaussianLikelihood):
    r"""
    BBN :math:`omega_{b}` likelihood.

    Parameters
    ----------
    mean : float
        :math:`omega_{b}` value.

    covariance : float
        :math:`omega_{b}` covariance.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.
    """
    config_fn = 'bbn.yaml'

    def initialize(self, mean, covariance=None, quantities=('omega_b',), cosmo=None):
        self.cosmo = cosmo
        self.quantities = list(quantities)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'params': {quantity: None for quantity in quantities}}
        elif self.cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            self.cosmo = Cosmoprimo()

        super(BaseBBNLikelihood, self).initialize(data=mean, covariance=covariance)

    @property
    def flattheory(self):
        return np.array([self.cosmo[quantity] for quantity in self.quantities])


class Schoneberg2024BBNLikelihood(BaseBBNLikelihood):

    r"""BBN :math:`\omega_{b}` measurement from https://arxiv.org/abs/2401.15054."""