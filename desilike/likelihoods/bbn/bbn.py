"""Big Bang Nucleosynthesis (BBN) likelihoods."""

import numpy as np

from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseGaussianLikelihood


class BaseBBNLikelihood(BaseGaussianLikelihood):
    r"""Cosmological likelihood from Big Bang Nucleosynthesis (BBN).

    The class provides access to cosmological results from BBN studies. The
    likelihoods are Gaussian likelihoods on cosmological parameters such as
    :math:`\omega_\mathrm{b}` and :math:`N_\mathrm{eff}`.
    """

    config_fn = "bbn.yaml"

    def initialize(self, mean, covariance, quantities,
                   cosmo=None):
        """Initialize the model.

        Parameters
        ----------
        mean : array_like
            Mean of cosmological parameters.
        covariance : array_like
            Covariance of cosmological parameters.
        quantities : array_like
            Name of cosmological parameters.
        cosmo : BasePrimordialCosmology, default=None
            Cosmology calculator. If None, defaults to ``Cosmoprimo()``.

        """
        self.cosmo = cosmo
        self.quantities = list(quantities)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {"params": {
                quantity: None for quantity in quantities}}
        elif self.cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            self.cosmo = Cosmoprimo()

        super().initialize(data=mean, covariance=covariance)

    @property
    def flattheory(self):
        """Theory predictions."""
        return np.array([self.cosmo[quantity] for quantity in self.quantities])


class Schoneberg2024BBNLikelihood(BaseBBNLikelihood):
    r"""BBN measurement from Schoneberg (2024).

    Reference
    ---------
    https://arxiv.org/abs/2401.15054

    """
