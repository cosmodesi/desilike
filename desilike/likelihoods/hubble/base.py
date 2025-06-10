"""Hubble parameter likelihoods."""

import numpy as np

from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseGaussianLikelihood


class H0Likelihood(BaseGaussianLikelihood):
    r"""Hubble parameter :math:`H_0` likelihood."""

    def initialize(self, mean, std, cosmo=None):
        r"""Initialize the model.

        Parameters
        ----------
        mean : array_like
            Mean of Hubble parameter :math:`H_0`.
        std : array_like
            Uncertaintity of Hubble parameter :math:`H_0`.
        cosmo : BasePrimordialCosmology, default=None
            Cosmology calculator. If None, defaults to ``Cosmoprimo()``.

        """
        self.cosmo = cosmo
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {"params": {"H0": None}}
        elif self.cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            self.cosmo = Cosmoprimo()
        super().initialize(data=mean, covariance=std**2)

    @property
    def flattheory(self):
        """Theory predictions."""
        return np.array([self.cosmo["H0"]])


class MbLikelihood(BaseGaussianLikelihood):
    r"""Magnitude likelihood, to be combined with SN likelihood."""

    def initialize(self, mean, std):
        r"""Initialize the model.

        Parameters
        ----------
        mean : array_like
            Mean of :math:`Mb`.
        std : array_like
            Uncertaintity of :math:`Mb`.

        """
        super().initialize(data=mean, covariance=std**2)

    def calculate(self, Mb):
        """Calculate model."""
        self.flattheory = np.array([Mb])
        super().calculate()
