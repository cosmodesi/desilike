import numpy as np

from desilike.likelihoods.base import BaseGaussianLikelihood


class H0Likelihood(BaseGaussianLikelihood):
    r"""
    H0 likelihood.

    Parameters
    ----------
    mean : float
        :math:`H_{0}` value.

    std : float
        :math:`H_{0}` uncertainty.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.
    """
    def initialize(self, mean, std, cosmo=None):
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        super(H0Likelihood, self).initialize(flatdata=mean, covariance=std**2)

    @property
    def flattheory(self):
        return np.asarray(self.cosmo.H0)


class MbLikelihood(BaseGaussianLikelihood):
    r"""
    Magnitude likelihood, to be combined with SN likelihood.

    Parameters
    ----------
    mean : float
        :math:`Mb` value.

    std : float
        :math:`Mb` uncertainty.
    """
    def initialize(self, mean, std):
        super(MbLikelihood, self).initialize(flatdata=mean, covariance=std**2)

    def calculate(self, Mb):
        self.flattheory = np.asarray(Mb)
        super(MbLikelihood, self).calculate()
