import os

import numpy as np

from desilike.likelihoods.base import BaseGaussianLikelihood


class H0Likelihood(BaseGaussianLikelihood):

    def initialize(self, mean, std, cosmo=None):
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        super(H0Likelihood, self).initialize(flatdata=mean, covariance=std**2)

    @property
    def flattheory(self):
        return self.cosmo.H0


class MbLikelihood(BaseGaussianLikelihood):

    def initialize(self, mean, std):
        super(MbLikelihood, self).initialize(flatdata=mean, covariance=std**2)

    def calculate(self, Mb):
        self.flattheory = Mb
        super(MbLikelihood, self).calculate()
