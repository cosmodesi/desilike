import numpy as np

from .differentiation import Differentiation
from .parameter import ParameterPrecision
from .base import BaseCalculator
from .utils import BaseClass
from . import utils


class PriorCalculator(BaseCalculator):

    def calculate(self, **params):
        params = {self.runtime_info.base_params[basename].name: value for basename, value in params.items()}
        self.logprior = self.runtime_info.params.prior(**params)

    def get(self):
        return self.logprior


class Fisher(BaseClass):

    def __init__(self, likelihood, method=None, ref_scale=0.5, accuracy=2, mpicomm=None):
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.mpicomm = mpicomm
        self.likelihood = likelihood

        solved_params = self.likelihood.all_params.select(solved=True)
        if solved_params:
            import warnings
            if self.mpicomm.rank == 0:
                warnings.warn('solved parameters: {}; cannot proceed with solved parameters, so we will work with likelihood.deepcopy(), varying solved parameters'.format(solved_params))
            self.likelihood = self.likelihood.deepcopy()
            for param in solved_params:
                self.likelihood.all_params[param].update(derived=False)
        self.varied_params = self.likelihood.varied_params

        prior_calculator = PriorCalculator()
        prior_calculator.params = [param for param in self.likelihood.all_params if param.depends or (not param.derived)]

        def prior_getter():
            return prior_calculator.logprior

        likelihoods = getattr(self.likelihood, 'likelihoods', [self.likelihood])
        from desilike.likelihoods import BaseGaussianLikelihood
        is_gaussian = all(isinstance(likelihood, BaseGaussianLikelihood) for likelihood in likelihoods)

        if is_gaussian:

            def getter():
                return [likelihood.flatdiff for likelihood in likelihoods]

            order = 1

            def finalize(derivs):
                loglikelihood = 0.
                for likelihood, flatdiff in zip(likelihoods, derivs):
                    flatdiff = np.array([flatdiff[param] for param in self.varied_params])
                    loglikelihood += flatdiff.dot(likelihood.precision).dot(flatdiff.T)
                return loglikelihood

        else:

            def getter():
                return - self.likelihood.loglikelihood

            order = 2

            def finalize(derivs):
                return np.array([[derivs[param1, param2] for param2 in self.varied_params] for param1 in self.varied_params])

        self.prior_differentiation = Differentiation(prior_calculator, getter=prior_getter, method=method, order=2, ref_scale=ref_scale, accuracy=accuracy, mpicomm=self.mpicomm)
        self.differentiation = Differentiation(self.likelihood, getter=getter, method=method, order=order, ref_scale=ref_scale, accuracy=accuracy, mpicomm=self.mpicomm)
        self._finalize = finalize

    def run(self, **params):
        self.prior_precision = ParameterPrecision(- self.prior_differentiation(**params), params=self.varied_params, center=[self.prior_differentiation.center[str(param)] for param in self.varied_params])
        self.precision = ParameterPrecision(self._finalize(self.differentiation(**self.prior_differentiation.center)), params=self.varied_params, center=self.prior_precision._center)

    def __call__(self, **params):
        self.run(**params)
        return self.prior_precision + self.precision
