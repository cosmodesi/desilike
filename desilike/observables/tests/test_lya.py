import numpy as np

from desilike import setup_logging
from desilike.likelihoods import ObservablesGaussianLikelihood


def test_compression():

    from desilike.observables.lya import P1DCompressionObservable

    observable = P1DCompressionObservable(data={}, covariance=np.diag([0.01, 0.01]), quantities=['delta2star', 'nstar'])
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    print(likelihood.varied_params)
    print(likelihood())
    print(likelihood.flattheory)


if __name__ == '__main__':

    setup_logging()
    test_compression()