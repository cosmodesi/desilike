import numpy as np

from desilike.likelihoods.bbn import Schoneberg2024BBNLikelihood
from desilike.likelihoods.hubble import Riess2020H0Likelihood


def test_bbn():
    # Test that the BBN likelihood behaves correctly, i.e., gives results
    # consistent with Schoneberg (2024).

    mean = [0.02196, 2.944]
    error = [0.00063, 0.21]

    likelihood = Schoneberg2024BBNLikelihood()

    # The likelihood should peak at the reported means.
    assert np.isclose(likelihood(omega_b=mean[0], N_eff=mean[1]), 0)

    # The marginal posteriors should agree.
    np.random.seed(42)
    samples = 8 * (np.random.random(size=(1000, 2)) - 0.5)
    samples[:, 0] = mean[0] + samples[:, 0] * error[0]
    samples[:, 1] = mean[1] + samples[:, 1] * error[1]
    w = np.array([np.exp(likelihood(omega_b=x[0], N_eff=x[1]))
                  for x in samples])
    n_eff = np.sum(w)**2 / np.sum(w**2)

    for i in range(2):
        assert np.isclose(np.sqrt(np.cov(samples[:, i], aweights=w)),
                          error[i], atol=0, rtol=0.1)
        assert np.isclose(np.average(samples[:, i], weights=w), mean[i],
                          atol=5 * mean[i] / np.sqrt(n_eff), rtol=0)


def test_hubble():
    # Test that the Hubble likelihoods behaves correctly, i.e., give results
    # consistent with the published papers.

    mean = 0.732
    error = 0.013

    likelihood = Riess2020H0Likelihood()
    assert np.isclose(likelihood(h=mean), 0)
    assert np.isclose(likelihood(h=mean + error), -1/2)
