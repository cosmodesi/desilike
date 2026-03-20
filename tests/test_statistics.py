import arviz
import emcee
import numpy as np
import pytest

from desilike import statistics


@pytest.mark.mpi_skip
def test_samples_basic():
    # Test the basic functionality of Samples.

    samples = statistics.Samples(
        a=np.linspace(0, 1, 10), b=range(10), c=np.arange(30).reshape(10, 3))
    samples['d'] = np.arange(20).reshape(10, 2)

    with pytest.raises(ValueError):
        samples['e'] = np.arange(5)  # wrong length

    assert set(samples.keys) == {'a', 'b', 'c', 'd'}
    assert len(samples[:5]) == 5
    samples.append(samples[:5])
    assert len(samples) == 15

    for key in samples.keys:
        assert isinstance(samples[key], np.ndarray)
        assert len(samples[key]) == 15

    assert isinstance(samples[0], dict)

    with pytest.raises(ValueError):
        samples.append(statistics.Samples(a=[1, 2]))  # not all keys present


@pytest.mark.mpi_skip
def test_diagnostics():
    # Test that the diagnostics agree with external libraries.

    np.random.seed(42)

    n_chains = 4
    n_dim = 5
    n_steps = 10000
    rho = np.random.uniform(0.9, 1.0, size=n_dim)

    chains = np.zeros((n_chains, n_steps, n_dim))
    for i in range(1, n_steps):
        chains[:, i] = (chains[:, i - 1] * rho[np.newaxis, np.newaxis, :] +
                        np.random.normal(size=(n_chains, n_dim)))

    # Most autocorrelation times agree to within machine precision. However,
    # small number is off by less than 1%.
    assert np.allclose(
        emcee.autocorr.integrated_time(np.transpose(chains, [1, 0, 2]), tol=0),
        statistics.diagnostics.integrated_autocorrelation_time(chains),
        rtol=1e-2)

    gr_arviz = arviz.rhat({str(i): chains[:, :, i] for i in range(n_dim)},
                          method='identity')
    gr_arviz = [gr_arviz[key].values for key in gr_arviz]
    assert np.allclose(
        gr_arviz, statistics.diagnostics.gelman_rubin(chains), rtol=1e-6)
