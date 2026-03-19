import emcee
import numpy as np
import pytest

from desilike import statistics


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


def test_autocorrelation_time():
    # Test that autocorrelation time compuations agree with those of
    # ``emcee``.

    np.random.seed(42)

    n_chains = 4
    n_dim = 25
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
        statistics.diagnostics.autocorrelation_time(chains), rtol=1e-2)
