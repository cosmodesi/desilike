import numpy as np
from scipy.linalg import eigvalsh

from desilike.samples import MCSamples
from desilike.samples.diagnostics import gelman_rubin


def test_eigen_gelman_rubin_generalized_eigenproblem():
    rng = np.random.default_rng(1234)
    covariance = np.array([[1.0, 0.95], [0.95, 1.0]])
    means = 0.1 * np.array([
        [-1.0, 1.0],
        [-0.3, 0.5],
        [0.5, -0.2],
        [1.2, -1.0],
    ])

    chains = []
    for mean in means:
        values = rng.multivariate_normal(mean, covariance, size=200)
        chains.append(MCSamples({'x': values[:, 0], 'y': values[:, 1]}))

    gr_xy, (V, W) = gelman_rubin(
        chains, params=['x', 'y'], method='eigen', return_matrices=True
    )
    gr_yx = gelman_rubin(chains, params=['y', 'x'], method='eigen')

    np.testing.assert_allclose(gr_xy, eigvalsh(V, W), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gr_xy, gr_yx, rtol=1e-12, atol=1e-12)
