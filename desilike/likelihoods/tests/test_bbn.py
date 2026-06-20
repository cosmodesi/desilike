"""Tests for BBN likelihoods."""

import numpy as np
import jax

from desilike.base import compile, get_params
from desilike.likelihoods.bbn import Schoneberg2024BBNLikelihood


def test_schoneberg2024():
    """Schoneberg2024BBNLikelihood matches a hand-computed chi2 from its hardcoded
    mean/covariance against the DESI fiducial's omega_b/N_eff, and is jit/grad-compatible."""
    like = Schoneberg2024BBNLikelihood()
    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}

    logpdf = pipe(defaults)
    assert np.isfinite(logpdf)

    import cosmoprimo
    fiducial = cosmoprimo.fiducial.DESI()
    theory = np.array([fiducial['omega_b'], fiducial['N_eff']])
    mean = np.array([0.02196, 2.904])
    covariance = np.array([[4.03112260e-07, 7.30390042e-05],
                            [7.30390042e-05, 4.52831584e-02]])
    r = mean - theory
    expected_logpdf = -0.5 * r @ np.linalg.inv(covariance) @ r
    assert np.isclose(float(logpdf), expected_logpdf, rtol=1e-6)

    # jit
    jit_logpdf = jax.jit(pipe)(defaults)
    assert np.isclose(float(logpdf), float(jit_logpdf))

    # grad: omega_b's prior is far tighter (variance ~4e-7 vs ~4.5e-2 for N_eff),
    # so its gradient should dominate.
    grad = jax.grad(pipe)(defaults)
    assert np.isfinite(grad['omega_b']) and grad['omega_b'] != 0.
    assert np.isfinite(grad['N_eff']) and grad['N_eff'] != 0.
    assert abs(grad['omega_b']) > abs(grad['N_eff'])


if __name__ == '__main__':

    test_schoneberg2024()
