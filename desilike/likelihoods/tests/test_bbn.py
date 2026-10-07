"""Tests for BBN likelihoods."""

import numpy as np
import jax

from desilike.base import build, get_params
from desilike.likelihoods.bbn import Schoneberg2024BBNLikelihood


def test_schoneberg2024():
    """Schoneberg2024BBNLikelihood matches a hand-computed chi2 from its hardcoded
    mean/covariance against the DESI fiducial's omega_b/N_eff, and is jit/grad-compatible."""
    like = Schoneberg2024BBNLikelihood()
    params = get_params(like)
    pipe = build(like)
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


def test_linx():
    """LINXBBNLikelihood: LINXAbundances reproduces a direct LINX call at the DESI fiducial,
    logpdf matches the hand-computed chi2 against the measurements, and jit/grad agree with eager
    and finite differences."""
    import pytest
    pytest.importorskip('linx')
    import jax.numpy as jnp
    import linx.const as const
    from linx.background import BackgroundModel
    from linx.abundances import AbundanceModel
    from linx.nuclear import NuclearRates
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology
    from desilike.theories.bbn import LINXAbundances
    from desilike.likelihoods.bbn import LINXBBNLikelihood, BBN_MEASUREMENTS

    import cosmoprimo
    # BBN needs only omega_b and N_eff: any engine will do, and eisenstein_hu is jax-native
    # (the fiducial is passed as an object so that building it does not require CLASS either)
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial=cosmoprimo.fiducial.DESI(engine='eisenstein_hu'))
    like = LINXBBNLikelihood(theory=LINXAbundances(cosmo=cosmo), measurements=('Aver2021', 'Cooke2018'))
    # neutron lifetime and nuclear-rate nuisances are varied by default
    varied = [param.name for param in get_params(like).select(varied=True)]
    assert 'tau_n' in varied and len([name for name in varied if name.startswith('q_')]) == 12

    pipe = build(like)
    defaults = {param.name: param._value for param in get_params(like)}
    logpdf = pipe(defaults)

    # the same LINX run by hand: extra radiation from N_eff - N_eff_sm, eta_fac = omega_b / Omegabh2
    background = BackgroundModel()(jnp.asarray((defaults['N_eff'] - 3.044) / LINXAbundances._neff_slope))
    time, scale_factor, rho_photon, rho_neutrino, rho_extra, pressure_extra = background[:6]
    yields = AbundanceModel(NuclearRates(nuclear_net='key_PRIMAT_2023'))(
        rho_photon, rho_neutrino, rho_extra, pressure_extra, t_vec=time, a_vec=scale_factor,
        eta_fac=jnp.asarray(defaults['omega_b'] / const.Omegabh2))
    # LINX's adaptive solvers (abundance rtol=1e-6) take a different step sequence in each XLA
    # compilation, so separately compiled runs of the same point agree to ~1e-5 only
    theory = like.theory
    assert np.isclose(float(theory.Y_p), 4. * float(yields[5]), rtol=1e-4)
    assert np.isclose(float(theory.D_H), float(yields[2] / yields[1]), rtol=1e-4)
    assert np.isclose(float(theory.N_eff_bbn), float(background[-1][-1]), rtol=1e-6)

    measurements = [BBN_MEASUREMENTS[name] for name in ('Aver2021', 'Cooke2018')]
    flattheory = np.array([float(theory.Y_p), float(theory.D_H)])
    expected_logpdf = -0.5 * sum(((mean - model) / std)**2 for (quantity, mean, std), model in zip(measurements, flattheory))
    assert np.isclose(float(logpdf), expected_logpdf, rtol=1e-8)

    # jit: same solver noise -- 1e-5 in D/H moves logpdf by ~5e-3 at a ~3 sigma D/H pull
    assert np.isclose(float(jax.jit(pipe)(defaults)), float(logpdf), atol=2e-2)

    # grad vs central finite differences; steps large enough to beat the solver noise
    # (eps = 1e-5 in omega_b is 4% off for that reason)
    grad = jax.jit(jax.grad(pipe))(defaults)
    for name, eps in [('omega_b', 2e-4), ('N_eff', 1e-1)]:
        finite_difference = (pipe({**defaults, name: defaults[name] + eps}) - pipe({**defaults, name: defaults[name] - eps})) / (2. * eps)
        assert np.isclose(float(grad[name]), float(finite_difference), rtol=1e-2)


if __name__ == '__main__':

    test_schoneberg2024()
    test_linx()
