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
    from desilike.theories.bbn import LINXAbundances, delta_neff_init_from_final
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

    # the spline inversion round-trips between its nodes to LINX's solver noise (~1e-5)
    neff_nodes = like.theory._neff_nodes
    for neff in (2.5, 3.5):
        neff_solved = BackgroundModel()(delta_neff_init_from_final(jnp.asarray(neff), neff_nodes))[-1][-1]
        assert np.isclose(float(neff_solved), neff, atol=3e-5)

    # the same LINX run by hand: initial Delta N_eff from the spline, eta_fac = omega_b / Omegabh2
    background = BackgroundModel()(delta_neff_init_from_final(jnp.asarray(defaults['N_eff']), neff_nodes))
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
    # dNeff_convention='final': LINX delivers the sampled N_eff
    assert np.isclose(float(theory.N_eff_bbn), defaults['N_eff'], atol=3e-5)

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


def test_linx_init():
    """dNeff_convention='init': Delta_N_eff_init goes to LINX unchanged."""
    import pytest
    pytest.importorskip('linx')
    import jax.numpy as jnp
    from linx.background import BackgroundModel
    from desilike.parameter import Parameter
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology
    from desilike.theories.bbn import LINXAbundances
    from desilike.likelihoods.bbn import LINXBBNLikelihood

    import cosmoprimo
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial=cosmoprimo.fiducial.DESI(engine='eisenstein_hu'))
    theory = LINXAbundances(cosmo=cosmo, dNeff_convention='init',
                            params=[Parameter('Delta_N_eff_init', value=0.5, fixed=True)])
    like = LINXBBNLikelihood(theory=theory, measurements=('Aver2026', 'Cooke2018'))
    pipe = build(like)
    defaults = {param.name: param._value for param in get_params(like)}
    pipe(defaults)
    assert np.isclose(float(like.theory.N_eff_bbn), float(BackgroundModel()(jnp.asarray(0.5))[-1][-1]), rtol=1e-6)


#: LINXBBNLikelihood(measurements=('Aver2026', 'Cooke2018')) outputs, eager, PRIMAT 2023 network, at the DESI fiducial
#: (omega_b = 0.02237, N_eff = 3.046, tau_n = 879.4, q = 0) shifted as named. Generated with LINX 34f4cbf (2026-10-24).
_LINX_REGRESSION_SHIFTS = {
    'defaults': {},
    'omega_b': {'omega_b': 0.02287},
    'N_eff': {'N_eff': 3.3},
    'nuisances': {'tau_n': 880.6, 'q_npdg': 1., 'q_ddHe3n': -1.},
}
_LINX_REGRESSION_REFERENCES = {
    'defaults': dict(Y_p=2.47236588e-01, D_H=2.44522171e-05, YHe=2.45904749e-01, N_eff_bbn=3.04600968, logpdf=-4.325971),
    'omega_b': dict(Y_p=2.47449189e-01, D_H=2.35792660e-05, YHe=2.46116580e-01, N_eff_bbn=3.04600968, logpdf=-16.685689),
    'N_eff': dict(Y_p=2.50569852e-01, D_H=2.52885976e-05, YHe=2.49226002e-01, N_eff_bbn=3.30000339, logpdf=-6.733132),
    'nuisances': dict(Y_p=2.47471035e-01, D_H=2.45899612e-05, YHe=2.46138346e-01, N_eff_bbn=3.04600968, logpdf=-3.395323),
}


def test_linx_regression():
    """LINXBBNLikelihood outputs at fixed points have not moved since the references were generated.

    test_linx compares desilike with LINX itself, so it passes whatever LINX predicts; this test catches a change in
    the predictions (a LINX update, a different omega_b <-> eta or N_eff mapping, ...). Tolerances sit above the ~1e-5
    noise of LINX's adaptive solvers between compilations. If a change is intended, regenerate the references and say why
    in the commit message."""
    import pytest
    pytest.importorskip('linx')
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology
    from desilike.theories.bbn import LINXAbundances
    from desilike.likelihoods.bbn import LINXBBNLikelihood

    import cosmoprimo
    cosmo = CosmoprimoCosmology(engine='eisenstein_hu', fiducial=cosmoprimo.fiducial.DESI(engine='eisenstein_hu'))
    like = LINXBBNLikelihood(theory=LINXAbundances(cosmo=cosmo), measurements=('Aver2026', 'Cooke2018'))
    pipe = build(like)
    defaults = {param.name: param._value for param in get_params(like)}
    mismatches = []
    for label, shift in _LINX_REGRESSION_SHIFTS.items():
        logpdf = float(pipe({**defaults, **shift}))
        computed = {name: float(getattr(like.theory, name)) for name in ('Y_p', 'D_H', 'YHe', 'N_eff_bbn')}
        computed['logpdf'] = logpdf
        for name, reference in _LINX_REGRESSION_REFERENCES[label].items():
            # abundances: relative; N_eff: absolute (the spline inversion's accuracy); logpdf: absolute, mostly a D/H pull
            # of up to ~6 sigma times the 1e-4 relative tolerance
            close = {'N_eff_bbn': np.isclose(computed[name], reference, rtol=0., atol=3e-5),
                     'logpdf': np.isclose(computed[name], reference, rtol=0., atol=0.1)}.get(name, np.isclose(computed[name], reference, rtol=1e-4, atol=0.))
            if not close:
                mismatches.append(f'{label} {name}: {computed[name]:.8e} vs reference {reference:.8e} (relative {computed[name] / reference - 1.:+.1e})')
    assert not mismatches, 'LINX BBN predictions moved:\n' + '\n'.join(mismatches)


if __name__ == '__main__':

    test_schoneberg2024()
    test_linx()
    test_linx_init()
    test_linx_regression()
