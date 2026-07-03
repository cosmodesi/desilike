"""Tests for :mod:`desilike.likelihoods.cobaya` (the generic cobaya -> desilike adapter).

Uses small synthetic cobaya ``Likelihood`` classes (defined below, no external data) that
together exercise every requirement kind ``CobayaLikelihood`` supports -- ``Cl``,
``unlensed_Cl``, ``Hubble``, ``angular_diameter_distance``, ``comoving_radial_distance``,
``sigma8_z``, ``fsigma8``, ``sigma_R``, ``Omega_b``/``Omega_cdm``/``Omega_nu_massive``,
linear ``Pk_grid`` (including a cross-spectrum), a derived parameter (``rdrag``), and an own
nuisance parameter -- and cross-check the values seen by the wrapped likelihood's ``logp()``
against independent ``cosmoprimo`` calls at the same (fiducial) parameter point.

Two further tests cross-check the wrapper end-to-end against a real hand-ported cobaya
likelihood, on a small synthetic CamSpec-NPIPE-lite SACC fixture (skipped if the non-pip,
sibling-checkout ``camspec_npipe_lite`` package isn't available):

* ``test_matches_native_jax_camspec_port``: the unmodified cobaya
  ``camspec_npipe_lite.planck_Camspec_NPIPE_lite`` class, wrapped via
  :func:`~desilike.likelihoods.cobaya.wrap_cobaya_likelihood`, must give the same ``logpdf``
  (with a real ``camb`` Boltzmann engine) as desilike's hand JAX-ported
  :class:`~desilike.likelihoods.cmb.camspec.CamspecNPIPELiteLikelihood`.
* ``test_matches_full_native_cobaya_pipeline``: the desilike-side pipeline must also agree
  with the *fully native* cobaya pipeline (cobaya's own ``camb`` theory, run through
  ``cobaya.model.get_model``, not just cosmoprimo's), once precision settings are matched.
"""

import inspect
import shutil
import sys
from pathlib import Path

import numpy as np
import jax
import pytest
import yaml

from desilike.base import compile, get_params
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from desilike.likelihoods.cobaya import wrap_cobaya_likelihood

cobaya = pytest.importorskip('cobaya')
from cobaya.likelihood import Likelihood as CobayaBaseLikelihood  # noqa: E402


_ELLMAX = 30
_ZS = np.array([0.3, 0.7])


class _SyntheticLikelihood(CobayaBaseLikelihood):
    """Throwaway cobaya likelihood exercising the full supported requirement vocabulary."""

    params = {'amp': {'prior': {'min': 0., 'max': 2.},
                       'ref': {'dist': 'norm', 'loc': 1., 'scale': 0.1},
                       'latex': 'A'}}

    def get_requirements(self):
        return {
            'Cl': {'tt': _ELLMAX, 'te': _ELLMAX, 'ee': _ELLMAX},
            'Hubble': {'z': _ZS},
            'angular_diameter_distance': {'z': _ZS},
            'comoving_radial_distance': {'z': _ZS},
            'sigma8_z': {'z': _ZS},
            'fsigma8': {'z': _ZS},
            'Pk_grid': {'z': _ZS, 'k_max': 1., 'nonlinear': False, 'vars_pairs': [['delta_tot', 'delta_tot']]},
            'rdrag': None,
        }

    def logp(self, **params_values):
        self.last = dict(
            cl=self.provider.get_Cl(ell_factor=False, units='FIRASmuK2'),
            Hz=self.provider.get_Hubble(_ZS, units='km/s/Mpc'),
            DA=self.provider.get_angular_diameter_distance(_ZS),
            DC=self.provider.get_comoving_radial_distance(_ZS),
            sigma8_z=self.provider.get_sigma8_z(_ZS),
            fsigma8=self.provider.get_fsigma8(_ZS),
            rdrag=self.provider.get_param('rdrag'),
        )
        k, z, pk = self.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_tot'), nonlinear=False)
        self.last.update(pk_k=k, pk_z=z, pk=pk)
        amp = params_values['amp']
        return -0.5 * (amp - 1.) ** 2


def _make_cosmo():
    return CosmoprimoCosmology(engine='class', fiducial=('DESI', dict(lensing=True, ellmax_cl=_ELLMAX, non_linear='mead')))


def test_nuisance_parameter_translated():
    """The cobaya likelihood's own `params:` block becomes a desilike free Parameter."""
    like = wrap_cobaya_likelihood(_SyntheticLikelihood, cosmo=_make_cosmo())
    assert 'amp' in like.params
    amp = like.params['amp']
    assert not amp.fixed
    assert amp.prior.limits == (0., 2.)
    assert amp.latex() is not None


def test_evaluate_and_cross_check_against_cosmoprimo():
    """Compile and evaluate; cross-check every provider value against a direct cosmoprimo call
    at the same (fiducial) parameter point."""
    cosmo = _make_cosmo()
    like = wrap_cobaya_likelihood(_SyntheticLikelihood, cosmo=cosmo)

    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}
    logpdf = pipe(defaults)
    assert np.isfinite(float(logpdf))
    assert np.isclose(float(logpdf), -0.5 * (1. - 1.) ** 2)  # amp defaults to 1.

    # _is_external=True: runs through pure_callback + finite-difference JVP; jit must
    # reproduce the plain forward evaluation, and logpdf must respond to its nuisance param.
    jit_logpdf = jax.jit(pipe)(defaults)
    assert np.isclose(float(logpdf), float(jit_logpdf))
    perturbed = dict(defaults, amp=1.3)
    assert np.isclose(float(pipe(perturbed)), -0.5 * (1.3 - 1.) ** 2)

    last = like._cobaya_like.last

    # Independent reference: build the same fiducial cosmology directly via cosmoprimo.
    from cosmoprimo.fiducial import DESI
    ref = DESI(engine='class', lensing=True, ellmax_cl=_ELLMAX, non_linear='mead')
    h = ref['h']
    bg, fo = ref.get_background(), ref.get_fourier()

    assert np.allclose(last['Hz'], 100. * h * bg.efunc(_ZS))
    assert np.allclose(last['DA'], bg.angular_diameter_distance(_ZS) / h)
    assert np.allclose(last['DC'], bg.comoving_radial_distance(_ZS) / h)
    assert np.allclose(last['sigma8_z'], fo.sigma8_z(_ZS, of='delta_m'))
    assert np.allclose(last['fsigma8'], bg.growth_rate(_ZS) * fo.sigma8_z(_ZS, of='delta_m'))
    assert np.isclose(last['rdrag'], ref.get_thermodynamics().rs_drag / h)

    cl_ref = ref.get_harmonic().lensed_cl(ellmax=_ELLMAX)
    units_factor = 2.7255e6
    assert np.allclose(last['cl']['tt'], cl_ref['tt'] * units_factor ** 2)
    assert np.allclose(last['cl']['te'], cl_ref['te'] * units_factor ** 2)

    k_phys = last['pk_k']
    expected_pk = fo.pk_interpolator(of='delta_m')(z=last['pk_z'], k=k_phys / h).T / h ** 3
    assert np.allclose(last['pk'], expected_pk, rtol=1e-4)


class _ExtendedLikelihood(CobayaBaseLikelihood):
    """Exercises unlensed_Cl, sigma_R, z-dependent Omega_*, and cross-spectra Pk_grid."""

    def get_requirements(self):
        return {
            'unlensed_Cl': {'tt': _ELLMAX, 'ee': _ELLMAX},
            'sigma_R': {'z': _ZS, 'R': [8., 20.], 'k_max': 1.},
            'Omega_b': {'z': _ZS},
            'Omega_cdm': {'z': _ZS},
            'Omega_nu_massive': {'z': _ZS},
            'Pk_grid': {'z': _ZS, 'k_max': 1., 'nonlinear': False, 'vars_pairs': [['delta_tot', 'delta_nonu']]},
        }

    def logp(self, **params_values):
        self.last = dict(
            unlensed_cl=self.provider.get_unlensed_Cl(ell_factor=False, units='FIRASmuK2'),
            Omega_b=self.provider.get_Omega_b(_ZS),
            Omega_cdm=self.provider.get_Omega_cdm(_ZS),
            Omega_nu=self.provider.get_Omega_nu_massive(_ZS),
        )
        z, R, sigma_r = self.provider.get_sigma_R(var_pair=('delta_tot', 'delta_tot'))
        self.last.update(sigma_r_z=z, sigma_r_R=R, sigma_r=sigma_r)
        k, z2, pk_cross = self.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_nonu'), nonlinear=False)
        self.last.update(pk_cross_k=k, pk_cross_z=z2, pk_cross=pk_cross)
        return 0.


def test_extended_requirements_cross_check():
    """unlensed_Cl, sigma_R, Omega_b/cdm/nu_massive, and cross-spectra Pk_grid all cross-check
    against direct cosmoprimo calls at the fiducial point."""
    cosmo = _make_cosmo()
    like = wrap_cobaya_likelihood(_ExtendedLikelihood, cosmo=cosmo)
    params = get_params(like)
    pipe = compile(like)
    defaults = {p.name: p._value for p in params}
    assert np.isfinite(float(pipe(defaults)))

    last = like._cobaya_like.last

    from cosmoprimo.fiducial import DESI
    ref = DESI(engine='class', lensing=True, ellmax_cl=_ELLMAX, non_linear='mead')
    h = ref['h']
    bg, fo = ref.get_background(), ref.get_fourier()

    cl_ref = ref.get_harmonic().unlensed_cl(ellmax=_ELLMAX)
    units_factor = 2.7255e6
    assert np.allclose(last['unlensed_cl']['tt'], cl_ref['tt'] * units_factor ** 2)

    assert np.allclose(last['Omega_b'], bg.Omega_b(_ZS))
    assert np.allclose(last['Omega_cdm'], bg.Omega_cdm(_ZS))
    assert np.allclose(last['Omega_nu'], bg.Omega_ncdm_tot(_ZS))

    R_phys = last['sigma_r_R']
    assert R_phys.min() <= 8. and R_phys.max() >= 20.  # generous grid covers the requested R's
    expected_sigma_r = fo.sigma_rz(R_phys * h, last['sigma_r_z'], of='delta_m').T
    assert np.allclose(last['sigma_r'], expected_sigma_r, rtol=1e-3)

    k_phys = last['pk_cross_k']
    expected_pk_cross = fo.pk_interpolator(of=('delta_m', 'delta_cb'))(z=last['pk_cross_z'], k=k_phys / h).T / h ** 3
    assert np.allclose(last['pk_cross'], expected_pk_cross, rtol=1e-4)


def _import_camspec_npipe_lite():
    """Import the third-party ``camspec_npipe_lite`` cobaya likelihood package.

    Not a desilike/cobaya dependency -- a sibling checkout in the cosmodesi meta-repo
    (``cosmodesi/camspec_npipe-lite``), not pip-installed by default. Tries a plain import
    first, else adds the conventional sibling path, else skips the test.
    """
    try:
        import camspec_npipe_lite
    except ImportError:
        sibling = Path(__file__).resolve().parents[4] / 'camspec_npipe-lite'
        if not sibling.is_dir():
            pytest.skip(f'camspec_npipe_lite package not found (tried a plain import and {sibling})')
        sys.path.insert(0, str(sibling))
        try:
            import camspec_npipe_lite
        except ImportError:
            pytest.skip('camspec_npipe_lite package not importable even after adding the sibling checkout to sys.path')
    return camspec_npipe_lite.planck_Camspec_NPIPE_lite


def _write_camspec_lite_sacc_fixture(path):
    """Write a small synthetic CamSpec-NPIPE-lite-format SACC file: TT/TE/EE at a handful of
    ell's within the default ell_cuts, with a diagonal (hence positive-definite) covariance.

    Matches the SACC layout both :meth:`CamspecNPIPELiteLikelihood._load_data` and cobaya's
    ``planck_Camspec_NPIPE_lite.initialize()`` read (same ``pol -> cl_XX`` tracer-combination
    convention), so the same file can be fed to both implementations.
    """
    sacc = pytest.importorskip('sacc')
    rng = np.random.default_rng(42)
    ells = np.arange(30, 61, 5)
    pol_to_sacc_dt = {'TT': 'cl_00', 'TE': 'cl_0e', 'EE': 'cl_ee'}
    amps = {'TT': 2000., 'TE': 50., 'EE': 20.}

    sacc_data = sacc.Sacc()
    sacc_data.add_tracer('misc', 'cmb')
    for pol, dt in pol_to_sacc_dt.items():
        dl = amps[pol] * (1. + 0.05 * rng.standard_normal(len(ells)))
        sacc_data.add_ell_cl(dt, 'cmb', 'cmb', ells, dl)
    covariance = np.eye(len(sacc_data.mean)) * (0.02 * np.abs(sacc_data.mean)) ** 2
    sacc_data.add_covariance(covariance)
    sacc_data.save_fits(str(path), overwrite=True)


_CAMSPEC_ELLMAX_CL = 100


def _setup_camspec_lite(tmp_path):
    """Build a synthetic CamSpec-NPIPE-lite SACC fixture plus the cobaya ``packages_path``
    layout ``planck_Camspec_NPIPE_lite`` needs to find it.

    Returns ``(planck_Camspec_NPIPE_lite, fits_path, packages_path, info_yaml)``, where
    ``info_yaml`` is ``camspec_npipe_lite.yaml``'s content, loaded explicitly as a workaround
    (see the note in :func:`test_matches_native_jax_camspec_port`).
    """
    planck_Camspec_NPIPE_lite = _import_camspec_npipe_lite()
    fits_path = tmp_path / 'CamSpec_NPIPE_cmb_sacc.fits'
    _write_camspec_lite_sacc_fixture(fits_path)

    yaml_path = Path(inspect.getfile(planck_Camspec_NPIPE_lite)).with_suffix('.yaml')
    with open(yaml_path) as file:
        info_yaml = yaml.safe_load(file)

    packages_path = tmp_path / 'packages'
    data_dir = packages_path / 'data' / 'Camspec_NPIPE_lite'
    data_dir.mkdir(parents=True)
    shutil.copy(fits_path, data_dir / 'CamSpec_NPIPE_cmb_sacc.fits')
    return planck_Camspec_NPIPE_lite, fits_path, packages_path, info_yaml


def test_matches_native_jax_camspec_port(tmp_path):
    """CobayaLikelihood-wrapped cobaya ``planck_Camspec_NPIPE_lite`` must give the same
    ``logpdf`` as desilike's hand-ported JAX ``CamspecNPIPELiteLikelihood``, at fiducial and
    at a perturbed nuisance parameter, with a real Boltzmann engine (``camb``).

    Note
    ----
    ``camspec_npipe_lite.py``'s class shadows cobaya's reserved ``file_base_name`` attribute
    with its *data* file's base name, which breaks ``HasDefaults.get_yaml_file()``'s
    auto-discovery of ``camspec_npipe_lite.yaml`` -- its params/priors never load in
    standalone construction (``self.params`` comes back empty). ``_setup_camspec_lite`` loads
    the yaml explicitly as a workaround and passes it via ``info=``.
    """
    from desilike.likelihoods.cmb.camspec import CamspecNPIPELiteLikelihood

    planck_Camspec_NPIPE_lite, fits_path, packages_path, info_yaml = _setup_camspec_lite(tmp_path)

    fiducial = ('DESI', dict(lensing=True, ellmax_cl=_CAMSPEC_ELLMAX_CL, non_linear='mead'))
    like_native = CamspecNPIPELiteLikelihood(data_file=str(fits_path), cosmo=CosmoprimoCosmology(engine='camb', fiducial=fiducial))
    like_wrapped = wrap_cobaya_likelihood(planck_Camspec_NPIPE_lite, info=info_yaml, packages_path=str(packages_path),
                                           cosmo=CosmoprimoCosmology(engine='camb', fiducial=fiducial))

    pipe_native = compile(like_native)
    pipe_wrapped = compile(like_wrapped)
    defaults_native = {p.name: p._value for p in get_params(like_native)}
    defaults_wrapped = {p.name: p._value for p in get_params(like_wrapped)}

    logpdf_native = float(pipe_native(defaults_native))
    logpdf_wrapped = float(pipe_wrapped(defaults_wrapped))
    assert np.isfinite(logpdf_native)
    assert np.isclose(logpdf_native, logpdf_wrapped, rtol=1e-6)

    perturbed_native = dict(defaults_native, A_planck=1.01)
    perturbed_wrapped = dict(defaults_wrapped, A_planck=1.01)
    assert np.isclose(float(pipe_native(perturbed_native)), float(pipe_wrapped(perturbed_wrapped)), rtol=1e-6)


def test_matches_full_native_cobaya_pipeline(tmp_path):
    """The desilike-side pipeline (``CosmoprimoCosmology(engine='camb')`` +
    ``CobayaLikelihood``) must agree with the fully native cobaya pipeline -- cobaya's own
    ``camb`` theory plus the *unmodified* ``camspec_npipe_lite`` likelihood, run through
    ``cobaya.model.get_model`` -- once precision settings are matched by hand.

    Two precision knobs matter and are **not** inherited automatically from cosmoprimo by
    cobaya's native ``camb`` theory (found by sweeping them until the gap closed):

    * ``lmax``: cosmoprimo's ``Harmonic.ellmax_cl`` is fixed at the *fiducial* cosmology's
      construction (the ``ellmax_cl=`` kwarg), not by the per-call
      ``harmonic.lensed_cl(ellmax=...)`` argument (which only slices the precomputed table)
      -- so cobaya's own ``lmax`` extra_arg must match the fiducial's ``ellmax_cl``, not the
      (possibly smaller) likelihood-requested ellmax.
    * ``lens_potential_accuracy``: cosmoprimo always sets this to 1 internally for lensed Cl;
      cobaya's native theory defaults it differently unless set explicitly.

    Leaving either unmatched gives an O(1) discrepancy in logpdf out of ~1.8e4 (checked by
    hand while writing this test); matched, the remaining difference is O(0.01) -- ordinary
    ``AccuracyBoost``-level numerical noise between the two CAMB-driving codepaths, not a
    real disagreement. ``halofit_version`` barely matters at these low ell (30-60).
    """
    pytest.importorskip('camb')
    from cobaya.model import get_model

    planck_Camspec_NPIPE_lite, fits_path, packages_path, info_yaml = _setup_camspec_lite(tmp_path)

    fiducial = ('DESI', dict(lensing=True, ellmax_cl=_CAMSPEC_ELLMAX_CL, non_linear='mead'))
    like_wrapped = wrap_cobaya_likelihood(planck_Camspec_NPIPE_lite, info=info_yaml, packages_path=str(packages_path),
                                           cosmo=CosmoprimoCosmology(engine='camb', fiducial=fiducial))
    pipe_wrapped = compile(like_wrapped)
    defaults_wrapped = {p.name: p._value for p in get_params(like_wrapped)}
    logpdf_wrapped = float(pipe_wrapped(defaults_wrapped))
    assert np.isfinite(logpdf_wrapped)

    point = {
        'H0': 100. * defaults_wrapped['h'],
        'ombh2': defaults_wrapped['omega_b'],
        'omch2': defaults_wrapped['omega_cdm'],
        'As': np.exp(defaults_wrapped['logA']) * 1e-10,
        'ns': defaults_wrapped['n_s'],
        'tau': defaults_wrapped['tau_reio'],
        'mnu': defaults_wrapped['m_ncdm'],
        'nnu': defaults_wrapped['N_eff'],
        'A_planck': defaults_wrapped['A_planck'],
        'calTE': defaults_wrapped['calTE'],
        'calEE': defaults_wrapped['calEE'],
    }
    info = {
        'params': point,
        'theory': {'camb': {'extra_args': {'lmax': _CAMSPEC_ELLMAX_CL, 'halofit_version': 'mead', 'lens_potential_accuracy': 1}}},
        'likelihood': {'camspec_test': {'external': planck_Camspec_NPIPE_lite, 'input_file': 'CamSpec_NPIPE_cmb_sacc.fits'}},
    }
    model = get_model(info, packages_path=str(packages_path))
    loglikes, _ = model.loglikes(point)
    logpdf_native = float(loglikes[0])

    assert np.isclose(logpdf_native, logpdf_wrapped, atol=0.5)


def test_unsupported_requirement_raises():
    """A requirement kind with no desilike cosmo equivalent raises NotImplementedError,
    naming the offending key."""

    class _UnsupportedLikelihood(CobayaBaseLikelihood):
        def get_requirements(self):
            return {'source_Cl': {'sources': {'my_source': {'function': 'spline'}}}}

        def logp(self, **params_values):
            return 0.

    like = wrap_cobaya_likelihood(_UnsupportedLikelihood, cosmo=_make_cosmo())
    with pytest.raises(NotImplementedError, match='source_Cl'):
        compile(like)


if __name__ == '__main__':
    import tempfile

    test_nuisance_parameter_translated()
    test_evaluate_and_cross_check_against_cosmoprimo()
    test_extended_requirements_cross_check()
    with tempfile.TemporaryDirectory() as tmp:
        test_matches_native_jax_camspec_port(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_matches_full_native_cobaya_pipeline(Path(tmp))
    test_unsupported_requirement_raises()
