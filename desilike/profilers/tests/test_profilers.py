import numpy as np
import pytest
from jax import numpy as jnp

import desilike.profilers as profilers
from desilike.samples import Profiles
from desilike.base import compile, GaussianLikelihood as BaseGaussianLikelihood, Prior, Posterior
from desilike.parameter import Parameter


# ── Toy likelihood: 2-D Gaussian ─────────────────────────────────────────────
#
# True best-fit: (MU_X, MU_Y) = (0.3, -0.7)
# True 1-sigma errors: (SX, SY) = (0.05, 0.1)
# Derived: z = x + y

MU_X, MU_Y = 0.3, -0.7
SX, SY     = 0.05, 0.1

PROFILER_CLS = dict(
    scipy=profilers.ScipyProfiler,
    minuit=profilers.MinuitProfiler,
    bobyqa=profilers.BOBYQAProfiler,
    optax=profilers.OptaxProfiler)
_VECTOR_PROFILER_NAMES = ['scipy', 'bobyqa']


@pytest.fixture
def likelihood():

    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.flatdata  = jnp.array([MU_X, MU_Y])
            self.precision = jnp.diag(jnp.array([1. / SX**2, 1. / SY**2]))
            self.z = Parameter('z', value=0., derived=True)

        def __call__(self):
            self.flattheory = jnp.array([self.x, self.y])
            self.z.value = self.x + self.y
            return super().__call__()

    x = Parameter('x', value=MU_X, prior=dict(dist='uniform', limits=[-1, 1]),
                  ref=dict(dist='norm', loc=MU_X, scale=0.2), proposal=SX)
    y = Parameter('y', value=MU_Y, prior=dict(dist='uniform', limits=[-1, 1]),
                  ref=dict(dist='norm', loc=MU_Y, scale=0.2), proposal=SY)
    return compile(Posterior(Likelihood(x, y), Prior(x, y)))


def make_profiler(key, likelihood, seed=42, **kwargs):
    """Instantiate the named profiler, skipping the test if the backend is absent."""
    optional_deps = dict(minuit='iminuit', bobyqa='pybobyqa', optax='optax')
    if key in optional_deps:
        pytest.importorskip(optional_deps[key])
    if key == 'optax':
        return PROFILER_CLS[key](likelihood, seed=seed, method='adam', **kwargs)
    return PROFILER_CLS[key](likelihood, seed=seed, **kwargs)


def make_vec_likelihood():
    """Compiled VecGaussian with a vector param v(2,) and scalar z."""
    mu_v  = np.array([0.3, 0.5])
    sig_v = np.array([0.05, 0.08])
    mu_z, sig_z = 0.1, 0.02

    v_param = Parameter('v', value=mu_v, shape=(2,), proposal=0.06, fixed=False)
    z_param = Parameter('z', value=mu_z, proposal=sig_z,
                        prior=dict(dist='norm', loc=mu_z, scale=0.1),
                        ref=dict(dist='norm', loc=mu_z, scale=0.1))

    class VecGaussian(BaseGaussianLikelihood):

        def __init__(self, v, z):
            self.v = v
            self.z = z
            self.flatdata  = jnp.concatenate([jnp.asarray(mu_v), jnp.array([mu_z])])
            self.precision = jnp.diag(jnp.array([
                1. / sig_v[0]**2, 1. / sig_v[1]**2, 1. / sig_z**2]))

        def __call__(self):
            self.flattheory = jnp.concatenate([self.v, jnp.array([self.z])])
            return super().__call__()

    return compile(Posterior(VecGaussian(v_param, z_param), Prior(z_param)))


@pytest.fixture
def vec_likelihood():
    return make_vec_likelihood()


# ── accuracy / solved / derived ───────────────────────────────────────────────

@pytest.mark.parametrize('key', PROFILER_CLS.keys())
def test_accuracy_and_derived(likelihood, key):
    """All profilers find the correct best-fit, errors, and derived parameter z."""
    p = make_profiler(key, likelihood)
    p.maximize(niterations=3)
    p.covariance()
    idx = p.profiles.argmax
    assert abs(float(p.profiles.best['x'][idx]) - MU_X) < 0.01
    assert abs(float(p.profiles.best['y'][idx]) - MU_Y) < 0.01
    assert abs(float(p.profiles.error['x'][idx]) - SX) < 0.02
    assert abs(float(p.profiles.error['y'][idx]) - SY) < 0.02
    assert 'z' in p.profiles.best
    assert abs(float(p.profiles.best['z'][idx]) - (MU_X + MU_Y)) < 0.02


@pytest.mark.parametrize('key', ['scipy', 'minuit'])
def test_solved(likelihood, key):
    """Solved (best-fit) parameter is correctly profiled."""
    class Likelihood(BaseGaussianLikelihood):

        def __init__(self, x, y):
            self.x = x; self.y = y
            self.flatdata  = jnp.array([MU_X, MU_Y])
            self.precision = jnp.diag(jnp.array([1. / SX**2, 1. / SY**2]))

        def __call__(self):
            self.flattheory = jnp.array([self.x, self.y])
            return super().__call__()

    x = Parameter('x', value=MU_X, prior=dict(dist='uniform', limits=[-1, 1]),
                  ref=dict(dist='norm', loc=MU_X, scale=0.2), proposal=SX)
    y = Parameter('y', value=MU_Y, derived='best')
    p = make_profiler(key, compile(Posterior(Likelihood(x, y), Prior(x))))
    p.maximize(niterations=3)
    assert abs(float(p.profiles.best['x'][p.profiles.argmax]) - MU_X) < 0.01


# ── construction ──────────────────────────────────────────────────────────────

class TestBaseProfiler:

    def test_varied_params(self, likelihood):
        assert profilers.ScipyProfiler(likelihood, seed=42).varied_params.names() == ['x', 'y']

    def test_transforms(self, likelihood):
        """Forward/backward transforms are inverses; rescale changes the scale."""
        p_flat = profilers.ScipyProfiler(likelihood, seed=42)
        p_resc = profilers.ScipyProfiler(likelihood, seed=42, rescale=True)
        x = np.array([0.5, -0.3])
        np.testing.assert_allclose(p_flat._forward(p_flat._backward(x)), x)
        np.testing.assert_allclose(p_flat._backward(p_flat._forward(x)), x)
        np.testing.assert_allclose(p_resc._forward(p_resc._backward(x)), x)
        assert not np.allclose(p_resc._scale, 1.)

    def test_chi2_and_starts(self, likelihood):
        """chi2 is zero at truth; _get_starts produces finite starting points."""
        p = profilers.ScipyProfiler(likelihood, seed=42)
        truth_rescaled = p._backward(np.array([MU_X, MU_Y]))
        np.testing.assert_allclose(float(p._chi2_rescaled(truth_rescaled)), 0., atol=1e-12)
        assert p._jit_chi2 is not None and p._jit_chi2 is not p._chi2_rescaled
        s = p._get_starts(5)
        assert s.shape == (5, 2)
        assert all(np.isfinite(float(p._jit_chi2(row))) for row in s)


# ── maximize ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestMaximize:

    def _check(self, profiles, atol_bf=0.01):
        idx = profiles.argmax
        assert abs(float(profiles.best['x'][idx]) - MU_X) < atol_bf
        assert abs(float(profiles.best['y'][idx]) - MU_Y) < atol_bf
        assert float(profiles.best['logpdf'][idx]) > -0.5

    def test_maximize(self, likelihood, profiler_name):
        """Basic and rescaled maximize converge; multiple runs accumulate."""
        p = make_profiler(profiler_name, likelihood)
        self._check(p.maximize(niterations=3))
        p.maximize(niterations=1)
        assert p.profiles.nruns == 4
        self._check(make_profiler(profiler_name, likelihood, rescale=True).maximize(niterations=3))

    def test_maximize_fixed_start(self, likelihood, profiler_name):
        profiles = make_profiler(profiler_name, likelihood).maximize(start=np.array([[0.35, -0.65]]))
        self._check(profiles)
        assert profiles.nruns == 1

    def test_scipy_gradient(self, likelihood, profiler_name):
        """Gradient-based L-BFGS-B maximisation (ScipyProfiler only)."""
        p = profilers.ScipyProfiler(likelihood, seed=42, gradient=True, method='L-BFGS-B')
        self._check(p.maximize(niterations=2))


# ── covariance ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestCovariance:

    def test_covariance(self, likelihood, profiler_name):
        """Errors and full covariance matrix match the known Gaussian widths."""
        p = make_profiler(profiler_name, likelihood)
        p.maximize(niterations=2)
        p.covariance()
        idx = p.profiles.argmax
        ex, ey = float(p.profiles.error['x'][idx]), float(p.profiles.error['y'][idx])
        assert abs(ex - SX) < 0.02, f'x error {ex:.4f} != {SX}'
        assert abs(ey - SY) < 0.02, f'y error {ey:.4f} != {SY}'
        cov = np.asarray(p.profiles.covariance)
        assert cov.shape == (2, 2)
        np.testing.assert_allclose(np.sqrt(cov[0, 0]), SX, atol=0.02)
        np.testing.assert_allclose(np.sqrt(cov[1, 1]), SY, atol=0.02)

    def test_covariance_auto_trigger(self, likelihood, profiler_name):
        """covariance() triggers maximize() when no best-fit exists."""
        p = make_profiler(profiler_name, likelihood)
        p.covariance()
        assert p.profiles.best is not None and p.profiles.error is not None


# ── interval ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestInterval:

    def test_interval(self, likelihood, profiler_name):
        """Single-param, 1-σ values, cl-probability equivalence, and 2-σ wider."""
        from scipy import stats
        p = make_profiler(profiler_name, likelihood)
        p.maximize(niterations=2)
        p.covariance()
        # Single param: only 'x' computed.
        p.interval(params='x')
        assert 'x' in p.profiles.interval and 'y' not in p.profiles.interval
        # 1-sigma full interval: correct shape, sign, and width.
        p.interval(cl=1, xtol=1e-4)
        lo_x, hi_x = p.profiles.interval['x']
        lo_y, hi_y = p.profiles.interval['y']
        assert lo_x.shape == (1,) and lo_x[0] < 0. and hi_x[0] > 0.
        assert lo_y[0] < 0. and hi_y[0] > 0.
        np.testing.assert_allclose(lo_x[0], -SX, atol=0.01)
        np.testing.assert_allclose(hi_x[0],  SX, atol=0.01)
        np.testing.assert_allclose(lo_y[0], -SY, atol=0.01)
        np.testing.assert_allclose(hi_y[0],  SY, atol=0.01)
        # Integer cl=1 and equivalent CDF probability give the same result.
        cl_prob = float(stats.chi2(df=1).cdf(1.))
        p.interval(cl=cl_prob, xtol=1e-5)
        lo_p, hi_p = p.profiles.interval['x']
        np.testing.assert_allclose(lo_p[0], lo_x[0], atol=0.005)
        np.testing.assert_allclose(hi_p[0], hi_x[0], atol=0.005)
        # 2-sigma is wider than 1-sigma.
        p.interval(cl=2, xtol=1e-4)
        lo2, hi2 = p.profiles.interval['x']
        assert lo2[0] < lo_x[0] and hi2[0] > hi_x[0]

    def test_interval_auto_trigger(self, likelihood, profiler_name):
        p = make_profiler(profiler_name, likelihood)
        p.interval()
        assert p.profiles.best is not None and p.profiles.interval is not None

    def test_interval_vector_param_raises(self, vec_likelihood, profiler_name):
        p = profilers.ScipyProfiler(vec_likelihood, seed=0)
        p.maximize(niterations=1)
        p.covariance()
        with pytest.raises(ValueError, match='scalar'):
            p.interval(params='v')


# ── contour ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestContour:

    def test_contour(self, likelihood, profiler_name):
        """1-σ contour shape/closure/centering/param-specs; 2-σ wider than 1-σ."""
        p = make_profiler(profiler_name, likelihood)
        p.maximize(niterations=2)
        p.covariance()
        size = 32
        # Pair-list spec.
        p.contour(params=[('x', 'y')], cl=1, size=size)
        assert 1 in p.profiles.contour and ('x', 'y') in p.profiles.contour[1]
        x1, x2 = p.profiles.contour[1][('x', 'y')]
        assert x1.shape == (size + 1,) and x2.shape == (size + 1,)
        assert float(x1[0]) == float(x1[-1]) and float(x2[0]) == float(x2[-1])  # closed
        mask = np.isfinite(x1) & np.isfinite(x2)
        assert mask.any()
        assert abs(float(np.mean(x1[mask])) - MU_X) < 0.05
        assert abs(float(np.mean(x2[mask])) - MU_Y) < 0.05
        # Name-list spec produces the same pair key.
        p.contour(params=['x', 'y'], cl=1, size=10)
        assert ('x', 'y') in p.profiles.contour[1]
        # 2-sigma has a wider x-range than 1-sigma.
        p.contour(cl=2, size=size)
        x1_2, _ = p.profiles.contour[2][('x', 'y')]
        range1 = float(np.ptp(x1[np.isfinite(x1)]))
        range2 = float(np.ptp(x1_2[np.isfinite(x1_2)]))
        assert range2 > range1

    def test_contour_auto_trigger(self, likelihood, profiler_name):
        p = make_profiler(profiler_name, likelihood)
        p.contour(size=10)
        assert p.profiles.best is not None and p.profiles.contour is not None

    def test_contour_vector_param_raises(self, vec_likelihood, profiler_name):
        p = profilers.ScipyProfiler(vec_likelihood, seed=0)
        p.maximize(niterations=1)
        p.covariance()
        with pytest.raises(ValueError, match='scalar'):
            p.contour(params=[('v', 'z')])


# ── profile ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestProfile:

    def test_profile(self, likelihood, profiler_name):
        """Profile x (auto grid and explicit grid) and both params."""
        p = make_profiler(profiler_name, likelihood)
        p.maximize(niterations=2)
        p.covariance()
        # Auto grid for x.
        scan, lp = p.profile(params='x', size=11).profile['x']
        assert len(scan) == 11 and abs(float(scan[np.argmax(lp)]) - MU_X) < 0.05
        # Explicit grid.
        scan_vals = np.linspace(0.1, 0.5, 7)
        scan2, lp2 = p.profile(params='x', grid=scan_vals).profile['x']
        np.testing.assert_array_equal(scan2, scan_vals)
        assert len(lp2) == 7
        # Both params.
        prof = p.profile(params=['x', 'y'], size=7)
        assert 'x' in prof.profile and 'y' in prof.profile


# ── grid ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestGrid:

    def test_grid(self, likelihood, profiler_name):
        """2-D and 1-D grids have correct shapes and the peak is near truth."""
        p = make_profiler(profiler_name, likelihood)
        p.maximize(niterations=2)
        p.covariance()
        # 2-D
        lp_2d = p.grid(params=['x', 'y'], size=5).grid['logpdf']
        assert lp_2d.shape == (5, 5)
        peak = np.unravel_index(np.argmax(lp_2d), lp_2d.shape)
        assert 1 <= peak[0] <= 3 and 1 <= peak[1] <= 3
        # 1-D
        assert p.grid(params=['x'], size=5).grid['logpdf'].shape == (5,)


# ── Profiles.start ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestStart:

    def test_start(self, likelihood, profiler_name):
        """start dict is populated in original space for both default and rescaled."""
        p = make_profiler(profiler_name, likelihood)
        profiles = p.maximize(niterations=3)
        assert profiles.start is not None
        assert profiles.start['x'].shape == (profiles.nruns,)
        assert profiles.start['y'].shape == (profiles.nruns,)
        # Rescaled profiler: starts are still in original parameter space.
        p2 = make_profiler(profiler_name, likelihood, rescale=True)
        profiles2 = p2.maximize(niterations=3)
        assert all(abs(float(v) - MU_X) < 1.0 for v in profiles2.start['x'])
        assert all(abs(float(v) - MU_Y) < 1.0 for v in profiles2.start['y'])


# ── Non-scalar parameters ─────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', _VECTOR_PROFILER_NAMES)
class TestVectorParams:

    MU_V  = np.array([0.3, 0.5])
    SIG_V = np.array([0.05, 0.08])
    MU_Z  = 0.1

    @pytest.fixture
    def vec_likelihood(self):
        return make_vec_likelihood()

    def test_flat_layout(self, vec_likelihood, profiler_name):
        p = make_profiler(profiler_name, vec_likelihood, seed=0)
        assert p._flat_size == 3
        assert p._param_slices['v'] == slice(0, 2)
        assert p._param_slices['z'] == slice(2, 3)

    def test_maximize_and_start(self, vec_likelihood, profiler_name):
        """maximize() converges for vector param; start dict has the right shapes."""
        p = make_profiler(profiler_name, vec_likelihood, seed=0)
        profiles = p.maximize(niterations=2)
        idx = profiles.argmax
        assert profiles.best['v'].shape == (profiles.nruns, 2)
        np.testing.assert_allclose(profiles.best['v'][idx], self.MU_V, atol=0.02)
        np.testing.assert_allclose(float(profiles.best['z'][idx]), self.MU_Z, atol=0.02)
        assert profiles.start['v'].shape == (profiles.nruns, 2)
        assert profiles.start['z'].shape == (profiles.nruns,)

    def test_profile_with_vector(self, vec_likelihood, profiler_name):
        """profile() raises for vector param; works for scalar param alongside vector."""
        p = make_profiler(profiler_name, vec_likelihood, seed=0)
        p.maximize(niterations=2)
        p.covariance()
        with pytest.raises(ValueError, match='scalar'):
            p.profile(params='v')
        scan, lp = p.profile(params='z', size=7).profile['z']
        assert len(scan) == 7 and abs(float(scan[np.argmax(lp)]) - self.MU_Z) < 0.05


# ── I/O roundtrip ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profiler_name', PROFILER_CLS.keys())
class TestIO:

    def test_save_and_reload(self, likelihood, profiler_name, tmp_path):
        fn  = str(tmp_path / 'profiles.h5')
        p   = make_profiler(profiler_name, likelihood, save_fn=fn)
        p.maximize(niterations=2)
        loaded = Profiles.read(fn)
        assert loaded.best is not None and 'x' in loaded.best
        np.testing.assert_allclose(loaded.best['logpdf'], p.profiles.best['logpdf'])
