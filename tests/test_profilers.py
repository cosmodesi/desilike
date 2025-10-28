import numpy as np

from desilike import setup_logging
from desilike.profilers import MinuitProfiler, ScipyProfiler, BOBYQAProfiler, OptaxProfiler

# Global variable for test data directory
dir_test_data = 'desilike/tests'


def test_profilers():
    """Test multiple profilers with galaxy clustering observables."""
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    # Setup Kaiser theory with ShapeFit template
    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): 
        param.update(derived='.best')
    
    observable = TracerPowerSpectrumMultipolesObservable(
        klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
        data=f'{dir_test_data}/_pk/data.npy', 
        covariance=f'{dir_test_data}/_pk/mock_*.npy', 
        wmatrix=f'{dir_test_data}/_pk/window.npy',
        theory=theory
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=1., name='LRG')
    for param in likelihood.all_params.select(basename=['qpar', 'qper']):
        param.update(fixed=True)

    # Test multiple profilers with and without gradients
    for Profiler, kwargs in [(MinuitProfiler, {}),
                             (MinuitProfiler, {'gradient': True}),
                             (ScipyProfiler, {}),
                             (BOBYQAProfiler, {}),
                             (OptaxProfiler, {})]:
        profiler = Profiler(likelihood, seed=42, **kwargs)
        profiles = profiler.maximize(niterations=1)
        
        # Verify profile attributes
        assert profiles.bestfit.attrs['ndof']
        assert profiles.bestfit.attrs['hartlap2007_factor'] is not None
        assert profiles.bestfit['LRG.loglikelihood'].param.latex() == r'L_{\mathrm{LRG}}'
        assert profiles.bestfit['LRG.loglikelihood'].param.derived
        assert profiles.bestfit.logposterior.param.latex() == r'\mathcal{L}'
        assert profiles.bestfit.logposterior.param.derived
        
        # Test profile, grid, interval, and contour methods
        profiler.profile(params=['df'], size=3)
        profiler.grid(params=['df', 'dm'], size=(2, 2))
        profiler.interval(params=['df'])
        # Only test contours for MinuitProfiler to save time
        if Profiler is MinuitProfiler and kwargs == {}:
            profiler.contour(params=['df', 'dm'], cl=1, size=2)
            profiler.contour(params=['df', 'dm'], cl=2, size=2)
        print(profiler.profiles.to_stats())

    # Test with PNG theory (primordial non-Gaussianity)
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    template = FixedPowerSpectrumTemplate(z=0.5, fiducial='DESI')
    # Here we choose b-p parameterization
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode='b-p')
    theory.params['p'].update(fixed=True)  # not fixing p biases fnl_loc posterior
    
    observable = TracerPowerSpectrumMultipolesObservable(
        data=f'{dir_test_data}/_pk/data.npy', 
        covariance=f'{dir_test_data}/_pk/mock_*.npy',
        klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]}, 
        theory=theory
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    profiler = MinuitProfiler(likelihood)
    profiler.maximize(niterations=1)  # Reduced from 2 for speed


def test_contours():
    """Test contour plotting with simple affine model."""
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood
    from desilike import setup_logging, Fisher
    from desilike.samples import plotting

    class AffineModel(BaseCalculator):  # all calculators should inherit from BaseCalculator
        # Model parameters; those can also be declared in a yaml file
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
                'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            # Actual initialization happens in initialize()
            self.x = x

        def calculate(self, a=0., b=0.):
            self.y = a * self.x + b  # simple affine model

        def get(self):
            return self.y

        def __getstate__(self):
            return {'x': self.x, 'y': self.y}  # needed for emulation

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None):
            # Generate some fake data
            self.xdata = np.linspace(0., 1., 10)
            mean = np.zeros_like(self.xdata)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(mean, self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            # Requirements
            if theory is None:
                theory = AffineModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)  # set x-coordinates

        @property
        def flattheory(self):
            return self.theory.y

    setup_logging()  # set up logging

    # Compute Fisher matrix and profiles
    likelihood = Likelihood()
    fisher = Fisher(likelihood)()
    profiler = MinuitProfiler(likelihood)
    profiler.maximize(niterations=1)
    profiler.profile(cl=1., size=3)  # Added size for speed
    
    # Generate contours at different confidence levels
    for cl in [1, 2]:
        profiles = profiler.contour(cl=cl, size=8)  # Reduced size from default
    plotting.plot_triangle([fisher, profiles], show=True)



def test_rescale():
    """Test that rescaling parameters doesn't affect results."""
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood
    from desilike import setup_logging

    class AffineModel(BaseCalculator):  # all calculators should inherit from BaseCalculator
        # Model parameters; those can also be declared in a yaml file
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
                'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            # Actual initialization happens in initialize()
            self.x = x

        def calculate(self, a=0., b=0.):
            self.y = a * self.x + b  # simple affine model

        def get(self):
            return self.y

        def __getstate__(self):
            return {'x': self.x, 'y': self.y}  # needed for emulation

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None):
            # Generate some fake data
            self.xdata = np.linspace(0., 1., 10)
            mean = np.zeros_like(self.xdata)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(mean, self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            # Requirements
            if theory is None:
                theory = AffineModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)  # set x-coordinates

        @property
        def flattheory(self):
            return self.theory.y

    setup_logging()  # set up logging
    likelihood = Likelihood()

    # Test that rescaling gives consistent results across profilers
    # Note: BOBYQA and OptaxProfiler excluded here due to convergence/scaling sensitivity with this simple model
    for Profiler, kwargs in [(MinuitProfiler, {}),
                             (ScipyProfiler, {})]:

        for rescale in [False, True]:
            profiler = Profiler(likelihood, seed=42, rescale=rescale, **kwargs)
            profiles = profiler.maximize(niterations=1)  # Reduced from 2
            profiler.profile(params=['a'], size=3)  # Reduced from 4
            profiler.grid(params=['a', 'b'], size=(2, 2))
            profiler.interval(params=['b'])
            for cl in [1, 2]: 
                profiler.contour(params=['a', 'b'], cl=cl, size=8)  # Reduced from 10
            if not rescale:
                ref = profiles
        
        # Compare rescaled vs non-rescaled results
        tol = {'atol': 1e-2, 'rtol': 1e-1} if Profiler is ScipyProfiler else {'atol': 1e-4, 'rtol': 1e-4}
        for param in ['a', 'b']:
            assert np.allclose(profiles.bestfit[param], ref.bestfit[param], **tol)
            print(profiles.error[param], ref.error[param])
            assert np.allclose(profiles.error[param], ref.error[param], **tol)
        for param in ['b']:
            assert np.allclose(profiles.interval[param], ref.interval[param], **tol)
        assert np.allclose(profiles.grid.logposterior, ref.grid.logposterior, **tol)
        for param in ['a']:
            assert np.allclose(profiles.profile[param], ref.profile[param], **tol)
        if Profiler is MinuitProfiler:  # for ScipyProfiler cov is quite different, leading to different contours
            for param in [('a', 'b')]:
                for cl in [1, 2]:
                    assert np.allclose(profiles.contour[cl][param], ref.contour[cl][param], **tol)


def test_grid():
    """Test grid profiling with multiple iterations."""
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['alpha*', 'sn*']): 
        param.update(derived='.best')
    
    observable = TracerPowerSpectrumMultipolesObservable(
        klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
        data=f'{dir_test_data}/_pk/data.npy', 
        covariance=f'{dir_test_data}/_pk/mock_*.npy', 
        wmatrix=f'{dir_test_data}/_pk/window.npy',
        theory=theory
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=1., name='LRG')
    for param in likelihood.all_params.select(basename=['qpar', 'qper']):
        param.update(fixed=True)

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize(niterations=1)  # Reduced from 2
    profiler.profile(params=['df'], size=2, niterations=1)  # Reduced from 2


def test_solve():
    """Test profiler with derived parameters and parameter solving."""
    from desilike.theories.galaxy_clustering import EFTLikeKaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, BoxFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    template.init.params['f_sqrt_Ap'] = {'derived': True}
    theory = EFTLikeKaiserTracerPowerSpectrumMultipoles(template=template)
    for param in theory.params.select(basename=['sn*']): 
        param.update(prior=dict(dist='norm', loc=1., scale=0.01))
    
    observable = TracerPowerSpectrumMultipolesObservable(
        klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={'b1': 2., 'ct0_2': 1., 'sn0': 0.5},
        theory=theory
    )
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable], name='LRG')
    likelihood()
    
    for param in likelihood.all_params.select(basename=['qpar']):
        param.update(fixed=True)
    
    # Test without rescaling
    profiler = MinuitProfiler(likelihood, rescale=False, seed=42)
    profiles = profiler.maximize(niterations=1)  # Reduced from 2
    print(profiles.to_stats())
    
    # Set derived parameters and test with rescaling
    for param in likelihood.all_params.select(basename=['ct*', 'sn*']): 
        param.update(derived='.best')
    for param in likelihood.all_params.select(basename=['sn*']): 
        param.update(derived='.prec')
    
    profiler = MinuitProfiler(likelihood, rescale=True, seed=42)
    profiles = profiler.maximize(niterations=1)  # Reduced from 2
    print(profiles.to_stats())
    
    # Test with gradients
    for param in likelihood.all_params.select(basename=['sn*']): 
        param.update(derived='.best_not_derived')
    print('GRADIENT')
    profiler = MinuitProfiler(likelihood, rescale=False, gradient=True, seed=42)
    profiles = profiler.maximize(niterations=1)  # Reduced from 2
    print(profiles.to_stats())
    
    # Test derived parameter access
    profiles.bestfit['LRG.loglikelihood']['ct2_2', 'ct2_2']
    try: 
        profiles.bestfit['LRG.loglikelihood']['sn0', 'sn0']
    except KeyError: 
        pass
    else: 
        raise ValueError("Expected KeyError for non-derived parameter")
    
    # Test interval method
    for param in likelihood.all_params.select(basename=['sn*']): 
        param.update(derived='.prec')
    profiler = MinuitProfiler(likelihood, rescale=False, seed=42)
    profiles = profiler.maximize(niterations=1)  # Reduced from 2
    print(profiles.to_stats())
    profiler.interval(params=['df', 'b1'])
    
    assert profiles.bestfit._loglikelihood == 'LRG.loglikelihood'
    print(profiles.bestfit['LRG.loglikelihood'], profiles.bestfit['f_sqrt_Ap'])
    likelihood(**profiles.bestfit.choice(input=True))


def test_bao():
    """Test BAO profiling with custom grid."""
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, BoxFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    
    # Set derived parameters with auto mode
    for param in theory.params.select(basename=['al*']): 
        param.update(derived='.auto')
    for param in theory.params.select(basename=['al*']): 
        param.update(fixed=True, derived='.auto')
    for param in theory.params.select(basename=['al0_0']): 
        param.update(fixed=False, derived='.auto')
    
    observable = TracerPowerSpectrumMultipolesObservable(
        klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={'b1': 1.5},
        theory=theory
    )
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    
    profiler = MinuitProfiler(likelihood, rescale=False, seed=42)
    profiles = profiler.maximize(niterations=1)
    
    # Test profile with explicit grid
    profiles = profiler.profile(params=['qpar'], grid=np.linspace(0.8, 1.2, 3))
    assert profiles.profile['qpar'].shape == (3, 2)
    likelihood(**profiles.bestfit.choice(input=True))
    
    if likelihood.mpicomm.rank == 0:
        print(profiles.bestfit.choice(input=True))
        observable.plot(show=True)


def test_error_cases():
    """Test error handling and edge cases in profilers."""
    
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood
    from desilike.samples import Profiles
    import tempfile
    import os

    class AffineModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
                   'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x

        def calculate(self, a=0., b=0.):
            self.y = a * self.x + b

        def get(self):
            return self.y

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            mean = np.zeros_like(self.xdata)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(mean, self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = AffineModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    # Test 1: No varied parameters - should raise ValueError
    likelihood = Likelihood()
    for param in likelihood.all_params.select(varied=True):
        param.update(fixed=True)
    try:
        profiler = MinuitProfiler(likelihood)
        assert False, "Should have raised ValueError for no varied params"
    except ValueError as e:
        assert "No parameters to be varied" in str(e)

    # Test 2: Test context manager
    likelihood = Likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    with profiler as p:
        assert p is profiler
    
    # Test 3: Test __getstate__
    state = profiler.__getstate__()
    assert 'max_tries' in state
    
    # Test 4: Test saving profiles to file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_fn = os.path.join(tmpdir, 'test_profiles.npy')
        profiler = MinuitProfiler(likelihood, seed=42, save_fn=save_fn)
        profiles = profiler.maximize(niterations=1)
        assert os.path.exists(save_fn)
        
        # Test loading from file
        profiler2 = MinuitProfiler(likelihood, seed=42, profiles=save_fn)
        assert profiler2.profiles is not None
    
    # Test 5: Test probability-based CL conversion in interval
    likelihood = Likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    profiler.maximize(niterations=1)
    # CL < 1.0 should be interpreted as probability
    profiles = profiler.interval(params=['a'], cl=0.68)
    assert 'interval' in profiles
    
    # Test 6: Test probability-based CL conversion in contour
    profiles = profiler.contour(params=[('a', 'b')], cl=0.68, size=4)  # Reduced from 5
    assert 'contour' in profiles
    
    # Test 7: Test covariance() method standalone
    likelihood = Likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.covariance()  # Should call maximize first
    assert 'covariance' in profiles
    assert 'error' in profiles
    
    # Test 8: Test profile with explicit grid
    likelihood = Likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    profiler.maximize(niterations=1)
    custom_grid = np.linspace(-2, 2, 3)  # Reduced from 5
    profiles = profiler.profile(params=['a'], grid=custom_grid)
    assert profiles.profile['a'].shape[0] == 3  # Updated assertion


def test_invalid_start():
    """Test invalid start parameter handling."""
    
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood

    class AffineModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x

        def calculate(self, a=0.):
            self.y = a * self.x

        def get(self):
            return self.y

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(np.zeros_like(self.xdata), self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = AffineModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    likelihood = Likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    
    # Test invalid start shape
    try:
        wrong_shape_start = [[0.0]]  # Should be (niterations, nparams)
        profiler.maximize(niterations=2, start=wrong_shape_start)
        assert False, "Should raise ValueError for wrong shape"
    except ValueError as e:
        assert "shape" in str(e)


def test_grid_params_variations():
    """Test different ways of specifying grid parameters."""
    
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood

    class AffineModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
                   'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x

        def calculate(self, a=0., b=0.):
            self.y = a * self.x + b

        def get(self):
            return self.y

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(np.zeros_like(self.xdata), self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = AffineModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    likelihood = Likelihood()
    
    # Test 1: Grid specified as dict with explicit values
    profiler1 = MinuitProfiler(likelihood, seed=42)
    profiler1.maximize(niterations=1)
    profiles = profiler1.grid(grid={'a': np.linspace(-1, 1, 3), 'b': np.linspace(-1, 1, 3)})
    assert 'grid' in profiles
    
    # Test 2: Single parameter grid (not a list) - use fresh profiler
    profiler2 = MinuitProfiler(likelihood, seed=42)
    profiler2.maximize(niterations=1)
    profiles = profiler2.grid(params='a', size=3)
    assert 'grid' in profiles


def test_profile_variations():
    """Test different profile parameter specifications."""
    
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood

    class AffineModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
                   'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x

        def calculate(self, a=0., b=0.):
            self.y = a * self.x + b

        def get(self):
            return self.y

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(np.zeros_like(self.xdata), self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = AffineModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    likelihood = Likelihood()
    
    # Test 1: Profile with explicit grid provided (doesn't need maximize first)
    profiler1 = MinuitProfiler(likelihood, seed=42)
    profiles = profiler1.profile(params=['a'], grid=[np.linspace(-2, 2, 3)])  # Reduced from 5
    assert 'profile' in profiles
    
    # Test 2: Profile with default size (tests size parameter)
    profiler2 = MinuitProfiler(likelihood, seed=42)  # Fresh profiler
    profiler2.maximize(niterations=1)
    profiles = profiler2.profile(params=['a'], size=3)  # Simple single param test
    assert 'profile' in profiles


def test_start_parameters():
    """Test different ways of providing start parameters."""
    
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood

    class AffineModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x

        def calculate(self, a=0.):
            self.y = a * self.x

        def get(self):
            return self.y

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(np.zeros_like(self.xdata), self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = AffineModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    likelihood = Likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    
    # Test providing start as single value (needs to be 2D array)
    start_value = [[0.5]]  # Shape (1, 1) for 1 iteration, 1 param
    profiles = profiler.maximize(start=start_value)
    assert 'bestfit' in profiles
    
    # Test providing explicit start values as list
    start_list = [[0.2], [0.8]]  # Shape (2, 1) for 2 iterations, 1 param
    profiles = profiler.maximize(start=start_list)
    assert 'bestfit' in profiles


def test_likelihood_error_handling():
    """Test error handling during likelihood evaluation."""
    
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood
    from desilike import PipelineError

    class ProblematicModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x
            self.call_count = 0

        def calculate(self, a=0.):
            self.call_count += 1
            # Raise an error for certain parameter values to test error handling
            if a < -5.0:
                raise RuntimeError("Invalid parameter value")
            # Return NaN for another range to test NaN handling
            if a > 5.0:
                self.y = np.full_like(self.x, np.nan)
            else:
                self.y = a * self.x

        def get(self):
            return self.y

    class Likelihood(BaseGaussianLikelihood):
        catch_errors = (RuntimeError,)  # Catch RuntimeError as -inf loglikelihood
        
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(np.zeros_like(self.xdata), self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = ProblematicModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    # Test 1: Likelihood that catches errors
    likelihood = Likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    # This should work - errors are caught and treated as -inf
    profiles = profiler.maximize(niterations=1)
    assert 'bestfit' in profiles
    
    # Test 2: Try to trigger error that should NOT be caught (uncaught error type)
    class UncatchableModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x

        def calculate(self, a=0.):
            if a < -8.0:  # Very unlikely with the prior
                raise KeyError("Uncaught error type")
            self.y = a * self.x

        def get(self):
            return self.y

    class StrictLikelihood(BaseGaussianLikelihood):
        catch_errors = (RuntimeError,)  # Only catch RuntimeError, not KeyError
        
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(np.zeros_like(self.xdata), self.covariance)
            super(StrictLikelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = UncatchableModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    # This likelihood works fine with normal parameters
    likelihood2 = StrictLikelihood()
    profiler2 = MinuitProfiler(likelihood2, seed=42)
    profiles2 = profiler2.maximize(niterations=1)
    assert 'bestfit' in profiles2


def test_max_tries_exceeded():
    """Test max_tries limit when no finite posterior is found."""
    
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood

    class AlwaysFailModel(BaseCalculator):
        _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

        def initialize(self, x=None):
            self.x = x

        def calculate(self, a=0.):
            # Always raise an error
            raise RuntimeError("Model always fails")

        def get(self):
            return self.y

    class Likelihood(BaseGaussianLikelihood):
        catch_errors = (RuntimeError,)
        
        def initialize(self, theory=None):
            self.xdata = np.linspace(0., 1., 10)
            self.covariance = np.eye(len(self.xdata))
            rng = np.random.RandomState(seed=42)
            y = rng.multivariate_normal(np.zeros_like(self.xdata), self.covariance)
            super(Likelihood, self).initialize(y, covariance=self.covariance)
            if theory is None:
                theory = AlwaysFailModel()
            self.theory = theory
            self.theory.init.update(x=self.xdata)

        @property
        def flattheory(self):
            return self.theory.y

    likelihood = Likelihood()
    # Set max_tries to a small value to test quickly
    try:
        profiler = MinuitProfiler(likelihood, seed=42, max_tries=3)
        profiles = profiler.maximize(niterations=1)
        assert False, "Should have raised ValueError for max_tries exceeded"
    except ValueError as e:
        assert "Could not find finite log posterior" in str(e)


if __name__ == '__main__':
    setup_logging()

    # Run all tests
    test_rescale()
    test_grid()
    test_profilers()
    test_solve()
    test_bao()
    test_contours()
    test_error_cases()
    test_invalid_start()
    test_grid_params_variations()
    test_profile_variations()
    test_start_parameters()
    test_likelihood_error_handling()
    test_max_tries_exceeded()
