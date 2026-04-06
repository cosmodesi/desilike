.. _user-getting-started:


Getting started
===============

In this page we will describe **desilike**'s basics' with a practical example,
but further examples can be found in the provided `notebooks <https://github.com/cosmodesi/desilike/blob/main/nb>`_.

**desilike** provides a framework to specify DESI likelihoods.

Clustering likelihood
---------------------
Let's describe how to specify the likelihood for power spectrum multipoles.

First, we specify a template, i.e. how the linear power spectrum as input of the theory codes is parameterized.
Several options are possible:

- standard (as in BOSS/eBOSS) parameterization, in terms of :math:`q_{\parallel}`, :math:`q_{\perp}` (scaling parameters),
  :math:`df` (variation in the growth rate of structure: :math:`f / f^{\mathrm{fid}}`) with :class:`~desilike.theories.galaxy_clustering.power_template.StandardPowerSpectrumTemplate`;
- `ShapeFit <https://arxiv.org/abs/2106.07641>`_ parameterization, in terms of :math:`q_{\parallel}`, :math:`q_{\perp}` (scaling parameters),
  :math:`df` (variation in the growth rate of structure: :math:`f / f^{\mathrm{fid}}`), :math:`dm` (ShapeFit tilt parameter) with :class:`~desilike.theories.galaxy_clustering.power_template.ShapeFitPowerSpectrumTemplate`;
- parameterization in terms of base cosmological parameters, with :class:`~desilike.theories.galaxy_clustering.power_template.DirectPowerSpectrumTemplate`

See :mod:`~desilike.theories.galaxy_clustering.power_template` for all templates.

.. code-block:: python

  from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate

  # Or StandardPowerSpectrumTemplate, DirectPowerSpectrumTemplate
  template = ShapeFitPowerSpectrumTemplate(z=0.8)  # effective redshift

.. note::

  In python, ``help(Calculator)``, for any Calculator, like :class:`~desilike.theories.galaxy_clustering.power_template.ShapeFitPowerSpectrumTemplate`,
  will provide useful information, in particular the possible arguments.

  Any calculator, profiler, sampler, etc. can be installed with :class:`~desilike.install.Installer`.

Next, let's specify the theory model.
Several options are possible, with the most notable one being:

- simple Kaiser model, with :class:`~desilike.theories.galaxy_clustering.full_shape.KaiserTracerPowerSpectrumMultipoles`,
  or :class:`~desilike.theories.galaxy_clustering.full_shape.KaiserTracerCorrelationFunctionMultipoles`
- `velocileptors <https://github.com/sfschen/velocileptors>`_ model (LPT_RSD), with :class:`~desilike.theories.galaxy_clustering.full_shape.LPTVelocileptorsTracerPowerSpectrumMultipoles`,
  or :class:`~desilike.theories.galaxy_clustering.full_shape.LPTVelocileptorsTracerCorrelationFunctionMultipoles`
- `pybird <https://github.com/pierrexyz/pybird>`_ model, with :class:`~desilike.theories.galaxy_clustering.full_shape.PyBirdTracerPowerSpectrumMultipoles`,
  or :class:`~desilike.theories.galaxy_clustering.full_shape.PyBirdTracerCorrelationFunctionMultipoles`
- `folps-nu <https://github.com/henoriega/FOLPS-nu>`_ model, with :class:`~desilike.theories.galaxy_clustering.full_shape.FOLPSTracerPowerSpectrumMultipoles`,
  or :class:`~desilike.theories.galaxy_clustering.full_shape.FOLPSTracerCorrelationFunctionMultipoles`
- empirical BAO model, with :class:`~desilike.theories.galaxy_clustering.bao.DampedBAOWigglesPowerSpectrumMultipoles`,
  or :class:`~desilike.theories.galaxy_clustering.bao.ResummedBAOWigglesPowerSpectrumMultipoles`
- power spectrum with scale-dependent bias (primordial non-gaussianity), with :class:`~desilike.theories.galaxy_clustering.primordial_non_gaussianity.PNGTracerPowerSpectrumMultipoles`

See :mod:`~desilike.theories.galaxy_clustering.full_shape` for all full shape models, and :mod:`~desilike.theories.galaxy_clustering.bao` for all BAO models.

.. code-block:: python

  from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles

  # Or LPTVelocileptorsTracerPowerSpectrumMultipoles, PyBirdTracerPowerSpectrumMultipoles, etc.
  theory = KaiserTracerPowerSpectrumMultipoles(template=template)

One can update the template (or any relevant calculator's options passed at initialization) with ``calculator.init.update(...)``:

.. code-block:: python

  theory.init.update(template=ShapeFitPowerSpectrumTemplate(z=1.))

Then, we want to compare the theory to data (an *observable*), typically:

- power spectrum multipoles, with :class:`~desilike.observables.galaxy_clustering.power_spectrum.TracerPowerSpectrumMultipolesObservable`,
- correlation function multipoles, with :class:`~desilike.observables.galaxy_clustering.correlation_function.TracerCorrelationFunctionMultipolesObservable`

.. code-block:: python

  from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable

  # Or TracerCorrelationFunctionMultipolesObservable
  observable = TracerPowerSpectrumMultipolesObservable(data={'b1': 1.2},  # a (list of) (path to) *pypower* power spectrum measurement, flat array, or dictionary of parameters where to evaluate the theory to take as a mock data vector
                                                       covariance=None,  # a (list of) (path to) mocks, array (covariance matrix), or None
                                                       klim={0: [0.01, 0.2, 0.005], 2: [0.01, 0.2, 0.005]},  # k-limits, between 0.01 and 0.2 h/Mpc with 0.005 h/Mpc step size for ell = 0, 2
                                                       theory=theory)  # previously defined theory

In this (runnable!) example, we do not have a covariance yet; let's estimate it on-the-fly (Gaussian approximation).

.. code-block:: python

  from desilike.observables.galaxy_clustering import BoxFootprint, ObservablesCovarianceMatrix

  footprint = BoxFootprint(volume=1e9, nbar=1e-3)  # box with volume of 1e9 (Mpc/h)^3 and density of 1e-3 (h/Mpc)^3
  covariance = ObservablesCovarianceMatrix(observables=[observable], footprints=[footprint])
  cov = covariance(b1=1.2)   # evaluate covariance matrix at this parameter

Now we can define the likelihood:

.. code-block:: python

  from desilike.likelihoods import ObservablesGaussianLikelihood

  # No need to specify covariance if already given to the observable (TracerPowerSpectrumMultipolesObservable)
  # If mocks are given to each observable, the likelihood covariance matrix is computed on-the-fly, using mocks from each observable (taking into account correlations)
  likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)

To sum several independent likelihoods:

.. code-block:: python

  from desilike.likelihoods import SumLikelihood

  likelihood = SumLikelihood(likelihoods=[likelihood1, likelihood2])


The likelihood (and any other calculator) can be called at any point with:

.. code-block:: python

  likelihood(b1=1., sn0=1000.)  # update linear bias b1, and shot noise sn0
  likelihood(qpar=0.99)  # update scaling parameter qpar; b1 and sn0 are kept to 1. and 1000.
  likelihood(sn0=100.)  # update shot noise, the template is to re-calculated

  theory.power  # contains multipoles of the power spectrum, evaluated at b1=1., qpar=0.99 and sn0=100.
  theory(sn0=1000.)  # recomputes the theory at sn0=1000.


Parameters
----------

Parameters of all calculators in the pipeline can be accessed e.g. with:

.. code-block:: python

  likelihood.all_params  # b1, sn0, df, qpar, qper, dm
  template.all_params  # df, qpar, qper, dm
  template.all_params.select(basename='q*')  # restrict to parameters with base name starting with q*: qpar, qper

To get only the parameters of the calculator:

.. code-block:: python

  theory.init.params  # b1, sn0

Above objects are :class:`~desilike.parameter.ParameterCollection`.

Main parameter's attributes are (see :class:`~desilike.parameter.Parameter`):

- its (base) name (basename)
- its default value (value)
- its prior (prior)
- its reference distribution (ref), to randomly sample initial points for sampling / profiling
- variation range to use when performing finite differentiation (delta); see :ref:`user-getting-started-fisher`
- whether the parameter is fixed (fixed)
- latex string (latex)

They can be all updated with :meth:`~desilike.parameter.Parameter.update`. This can be done at the calculator level:

.. code-block:: python

  from desilike.theories import Cosmoprimo
  from desilike.galaxy_clustering import DirectPowerSpectrumTemplate

  cosmo = Cosmoprimo()
  cosmo.init.params = {'Omega_m': {'value': 0.3}, 'h': {'value': 0.7}, 'sigma8': {'value': 0.8}}
  cosmo.init.params['n_s'].update(value=0.96)

  template = DirectPowerSpectrumTemplate(cosmo=cosmo, z=1.)

Or at the pipeline level:

.. code-block:: python

  # Update parameter dm's reference distribution with uniform distribution in [-0.01, 0.01]
  likelihood.all_params['dm'].update(ref={'limits': [-0.01, 0.01]})
  # Update parameter df's prior distribution with normal distribution centered on 1 and with standard deviation 2
  likelihood.all_params['df'].update(prior={'dist': 'norm', 'loc': 1., 'scale': 2.})
  # Set b1=2. and fix it
  likelihood.all_params['b1'].update(value=2., fixed=True)
  # Now varied likelihood parameters are:
  likelihood.varied_params  # dm, df, sn0, qpar, qper

.. note::

  Changes to parameters at the calculator level (``calculator.init.params``) will be shared by all pipelines using this calculator instance;
  e.g. if above ``cosmo`` is used by multiple templates ``template1``, ``template2``, etc.,
  ``template1.all_params`` and ``template2.all_params`` would share the same Omega_m, h, sigma8 parameters.
  However, updating parameters at the pipeline level, e.g. ``template1.all_params['n_s']`` leaves ``template2.all_params['n_s']`` untouched.
  Also, if ``template1`` needs to be reinitialized, e.g. because it is passed to a theory or ``template1.init.params`` is updated,
  then changes to ``template1.all_params['n_s']`` will be lost.
  Therefore, updates to ``calculator.all_params`` are only useful for the final calculator of the pipeline, typically the likelihood as illustrated above.


The likelihood can be analytically marginalized over linear parameters (here ``sn0``):

.. code-block:: python

  # '.best': set sn0 at best fit
  # '.marg': marginalization, assuming Gaussian likelihood
  # '.auto': automatically choose between '.best' (likelihood profiling) and '.marg' (likelihood sampling)
  likelihood.all_params['sn0'].update(derived='.auto')
  # Now the likelihood has for varied parameters (no sn0)
  likelihood.varied_params  # b1, df, dm, qiso, qap

One can reparameterize the whole likelihood as:

.. code-block:: python

  likelihood.all_params['qpar'].update(derived='{qiso} * {qap}**(2. / 3.)')
  likelihood.all_params['qper'].update(derived='{qiso} * {qap}**(- 1. / 3.)')
  # Then add qiso, qap to the parameter collection
  likelihood.all_params['qiso'] = {'prior': {'limits': [0.9, 1.1]}, 'latex': 'q_{\mathrm{iso}}'}
  likelihood.all_params['qap'] = {'prior': {'limits': [0.9, 1.1]}, 'latex': 'q_{\mathrm{ap}}'}
  # Now the likelihood has for varied parameters
  likelihood.varied_params  # b1, sn0, df, dm, qiso, qap

(a reparameterization we could have achieved in this particular case by passing ``apmode='qparqper'`` to ``ShapeFitPowerSpectrumTemplate``)


Bindings
--------

Now we have our likelihood, we can bind it to external cosmological inference codes (montepython, cosmosis, cobaya).

.. code-block:: python

    # Let's recap the likelihood definition in this function
    def MyLikelihood():

      from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles
      from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
      from desilike.likelihoods import ObservablesGaussianLikelihood

      # 'external' means "get primordial quantities from external source, e.g. cobaya
      template = DirectPowerSpectrumTemplate(z=1.1, cosmo='external')
      theory = KaiserTracerPowerSpectrumMultipoles(template=template)
      observable = TracerPowerSpectrumMultipolesObservable(data={'b1': 1.2}, covariance=None,
                                                           klim={0: [0.01, 0.2, 0.005], 2: [0.01, 0.2, 0.005]}, theory=theory)
      footprint = BoxFootprint(volume=1e9, nbar=1e-3)
      covariance = ObservablesCovarianceMatrix(observables=observable, footprints=footprint)
      cov = covariance(b1=1.2)
      return ObservablesGaussianLikelihood(observables=observable, covariance=cov)


    from desilike import setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator

    setup_logging('info')
    # Pass the function above to the generators, that will write the necessary files to import it as an external likelihood
    # in cobaya, cosmosis, montepython
    CobayaLikelihoodGenerator()(MyLikelihood)
    CosmoSISLikelihoodGenerator()(MyLikelihood)
    MontePythonLikelihoodGenerator()(MyLikelihood)


.. note::

  All the calculation below (emulation, profiling, sampling) benefits from MPI parallelization;
  just run the code with multiple MPI processes.


Emulators
---------

Had we chosen a slower theory model, e.g. :class:`~desilike.theories.galaxy_clustering.full_shape.LPTVelocileptorsTracerPowerSpectrumMultipoles`,
we would probably have wanted to emulate it, with:

- Taylor expansion, up to a given order, with :class:`~desilike.emulators.TaylorEmulatorEngine`
- Neural net (multilayer perceptron), with :class:`~desilike.emulators.MLPEmulatorEngine`

See also the base emulator class, :class:`~desilike.emulators.Emulator`.

.. code-block:: python

  from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles

  theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=0.8))

  from desilike.emulators import Emulator, TaylorEmulatorEngine, EmulatedCalculator

  # Let's emulate the perturbation theory part (.pt) by performing a Taylor expansion of order 3
  emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=3))
  emulator.set_samples()  # evaluate the theory derivatives (with jax auto-differentiation if possible, else finite differentiation)
  emulator.fit()  # set Taylor expansion

  # Emulator can be saved with:
  emulator.save('emulator.npy')
  # And reloaded with:
  pt = EmulatedCalculator.load('emulator.npy')

  theory.init.update(pt=pt)
  # Now the theory will run much faster!
  theory(logA=3.)


.. _user-getting-started-fisher:

Fisher
------

We provide a routine for Fisher estimation.

.. code-block:: python

  from desilike import Fisher

  fisher = Fisher(likelihood)
  # Estimate Fisher (precision) matrix at b1=2, using jax auto-differentiation where possible, else finite differentiation (with step :attr:`Parameter.delta`)
  fish = fisher(b1=2.)
  # To sum independent likelihood's Fisher matrices:
  # fish1 + fish2
  # To get covariance matrix
  covariance = fish.covariance()

See :class:`~desilike.parameter.ParameterPrecision` and :class:`~desilike.parameter.ParameterCovariance` to know more about precision and covariance
data classes.

Profilers
---------

Because we may want to test cosmological inference in-place (without resorting to montepython, cosmosis or cobaya),
we provide wrapping for some profilers and samplers.

Profilers currently available are:
- `minuit <https://github.com/scikit-hep/iminuit>`_, used by the high-energy physics community, with :class:`~desilike.profilers.MinuitProfiler`
- `bobyqa <https://github.com/numericalalgorithmsgroup/pybobyqa>`_, with :class:`~desilike.profilers.BOBYQAProfiler`
- `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_, with :class:`~desilike.profilers.ScipyProfiler`


These can be used with e.g.:

.. code-block:: python

  from desilike.profilers import MinuitProfiler

  profiler = MinuitProfiler(likelihood)  # optinally, provide save_fn = 'profiles.npy' to save profiles to save_fn
  profiles = profiler.maximize(niterations=5)
  profiles = profiler.interval(params=['b1'])
  # To print relevant information
  if profiler.mpicomm.rank == 0:
    print(profiles.to_stats(tablefmt='pretty'))
    # If you saved profiles to 'profiles.npy', you can load the object with:
    # from desilike.samples import Profiles
    # profiles = Profiles.load('profiles.npy')

See :class:`~desilike.samples.profiles.Profiles` to know more about this data class.


Samplers
--------

Samplers currently available are:
- `Antony Lewis <https://github.com/CobayaSampler/cobaya/tree/master/cobaya/samplers/mcmc>`_ MCMC sampler, with :class:`~desilike.samplers.MCMCSampler`
- `emcee <https://github.com/dfm/emcee>`_ ensemble sampler, with :class:`~desilike.samplers.EmceeSampler`
- `zeus <https://github.com/minaskar/zeus>`_ ensemble slicing sampler, with :class:`~desilike.samplers.ZeusSampler`
- `pocomc <https://github.com/minaskar/pocomc>`_ pre-conditioned Monte-Carlo sampler, with :class:`~desilike.samplers.PocoMCSampler`
- `dynesty <https://github.com/joshspeagle/dynesty>`_ nested sampler, with :class:`~desilike.samplers.DynamicDynestySampler`
- `polychord <https://github.com/PolyChord/PolyChordLite>`_ nested sampler, with :class:`~desilike.samplers.NestedSampler`


These can be used with e.g.:

.. code-block:: python

  from desilike.samplers import EmceeSampler

  sampler = EmceeSampler(likelihood, chains=4)  # optinally, provide save_fn = 'chain_*.npy' to save chains to save_fn
  chains = sampler.run(check={'max_eigen_gr': 0.05})  # run until Gelman-Rubin criterion < 0.05
  # To print relevant information
  if sampler.mpicomm.rank == 0:  # chains only available on rank 0
    chain = chains[0].concatenate([chain.remove_burnin(0.5)[::10] for chain in chains])  # removing burnin and thinning
    print(chain.to_stats(tablefmt='pretty'))
    # If you saved chains to 'chain_*.npy', you can load them with:
    # from desilike.samples import Chain
    # chain = Chain.concatenate([Chain.load('chain_{:d}.npy'.format(i)).remove_burnin(0.5)[::10] for i in range(4)])  # remove burnin and thin by a factor 10

See :class:`~desilike.samples.chain.Chain` to know more about this data class.


MPI
---
All costly operations, e.g. emulation, Fisher (computation of numerical derivatives), profiling, sampling, are MPI-parallelized.
Look at the emulator, profiler and sampler documentation for more information.
In a nutshell, the code in the above sections (Emulators, Fisher, Profilers, Samplers) can be run without any change on several MPI processes;
if written in a script ``yourscript.py``, it can be launched with e.g. 8 processes (or more, depending on your setup): ``mpiexec -np 8 yourscript.py``.
On supercomputers using Slurm workload manager, one may have to use ``srun -n`` instead of ``mpiexec -np``.