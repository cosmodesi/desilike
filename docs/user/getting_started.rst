.. _user-getting-started:


Getting started
===============

In this page we describe **desilike**'s basics with a practical example.
Further examples can be found in the provided `notebooks <https://github.com/cosmodesi/desilike/blob/main/nb>`_.

**desilike** provides a framework to specify DESI likelihoods and connect them to
profilers, samplers, and external codes.

Pipeline basics
---------------

**desilike** builds pipelines of *calculators* — Python objects whose
``__init__`` defines parameters and dependencies, and whose ``__call__`` runs the
computation.  Once the graph of calculators is assembled, :func:`~desilike.base.compile`
traces the dependencies, assigns a topological evaluation order, and returns an
executable :class:`~desilike.base.CompiledGraph`:

.. code-block:: python

  from desilike.base import compile

  pipe = compile(likelihood)       # compile the calculator graph
  params = {p.name: p.value for p in pipe.params}  # collect default values
  pipe(params)                     # evaluate the pipeline at these parameters

The compiled graph exposes ``pipe.params`` (a :class:`~desilike.parameter.VariableCollection`
of all parameters), and calling it with a ``dict`` re-evaluates the full graph.


JAX transforms and derived parameters
--------------------------------------

The compiled graph is a plain callable compatible with all JAX transforms.

**JIT compilation**:

.. code-block:: python

  import jax

  logL = jax.jit(pipe)(params)

The first call traces and compiles the graph; subsequent calls with the same
parameter shapes hit the compiled kernel directly.

**Gradient**:

.. code-block:: python

  grad = jax.grad(pipe)(params)   # dict of partial derivatives

For parameters that feed only JAX calculators the gradient is exact (forward-mode AD).
For parameters that feed a non-JAX (``_is_external = True``) calculator, the gradient
uses finite differences; step size and accuracy order are set per-parameter via
``param.fd_eps`` and ``param.fd_acc``.

**Vectorised map (vmap)**:

.. code-block:: python

  n = 200
  batch = {**params,
           'b1': jnp.linspace(1., 3., n),
           'sn0': jnp.zeros(n)}
  logL_batch = jax.vmap(pipe)(batch)   # shape (n,)

Every parameter in the batch dict must have a leading axis of the same size ``n``;
scalar parameters must be broadcast to ``jnp.full(n, value)`` first.

**Derived parameters**:

Some parameters are declared as derived (``derived=True``) and written by
``__call__`` rather than read as inputs (e.g. a chi-squared diagnostic set
inside a likelihood).  To retrieve their values alongside the return value,
pass ``return_derived=True``:

.. code-block:: python

  logL, derived = pipe(params, return_derived=True)
  # derived: dict mapping derived parameter name → value
  print(derived['chi2'])

Under ``jax.jit`` or ``jax.vmap``, pass ``return_derived`` as a closed-over
constant (not a traced kwarg):

.. code-block:: python

  logL, derived = jax.jit(lambda p: pipe(p, return_derived=True))(params)


Clustering likelihood
---------------------

We now build a complete galaxy-clustering likelihood step by step.

Template
~~~~~~~~

First choose a power-spectrum template — i.e. how the linear matter power
spectrum is parameterized.  Several options are available:

- :class:`~desilike.theories.galaxy_clustering.ShapeFitSpectrum2Template`:
  ShapeFit parameterization (:math:`q_\parallel`, :math:`q_\perp`, :math:`df`, :math:`dm`).
- :class:`~desilike.theories.galaxy_clustering.BAOSpectrum2Template`:
  BAO-specific parameterization with a fiducial cosmology.
- :class:`~desilike.theories.galaxy_clustering.DirectSpectrum2Template`:
  Direct base cosmological parameters.

.. code-block:: python

  from desilike.theories.galaxy_clustering import ShapeFitSpectrum2Template

  template = ShapeFitSpectrum2Template(z=0.8)  # effective redshift

.. note::

  ``help(calculator)`` for any calculator provides useful information,
  in particular the constructor arguments.

  Any calculator, profiler, sampler, etc. can be installed with :class:`~desilike.install.Installer`.

Theory
~~~~~~

Next, specify the theory model.  The most notable options are:

- Kaiser model: :class:`~desilike.theories.galaxy_clustering.KaiserTracerSpectrum2Poles`
  or :class:`~desilike.theories.galaxy_clustering.KaiserTracerCorrelation2Poles`
- `velocileptors <https://github.com/sfschen/velocileptors>`_ (LPT_RSD):
  :class:`~desilike.theories.galaxy_clustering.LPTVelocileptorsTracerSpectrum2Poles`
  or :class:`~desilike.theories.galaxy_clustering.LPTVelocileptorsTracerCorrelation2Poles`
- `pybird <https://github.com/pierrexyz/pybird>`_:
  :class:`~desilike.theories.galaxy_clustering.PyBirdTracerSpectrum2Poles`
  or :class:`~desilike.theories.galaxy_clustering.PyBirdTracerCorrelation2Poles`
- `FOLPS-D <https://github.com/cosmodesi/FolpsD>`_:
  :class:`~desilike.theories.galaxy_clustering.FOLPSTracerSpectrum2Poles`
  or :class:`~desilike.theories.galaxy_clustering.FOLPSTracerCorrelation2Poles`
- Empirical BAO model:
  :class:`~desilike.theories.galaxy_clustering.DampedBAOWigglesTracerSpectrum2Poles`
  or :class:`~desilike.theories.galaxy_clustering.ResummedBAOWigglesTracerSpectrum2Poles`
- Primordial non-Gaussianity:
  :class:`~desilike.theories.galaxy_clustering.PNGTracerSpectrum2Poles`

See :mod:`~desilike.theories.galaxy_clustering.full_shape` for all full-shape
models and :mod:`~desilike.theories.galaxy_clustering.bao` for all BAO models.

.. code-block:: python

  import numpy as np
  from desilike.theories.galaxy_clustering import KaiserTracerSpectrum2Poles

  k = np.linspace(0.01, 0.2, 101)
  ells = (0, 2)
  # Or LPTVelocileptorsTracerSpectrum2Poles, PyBirdTracerSpectrum2Poles, etc.
  theory = KaiserTracerSpectrum2Poles(k=k, ells=ells, template=template)

One can update the template (or any constructor argument) with ``calculator.init.update(...)``:

.. code-block:: python

  theory.update(template=ShapeFitSpectrum2Template(z=1.))

Observable
~~~~~~~~~~

Wrap the theory in an observable that compares it to data:

- Power spectrum multipoles: :class:`~desilike.observables.Spectrum2PolesObservable`
- Correlation function multipoles: :class:`~desilike.observables.Correlation2PolesObservable`

.. code-block:: python

  from desilike.observables import Spectrum2PolesObservable

  # data: flat array, dict of params (to evaluate the theory as mock data), or None
  # covariance: 2-D array, 1-D diagonal, or None
  obs = Spectrum2PolesObservable(data={'b1': 1.2},
                                 covariance=None,
                                 theory=theory,
                                 k=k,
                                 ells=ells)

Likelihood
~~~~~~~~~~

Now define the likelihood.  The covariance can be provided to the observable,
to the likelihood, or estimated on-the-fly from mocks:

.. code-block:: python

  from desilike.likelihoods import ObservablesGaussianLikelihood

  n = len(ells) * len(k)
  cov = np.diag(np.full(n, 1e4))   # diagonal covariance for illustration

  likelihood = ObservablesGaussianLikelihood(observables=[obs], covariance=cov)

To sum independent likelihoods:

.. code-block:: python

  from desilike.likelihoods import SumLikelihood

  combined = SumLikelihood(likelihoods=[likelihood1, likelihood2])


Compile and evaluate
~~~~~~~~~~~~~~~~~~~~

Compile the graph and call it with a parameter dict:

.. code-block:: python

  from desilike.base import compile

  pipe = compile(likelihood)

  # Evaluate at default parameter values
  params = {p.name: p.value for p in pipe.params}
  pipe(params)

  # Access theory outputs after evaluation
  theory.poles   # multipoles of the power spectrum


Parameters
----------

All parameters in the pipeline are accessible through ``pipe.params``:

.. code-block:: python

  pipe.params                          # all parameters
  pipe.params.select(varied=True)      # only varied parameters
  pipe.params.select(basename='q*')    # glob-filter on basename

To access parameters of a single calculator before compilation:

.. code-block:: python

  from desilike import get_params
  params = get_params(theory)          # b1, sn0 (declared by this calculator)

Parameters are :class:`~desilike.parameter.Parameter` objects; their main attributes are:

- ``basename``, ``name`` — short and fully-qualified name
- ``value`` — default value
- ``prior`` — prior distribution
- ``ref`` — reference distribution (used for initial samples)
- ``delta`` — finite-difference step
- ``fixed`` — whether the parameter is fixed
- ``latex`` — LaTeX string

They can be updated as:

.. code-block:: python

  # Tighten the prior on df
  params['df'].update(prior={'dist': 'norm', 'loc': 1., 'scale': 2.})
  # Fix b1 = 2
  params['b1'].update(value=2., fixed=True)

Parameters can be analytically marginalized (useful for linear nuisance parameters):

.. code-block:: python

  # 'marg': Gaussian marginalization; 'best': set at best-fit (= Jeffreys prior)
  params['sn0'].update(derived='marg')

Or reparameterized via expressions:

.. code-block:: python

  params['qpar'].update(derived='qiso * qap**(2. / 3.)')
  params['qper'].update(derived='qiso * qap**(-1. / 3.)')
  params['qiso'].update(prior={'limits': [0.9, 1.1]}, latex=r'q_\mathrm{iso}')
  params['qap'].update(prior={'limits': [0.9, 1.1]}, latex=r'q_\mathrm{ap}')


Posterior
~~~~~~~~~

To run a Bayesian analysis, combine the likelihood with a
:class:`~desilike.base.Prior` inside a :class:`~desilike.base.Posterior`:

.. code-block:: python

  from desilike import get_params
  from desilike.base import Prior, Posterior

  # Prior: pass Parameter objects (keyword args) or a VariableCollection
  # (positional arg).  Parameters with no prior distribution contribute 0.
  prior = Prior(get_params(likelihood))

  posterior = Posterior(likelihood, prior)
  pipe = compile(posterior)

  params = {p.name: p.value for p in pipe.params}
  pipe(params)   # returns log-posterior = log-likelihood + log-prior

The prior is evaluated first; if the log-prior is :math:`-\infty` (any
parameter outside its support), the likelihood is skipped.  The scalar
log-posterior is the return value; ``loglikelihood``, ``logprior``, and
``logposterior`` are also available as derived parameters:

.. code-block:: python

  logP, derived = pipe(params, return_derived=True)
  print(derived['loglikelihood'], derived['logprior'])

**Custom prior with extra conditions**

Subclass :class:`~desilike.base.Prior` to add constraints beyond standard
per-parameter distributions.  Call ``super().__call__()`` to get the standard
log-prior, then apply your condition:

.. code-block:: python

  import jax.numpy as jnp
  from desilike.base import Prior, Posterior

  class CustomPrior(Prior):
      """Hard constraint w0 + wa < 0, on top of individual parameter priors."""

      def __call__(self):
          logpdf = super().__call__()
          w0, wa = self.params['w0_fld'], self.params['wa_fld']
          self.logpdf = jnp.where(w0 + wa < 0., logpdf, -jnp.inf)
          return self.logpdf

  params = get_params(likelihood)
  posterior = Posterior(likelihood, prior=CustomPrior(params))
  pipe = compile(posterior)


**Reparametrization inside the prior**

A custom prior can also override parameter values before the likelihood runs —
for example to tie two parameters together.  Set the parameter's ``.value``
inside ``__call__``; :class:`~desilike.base.Posterior` automatically picks up
the new value and passes it to the likelihood:

.. code-block:: python

  class ReparPrior(Prior):
      """Force omega_m = A at every likelihood evaluation."""

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self._omega_m = self.params['omega_m']
          self._omega_m.update(derived=True)  # such that it isn't listed as input parameter
          self._A = self.params['A']

      def __call__(self):
          logpdf = super().__call__()
          self._omega_m.value = self._A.value   # override before likelihood
          return logpdf

  params = get_params(likelihood)
  posterior = Posterior(likelihood, prior=ReparPrior(params))
  pipe = compile(posterior)

The reparametrization is fully differentiable under ``jax.jit`` and
``jax.grad``.



Emulators
---------

For slower theory models, such as :class:`~desilike.theories.galaxy_clustering.LPTVelocileptorsTracerSpectrum2Poles`,
the perturbation-theory (PT) sub-calculator can be emulated with a Taylor
expansion using :class:`~desilike.TaylorEmulator`:

.. code-block:: python

  from desilike import compile, TaylorEmulator
  from desilike.theories.galaxy_clustering import (
      ShapeFitSpectrum2Template, KaiserPTSpectrum2Poles, KaiserTracerSpectrum2Poles)

  k = np.linspace(0.01, 0.2, 101)
  ells = (0, 2)
  template = ShapeFitSpectrum2Template(z=0.8)
  pt = KaiserPTSpectrum2Poles(k=k, ells=ells)
  pt.update(template=template)

  # Compile just the PT sub-graph, then emulate it
  pt_pipe = compile(pt)
  emulator = TaylorEmulator(pt_pipe, order=3)
  emulator.fit()          # compute Taylor coefficients (uses JAX auto-differentiation where possible)
  emulated_pt = emulator.to_calculator()

  # Emulator can be saved and reloaded
  emulator.write('emulator.h5')
  emulator2 = TaylorEmulator.read('emulator.h5')

  # Drop the emulated PT into the full tracer pipeline
  theory = KaiserTracerSpectrum2Poles(k=k, ells=ells, pt=emulated_pt)
  pipe = compile(theory)   # now runs much faster!


Derivatives
-----------

:func:`~desilike.base.differentiate` builds an arbitrary-order partial derivative
of a compiled graph, mixing exact JAX forward-mode AD (for JAX-native nodes) with
direct finite-difference stencils (for non-JAX nodes) at linear cost in the
derivative order:

.. code-block:: python

  from desilike.base import differentiate

  pipe = compile(likelihood)

  # First derivative w.r.t. b1
  grad_fn = differentiate(pipe, {'b1': 1})
  g = grad_fn()                          # evaluate at default param values
  g = grad_fn({'b1': 1.5})              # evaluate at a specific point

  # Second derivative w.r.t. qpar
  d2_fn = differentiate(pipe, {'qpar': 2})

  # Mixed partial ∂²/(∂b1 ∂sn0)
  cross_fn = differentiate(pipe, {'b1': 1, 'sn0': 1})

  # Override FD step and accuracy for non-JAX parameters
  d_fn = differentiate(pipe, {'omega_m': 1}, fd_eps=1e-4, fd_acc=4)

  # Retrieve derived parameters simultaneously
  dval, d_derived = differentiate(pipe, {'b1': 1})(return_derived=True)

The returned callable accepts the same ``params`` dict and ``**kwargs`` as
``pipe``, and is compatible with ``jax.jit``.


Parallel map
------------

:func:`~desilike.base.pmap` is a distributed ``vmap``: it maps a function over
the leading batch axis of its inputs, distributing the work across MPI ranks and/or
local JAX devices.

.. code-block:: python

  from desilike.base import compile, pmap

  pipe = compile(likelihood)

  n = 1000
  batch = {p.name: jnp.full(n, p.value) for p in pipe.params}
  batch['b1'] = jnp.linspace(0.5, 3., n)

  logL_batch = pmap(pipe)(batch)          # shape (n,), distributed across ranks/devices

Three backends are available via the ``backend`` keyword:

- ``'jax'`` — shard across local JAX devices only.
- ``'mpi'`` — split across MPI ranks; each rank uses a single device.
- ``'mpi_and_jax'`` *(default)* — MPI outer split + JAX inner sharding.

On a single-device, single-rank machine all three reduce to ``jax.vmap``.

``pmap`` wraps any callable, not just compiled graphs:

.. code-block:: python

  grad_batch = pmap(jax.grad(pipe))(batch)   # batch of gradients


Profilers
---------

In-place profiling without an external code.  Available kernels:

- `iminuit <https://github.com/scikit-hep/iminuit>`_: :class:`~desilike.profilers.Minuit`
- `pybobyqa <https://github.com/numericalalgorithmsgroup/pybobyqa>`_: :class:`~desilike.profilers.BOBYQA`
- `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_: :class:`~desilike.profilers.Scipy`
- `optax <https://github.com/google-deepmind/optax>`_ (JAX gradient-based): :class:`~desilike.profilers.Optax`

.. code-block:: python

  from desilike.profilers import Profiler, Minuit

  pipe = compile(posterior)
  profiler = Profiler(pipe, kernel=Minuit(), output_dir='profiles/')

  profiler.maximize(niterations=5)
  profiler.interval(params=['b1'])

  if profiler.mpicomm.rank == 0:
    print(profiler.profiles.to_stats(tablefmt='pretty'))
    # profiles saved in output_dir; reload with:
    # from desilike.samples import Profiles
    # profiles = Profiles.read('profiles/profiles.h5')

See :class:`~desilike.samples.profiles.Profiles` for this data class.


Samplers
--------

In-place sampling.  Pass a kernel to :func:`~desilike.samplers.Sampler`:

- Metropolis-Hastings: :class:`~desilike.samplers.MH`
- emcee: :class:`~desilike.samplers.Emcee`
- zeus: :class:`~desilike.samplers.Zeus`
- blackjax (HMC/NUTS/MCLMC): :class:`~desilike.samplers.BlackjaxHMC`, :class:`~desilike.samplers.BlackjaxNUTS`, :class:`~desilike.samplers.BlackjaxMCLMC`
- numpyro (NUTS/HMC/BarkerMH): :class:`~desilike.samplers.NumpyroNUTS`, :class:`~desilike.samplers.NumpyroHMC`, :class:`~desilike.samplers.NumpyroBarkerMH`
- dynesty (nested): :class:`~desilike.samplers.Dynesty`
- nautilus (importance nested): :class:`~desilike.samplers.Nautilus`
- pocomc (SMC): :class:`~desilike.samplers.PocoMC`

.. code-block:: python

  from desilike.samplers import Sampler, Emcee

  pipe = compile(posterior)
  sampler = Sampler(pipe, kernel=Emcee(nwalkers=32), nparallel=4, output_dir='chains/')

  chains = sampler.run(gelman_rubin=1.05)  # run until Gelman-Rubin < 1.05

  if sampler.mpicomm.rank == 0:  # chains only available on rank 0
    chain = chains[0].concatenate([c.remove_burnin(0.5)[::10] for c in chains])
    print(chain.to_stats(tablefmt='pretty'))
    # chains saved in output_dir can be reloaded; see MCSamples.read()

See :class:`~desilike.samples.chain.MCSamples` for this data class.


MPI
---
Just add the top of your script:

.. code-block:: python

  import desilike
  desilike.distributed.initialize()

All costly operations (profiling, sampling) are MPI-parallelized.
To run across multiple processes:

.. code-block:: bash

  mpiexec -np 8 yourscript.py

On Slurm clusters, use ``srun -n`` instead of ``mpiexec -np``.