.. title:: desilike docs

**************************************
Welcome to desilike's documentation!
**************************************

.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  api/api

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/documentation
  developer/tests
  developer/contributing
  developer/changes

.. toctree::
  :hidden:

************
Introduction
************

**desilike** is an attempt to provide a common framework for writing DESI likelihoods,
that can be imported in common cosmological inference codes (`cobaya <https://github.com/CobayaSampler/cobaya>`_,
`cosmosis <https://github.com/joezuntz/cosmosis>`_, `montepython <https://github.com/brinckmann/montepython_public>`_).

Stricly required, to define likelihoods (including theories) and import them in these inference codes, are:

  - theories
  - observables
  - likelihoods
  - bindings
  - (optionally: emulators)

Directories samples, samplers and profilers are provided for self-contained sampling / profiling of provided likelihoods, as discussed below.

**desilike** provides:

* tools to emulate calculators (in-place) at any step
* profilers
* samplers
* Fisher analysis tools

In detail, just as cosmological inference codes, **desilike** includes:

* an advanced parameterization infrastructure (priors, reference distributions, derived parameters, etc.)
* speed hierarchy between various parameters, exploited in some samplers (MCMCSampler, PolychordSampler)
* primordial cosmology computations with `cosmoprimo <https://github.com/cosmodesi/cosmoprimo>`_
* tools to install external data/packages
* convergence diagnostics
* MPI support to run several chains in parallel
* for convenience, some usual external likelihoods (Planck2018, SN, ...) --- though we emphasize full cosmological inference is not desilike's purpose

In addition:

* consistent parameterization between Boltzmann codes (through `cosmoprimo <https://github.com/cosmodesi/cosmoprimo>`_)
* transparent namespace scheme for parameters, to avoid keeping track of all possible parameter (base) names among all various calculators (and likelihoods)
* in-place emulation, to considerably speed-up inference, while allowing for easy checks of the emulation strategy at the posterior level (what we care about!)
* general differentiation routine (with jax for parameters that support automatic differentiation, else automatic finite difference)
  for e.g. Fisher analysis, emulators (Taylor expansion), analytic marginalization (next: gradient to be used in profilers and samplers)
* double parallelization level (several chains, and several processes per chain), for all samplers
* more likelihood profiling tools (1D and 2D profiles in addition to likelihood/posterior maximization)
* the possibility to save any array (of any shape) quantity (and derivative) to disk, no matter the sampler/profiler under use,
to facilitate debugging, set up model template bases, build emulators within the relevant parameter space, etc.

Development so far has focused on galaxy clustering / compression techniques.


Quick start-up
==============

For a quick start-up, see `notebooks <https://github.com/cosmodesi/desilike/blob/main/nb>`_.


Changelog
=========

* :doc:`developer/changes`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
