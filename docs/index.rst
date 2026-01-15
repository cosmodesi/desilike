.. title:: index

####################################
Welcome to desilike's Documentation!
####################################

**desilike** is a modern cosmological inference framework used by the Dark Energy Spectroscopic Instrument (DESI) experiment. While initially designed for DESI, it is a stand-alone inference code that can also be used with other cosmological datasets.  Conversely, `desilike` also supports exporting DESI likelihoods for use in other inference codes such as `Cobaya <https://github.com/CobayaSampler/cobaya>`_, `CosmoSIS <https://github.com/joezuntz/cosmosis>`_, or `MontePython <https://github.com/brinckmann/montepython_public>`_.

Key features:

* Advanced parameterization infrastructure (priors, reference distributions, derived parameters, etc.)
* MPI support for parallel posterior computations
* Consistent parameterization between Boltzmann codes (via `cosmoprimo <https://github.com/cosmodesi/cosmoprimo>`_)
* Transparent namespace scheme for parameters
* In-place emulation for faster inference
* General differentiation routines (with `JAX` for automatic differentiation)

*****************
Table of Contents
*****************

.. toctree::
  :maxdepth: 1
  :caption: User Documentation

  user/building
  user/getting_started

.. toctree::
  :maxdepth: 1
  :caption: API Reference

  api/core
  api/theories
  api/observables
  api/likelihoods
  api/bindings
  api/profilers
  api/samplers
  api/samples
  api/emulators

.. toctree::
  :maxdepth: 1
  :caption: Development

  dev/introduction
  dev/documentation
  dev/tests
