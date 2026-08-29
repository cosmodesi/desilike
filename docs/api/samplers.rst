Samplers
========

The unified entry point is the :func:`~desilike.samplers.Sampler` factory, which
selects the right infrastructure class based on the kernel passed to it.  Kernels
are plain configuration objects; the sampler base classes handle MPI, chain I/O,
and convergence checking.

Factory
-------

.. autosummary::
  :toctree: _autosummary

  desilike.samplers.Sampler

Static Kernels
--------------

.. autosummary::
  :toctree: _autosummary
  :template: class.rst

  desilike.samplers.Grid
  desilike.samplers.QMC
  desilike.samplers.Importance

Population Kernels
------------------

.. autosummary::
  :toctree: _autosummary
  :template: class.rst

  desilike.samplers.Dynesty
  desilike.samplers.Nautilus
  desilike.samplers.PocoMC
  desilike.samplers.SMC

Proposals
---------

Population kernels can start from a *proposal* instead of the prior -- the beta = 0
distribution of their tempered path. The posterior is unchanged whatever the proposal;
what shrinks is the distance the sampler has to anneal over. Pass one as
``Sampler(..., proposal=...)``; a :class:`~desilike.samples.Covariance` is accepted too,
and is wrapped into a :class:`~desilike.samplers.GaussianProposal`.

.. autosummary::
  :toctree: _autosummary
  :template: class.rst

  desilike.samplers.PriorProposal
  desilike.samplers.SamplesProposal
  desilike.samplers.GaussianProposal
  desilike.samplers.ProductProposal

MCMC Kernels
------------

.. autosummary::
  :toctree: _autosummary
  :template: class.rst

  desilike.samplers.MH
  desilike.samplers.BlackjaxHMC
  desilike.samplers.BlackjaxNUTS
  desilike.samplers.BlackjaxMCLMC
  desilike.samplers.NumpyroNUTS
  desilike.samplers.NumpyroHMC
  desilike.samplers.NumpyroBarkerMH
  desilike.samplers.NumpyroSA

Ensemble Kernels
----------------

.. autosummary::
  :toctree: _autosummary
  :template: class.rst

  desilike.samplers.Emcee
  desilike.samplers.Zeus

Base Classes
------------

.. autosummary::
  :toctree: _autosummary
  :template: class.rst

  desilike.samplers.base.MCMCSampler
  desilike.samplers.base.EnsembleSampler
  desilike.samplers.base.PopulationSampler
  desilike.samplers.base.StaticSampler
  desilike.samplers.base.Kernel
  desilike.samplers.base.PopulationKernel
  desilike.samplers.base.StaticKernel
  desilike.samplers.proposals.BaseProposal
