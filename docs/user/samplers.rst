.. _user-samplers:

Samplers
========

As a stand-alone cosmological inference framework, `desilike` provides easy access to a variety of Bayesian inference algorithms, commonly referred to as "samplers". To streamline the process, `desilike` offers a unified interface to a wide range of samplers.

`desilike` distinguishes between three types of samplers: **static samplers**, **population samplers**, and **Markov chain samplers**.

* **Static samplers** evaluate parts of parameter space and do not adapt to the actual likelihood surface. The parameter combinations to explore are set at the beginning of the algorithm.
* **Population samplers**, including nested samplers, importance samplers, and Sequential Monte Carlo algorithms, utilize large populations of previously evaluated parameter combinations to inform which part of parameter space to explore next.
* **Markov chain Monte Carlo (MCMC)** samplers produce a Markov chain of parameter combinations where the stable distribution approaches the Bayesian posterior. Unlike population samplers, the next state of the sampler depends only on the current state (i.e., parameter combination).

With the exception of the static importance sampler (:class:`desilike.samplers.ImportanceSampler`), all samplers can be run when providing just a likelihood using the following unifying syntax.

.. code-block:: python

  import numpy as np

  import desilike.samplers as samplers
  from desilike.likelihoods import BaseGaussianLikelihood

  class Likelihood(BaseGaussianLikelihood):

     def calculate(self, **kwargs):
          self.flattheory = np.array([kwargs[name] for name in
                                    self.varied_params.names()])
          super().calculate()

  likelihood = Likelihood(np.array([0.4, 0.6]), covariance=np.eye(2) * 0.01)
  likelihood.init.params = dict(
      a=dict(prior=dict(dist='uniform', limits=[0, 1])),
      b=dict(prior=dict(dist='uniform', limits=[0, 1])))

  sampler = samplers.MetropolisHastingsSampler(likelihood)
  posterior = sampler.run()
  print(posterior.mean(params=['a', 'b']))

In the above example, we chose the Metropolis-Hastings MCMC sampler, but the same code will work with virtually all other samplers, providing a common entry point to Bayesian sampling.

Static Samplers
---------------

Static samplers (:class:`desilike.samplers.base.StaticSampler`) do not adapt to the likelihood surface. As such, they are primarily used for low-dimensional problems or, in the case of importance sampling, when we already have a distribution that closely resembles the posterior distribution. The following samplers are supported:

* Grid Sampler (:class:`desilike.samplers.StaticSampler`): This simple sampler evaluates the likelihood on a regular grid. The user provides the number of grid points per dimension. This sampler should not be used for high-dimensional problems, as the total number of grid points grows exponentially with the dimensionality.
* Quasi-Monte Carlo (QMC) Sampler (:class:`desilike.samplers.QMCSampler`): This sampler is similar to the grid sampler but evaluates points on a non-regular "grid" derived from a `Quasi-Monte Carlo <https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method>`_ method. Compared to purely random uniform sampling, QMC-derived points have lower variance. This sampler can be very effective for low-dimensional problems, provided the volume of the typical set is not much smaller than the prior. `desilike` supports Halton sequences, Latin hypercube sampling, Sobol sequences (all implemented via `SciPy <https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>`_), and Kronecker sequences.
* Importance Sampler (:class:`desilike.samplers.ImportanceSampler`): The importance sampler adjusts the weights of an existing posterior sample based on a new likelihood. The outcome is either:
  * ``mode='resample'``: Resamples the existing posterior samples based on the new likelihood, resulting in a new posterior distribution that is independent of the old posterior.
  * ``mode='combine'``: Combines the new likelihood with the old posterior by adjusting the weights of the existing samples. The resulting posterior is the product of the old posterior and the new likelihood (i.e., the product of the prior, the old likelihood, and the new likelihood). This mode is useful when you want to update your posterior based on new observations.

Population Samplers
-------------------

Population samplers are dynamic samplers that utilize most or all previously evaluated points in parameter space to guide which regions to explore next. Unlike MCMC samplers, the resulting posterior samples are typically weighted. These samplers are also the only ones supported by `desilike` capable of computing Bayesian evidence.

`desilike` provides interfaces to a wide range of population samplers, each with its own terminology, convergence criteria, and other specifics. Unless unifying the interface improves the user experience, `desilike` allows users to interact directly with the underlying packages. For detailed information on how `desilike` interfaces with each sampler, please refer to their `run_sampler` methods.

The following population samplers are supported:

* `Dynesty` (:class:`desilike.samplers.DynestySampler`, `Paper <https://doi.org/10.1093/mnras/staa278>`_, `Code <https://github.com/joshspeagle/dynesty>`_): A pure-Python dynamic nested sampling code. The slice sampling mode in this sampler is similar to that of `PolyChord`.
* `nautilus` (:class:`desilike.samplers.NautilusSampler`, `Paper <https://doi.org/10.1093/mnras/stad2441>`_, `Code <https://github.com/johannesulf/nautilus>`_): A pure-Python importance nested sampling code. This is an evolution of `MultiNest`'s importance nested sampling (INS) mode and incorporates neural networks to improve sampling efficiency.
* `pocoMC` (:class:`desilike.samplers.PocoMCSampler`, `Paper 1 <https://doi.org/10.21105/joss.04634>`_, `Paper 2 <https://doi.org/10.1093/mnras/stac2272>`_, `Code <https://github.com/minaskar/pocomc>`_): An implementation of preconditioned Monte Carlo (PMC), which is an extension of Sequential Monte Carlo. This code leverages normalizing flows to enhance sampling efficiency.

