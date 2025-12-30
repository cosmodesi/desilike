.. _user-samplers:

Samplers
========

As a stand-alone cosmological inference framework, `desilike` provides easy access to a variety of Bayesian inference algorithms, commonly referred to as "samplers". To streamline the process, `desilike` offers a unified interface to a wide range of samplers.

Overview
--------

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
* Quasi-Monte Carlo (QMC) Sampler (:class:`desilike.samplers.QMCSampler`): This sampler is similar to the grid sampler but evaluates points on a non-regular "grid" derived from a `Quasi-Monte Carlo <https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method>`_ method. Compared to purely random uniform sampling, QMC-derived points have lower variance. This sampler can be very effective for low-dimensional problems, provided the typical set occupies a significant fraction of the prior volume. `desilike` supports Halton sequences, Latin hypercube sampling, Sobol sequences (all implemented via `SciPy <https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>`_), and Kronecker sequences.
* Importance Sampler (:class:`desilike.samplers.ImportanceSampler`): The importance sampler adjusts the weights of an existing posterior sample based on a new likelihood. The outcome is either:
  * ``mode='resample'``: Resamples the existing posterior samples based on the new likelihood, resulting in a new posterior distribution that is independent of the old posterior.
  * ``mode='combine'``: Combines the new likelihood with the old posterior by adjusting the weights of the existing samples. The resulting posterior is the product of the old posterior and the new likelihood (i.e., the product of the prior, the old likelihood, and the new likelihood). This mode is useful when you want to update your posterior based on new observations.

Population Samplers
-------------------

Population samplers (:class:`desilike.samplers.base.PopulationSampler`) are dynamic samplers that utilize most or all previously evaluated points in parameter space to guide which regions to explore next. Unlike MCMC samplers, the resulting posterior samples are typically weighted. These samplers are also the only ones supported by `desilike` capable of computing Bayesian evidence.

`desilike` provides interfaces to a wide range of population samplers, each with its own terminology, convergence criteria, and other specifics. Unless unifying the interface improves the user experience, `desilike` allows users to interact directly with the underlying packages. For detailed information on how `desilike` interfaces with each sampler, please refer to their `run_sampler` methods.

The following population samplers are supported:

* `Dynesty` (:class:`desilike.samplers.DynestySampler`, `Speagle (2020) <https://doi.org/10.1093/mnras/staa278>`_, `dynesty repo <https://github.com/joshspeagle/dynesty>`_): A pure-Python dynamic nested sampling code. The slice sampling mode in this sampler is similar to that of `PolyChord`.
* `nautilus` (:class:`desilike.samplers.NautilusSampler`, `Lange (2023) <https://doi.org/10.1093/mnras/stad2441>`_, `nautilus repo <https://github.com/johannesulf/nautilus>`_): A pure-Python importance nested sampling code. This is an evolution of `MultiNest`'s importance nested sampling (INS) mode and incorporates neural networks to improve sampling efficiency.
* `pocoMC` (:class:`desilike.samplers.PocoMCSampler`, `Karamanis et al. (2022a) <https://doi.org/10.1093/mnras/stac2272>`_, `Karamanis et al. (2022b) <https://doi.org/10.21105/joss.04634>`_, `pocoMC repo <https://github.com/minaskar/pocomc>`_): An implementation of preconditioned Monte Carlo (PMC), which is an extension of Sequential Monte Carlo. This code leverages normalizing flows to enhance sampling efficiency.

If you use any of these classes in your published work, please make sure to cite the corresponding papers.

MCMC Samplers
-------------

MCMC samplers (:class:`desilike.samplers.base.MarkovChainSampler`) approximate the posterior distribution by creating a Markov chain whose stationary distribution approaches the posterior. Unlike population samplers, all MCMC algorithms in `desilike` follow the same unified interface and differ only in how they advance the chain.

* Metropolis-Hastings MCMC (:class:`desilike.samplers.MetropolisHastingsSampler`): The classical MCMC algorithm. The version employed in `desilike` supports the fast-and-slow decomposition described in `Lewis (2013) <https://doi.org/10.1103/PhysRevD.87.103529>`_.
* Hamiltonian Monte-Carlo (HMC, :class:`desilike.samplers.HMCSampler`), No-U-Turn Sampler (NUTS, :class:`desilike.samplers.NUTSSampler`, `Hoffman & Gelman (2014) <https://jmlr.org/papers/v15/hoffman14a.html>`_), and Microcanonical Langevin Monte Carlo (MCLMC, :class:`desilike.samplers.MCLMCSampler`, `Robnik & Seljak (2024) <https://proceedings.mlr.press/v253/robnik24a.html>`_): Classes of MCMC algorithms that leverage the derivative of the posterior to improve chain mixing. All three samplers are implemented via the `blackjax package <https://github.com/blackjax-devs/blackjax>`_.
* emcee (:class:`desilike.samplers.EmceeSampler`, `Foreman-Mackey et al. (2013) <https://doi.org/10.1086/670067>`_, `emcee repo <https://github.com/dfm/emcee>`_): The affine-invariant sampler popular in astronomy. It proposes new points by utilizing the positions of other walkers, i.e., chains.
* zeus (:class:`desilike.samplers.ZeusSampler`, `Karamanis, Beutler, & Peacock (2021) <https://doi.org/10.1093/mnras/stab2867>`_, `zeus repo <https://github.com/minaskar/zeus>`_): An ensemble slice sampling MCMC algorithm. Like `emcee`, this sampler is insensitive to linear correlations.

The implementation of all samplers is derived from the :class:`desilike.samplers.base.MarkovChainSampler` and differ only in their initialization and how they advance the chain via the `run_sampler(n_steps)` method. In particular, all samplers use the same run method, :meth:`desilike.samplers.base.MarkovChainSampler.run`. The run method defines how long the chain is run. The following convergence criterion can be employed.

* Gelman-Rubin statistic (``gelman_rubin``): Maximum value of the `Gelman-Rubin statistic <https://en.wikipedia.org/wiki/Gelman-Rubin_statistic>`_ :math:`R` across all parameters.
* Geweke statistic (``geweke``): Maximum absolute value of the `Geweke statistic <https://math.arizona.edu/~piegorsch/675/GewekeDiagnostics.pdf>`_ :math:`T` across all parameters.
* Effective sample size (ESS, ``ess``): Minimum value for the ESS per chain. The ESS is defined as :math:`n / \tau` where :math:`n` is the length of the chain and :math:`\tau` the integrated autocorrelation time.

Parallelization
---------------

All samplers in `desilike` natively support parallelization via the Message Passing Interface (MPI). `desilike` will distribute multiple concurrent likelihood calculation but will typically not parallelize individual likelihood computations. As a result, the efficiency of parallelization depends on the sampling algorithm. For example, MCMC chains typically do not scale beyond the number of chains, whereas population samplers may benefit more from parallel execution.

To run a sampler in parallel, simply execute your Python script with MPI (e.g., using ``mpirun``). No modifications to your code are required.

Saving Progress
---------------

Bayesian sampling can be computationally expensive. To make long runs more manageable, `desilike` allows saving the progress of a sampler. Specify a directory to store results using the ``directory`` argument. When the sampler is re-run, `desilike` will automatically detect any existing results in that directory and resume the run from the previous state, if available.

**Warning**: Do **not** resume runs from a directory created with different settings, likelihoods, or parameters, as this may lead to incorrect results or unexpected errors.
