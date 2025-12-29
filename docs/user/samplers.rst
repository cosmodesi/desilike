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

  from desilike.samplers import MHSampler

  sampler = MHSampler(likelihood)
  posterior = sampler.run()

In the above example, we chose the Metropolis-Hastings MCMC sampler, but the same code will work with virtually all other samplers, providing a common entry point to Bayesian sampling.

Static Samplers
---------------

Static samplers (:class:`desilike.samplers.base.StaticSampler`) do not adapt to the likelihood surface. As such, they are primarily used for low-dimensional problems or, in the case of importance sampling, we already have a distribution who closely resembles the posterior distribution. The following samplers are supported.
