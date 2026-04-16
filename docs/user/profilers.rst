.. _user-profilers:

Profilers
=========

In addition to traditional Bayesian sampling algorithms, `desilike` supports profiling the likelihood and/or posterior.

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

Parallelization
---------------

All samplers in `desilike` natively support parallelization via the Message Passing Interface (MPI). `desilike` will distribute multiple concurrent likelihood calculation but will typically not parallelize individual likelihood computations. As a result, the efficiency of parallelization depends on the sampling algorithm. For example, MCMC chains typically do not scale beyond the number of chains, whereas population samplers may benefit more from parallel execution.

To run a sampler in parallel, simply execute your Python script with MPI (e.g., using ``mpirun``). No modifications to your code are required.

Saving Progress
---------------

Bayesian sampling can be computationally expensive. To make long runs more manageable, `desilike` allows saving the progress of a sampler. Specify a directory to store results using the ``directory`` argument. When the sampler is re-run, `desilike` will automatically detect any existing results in that directory and resume the run from the previous state, if available.

**Warning**: Do **not** resume runs from a directory created with different settings, likelihoods, or parameters, as this may lead to incorrect results or unexpected errors.
