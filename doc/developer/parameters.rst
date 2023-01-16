.. _user-parameters:


Parameters
==========

Specifying calculator's parameters requires some syntax. Below some parameter specification.

.. code-block:: yaml

  params:
    .fixed: 'al*'  # Fix all parameters matching al* pattern
    .varied: ['al2_2']  # Allow parameter al2_2 to vary
    qpar:
      value: 1.  # This is the value the parameter will take as default
      prior:  # Prior for this parameter is Gaussian, centered around 1, with standard deviation of 0.1
        dist: norm
        loc: 1.
        scale: 0.1
      ref:  # To start sampling/profiling, generate first parameter values in this distribution (defaults to prior)
        dist: norm
        loc: 1.
        scale: 0.01
      proposal: 0.02  # Proposal standard deviation for this parameter, used when setting initial covariance matrices (MCCMSampler)
      latex: '\alpha_\parallel'  # Latex
    qpar:
      prior:
        limits: [0.9, 1.1]  # uniform prior
   qiso:
      # A derived parameter, expressed as a function of the others
     derived: 'qper**(1./3.) * qper**(2./3.)'
     # One can specify a prior for this derived parameter:
    prior:
      limits: [0.95, 1.05]
    # If this parameter is not used by the calculator, drop it!
    drop: True
  power:
    # A derived parameter, as an attribute (or given by __getstate__(self) method) of this calculator
    derived: True
