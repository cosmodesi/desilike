"""JAX-based framework for likelihood pipelines, samplers, profilers, and emulators in cosmology."""

import os

from ._version import __version__
from .install import Installer
Installer().setenv()
# folps (and fkptjax, which imports it) selects its backend once, at first import, from
# FOLPS_BACKEND; desilike always requires the JAX backend (the numpy one breaks tracing),
# so select it before anything — e.g. a test's importorskip('folps') — can import folps.
os.environ['FOLPS_BACKEND'] = 'jax'
from .parameter import Node, Variable, ParameterPrior, Parameter, VariableCollection
from .base import Calculator, CompiledGraph, Likelihood, SumLikelihood, GaussianLikelihood, Posterior, Prior, compile, differentiate, jacfwd, hessian, pmap, get_params
from .utils import read, write, setup_logging, round_measurement
from .emulators import TaylorEmulator
from .samples import Samples, MCSamples, Profiles, Covariance, Precision
from .profilers import Profiler, Scipy, Minuit, Optax, BOBYQA
from .samplers import (Sampler,
                       Emcee, Zeus, MH,
                       BlackjaxHMC, BlackjaxNUTS, BlackjaxMCLMC,
                       NumpyroNUTS, NumpyroHMC, NumpyroBarkerMH, NumpyroSA,
                       Dynesty, Nautilus, PocoMC,
                       Grid, QMC, Importance)
