"""JAX-based framework for likelihood pipelines, samplers, profilers, and emulators in cosmology."""

import os

from ._version import __version__
from .install import Installer
Installer().setenv()
# folps (and fkptjax, which imports it) selects its backend once, at first import, from
# FOLPS_BACKEND; desilike always requires the JAX backend (the numpy one breaks tracing),
# so select it before anything — e.g. a test's importorskip('folps') — can import folps.
os.environ['FOLPS_BACKEND'] = 'jax'
# equinox reads EQX_ON_ERROR once, at first import: make equinox/lineax runtime checks
# (e.g. interpax cubic-spline solves fed NaN, such as ACE emulator outputs outside their
# training ranges) return NaN instead of raising, so samplers reject the point instead of
# crashing. Set before any equinox-importing dependency (interpax, lineax, blackjax, ...);
# setdefault so EQX_ON_ERROR=default in the environment restores raising for debugging.
os.environ.setdefault('EQX_ON_ERROR', 'nan')
from .parameter import Node, Variable, ParameterPrior, Parameter, VariableCollection
from .base import Calculator, CompiledGraph, Likelihood, SumLikelihood, GaussianLikelihood, Posterior, Prior, build, differentiate, jacfwd, hessian, pmap, get_params
from .utils import read, write, setup_logging, round_measurement
from .samples import Samples, MCSamples, Profiles, Covariance, Precision
from .profilers import Profiler, Scipy, Minuit, Optax, BOBYQA
from .samplers import (Sampler, PriorProposal, SamplesProposal, GaussianProposal, ProductProposal)


def __getattr__(name):
    """Resolve a sampler kernel on first use, through `desilike.samplers` (PEP 562).

    Naming them here eagerly would import every backend and undo the laziness there -- see
    `desilike.samplers._KERNEL_MODULES`. `from desilike import PocoMC` still works and imports
    only what PocoMC needs.
    """
    from .samplers import _KERNEL_MODULES

    if name not in _KERNEL_MODULES:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from . import samplers

    kernel = getattr(samplers, name)
    globals()[name] = kernel
    return kernel


def __dir__():
    from .samplers import _KERNEL_MODULES

    return sorted(set(globals()) | set(_KERNEL_MODULES))
