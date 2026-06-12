"""desilike.samplers — wrappers for commonly used posterior samplers."""

from .base import Sampler, MCMCSampler, EnsembleKernelSampler, NestedSampler
from .blackjax import HMCSampler, MCLMCSampler, NoUTurnSampler
from .dynesty import DynestySampler
from .emcee import EmceeSampler
from .grid import GridSampler
from .importance import ImportanceSampler
from .mhmcmc import MetropolisHastingsSampler
from .nautilus import NautilusSampler
from .numpyro import NumpyroBarkerMHSampler, NumpyroHMCSampler, NumpyroNUTSSampler, NumpyroSASampler
from .pocomc import PocoMCSampler
from .qmc import QMCSampler
from .zeus import ZeusSampler
from .kernels import (Kernel, NestedKernel,
                      Emcee, Zeus, MetropolisHastings,
                      HMC, NUTS, MCLMC,
                      NumpyroNUTS, NumpyroHMC, NumpyroBarkerMH, NumpyroSA,
                      Dynesty, Nautilus, PocoMC)

__all__ = [
    'Sampler',
    'MCMCSampler',
    'EnsembleKernelSampler',
    'NestedSampler',
    'Kernel',
    'NestedKernel',
    'Emcee',
    'Zeus',
    'MetropolisHastings',
    'HMC',
    'NUTS',
    'MCLMC',
    'NumpyroNUTS',
    'NumpyroHMC',
    'NumpyroBarkerMH',
    'NumpyroSA',
    'Dynesty',
    'Nautilus',
    'PocoMC',
    # Legacy class names kept for backward compatibility
    'DynestySampler',
    'EmceeSampler',
    'GridSampler',
    'HMCSampler',
    'ImportanceSampler',
    'MCLMCSampler',
    'MetropolisHastingsSampler',
    'NautilusSampler',
    'NoUTurnSampler',
    'NumpyroBarkerMHSampler',
    'NumpyroHMCSampler',
    'NumpyroNUTSSampler',
    'NumpyroSASampler',
    'PocoMCSampler',
    'QMCSampler',
    'ZeusSampler',
]
