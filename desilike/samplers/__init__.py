"""desilike.samplers — wrappers for commonly used posterior samplers."""

from .base import (Sampler, AffineConditioner, MCMCSampler, EnsembleSampler, PopulationSampler,
                   StaticSampler, Kernel, PopulationKernel, StaticKernel)
from .proposals import (BaseProposal, PriorProposal, SamplesProposal, GaussianProposal,
                        ProductProposal, MixtureProposal)

#: Kernel name -> the module that defines it. Imported on first use, not here: each kernel drags
#: in its own third-party backend, and importing them all cost 8.9 s of a 16.2 s `import
#: desilike` -- 3.8 s for pocomc (through zuko, torch) and 3.0 s for blackjax alone -- paid by
#: everyone who only wanted to build a likelihood. `base` and `proposals` stay eager: they are
#: the shared surface every kernel and every caller uses, and they carry no backend of their own.
_KERNEL_MODULES = {
    'Emcee': 'emcee', 'Zeus': 'zeus', 'MH': 'mhmcmc',
    'BlackjaxHMC': 'blackjax', 'BlackjaxNUTS': 'blackjax', 'BlackjaxMCLMC': 'blackjax',
    'NumpyroNUTS': 'numpyro', 'NumpyroHMC': 'numpyro', 'NumpyroBarkerMH': 'numpyro',
    'NumpyroSA': 'numpyro', 'NumpyroAIES': 'numpyro', 'NumpyroESS': 'numpyro',
    'Dynesty': 'dynesty', 'Nautilus': 'nautilus', 'PocoMC': 'pocomc', 'SMC': 'smc',
    'Grid': 'grid', 'QMC': 'qmc', 'Importance': 'importance',
}


def __getattr__(name):
    """Import a kernel's module the first time the kernel is asked for (PEP 562).

    `from desilike.samplers import Emcee` reaches here and imports only `emcee`, so a caller
    pays for the backends of the kernels they actually name.
    """
    module_name = _KERNEL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    import importlib

    kernel = getattr(importlib.import_module(f'.{module_name}', __name__), name)
    globals()[name] = kernel   # bind it, so the next lookup skips this entirely
    return kernel


def __dir__():
    return sorted(set(globals()) | set(_KERNEL_MODULES))

__all__ = [
    'Sampler',
    'AffineConditioner',
    'MCMCSampler',
    'EnsembleSampler',
    'PopulationSampler',
    'StaticSampler',
    'Kernel',
    'PopulationKernel',
    'StaticKernel',
    'Emcee',
    'Zeus',
    'MH',
    'BlackjaxHMC',
    'BlackjaxNUTS',
    'BlackjaxMCLMC',
    'NumpyroNUTS',
    'NumpyroHMC',
    'NumpyroBarkerMH',
    'NumpyroSA',
    'NumpyroAIES',
    'NumpyroESS',
    'Dynesty',
    'Nautilus',
    'PocoMC',
    'SMC',
    'Grid',
    'QMC',
    'Importance',
    'BaseProposal',
    'PriorProposal',
    'SamplesProposal',
    'GaussianProposal',
    'ProductProposal',
    'MixtureProposal',
]
