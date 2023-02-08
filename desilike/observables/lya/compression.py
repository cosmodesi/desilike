from desilike.jax import numpy as jnp
from desilike.parameter import Parameter
from desilike.base import BaseCalculator
from desilike.samples import load_source
from desilike.theories.lya.power_template import P1DPowerSpectrumExtractor
from desilike.observables.galaxy_clustering.compression import BAOCompressionObservable


def _check_conflicts(quantities, conflicts):
    for conflict in conflicts:
        if all(c in quantities for c in conflict):
            raise ValueError('Found conflicting quantities: {}'.format(conflict))


class P1DCompressionObservable(BaseCalculator):
    """
    P1D compression observable: compare P1D compressed measurements to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        P1D compressed parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for P1D compressed parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['delta2star', 'nstar', 'alphastar']``.

    **kwargs : dict
        Optional arguments for :class:`P1DPowerSpectrumExtractor`, e.g. ``z``, ``qstar``.


    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    """
    def initialize(self, data=None, covariance=None, cosmo=None, quantities=None, **kwargs):
        self.quantities, self.flatdata, self.covariance = self.load_data(data=data, covariance=covariance, quantities=quantities)
        if self.mpicomm.rank == 0:
            self.log_info('Found quantities {}.'.format(self.quantities))
        # If cosmo is None, this will set default parameters for cosmology
        self.extractor = P1DPowerSpectrumExtractor(cosmo=cosmo, **kwargs).runtime_info.initialize()

    def load_data(self, data=None, covariance=None, quantities=None):
        data = load_source(data, params=quantities, choice=True, return_type='dict')
        quantities = [Parameter(quantity) for quantity in data.keys()]
        allowed_quantities = ['delta2star', 'nstar', 'alphastar']
        indices = []
        for iq, quantity in enumerate(quantities):
            if quantity.basename in allowed_quantities: indices.append(iq)
        quantities = [quantities[iq] for iq in indices]
        conflicts = []
        _check_conflicts(quantities, conflicts)
        flatdata = [data[quantity.name] for quantity in quantities]
        covariance = load_source(covariance, params=quantities, cov=True, return_type='nparray')
        return [quantity.basename for quantity in quantities], flatdata, covariance

    def calculate(self):
        self.flattheory = jnp.array([getattr(self.extractor, quantity) for quantity in self.quantities])

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'flattheory', 'quantities']:
            state[name] = getattr(self, name)
        return state
