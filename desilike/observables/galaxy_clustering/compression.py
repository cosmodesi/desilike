from desilike.jax import numpy as jnp
from desilike.parameter import Parameter
from desilike.base import BaseCalculator
from desilike.samples import load_source
from desilike.theories.galaxy_clustering.power_template import BAOExtractor, ShapeFitPowerSpectrumExtractor


def _check_conflicts(quantities, conflicts):
    for conflict in conflicts:
        if all(c in quantities for c in conflict):
            raise ValueError('Found conflicting quantities: {}'.format(conflict))


class BAOCompressionObservable(BaseCalculator):
    """
    BAO observable: compare (compressed) BAO measurements
    (in terms of ratios of distances to the sound horizon scale at the drag epoch) to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        BAO parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.
    
    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for BAO parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    
    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']``.

    z : float, default=None
        Effective redshift.
    
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    """
    def initialize(self, data=None, covariance=None, cosmo=None, quantities=None, z=None, fiducial='DESI'):
        self.bao_quantities, self.flatdata, self.covariance = self.load_data(data=data, covariance=covariance, quantities=quantities)
        if self.mpicomm.rank == 0:
            self.log_info('Found BAO quantities {}.'.format(self.bao_quantities))
        self.bao = BAOExtractor(z=z, fiducial=fiducial, cosmo=cosmo)

    def load_data(self, data=None, covariance=None, quantities=None):
        data = load_source(data if data is not None else covariance, params=quantities, choice=True, return_type='dict')
        quantities = [Parameter(quantity) for quantity in data.keys()]
        allowed_bao_quantities = ['DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']
        indices = []
        for iq, quantity in enumerate(quantities):
            if quantity.basename in allowed_bao_quantities:
                indices.append(iq)
        quantities = [quantities[iq] for iq in indices]
        conflicts = [('DM_over_rd', 'qper'), ('DH_over_rd', 'qper'), ('DM_over_DH', 'qap'), ('DV_over_rd', 'qsio')]
        _check_conflicts(quantities, conflicts)
        flatdata = [data[quantity.name] for quantity in quantities]
        covariance = load_source(covariance if covariance is not None else data, params=quantities, cov=True, return_type='nparray')
        return [quantity.basename for quantity in quantities], flatdata, covariance

    def calculate(self):
        bao = [getattr(self.bao, quantity) for quantity in self.bao_quantities]
        self.flattheory = jnp.array(bao)

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'flattheory', 'bao_quantities']:
            state[name] = getattr(self, name)
        return state


class ShapeFitCompressionObservable(BaseCalculator):
    """
    ShapeFit observable: compare ShapeFit measurements to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        ShapeFit parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.
    
    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for ShapeFit parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    
    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['m', 'n', 'f_sqrt_Ap', 'dm', 'dn', 'f', 'DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']``.

    z : float, default=None
        Effective redshift.
    
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance


    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    """
    def initialize(self, data=None, covariance=None, cosmo=None, quantities=None, z=None, fiducial='DESI'):
        self.bao_quantities, self.fs_quantities, self.flatdata, self.covariance = self.load_data(data=data, covariance=covariance, quantities=quantities)
        if self.mpicomm.rank == 0:
            self.log_info('Found BAO quantities {}.'.format(self.bao_quantities))
            self.log_info('Found FS quantities {}.'.format(self.fs_quantities))
        # If cosmo is None, this will set default parameters for cosmology
        self.fs = ShapeFitPowerSpectrumExtractor(z=z, n_varied='n' in self.fs_quantities, cosmo=cosmo, fiducial=fiducial).runtime_info.initialize()
        self.bao = BAOExtractor(z=z, fiducial=self.fs.fiducial, cosmo=self.fs.cosmo)

    def load_data(self, data=None, covariance=None, quantities=None):
        data = load_source(data, params=quantities, choice=True, return_type='dict')
        quantities = [Parameter(quantity) for quantity in data.keys()]
        allowed_bao_quantities = ['DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']
        allowed_fs_quantities = ['m', 'n', 'f_sqrt_Ap', 'dm', 'dn', 'df']
        bao_indices, fs_indices = [], []
        for iq, quantity in enumerate(quantities):
            if quantity.basename in allowed_bao_quantities:
                bao_indices.append(iq)
            elif quantity.basename in allowed_fs_quantities:
                fs_indices.append(iq)
        bao_quantities = [quantities[iq] for iq in bao_indices]
        conflicts = [('DM_over_rd', 'qper'), ('DH_over_rd', 'qper'), ('DM_over_DH', 'qap'), ('DV_over_rd', 'qsio')]
        _check_conflicts(bao_quantities, conflicts)
        fs_quantities = [quantities[iq] for iq in fs_indices]
        conflicts = [('m', 'dm'), ('n', 'dn'), ('f_sqrt_Ap', 'df')]
        _check_conflicts(fs_quantities, conflicts)
        quantities = bao_quantities + fs_quantities
        flatdata = [data[quantity.name] for quantity in quantities]
        covariance = load_source(covariance, params=quantities, cov=True, return_type='nparray')
        return [quantity.basename for quantity in bao_quantities], [quantity.basename for quantity in fs_quantities], flatdata, covariance

    def calculate(self):
        bao = [getattr(self.bao, quantity) for quantity in self.bao_quantities]
        fs = [getattr(self.fs, quantity) for quantity in self.fs_quantities]
        self.flattheory = jnp.array(bao + fs)

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'flattheory', 'bao_quantities', 'fs_quantities']:
            state[name] = getattr(self, name)
        return state
