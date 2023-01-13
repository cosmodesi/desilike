from desilike.jax import numpy as jnp
from desilike.parameter import Parameter
from desilike.base import BaseCalculator
from desilike.samples import load_source
from desilike.theories.galaxy_clustering.power_template import BAOExtractor, ShapeFitPowerSpectrumExtractor


class BAOCompression(BaseCalculator):

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


class ShapeFitCompression(BaseCalculator):

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
        allowed_fs_quantities = ['m', 'n', 'f_sqrt_Ap', 'dm', 'dn', 'f']
        bao_indices, fs_indices = [], []
        for iq, quantity in enumerate(quantities):
            if quantity.basename in allowed_bao_quantities:
                bao_indices.append(iq)
            elif quantity.basename in allowed_fs_quantities:
                fs_indices.append(iq)
        bao_quantities = [quantities[iq] for iq in bao_indices]
        fs_quantities = [quantities[iq] for iq in fs_indices]
        quantities = bao_quantities + fs_quantities
        flatdata = [data[quantity.name] for quantity in quantities]
        covariance = load_source(covariance, params=quantities, cov=True, return_type='nparray')
        fs_convert = {'f': 'df'}
        return [quantity.basename for quantity in bao_quantities], [fs_convert.get(quantity.basename, quantity.basename) for quantity in fs_quantities], flatdata, covariance

    def calculate(self):
        bao = [getattr(self.bao, quantity) for quantity in self.bao_quantities]
        fs = [getattr(self.fs, quantity) for quantity in self.fs_quantities]
        self.flattheory = jnp.array(bao + fs)

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'flattheory', 'bao_quantities', 'fs_quantities']:
            state[name] = getattr(self, name)
        return state
