import numpy as np

from desilike.base import BaseCalculator
from desilike.theories.galaxy_clustering.power_template import BAOExtractor, ShapeFitPowerSpectrumExtractor


class BAOCompression(BaseCalculator):

    def initialize(self, data=None, covariance=None, cosmo=None, quantities=None, fiducial='DESI'):
        self.bao_quantities, self.flatdata, self.covariance = self.load_data(data=data, covariance=covariance, quantities=quantities)
        if self.mpicomm.rank == 0:
            self.log_info('Found BAO quantities {}.'.format(self.bao_quantities))
        self.bao = BAOExtractor(z=self.z, fiducial=fiducial, cosmo=cosmo)

    def load_data(self, data=None, covariance=None, quantities=None):
        allowed_bao_quantities = ['DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']
        bao_indices = []
        if quantities is None:
            quantities = ['DM_over_rd', 'DH_over_rd']
        for iq, quantity in enumerate(quantities):
            if quantity in allowed_bao_quantities:
                bao_indices.append(iq)
        flatdata = [data[iq] for iq in bao_indices]
        covariance = np.asarray(covariance)[np.ix_(bao_indices, bao_indices)]
        return [quantities[iq] for iq in bao_indices], flatdata, covariance

    def calculate(self):
        bao = [getattr(self.bao, quantity) for quantity in self.bao_quantities]
        self.flatmodel = np.array(bao)


class ShapeFitCompression(BaseCalculator):

    def initialize(self, data=None, covariance=None, cosmo=None, quantities=None, fiducial='DESI'):
        self.bao_quantities, self.fs_quantities, self.flatdata, self.covariance = self.load_data(data=data, covariance=covariance, quantities=quantities)
        if self.mpicomm.rank == 0:
            self.log_info('Found BAO quantities {}.'.format(self.bao_quantities))
            self.log_info('Found FS quantities {}.'.format(self.fs_quantities))
        self.bao = BAOExtractor(z=self.z, fiducial=fiducial, cosmo=cosmo).runtime_info.initialize()
        self.fs = ShapeFitPowerSpectrumExtractor(z=self.z, n_varied='n' in self.fs_quantities, cosmo=self.bao.cosmo, fiducial=self.bao.fiducial)

    def load_data(self, data=None, covariance=None, quantities=None):
        allowed_bao_quantities = ['DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']
        allowed_fs_quantities = ['m', 'n', 'f_sqrt_Ap']
        bao_indices, fs_indices = [], []
        if quantities is None:
            quantities = ['DM_over_rd', 'DH_over_rd', 'm', 'f_sqrt_Ap']
        for iq, quantity in enumerate(quantities):
            if quantity in allowed_bao_quantities:
                bao_indices.append(iq)
            elif quantity in allowed_fs_quantities:
                fs_indices.append(iq)
        all_indices = bao_indices + fs_indices
        flatdata = [data[iq] for iq in all_indices]
        covariance = np.asarray(covariance)[np.ix_(all_indices, all_indices)]
        return [quantities[iq] for iq in bao_indices], [quantities[iq] for iq in fs_indices], flatdata, covariance

    def calculate(self):
        bao = [getattr(self.bao, quantity) for quantity in self.bao_quantities]
        fs = [getattr(self.fs, quantity) for quantity in self.fs_quantities]
        self.flatmodel = np.array(bao + fs)
