import numpy as np

from desilike.base import BaseCalculator
from desilike import utils


class GaussianLikelihood(BaseCalculator):

    def initialize(self, observables, covariance=None, covariance_scale=1.):
        if not utils.is_sequence(observables):
            observables = [observables]
        self.nobs = None
        self.observables = [obs.runtime_info.initialize() for obs in observables]
        if covariance is None:
            nmocks = [self.mpicomm.bcast(len(obs.mocks) if self.mpicomm.rank == 0 and obs.mocks is not None else 0) for obs in self.observables]
            self.nobs = nmocks[0]
            if not any(nmocks):
                raise ValueError('Observables must have mocks if global covariance matrix not provided')
            if not all(nmock == nmocks[0] for nmock in nmocks):
                raise ValueError('Provide the same number of mocks for each observable, found {}'.format(nmocks))
            if self.mpicomm.rank == 0:
                list_y = [np.concatenate(y, axis=0) for y in zip(*[obs.mocks for obs in self.observables])]
                covariance = np.cov(list_y, rowvar=False, ddof=1)
        self.covariance = np.atleast_2d(self.mpicomm.bcast(covariance_scale * covariance if self.mpicomm.rank == 0 else None, root=0))
        if self.covariance.shape != (self.covariance.shape[0],) * 2:
            raise ValueError('Covariance must be a square matrix')
        self.flatdata = np.concatenate([obs.flatdata for obs in self.observables], axis=0)
        if self.covariance.shape != (self.flatdata.size,) * 2:
            raise ValueError('Based on provided observables, covariance expected to be a matrix of shape ({0:d}, {0:d})'.format(self.flatdata.size))

        self.precision = utils.inv(self.covariance)
        size = self.precision.shape[0]
        if self.nobs is not None:
            self.hartlap = (self.nobs - size - 2.) / (self.nobs - 1.)
            if self.mpicomm.rank == 0:
                self.log_info('Covariance matrix with {:d} points built from {:d} observations.'.format(size, self.nobs))
                self.log_info('...resulting in Hartlap factor of {:.4f}.'.format(self.hartlap))
            self.precision *= self.hartlap
        self.runtime_info.requires = self.observables

    def calculate(self):
        flatdiff = self.flatdiff
        self.loglikelihood = -0.5 * flatdiff.dot(self.precision).dot(flatdiff)
        return self.loglikelihood

    @property
    def flatmodel(self):
        return np.concatenate([obs.flatmodel for obs in self.observables], axis=0)

    @property
    def flatdiff(self):
        return self.flatmodel - self.flatdata

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'precision', 'loglikelihood']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
