import numpy as np

from desilike.base import BaseCalculator, Parameter, ParameterArray
from desilike.utils import jnp
from desilike import utils


class BaseLikelihood(BaseCalculator):

    solved_default = '.marg'
    _loglikelihood_name = 'loglikelihood'
    _logprior_name = 'logprior'

    @property
    def varied_params(self):
        return self.runtime_info.pipeline.varied_params

    def get(self):
        pipeline = self.runtime_info.pipeline
        all_params = pipeline.params
        sum_loglikelihood = float(self.loglikelihood)
        sum_logprior = 0.
        solved_params, indices_best, indices_marg = [], [], []
        for param in all_params:
            solved = param.derived
            if param.solved:
                iparam = len(solved_params)
                solved_params.append(param)
                if solved == '.auto': solved = self.solved_default
                if solved == '.best':
                    indices_best.append(iparam)
                elif solved == '.marg':  # marg
                    indices_marg.append(iparam)
                else:
                    raise ValueError('Unknown option for solved = {}'.format(solved))
        flatdiff = self.flatdiff
        if solved_params:

            def getter():
                return self.flatdiff
            # flatdiff is model - data
            jac = pipeline.jac(getter, solved_params)
            zeros = np.zeros_like(self.precision, shape=self.precision.shape[0])
            jac = np.column_stack([jac[param.name] if param.name in jac else zeros for param in solved_params])
            projector = self.precision.dot(jac)
            projection = projector.T.dot(flatdiff)
            inverse_fisher = jac.T.dot(projector)
        dx, x = [], []
        if solved_params:
            inverse_priors, x0 = [], []
            for param in solved_params:
                scale = getattr(param.prior, 'scale', None)
                inverse_priors.append(0. if scale is None or param.fixed else scale**(-2))
                x0.append(pipeline._param_values[param.name])
            inverse_priors = np.array(inverse_priors)
            sum_inverse_fisher = inverse_fisher + np.diag(inverse_priors)
            dx = - np.linalg.solve(sum_inverse_fisher, projection)
            x = x0 + dx
        for param, xx in zip(solved_params, x):
            sum_logprior += all_params[param].prior(xx)
            pipeline._param_values[param.name] = xx
            pipeline.derived.set(ParameterArray(xx, param))
        #if self.stop_at_inf_prior and not np.isfinite(sum_logprior): return
        if indices_best:
            sum_loglikelihood -= 1. / 2. * dx[indices_best].dot(inverse_fisher[np.ix_(indices_best, indices_best)]).dot(dx[indices_best])
            sum_loglikelihood -= projection[indices_best].dot(dx[indices_best])
        if indices_marg:
            sum_loglikelihood += 1. / 2. * dx[indices_marg].dot(inverse_fisher[np.ix_(indices_marg, indices_marg)]).dot(dx[indices_marg])
        if indices_marg:
            sum_loglikelihood -= 1. / 2. * np.linalg.slogdet(sum_inverse_fisher[np.ix_(indices_marg, indices_marg)])[1]
            #sum_loglikelihood += 1. / 2. * len(indices_marg) * np.log(2. * np.pi)
            # Convention: in the limit of no likelihood constraint on dx, no change to the loglikelihood
            # This allows to ~ keep the interpretation in terms of -1./2. chi2
            ip = inverse_priors[indices_marg]
            sum_loglikelihood += 1. / 2. * np.sum(np.log(ip[ip > 0.]))  # logdet
            #sum_loglikelihood -= 1. / 2. * len(indices_marg) * np.log(2. * np.pi)

        toret = sum_logprior + sum_loglikelihood
        self.loglikelihood = sum_loglikelihood
        self.logprior = sum_logprior

        for param in all_params:
            if param.varied and not param.solved:
                if param.derived and not param.drop:
                    array = self.derived[param]
                    self.logprior += array.param.prior(array)
                else:
                    self.logprior += param.prior(pipeline._param_values[param.name])

        param = Parameter(self._loglikelihood_name, latex=utils.outputs_to_latex(self._loglikelihood_name), derived=True)
        self.runtime_info.derived.set(ParameterArray(self.loglikelihood, param))
        param = Parameter(self._logprior_name, latex=utils.outputs_to_latex(self._logprior_name), derived=True)
        self.runtime_info.derived.set(ParameterArray(self.logprior, param))
        return toret


class GaussianLikelihood(BaseLikelihood):

    def initialize(self, observables, covariance=None, scale_covariance=1.):
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
            if isinstance(scale_covariance, bool):
                if scale_covariance:
                    scale_covariance = 1. / self.nobs
                else:
                    scale_covariance = 1.
        if isinstance(scale_covariance, bool):
            import warnings
            if scale_covariance:
                warnings.warn('Got scale_covariance = {} (boolean), but I do not know the number of realizations; defaults to scale_covariance = 1.'.format(scale_covariance))
            else:
                warnings.warn('Got scale_covariance = {} (boolean), why? defaults to scale_covariance = 1.'.format(scale_covariance))
            scale_covariance = 1.
        self.covariance = np.atleast_2d(self.mpicomm.bcast(scale_covariance * covariance if self.mpicomm.rank == 0 else None, root=0))
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

    @property
    def flatmodel(self):
        return jnp.concatenate([obs.flatmodel for obs in self.observables], axis=0)

    @property
    def flatdiff(self):
        return self.flatmodel - self.flatdata

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'precision', 'loglikelihood']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
