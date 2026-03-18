import warnings

import numpy as np
import lsstypes as types

from desilike.base import BaseCalculator, Parameter, ParameterCollection, ParameterArray
from desilike.jax import numpy as jnp
from desilike.jax import jit
from desilike import plotting, utils


@jit
def chi2(flatdiff, precision):
    if precision.ndim == 1:
        return (flatdiff * precision).dot(flatdiff.T)
    return flatdiff.dot(precision).dot(flatdiff.T)



class FastFisher(object):

    alltogether = False

    def __init__(self, this, solved_params):  # this: current likelihood self
        pipeline = this.runtime_info.pipeline
        self.solved_params = ParameterCollection(solved_params)

        likelihoods = getattr(this, 'likelihoods', [this])

        def get_params(likelihood):

            calculators = []
            def callback(calculator):
                if calculator in calculators:
                    return
                calculators.append(calculator)
                for require in calculator.runtime_info.requires:
                    callback(require)

            callback(likelihood)
            return sum(calculator.runtime_info.params for calculator in calculators)

        self.solve_likelihoods = []
        likelihood_solved_params, solved_params_friends = [], {param.name: set() for param in self.solved_params}
        for likelihood in likelihoods:
            likelihood_params = get_params(likelihood)
            solved_params = ParameterCollection([param for param in likelihood_params if param in self.solved_params])
            if solved_params:
                self.solve_likelihoods.append(likelihood)
                likelihood_solved_params.append(solved_params)
                solved_names = solved_params.names()
                for name in solved_names:
                    solved_params_friends[name] |= set(solved_names)

        if self.alltogether:
            group_solve_likelihoods = [list(self.solve_likelihoods)]
        else:

            def get_all_levels_of_friends(friends_dict, person):
                from collections import deque

                if person not in friends_dict:
                    return []

                # Initialize a queue for BFS and a set for visited nodes
                queue = deque([person])
                visited = set([person])

                all_friends = set([person])  # To store all levels of friends

                # Perform BFS
                while queue:
                    current_person = queue.popleft()
                    # Get the direct friends of the current person
                    if current_person in friends_dict:
                        for friend in friends_dict[current_person]:
                            if friend not in visited:
                                visited.add(friend)  # Mark as visited
                                queue.append(friend)  # Add to queue for further exploration
                                all_friends.add(friend)  # Add to all_friends set

                return all_friends

            solved_params_groups = []
            for param in solved_params_friends:
                group = get_all_levels_of_friends(solved_params_friends, param)
                if group not in solved_params_groups:
                    solved_params_groups.append(group)

            group_solve_likelihoods = [[] for i in solved_params_groups]
            for likelihood, solved_params in zip(self.solve_likelihoods, likelihood_solved_params):
                param = solved_params[0].name
                for igroup, params_group in enumerate(solved_params_groups):
                    if param in params_group:
                        group_solve_likelihoods[igroup].append(likelihood)
                        break

        self.ilikelihood_solved_indices = [None for i in self.solve_likelihoods]
        self._group_solved_params, self._group_solved_indices, self._all_params_group = [], [], {}
        self._group_solve_likelihoods, self._group_solve_group_likelihoods_indices = [], []
        for igroup, likelihoods in enumerate(group_solve_likelihoods):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Derived parameter.*')
                likelihood = SumLikelihood(likelihoods)
                likelihood.mpicomm = this.mpicomm
                likelihood.runtime_info.pipeline.more_initialize = None
                likelihood.runtime_info.pipeline.more_calculate = lambda: None
                all_params = likelihood.all_params
                for param in pipeline.params:
                    if param in likelihood.all_params:
                        param = param.clone(derived=False if param in self.solved_params or param.depends else param.derived, fixed=param not in self.solved_params)
                        all_params.set(param)
                likelihood.all_params = all_params
            input_params = [param for param in likelihood.all_params if param.name in pipeline.input_values]
            values = {param.name: pipeline.input_values[param.name] for param in input_params}
            likelihood.runtime_info.pipeline.input_values = values
            self._group_solve_likelihoods.append(likelihood)
            self._group_solve_group_likelihoods_indices.append([self.solve_likelihoods.index(likelihood) for likelihood in likelihoods])
            for ilike in self._group_solve_group_likelihoods_indices[-1]:
                self.ilikelihood_solved_indices[ilike] = np.array([iparam for iparam, param in enumerate(self.solved_params) if param in likelihood.all_params])
            self._group_solved_params.append(ParameterCollection([param for param in likelihood.all_params if param in self.solved_params]))
            self._group_solved_indices.append(np.array([self.solved_params.index(param) for param in self._group_solved_params[-1]]))
            for param in likelihood.all_params:
                self._all_params_group[param.name] = self._all_params_group.get(param.name, []) + [igroup]
        self.all_params = sum(likelihood.all_params for likelihood in self._group_solve_likelihoods)
        self.input_params = ParameterCollection([param for param in self.all_params if param.name in pipeline.input_values])

    def __call__(self, values, gradient=True):
        import jax

        def _get_list():
            return [None for like in self.solve_likelihoods]

        likelihoods_gradient, likelihoods_hessian, likelihoods_flatdiff, likelihoods_flatderiv = (_get_list() for i in range(4))
        values_ilikelihood = [{} for igroup in range(len(self._group_solve_likelihoods))]

        for param, value in values.items():
            for igroup in self._all_params_group[param]:
                values_ilikelihood[igroup][param] = value

        multiple_groups = len(self._group_solve_likelihoods) > 1
        if multiple_groups:
            nsolved = len(self.solved_params)
            x, dx = jnp.zeros(nsolved), jnp.zeros(nsolved)
            posterior_hessian, prior_hessian = jnp.zeros((nsolved, nsolved)), jnp.zeros((nsolved, nsolved))

        for igroup, likelihood in enumerate(self._group_solve_likelihoods):
            diff_names = self._group_solved_params[igroup].names()
            all_values = values_ilikelihood[igroup]

            def getter(diff_values):
                likelihood({**all_values, **dict(zip(diff_names, diff_values))})
                return [likelihood.flatdiff for likelihood in likelihood.likelihoods]

            diff_values = jnp.array([all_values[name] for name in diff_names])
            if gradient: flatderivs = jax.jacfwd(getter, argnums=0, has_aux=False, holomorphic=False)(diff_values)
            flatdiffs = getter(diff_values)

            group_likelihoods_indices = self._group_solve_group_likelihoods_indices[igroup]
            for idx, flatdiff in zip(group_likelihoods_indices, flatdiffs):
                likelihoods_flatdiff[idx] = flatdiff.T

            if gradient:
                for idx, flatdiff, flatderiv, like in zip(group_likelihoods_indices, flatdiffs, flatderivs, likelihood.likelihoods):
                    precision = like.precision
                    flatderiv = flatderiv.T
                    likelihoods_flatderiv[idx] = flatderiv
                    if precision.ndim == 1:
                        derivp = flatderiv * precision
                    else:
                        derivp = flatderiv.dot(precision)
                    likelihoods_gradient[idx] = - derivp.dot(flatdiff.T)
                    likelihoods_hessian[idx] = - derivp.dot(flatderiv.T)

                group_prior_gradient, group_prior_hessian = [], []
                group_params = self._group_solved_params[igroup]
                for param in group_params:
                    value = all_values[param.name]
                    loc, scale = getattr(param.prior, 'loc', 0.), getattr(param.prior, 'scale', np.inf)
                    prec = scale**(-2)
                    group_prior_gradient.append(- (value - loc) * prec)
                    group_prior_hessian.append(- prec)
                group_prior_gradient = jnp.array(group_prior_gradient)
                group_prior_hessian = jnp.diag(jnp.array(group_prior_hessian))
                group_posterior_gradient = group_prior_gradient + sum(likelihoods_gradient[idx] for idx in group_likelihoods_indices)
                group_posterior_hessian = group_prior_hessian + sum(likelihoods_hessian[idx] for idx in group_likelihoods_indices)
                group_dx = - jnp.linalg.solve(group_posterior_hessian, group_posterior_gradient)
                group_x = jnp.array([all_values[param.name] for param in group_params]) + group_dx
                group_indices = self._group_solved_indices[igroup]
                if multiple_groups:
                    dx = dx.at[group_indices].set(group_dx)
                    x = x.at[group_indices].set(group_x)
                    posterior_hessian = posterior_hessian.at[np.ix_(group_indices, group_indices)].set(group_posterior_hessian)
                    prior_hessian = prior_hessian.at[np.ix_(group_indices, group_indices)].set(group_prior_hessian)
                else:
                    dx, x, posterior_hessian, prior_hessian = group_dx, group_x, group_posterior_hessian, group_prior_hessian
        if gradient:
            return x, dx, posterior_hessian, prior_hessian, likelihoods_hessian, likelihoods_gradient, likelihoods_flatdiff, likelihoods_flatderiv
        return likelihoods_flatdiff


class BaseLikelihood(BaseCalculator):

    """Base class for likelihood."""
    _attrs = ['loglikelihood', 'logprior']
    name = None
    solved_default = '.marg'

    def initialize(self, catch_errors=None, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs['name']
        for name in self._attrs:
            if name not in self.init.params.basenames():
                self.init.params.set(Parameter(basename=name, namespace=self.name, latex=utils.outputs_to_latex(name), derived=True))
            param = self.init.params.select(basename=name)
            if not len(param):
                raise ValueError('{} derived parameter not found'.format(name))
            elif len(param) > 1:
                raise ValueError('Several parameters with name {:0} found. Which one is the {:0}?'.format(name))
            param = param[0]
            param.update(derived=True)
            setattr(self, '_param_{}'.format(name), param)
        if catch_errors is not None:
            catch_errors = tuple(catch_errors)
        self._catch_errors = catch_errors
        #self.__fisher = None

    def more_initialize(self):
        pipeline = self.runtime_info.pipeline
        likelihoods = getattr(self, 'likelihoods', [self])

        # Reset precision and flatdata
        for likelihood in likelihoods:
            pipeline_initialize = getattr(likelihood, '_pipeline_initialize', None)
            if pipeline_initialize is not None:
                pipeline_initialize(pipeline)

        self._marginalize_precision()
        pipeline.more_calculate = self._solve

    def get(self):
        pipeline = self.runtime_info.pipeline
        self.logprior = pipeline.params.prior(**pipeline.input_values)  # does not include solved params
        return self.loglikelihood + self.logprior

    @property
    def catch_errors(self):
        toret = getattr(self, '_catch_errors', None)
        if toret is None:
            self._catch_errors = []
            for calculator in self.runtime_info.pipeline.calculators:
                self._catch_errors += [error for error in getattr(calculator, '_likelihood_catch_errors', [])]
            toret = self._catch_errors = tuple(self._catch_errors)
        return toret

    def _marginalize_precision(self):

        pipeline = self.runtime_info.pipeline
        all_params = pipeline._params

        prec_params, solved_params = [], []
        for param in all_params:
            solved = param.derived
            if param.solved:
                if solved.startswith('.prec'):
                    prec_params.append(param)
                else:
                    solved_params.append(param)
                    if not (solved.startswith('.auto') or solved.startswith('.marg') or solved.startswith('.best')):
                        raise ValueError('unknown option for solved = {}'.format(solved))

        # Reset precision and flatdata
        for likelihood in getattr(self, 'likelihoods', [self]):
            for name in ['precision', 'flatdata']:
                input_name = '_{}_input'.format(name)
                if hasattr(likelihood, input_name):
                    setattr(likelihood, name, getattr(likelihood, input_name))

        if prec_params:

            fisher = FastFisher(self, prec_params)
            if len(fisher.ilikelihood_solved_indices) > 1:
                intersection = set.intersection(*[set(indices) for indices in fisher.ilikelihood_solved_indices])
                if intersection:
                    raise ValueError('cannot use .prec for parameters that are shared between likelihoods (we would need to create a joint covariance matrix!); common block is {}'.format([fisher.solved_params[i] for i in intersection]))

            # Just to reject from ``values`` parameters from which base ones are derived, and are not kept in solve_likelihood.all_params
            values = {param.name: pipeline.input_values[param.name] for param in fisher.input_params}
            #print(values)
            for param in prec_params: values[param.name] = 0.

            fisher(values, gradient=False)  # to set flatdata
            for likelihood in fisher.solve_likelihoods:
                likelihood._precision_input = getattr(likelihood, '_precision_input', likelihood.precision)
                likelihood._flatdata_input = getattr(likelihood, '_flatdata_input', likelihood.flatdata)

            x, dx, posterior_hessian, prior_hessian, likelihoods_hessian, likelihoods_gradient, likelihoods_flatdiff, likelihoods_flatderiv = fisher(values)
            for param in prec_params: values[param.name] = getattr(param.prior, 'loc', 0.)
            fisher(values, gradient=False)
            for param in prec_params: values[param.name] = getattr(param.prior, 'loc', 0.)
            for likelihood, flatdiff, flatderiv, solved_indices in zip(fisher.solve_likelihoods, likelihoods_flatdiff, likelihoods_flatderiv, fisher.ilikelihood_solved_indices):
                precision = likelihood._precision_input
                if precision.ndim == 1:
                    derivp = flatderiv * precision
                else:
                    derivp = flatderiv.dot(precision)
                likelihood.precision = np.asarray(precision - derivp.T.dot(np.linalg.solve(- posterior_hessian[np.ix_(solved_indices, solved_indices)], derivp)))
                likelihood.flatdata = np.asarray(likelihood._flatdata_input - (likelihood.flatdiff - flatdiff))  # flatdiff = flattheory - flatdata

        self.__solved_params = ParameterCollection(solved_params)
        self.__fisher = None

    def _solve(self):
        # Analytic marginalization, to be called, if desired, in get()
        pipeline = self.runtime_info.pipeline
        self.logprior = pipeline.params.prior(**pipeline.input_values)  # does not include solved params

        fisher = None

        if self.__solved_params:

            derived = pipeline.derived
            #pipeline.more_calculate = lambda: None
            fisher = self.__fisher

            if fisher is None or fisher.mpicomm is not self.mpicomm or fisher.solved_default is not self.solved_default:
                #if self.fisher is not None: print(self.fisher.mpicomm is not self.mpicomm, self.fisher.varied_params != solved_params)
                fisher = FastFisher(self, self.__solved_params)
                fisher.mpicomm = self.mpicomm
                fisher.solved_default = self.solved_default

                marg_indices = []
                for iparam, param in enumerate(fisher.solved_params):
                    solved = param.derived
                    if param.solved and not solved.startswith('.prec'):
                        if solved.startswith('.auto'):
                            solved = solved.replace('.auto', self.solved_default)
                        if solved.startswith('.marg'):  # marg
                            marg_indices.append(iparam)
                fisher.marg_indices = np.array(marg_indices)
                derivs = [()]
                derivs_indices = [], []
                for iparam1, param1 in enumerate(fisher.solved_params):
                    if param1.derived.endswith('not_derived'): continue  # do not export to .derived
                    for iparam2, param2 in enumerate(fisher.solved_params[iparam1:]):
                        if param2.derived.endswith('not_derived'): continue
                        derivs.append((param1.name, param2.name))
                        derivs_indices[0].append(iparam1)
                        derivs_indices[1].append(iparam1 + iparam2)
                fisher.derivs = derivs
                fisher.derivs_indices = derivs_indices
                self.__fisher = fisher

            values = {param.name: pipeline.input_values[param.name] for param in fisher.input_params}
            x, dx, posterior_hessian, prior_hessian, likelihoods_hessian, likelihoods_gradient, likelihoods_flatdiff, likelihoods_flatderiv = fisher(values)

        derived = pipeline.derived
        sum_loglikelihood = jnp.zeros(len(fisher.derivs) if self.__solved_params and derived is not None else (), dtype='f8')
        sum_logprior = jnp.zeros((), dtype='f8')

        if fisher is not None:
            for param, xx in zip(self.__solved_params, x):
                sum_logprior += param.prior(xx)
                # hack to run faster than calling param.prior --- saving ~ 0.0005 s
                #sum_logprior += -0.5 * (xx - param.prior.attrs['loc'])**2 / param.prior.attrs['scale']**2 if param.prior.dist == 'norm' else 0.
                #pipeline.i                    print(vlikelihood(profiles.bestfit.to_dict(params=profiles.bestfit.params(input=True))))nput_values[param.name] = xx  # may lead to instabilities
                if derived is not None:
                    derived.set(ParameterArray(xx, param=param))

        if fisher is not None and derived is not None:
            sum_logprior = jnp.insert(prior_hessian[fisher.derivs_indices], 0, sum_logprior + self.logprior)
        else:
            sum_logprior += self.logprior

        for likelihood in getattr(self, 'likelihoods', [self]):
            loglikelihood = jnp.array(likelihood.loglikelihood)

            if fisher is not None and likelihood in fisher.solve_likelihoods:
                index_likelihood = fisher.solve_likelihoods.index(likelihood)
                ddx = dx[fisher.ilikelihood_solved_indices[index_likelihood]]
                likelihood_hessian = likelihoods_hessian[index_likelihood]
                # Here we plug in best x into L = dx.T.dot(likelihood_hessian).dot(dx) + dx.T.dot(likelihood_gradient) + likelihood_with_x_fixed
                # Note: priors of solved params have already been added
                loglikelihood += 1. / 2. * ddx.dot(likelihood_hessian).dot(ddx)
                loglikelihood += likelihoods_gradient[index_likelihood].dot(ddx)
                # Set derived values
                if derived is not None:
                    loglikelihood = jnp.insert(likelihood_hessian[fisher.derivs_indices], 0, loglikelihood)
                    derived.set(ParameterArray(loglikelihood, param=likelihood._param_loglikelihood, derivs=fisher.derivs))

            sum_loglikelihood += loglikelihood

        if fisher is not None and fisher.marg_indices.size:
            marg_likelihood = -1. / 2. * jnp.linalg.slogdet(- posterior_hessian[np.ix_(fisher.marg_indices, fisher.marg_indices)])[1]
            # sum_loglikelihood += 1. / 2. * len(marg_indices) * np.log(2. * np.pi)
            # Convention: in the limit of no likelihood constraint on dx, no change to the loglikelihood
            # This allows to ~ keep the interpretation in terms of -1. / 2. * chi2
            #ip = jnp.diag(prior_hessian)[fisher.marg_indices]
            #marg_likelihood += 1. / 2. * jnp.sum(jnp.log(jnp.where(ip < 0, -ip, 1.)))  # logdet
            # sum_loglikelihood -= 1. / 2. * len(marg_indices) * np.log(2. * np.pi)
            if derived is not None:
                marg_likelihood = marg_likelihood * np.array([1.] + [0.] * (len(fisher.derivs) - 1), dtype='f8')
            sum_loglikelihood += marg_likelihood

        self.loglikelihood = sum_loglikelihood
        self.logprior = sum_logprior

        if fisher is not None and derived is not None:
            derived.set(ParameterArray(self.loglikelihood, param=self._param_loglikelihood, derivs=fisher.derivs))
            derived.set(ParameterArray(self.logprior, param=self._param_logprior, derivs=fisher.derivs))

        return self.loglikelihood.ravel()[0] + self.logprior.ravel()[0]

    @classmethod
    def sum(cls, *others):
        """Sum likelihoods: return :class:`SumLikelihood` instance."""
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        likelihoods = []
        for likelihood in others:
            if isinstance(likelihood, SumLikelihood):
                if likelihood.runtime_info.initialized:
                    likelihoods += likelihood.likelihoods
                else:
                    likelihoods += list(likelihood.init.get('likelihoods', []))
            else:
                likelihoods.append(likelihood)
        return SumLikelihood(likelihoods=likelihoods)

    def __add__(self, other):
        """Sum likelihoods ``self`` and ``other``: return :class:`SumLikelihood` instance."""
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0:
            return self.sum(self)
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0:
            return self.sum(self)
        return self.__add__(other)

    @property
    def size(self):
        # Data vector size
        return len(self.flatdata)

    @property
    def nvaried(self):
        return len(self.varied_params) + len(self.all_params.select(solved=True))

    @property
    def ndof(self):
        return self.size - self.nvaried

    def __getstate__(self, varied=True, fixed=True):
        state = {}
        for name in (['loglikelihood'] if varied else []):
            if hasattr(self, name): state[name] = getattr(self, name)
        return state


class BaseGaussianLikelihood(BaseLikelihood):
    """
    Base class for Gaussian likelihood, which allows parameters the theory is linear with to be analytically marginalized over.

    Parameters
    ----------
    data : array
        Data.
    covariance : array, default=None
        Covariance matrix (or its diagonal).
    precision : array, default=None
        If ``covariance`` is not provided, precision matrix (or its diagonal).
    """
    _attrs = ['loglikelihood', 'logprior']

    def initialize(self, data, covariance=None, precision=None, **kwargs):
        self.flatdata = np.ravel(data)
        if precision is None:
            if covariance is None:
                raise ValueError('Provide either precision or covariance matrix to {}'.format(self.__class__))
            self.precision = utils.inv(np.atleast_2d(np.array(covariance, dtype='f8')))
        else:
            self.precision = np.atleast_1d(np.array(precision, dtype='f8'))
        super(BaseGaussianLikelihood, self).initialize(**kwargs)

    def calculate(self):
        self.flatdiff = self.flattheory - self.flatdata
        self.loglikelihood = -0.5 * chi2(self.flatdiff, self.precision)

    def __getstate__(self, varied=True, fixed=True):
        state = {}
        for name in (['flatdata', 'covariance', 'precision', 'transform'] if fixed else []) + (['flatdiff', 'loglikelihood'] if varied else []):
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class ObservablesGaussianLikelihood(BaseGaussianLikelihood):
    """
    Gaussian likelihood of observables.

    Parameters
    ----------
    observables : list, BaseCalculator
        List of (or single) observable, e.g. :class:`TracerPowerSpectrumMultipolesObservable` or :class:`TracerCorrelationFunctionMultipolesObservable`.
    covariance : array, types.CovarianceMatrix, default=None
        Covariance matrix (or its diagonal) for input ``observables``.
    scale_covariance : float, default=1.
        Scale precision by the inverse of this value.
    correct_covariance : dict, default=None
        'hartlap' to apply Hartlap 2007 factor (https://arxiv.org/abs/astro-ph/0608064).
        'percival2014' to apply Percival 2014 factor (https://arxiv.org/abs/1312.4841).
        A dictionary to specify the number of observations, ``{'nobs': nobs, 'correction': 'hartlap-percival2014'}``.
    precision : array, default=None
        Precision matrix to be used instead of the inverse covariance.
    """
    def initialize(self, observables, covariance=None, scale_covariance=1., correct_covariance=None, precision=None, **kwargs):
        if not utils.is_sequence(observables):
            observables = [observables]
        if isinstance(correct_covariance, dict):
            nobs = correct_covariance.get('nobs', self.nobs)
            correct_covariance = correct_covariance['correction']
        else:
            nobs = getattr(covariance, 'nobs', None)
        if correct_covariance is None:
            correct_covariance = {'correction': '', 'nobs': None}
        else:
            correct_covariance = {'correction': correct_covariance, 'nobs': nobs}
        self.observables = list(observables)
        for obs in self.observables: obs._mpicomm = self.mpicomm
        #for obs in observables: obs.all_params  # to set observable's pipelines, and initialize once (percival factor below requires all_params)
        covariance, scale_covariance, precision = (self.mpicomm.bcast(obj if self.mpicomm.rank == 0 else None, root=0) for obj in (covariance, scale_covariance, precision))
        if covariance is None:
            if len(self.observables) == 1 and getattr(self.observables[0], 'covariance', None) is not None:
                covariance = self.observables[0].covariance
                if correct_covariance['correction']:
                    correct_covariance.setdefault('nobs', getattr(covariance, 'nobs', None))
                covariance = covariance.clone(observable=types.ObservableTree([covariance.observable], observables=[self.observables[0].name]))
            elif precision is None:
                raise ValueError('Observables must have their own covariance if global covariance or precision matrix not provided')
        data = [observable.data for observable in self.observables]
        self.data = types.ObservableTree(data, observables=[observable.name for observable in self.observables])
        self.flatdata = self.data.value()

        def check_matrix(matrix, name):
            matrix = np.atleast_2d(matrix).copy()
            if matrix.shape != (matrix.shape[0],) * 2:
                raise ValueError('{} must be a square matrix, but found shape {}'.format(name, matrix.shape))
            mshape = '({0}, {0})'.format(matrix.shape[0])
            shape = '({0}, {0})'.format(self.flatdata.size)
            shape_obs = '({0}, {0})'.format(' + '.join([str(obs.flatdata.size) for obs in self.observables]))
            if matrix.shape[0] != self.flatdata.size:
                raise ValueError('based on provided observables, {} expected to be a matrix of shape {} = {}, but found {}'.format(name, shape, shape_obs, mshape))
            return matrix

        self.precision = check_matrix(precision, 'precision') if precision is not None else None

        self.covariance = None
        if isinstance(covariance, types.CovarianceMatrix):
            self.covariance = covariance.at.observable.match(self.data)
        elif covariance is not None:
            covariance = check_matrix(covariance, 'covariance')
            self.covariance = types.CovarianceMatrix(observable=self.data.clone(value=0. * self.data.value()), value=covariance)

        if self.precision is None:
            if self.covariance is None:
                raise ValueError('if precision is not provided, provide covariance')
            self.precision = self.covariance.inv(level=1) / scale_covariance
        self.correct_covariance = correct_covariance
        if self.correct_covariance['correction'] and self.correct_covariance['nobs'] is None:
            raise ValueError(f'provide nobs to apply correction {self.correct_covariance["correction"]}')
        if 'hartlap' in self.correct_covariance['correction']:
            nbins = self.precision.shape[0]
            nobs = self.correct_covariance['nobs']
            hartlap2007_factor = (nobs - nbins - 2.) / (nobs - 1.)
            if self.mpicomm.rank == 0:
                self.log_info(f'Covariance matrix with {nbins:d} points built from {nobs:d} observations.')
                self.log_info(f'...resulting in a Hartlap 2007 factor of {hartlap2007_factor:.4f}.')
            self.precision *= hartlap2007_factor

        self.runtime_info.requires = self.observables
        super(ObservablesGaussianLikelihood, self).initialize(self.flatdata, covariance=self.covariance, precision=self.precision, **kwargs)
        self.precision_hartlap2007 = self.precision.copy()

    def _pipeline_initialize(self, pipeline):
        varied_params = pipeline._params.select(varied=True, input=True)
        if 'percival' in self.correct_covariance['correction']:
            nbins = self.precision_hartlap2007.shape[0]
            nobs = self.correct_covariance['nobs']
            # Eq. 8 and 18 of https://arxiv.org/pdf/1312.4841.pdf
            A = 2. / (nobs - nbins - 1.) / (nobs - nbins - 4.)
            B = (nobs - nbins - 2.) / (nobs - nbins - 1.) / (nobs - nbins - 4.)

            params = set()

            def callback(calculator, params):
                params |= set(calculator.runtime_info.params.names())
                for require in calculator.runtime_info.requires:
                    callback(require, params)

            # Wrong, this will reinitialize calculators once more, which will result in unreferenced calculators if created at initialize() step in the current pipeline
            #for obs in self.observables: params |= set(obs.all_params.names())
            for obs in self.observables: callback(obs, params)
            params = [param for param in params if param in varied_params]
            nparams = len(params)
            self.percival2014_factor = (1 + B * (nbins - nparams)) / (1 + A + B * (nparams + 1))
            if self.mpicomm.rank == 0:
                self.log_info(f'Covariance matrix with {nbins:d} points built from {nobs:d} observations, varying {nparams:d} parameters.')
                self.log_info(f'...resulting in a Percival 2014 factor of {self.percival2014_factor:.4f}.')
            self.precision = self.precision_hartlap2007 / self.percival2014_factor

    def calculate(self):
        """Set :attr:`flatdiff` and :attr:`loglikelihood`."""
        self.flatdiff = self.flattheory - self.flatdata
        self.loglikelihood = -0.5 * chi2(self.flatdiff, self.precision)

    @property
    def flattheory(self):
        return jnp.concatenate([observable.flattheory for observable in self.observables], axis=0)


class SumLikelihood(BaseLikelihood):

    _attrs = ['loglikelihood', 'logprior']

    def initialize(self, likelihoods, **kwargs):
        if not utils.is_sequence(likelihoods): likelihoods = [likelihoods]
        self.likelihoods = list(likelihoods)
        super(SumLikelihood, self).initialize(**kwargs)
        self.runtime_info.requires = self.likelihoods

    def calculate(self):
        # more_calculate = solve doesn't apply to ``self.likelihoods``.
        self.loglikelihood = sum(likelihood.loglikelihood for likelihood in self.likelihoods)

    @property
    def size(self):
        # Theory vector size
        return sum(likelihood.size for likelihood in self.likelihoods)