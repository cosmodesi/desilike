import warnings

import numpy as np

from desilike.base import BaseCalculator, Parameter, ParameterCollection, ParameterArray
from desilike.observables import ObservableCovariance
from desilike.jax import numpy as jnp
from desilike.jax import jit
from desilike import plotting, utils


@jit
def chi2(flatdiff, precision):
    if precision.ndim == 1:
        return (flatdiff * precision).dot(flatdiff.T)
    return flatdiff.dot(precision).dot(flatdiff.T)


class BaseLikelihood(BaseCalculator):

    """Base class for likelihood."""
    _attrs = ['loglikelihood', 'logprior']
    name = None
    solved_default = '.marg'

    def initialize(self, catch_errors=None, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs['name']
        for name in self._attrs:
            if name not in self.params.basenames():
                self.params.set(Parameter(basename=name, namespace=self.name, latex=utils.outputs_to_latex(name), derived=True))
            param = self.params.select(basename=name)
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
        #self.fisher = None

    def more_initialize(self):
        pipeline = self.runtime_info.pipeline
        likelihoods = getattr(self, 'likelihoods', [self])

        # Reset precision and flatdata
        for likelihood in likelihoods:
            pipeline_initialize = getattr(likelihood, '_pipeline_initialize', None)
            if pipeline_initialize is not None:
                pipeline_initialize(pipeline)

        self._marginalize_precision()

    def get(self):
        pipeline = self.runtime_info.pipeline
        self.logprior = pipeline.params.prior(**pipeline.input_values)  # does not include solved params
        if pipeline.more_calculate is None:
            pipeline.more_calculate = self._solve
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
        likelihoods = getattr(self, 'likelihoods', [self])

        solved_params = []
        for param in all_params:
            solved = param.derived
            if param.solved and solved.startswith('.prec'):
                solved_params.append(param)

        # Reset precision and flatdata
        for likelihood in likelihoods:
            for name in ['precision', 'flatdata']:
                original_name = '_{}_original'.format(name)
                if hasattr(likelihood, original_name):
                    setattr(likelihood, name, getattr(likelihood, original_name))

        self.fisher = None

        if solved_params:
            solved_params = ParameterCollection(solved_params)

            from desilike.fisher import Fisher
            solve_likelihoods = [likelihood for likelihood in likelihoods if any(param in solved_params for param in likelihood.all_params)]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Derived parameter.*')
                solve_likelihood = SumLikelihood(solve_likelihoods)
                solve_likelihood.mpicomm = self.mpicomm
                solve_likelihood.runtime_info.pipeline.more_initialize = None
                solve_likelihood.runtime_info.pipeline.more_calculate = lambda: None
                all_params = solve_likelihood.all_params
                #solved_params = ParameterCollection(solved_params)
                for param in pipeline.params:
                    if param in solve_likelihood.all_params:
                        param = param.clone(derived=False if param in solved_params or param.depends else param.derived, fixed=param not in solved_params)
                        all_params.set(param)
                solve_likelihood.all_params = all_params

            # Just to reject from ``values`` parameters from which base ones are derived, and are not kept in solve_likelihood.all_params
            input_params = [param for param in solve_likelihood.all_params if param.name in pipeline.input_values]
            values = {param.name: pipeline.input_values[param.name] for param in input_params}
            #print(values)
            for param in solved_params: values[param.name] = 0.

            fisher = Fisher(solve_likelihood, method='auto')
            for likelihood in solve_likelihood.likelihoods:
                likelihood.precision = likelihood._precision_original = getattr(likelihood, '_precision_original', likelihood.precision)
                likelihood.flatdata = likelihood._flatdata_original = getattr(likelihood, '_flatdata_original', likelihood.flatdata)
            posterior_fisher = fisher(**values)
            derivs = fisher.mpicomm.bcast(fisher.likelihood_differentiation.samples, root=0)
            for param in solved_params: values[param.name] = getattr(param.prior, 'loc', 0.)
            solve_likelihood(**values)
            for likelihood, derivs in zip(solve_likelihoods, derivs):
                precision = likelihood._precision_original
                flatderiv = np.asarray(derivs)[1:]  # (len(solved_params), len(flatdata)) first is zero-lag
                if precision.ndim == 1:
                    derivp = flatderiv * precision
                else:
                    derivp = flatderiv.dot(precision)
                likelihood.precision = np.asarray(precision - derivp.T.dot(np.linalg.solve(- posterior_fisher._hessian, derivp)))
                likelihood.flatdata = np.asarray(likelihood._flatdata_original - (likelihood.flatdiff - derivs[()]))  # flatdiff = flattheory - flatdata

    def _solve(self):
        # Analytic marginalization, to be called, if desired, in get()
        pipeline = self.runtime_info.pipeline
        all_params = pipeline.params
        likelihoods = getattr(self, 'likelihoods', [self])

        solved_params, indices_marg = [], []
        for param in all_params:
            solved = param.derived
            if param.solved and not solved.startswith('.prec'):
                iparam = len(solved_params)
                solved_params.append(param)
                if solved.startswith('.auto'):
                    solved = solved.replace('.auto', self.solved_default)
                if solved.startswith('.marg'):  # marg
                    indices_marg.append(iparam)
                elif not solved.startswith('.best'):
                    raise ValueError('unknown option for solved = {}'.format(solved))

        indices_marg = np.array(indices_marg)
        x, dx, solve_likelihoods, derivs = [], [], [], None
        if solved_params:
            solved_params = ParameterCollection(solved_params)
            solve_likelihoods = []
            for likelihood in likelihoods:
                if any(param in solved_params for param in likelihood.all_params):
                    solve_likelihoods.append(likelihood)

            derived = pipeline.derived
            #pipeline.more_calculate = lambda: None
            self.fisher = getattr(self, 'fisher', None)

            if self.fisher is None or self.fisher.mpicomm is not self.mpicomm:
                #if self.fisher is not None: print(self.fisher.mpicomm is not self.mpicomm, self.fisher.varied_params != solved_params)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*Derived parameter.*')
                    solve_likelihood = SumLikelihood(solve_likelihoods)
                    solve_likelihood.runtime_info.pipeline.more_initialize = None
                    solve_likelihood.runtime_info.pipeline.more_calculate = lambda: None
                    all_params = solve_likelihood.all_params
                    #solved_params = ParameterCollection(solved_params)
                    for param in pipeline.params:
                        if param in solve_likelihood.all_params:
                            param = param.clone(derived=False if param in solved_params or param.depends else param.derived, fixed=param not in solved_params)
                            all_params.set(param)
                    solve_likelihood.all_params = all_params
                    # Such that when initializing, Fisher calls the pipeline (on all ranks of likelihood.mpicomm) at its current parameters
                    # and does not use default ones (call to self.fisher(**values) below only updates the calculator states on the last rank)
                    input_params = [param for param in solve_likelihood.all_params if param.name in pipeline.input_values]
                    values = {param.name: pipeline.input_values[param.name] for param in input_params}
                    solve_likelihood.runtime_info.pipeline.input_values = values

                def fisher(params):

                    import jax
                    names = solved_params.names()
                    values = jnp.array([params[name] for name in names])

                    def getter(values):
                        solve_likelihood({**params, **dict(zip(names, values))})
                        return [likelihood.flatdiff for likelihood in solve_likelihood.likelihoods]

                    #flatdiffs = [likelihood.flatdiff for likelihood in solve_likelihood.likelihoods]
                    #print('deriv')
                    flatderivs = jax.jacfwd(getter, argnums=0, has_aux=False, holomorphic=False)(values)
                    #print('diff')
                    flatdiffs = getter(values)
                    #print(values)
                    likelihoods_gradient, likelihoods_hessian = [], []
                    for ilike, likelihood in enumerate(solve_likelihood.likelihoods):
                        flatdiff, flatderiv = flatdiffs[ilike], flatderivs[ilike].T
                        precision = likelihood.precision
                        if precision.ndim == 1:
                            derivp = flatderiv * precision
                        else:
                            derivp = flatderiv.dot(precision)
                        likelihoods_gradient.append(- derivp.dot(flatdiff.T))
                        likelihoods_hessian.append(- derivp.dot(flatderiv.T))

                    prior_gradient, prior_hessian = [], []
                    for param, value in zip(solved_params, values):
                        loc, scale = getattr(param.prior, 'loc', 0.), getattr(param.prior, 'scale', np.inf)
                        prec = scale**(-2)
                        prior_gradient.append(- (value - loc) * prec)
                        prior_hessian.append(- prec)
                    prior_gradient = jnp.array(prior_gradient)
                    prior_hessian = jnp.diag(jnp.array(prior_hessian))
                    posterior_gradient = sum(likelihoods_gradient + [prior_gradient])
                    posterior_hessian = sum(likelihoods_hessian + [prior_hessian])
                    #print(np.diag(posterior_hessian), posterior_gradient)
                    dx = - jnp.linalg.solve(posterior_hessian, posterior_gradient)
                    x = values + dx
                    return x, dx, posterior_hessian, prior_hessian, likelihoods_hessian, likelihoods_gradient

                """
                    from desilike.fisher import Fisher
                    _fisher = Fisher(solve_likelihood, method='auto')

                    def fisher(params):
                        #print([params[name] for name in solved_params.names()])
                        p = _fisher(params)
                        x = p.mean()
                        dx = x - p._center
                        solved_indices = p._index(solved_params)
                        x, dx = x[solved_indices], dx[solved_indices]
                        posterior_hessian = p._hessian[np.ix_(solved_indices, solved_indices)]
                        prior_hessian = _fisher.prior_fisher._hessian[np.ix_(solved_indices, solved_indices)]
                        likelihoods_hessian = [fisher._hessian[np.ix_(solved_indices, solved_indices)] for fisher in _fisher.likelihood_fishers]
                        likelihoods_gradient = [fisher._gradient[solved_indices] for fisher in _fisher.likelihood_fishers]
                        #print(np.diag(posterior_hessian), p._gradient)
                        return x, dx, posterior_hessian, prior_hessian, likelihoods_hessian, likelihoods_gradient
                """
                fisher.input_params = input_params
                fisher.mpicomm = self.mpicomm
                self.fisher = fisher
                #self.fisher.varied_params = solved_params  # just to get same _derived attribute for solved_params != self.fisher.varied_params not to fail
                #assert self.fisher.varied_params == solved_params
                #params_bak, varied_params_bak = pipeline.params, pipeline.varied_params
                #pipeline._varied_params = solved_params  # to set varied_params
                #pipeline._params = ParameterCollection([param.clone(derived=False) if param in pipeline._varied_params else param.clone(fixed=True) for param in params_bak])
                #pipeline._varied_params.updated, pipeline._params.updated = False, False
                #self.fisher = Fisher(self, method='auto')
                #pipeline._params, pipeline._varied_params = params_bak, varied_params_bak
            #self.fisher.likelihood.runtime_info.pipeline.input_values = values
            #self.fisher.mpicomm = self.mpicomm
            #print('start fisher')
            values = {param.name: pipeline.input_values[param.name] for param in self.fisher.input_params}
            x, dx, posterior_hessian, prior_hessian, likelihoods_hessian, likelihoods_gradient = self.fisher(values)
            #print('stop fisher')
            #pipeline.derived = derived
            #pipeline.more_calculate = self._solve
            # flatdiff is theory - data
            derivs = [()]
            indices_derivs = [], []
            for iparam1, param1 in enumerate(solved_params):
                if param1.derived.endswith('not_derived'): continue  # do not export to .derived
                for iparam2, param2 in enumerate(solved_params[iparam1:]):
                    if param2.derived.endswith('not_derived'): continue
                    derivs.append((param1.name, param2.name))
                    indices_derivs[0].append(iparam1)
                    indices_derivs[1].append(iparam1 + iparam2)

        derived = pipeline.derived
        sum_loglikelihood = jnp.zeros(len(derivs) if solved_params and derived is not None else (), dtype='f8')
        sum_logprior = jnp.zeros((), dtype='f8')

        for param, xx in zip(solved_params, x):
            sum_logprior += param.prior(xx)
            # hack to run faster than calling param.prior --- saving ~ 0.0005 s
            #sum_logprior += -0.5 * (xx - param.prior.attrs['loc'])**2 / param.prior.attrs['scale']**2 if param.prior.dist == 'norm' else 0.
            #pipeline.input_values[param.name] = xx  # may lead to instabilities
            if derived is not None:
                derived.set(ParameterArray(xx, param=param))

        if solved_params and derived is not None:
            sum_logprior = jnp.insert(prior_hessian[indices_derivs], 0, sum_logprior + self.logprior)
        else:
            sum_logprior += self.logprior

        for likelihood in likelihoods:
            loglikelihood = jnp.array(likelihood.loglikelihood)
            if likelihood in solve_likelihoods:
                likelihood_index = solve_likelihoods.index(likelihood)
                likelihood_hessian = likelihoods_hessian[likelihood_index]
                # Note: priors of solved params have already been added
                loglikelihood += 1. / 2. * dx.dot(likelihood_hessian).dot(dx)
                loglikelihood += likelihoods_gradient[likelihood_index].dot(dx)
                # Set derived values
                if derived is not None:
                    loglikelihood = jnp.insert(likelihood_hessian[indices_derivs], 0, loglikelihood)
                    derived.set(ParameterArray(loglikelihood, param=likelihood._param_loglikelihood, derivs=derivs))
            sum_loglikelihood += loglikelihood

        if indices_marg.size:
            marg_likelihood = -1. / 2. * jnp.linalg.slogdet(- posterior_hessian[np.ix_(indices_marg, indices_marg)])[1]
            # sum_loglikelihood += 1. / 2. * len(indices_marg) * np.log(2. * np.pi)
            # Convention: in the limit of no likelihood constraint on dx, no change to the loglikelihood
            # This allows to ~ keep the interpretation in terms of -1. / 2. * chi2
            ip = jnp.diag(prior_hessian)[indices_marg]
            marg_likelihood += 1. / 2. * jnp.sum(jnp.log(jnp.where(ip > 0, ip, 1.)))  # logdet
            # sum_loglikelihood -= 1. / 2. * len(indices_marg) * np.log(2. * np.pi)
            if derived is not None:
                marg_likelihood = marg_likelihood * np.array([1.] + [0.] * (len(derivs) - 1), dtype='f8')
            sum_loglikelihood += marg_likelihood

        self.loglikelihood = sum_loglikelihood
        self.logprior = sum_logprior

        if derived is not None:
            derived.set(ParameterArray(self.loglikelihood, param=self._param_loglikelihood, derivs=derivs))
            derived.set(ParameterArray(self.logprior, param=self._param_logprior, derivs=derivs))

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
        """Sum likelihoods- ``self`` and ``other``: return :class:`SumLikelihood` instance."""
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

    def __getstate__(self):
        state = {}
        for name in ['flatdiff', 'flatdata', 'covariance', 'precision', 'transform', 'loglikelihood']:
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

    covariance : array, default=None
        Covariance matrix (or its diagonal) for input ``observables``.
        If ``None``, covariance matrix is computed on-the-fly using observables' mocks.

    scale_covariance : float, default=1.
        Scale precision by the inverse of this value.

    correct_covariance : str, default='hartlap-percival2014'
        Only applies if mocks are provided to input observables.
        'hartlap' to apply Hartlap 2007 factor (https://arxiv.org/abs/astro-ph/0608064).
        'percival2014' to apply Percival 2014 factor (https://arxiv.org/abs/1312.4841).
        Can be a dictionary to specify the number of observations, ``{'nobs': nobs, 'correction': 'hartlap-percival2014'}``.

    precision : array, default=None
        Precision matrix to be used instead of the inverse covariance.
    """
    def initialize(self, observables, covariance=None, scale_covariance=1., correct_covariance='hartlap-percival2014', precision=None, **kwargs):
        if not utils.is_sequence(observables):
            observables = [observables]
        self.nobs = getattr(covariance, 'nobs', None)
        if isinstance(correct_covariance, dict):
            self.nobs = correct_covariance.get('nobs', self.nobs)
            correct_covariance = correct_covariance['correction']
        self.observables = list(observables)
        for obs in observables: obs.all_params  # to set observable's pipelines, and initialize once (percival factor below requires all_params)
        covariance, scale_covariance, precision = (self.mpicomm.bcast(obj if self.mpicomm.rank == 0 else None, root=0) for obj in (covariance, scale_covariance, precision))
        if covariance is None:
            nmocks = [self.mpicomm.bcast(len(obs.mocks) if self.mpicomm.rank == 0 and getattr(obs, 'mocks', None) is not None else 0) for obs in self.observables]
            if any(nmocks):
                if self.nobs is None: self.nobs = nmocks[0]
                if not all(nmock == nmocks[0] for nmock in nmocks):
                    raise ValueError('Provide the same number of mocks for each observable, found {}'.format(nmocks))
                if self.mpicomm.rank == 0:
                    list_y = [np.concatenate(y, axis=0) for y in zip(*[obs.mocks for obs in self.observables])]
                    covariance = np.cov(list_y, rowvar=False, ddof=1)
                covariance = self.mpicomm.bcast(covariance if self.mpicomm.rank == 0 else None, root=0)
            elif all(getattr(obs, 'covariance', None) is not None for obs in self.observables):
                covariances = [obs.covariance for obs in self.observables]
                if self.nobs is None:
                    nobs = [getattr(obs, 'nobs', None) for obs in self.observables]
                    if all(nobs):
                        self.nobs = np.mean(nobs).astype('i4')
                size = sum(cov.shape[0] for cov in covariances)
                covariance = np.zeros((size, size), dtype='f8')
                start = 0
                for cov in covariances:
                    stop = start + cov.shape[0]
                    sl = slice(start, stop)
                    covariance[sl, sl] = cov
                    start = stop
            elif precision is None:
                raise ValueError('Observables must have mocks or their own covariance if global covariance or precision matrix not provided')
        self.flatdata = np.concatenate([obs.flatdata for obs in self.observables], axis=0)

        def check_matrix(matrix, name):
            if matrix is None:
                return None
            matrix = np.atleast_2d(matrix).copy()
            if matrix.shape != (matrix.shape[0],) * 2:
                raise ValueError('{} must be a square matrix, but found shape {}'.format(name, matrix.shape))
            mshape = '({0}, {0})'.format(matrix.shape[0])
            shape = '({0}, {0})'.format(self.flatdata.size)
            shape_obs = '({0}, {0})'.format(' + '.join(['{:d}'.format(obs.flatdata.size) for obs in self.observables]))
            if matrix.shape[0] != self.flatdata.size:
                raise ValueError('based on provided observables, {} expected to be a matrix of shape {} = {}, but found {}'.format(name, shape, shape_obs, mshape))
            return matrix

        if isinstance(covariance, ObservableCovariance):
            cov_nobservables = len(covariance.observables())
            if len(self.observables) != cov_nobservables:
                raise ValueError('provided {:d} observables, but the covariance contains {:d}'.format(len(self.observables), cov_nobservables))
            for iobs, obs in enumerate(self.observables):
                array = obs.to_array()
                x = [(edges[:-1] + edges[1:]) / 2. for edges in array.edges()]
                covariance = covariance.xmatch(observables=iobs, x=x, projs=array.projs, select_projs=True, method='mid')
            covariance = covariance.view()

        self.precision = check_matrix(precision, 'precision')
        self.covariance = check_matrix(covariance, 'covariance')

        self.runtime_info.requires = self.observables

        if self.covariance is not None:
            self.log_info('Rescaling covariance with a factor {:.4e}'.format(scale_covariance))
            self.covariance *= scale_covariance
            start, slices, covariances = 0, [], []
            for obs in observables:
                stop = start + len(obs.flatdata)
                sl = slice(start, stop)
                slices.append(sl)
                obs.covariance = self.covariance[sl, sl]  # Set each observable's (scaled) covariance (for, e.g., plots)
                start = stop
            if self.precision is None:
                # Block-inversion is usually more numerically stable
                self.precision = utils.blockinv([[self.covariance[sl1, sl2] for sl2 in slices] for sl1 in slices])
        else:
            self.log_info('Rescaling precision with a factor {:.4e}'.format(1/scale_covariance))
            self.precision /= scale_covariance
        self.correct_covariance = correct_covariance
        if self.nobs is not None and 'hartlap' in self.correct_covariance:
            nbins = self.precision.shape[0]
            self.hartlap2007_factor = (self.nobs - nbins - 2.) / (self.nobs - 1.)
            if self.mpicomm.rank == 0:
                self.log_info('Covariance matrix with {:d} points built from {:d} observations.'.format(nbins, self.nobs))
                self.log_info('...resulting in a Hartlap 2007 factor of {:.4f}.'.format(self.hartlap2007_factor))
            self.precision *= self.hartlap2007_factor
        super(ObservablesGaussianLikelihood, self).initialize(self.flatdata, covariance=self.covariance, precision=self.precision, **kwargs)
        self.precision_hartlap2007 = self.precision.copy()

    def _pipeline_initialize(self, pipeline):
        varied_params = pipeline._params.select(varied=True, input=True)
        if self.nobs is not None and 'percival' in self.correct_covariance:
            nbins = self.precision_hartlap2007.shape[0]
            # eq. 8 and 18 of https://arxiv.org/pdf/1312.4841.pdf
            A = 2. / (self.nobs - nbins - 1.) / (self.nobs - nbins - 4.)
            B = (self.nobs - nbins - 2.) / (self.nobs - nbins - 1.) / (self.nobs - nbins - 4.)
            params = set()
            for obs in self.observables: params |= set(obs.all_params.names())
            params = [param for param in params if param in varied_params]
            nparams = len(params)
            self.percival2014_factor = (1 + B * (nbins - nparams)) / (1 + A + B * (nparams + 1))
            if self.mpicomm.rank == 0:
                self.log_info('Covariance matrix with {:d} points built from {:d} observations, varying {:d} parameters.'.format(nbins, self.nobs, nparams))
                self.log_info('...resulting in a Percival 2014 factor of {:.4f}.'.format(self.percival2014_factor))
            self.precision = self.precision_hartlap2007 / self.percival2014_factor

    def calculate(self):
        self.flatdiff = self.flattheory - self.flatdata
        self.loglikelihood = -0.5 * chi2(self.flatdiff, self.precision)

    @property
    def flattheory(self):
        return jnp.concatenate([obs.flattheory for obs in self.observables], axis=0)

    def to_covariance(self):
        from desilike.observables import ObservableCovariance
        return ObservableCovariance(value=self.covariance, observables=[o.to_array() for o in self.observables])

    @plotting.plotter
    def plot_covariance_matrix(self, corrcoef=True, **kwargs):
        """
        Plot covariance matrix.

        Parameters
        ----------
        corrcoef : bool, default=True
            If ``True``, plot the correlation matrix; else the covariance.

        barlabel : str, default=None
            Optionally, label for the color bar.

        label1 : str, list of str, default=None
            Optionally, label(s) for the observable(s).

        figsize : int, tuple, default=None
            Optionally, figure size.

        norm : matplotlib.colors.Normalize, default=None
            Scales the covariance / correlation to the canonical colormap range [0, 1] for mapping to colors.
            By default, the covariance / correlation range is mapped to the color bar range using linear scaling.

        labelsize : int, default=None
            Optionally, size for labels.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``len(self.observables) * len(self.observables)`` axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from desilike.observables.plotting import plot_covariance_matrix
        cumsize = np.insert(np.cumsum([len(obs.flatdata) for obs in self.observables]), 0, 0)
        mat = [[self.covariance[start1:stop1, start2:stop2] for start2, stop2 in zip(cumsize[:-1], cumsize[1:])] for start1, stop1 in zip(cumsize[:-1], cumsize[1:])]
        return plot_covariance_matrix(mat, corrcoef=corrcoef, **kwargs)


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