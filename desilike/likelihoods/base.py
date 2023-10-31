import numpy as np

from desilike.base import BaseCalculator, Parameter, ParameterCollection, ParameterArray
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

    def initialize(self, catch_errors=None):
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

    def _solve(self):
        # Analytic marginalization, to be called, if desired, in get()
        pipeline = self.runtime_info.pipeline
        all_params = pipeline.params
        likelihoods = getattr(self, 'likelihoods', [self])

        solved_params, indices_marg = [], []
        for param in all_params:
            solved = param.derived
            if param.solved:
                iparam = len(solved_params)
                solved_params.append(param)
                if solved == '.auto':
                    solved = self.solved_default
                if solved == '.marg':  # marg
                    indices_marg.append(iparam)
                elif solved != '.best':
                    raise ValueError('Unknown option for solved = {}'.format(solved))

        dx, x, solve_likelihoods, derivs = [], [], [], None

        if solved_params:
            solved_params = ParameterCollection(solved_params)
            from desilike.fisher import Fisher

            solve_likelihoods = [likelihood for likelihood in likelihoods if any(param.solved for param in likelihood.all_params)]
            values = dict(pipeline.input_values)
            for param in solved_params:
                if not np.isfinite(values[param.name]): values[param.name] = param.value

            derived = pipeline.derived
            #pipeline.more_calculate = lambda: None
            self.fisher = getattr(self, 'fisher', None)
            if self.fisher is None or self.fisher.mpicomm is not self.mpicomm or self.fisher.varied_params != solved_params:
                #if self.fisher is not None: print(self.fisher.mpicomm is not self.mpicomm, self.fisher.varied_params != solved_params)
                solve_likelihood = SumLikelihood(solve_likelihoods)
                all_params = solve_likelihood.all_params
                #solved_params = ParameterCollection(solved_params)
                for param in pipeline.params:
                    if param in solve_likelihood.all_params:
                        param = param.clone(derived=False) if param in solved_params else param.clone(fixed=True)
                        all_params.set(param)
                solve_likelihood.all_params = all_params
                solve_likelihood.runtime_info.pipeline.more_calculate = lambda: None
                self.fisher = Fisher(solve_likelihood, method='auto')
                self.fisher.varied_params = solved_params  # just to get same _derived attribute
                #assert self.fisher.varied_params == solved_params
                #params_bak, varied_params_bak = pipeline.params, pipeline.varied_params
                #pipeline._varied_params = solved_params  # to set varied_params
                #pipeline._params = ParameterCollection([param.clone(derived=False) if param in pipeline._varied_params else param.clone(fixed=True) for param in params_bak])
                #pipeline._varied_params.updated, pipeline._params.updated = False, False
                #self.fisher = Fisher(self, method='auto')
                #pipeline._params, pipeline._varied_params = params_bak, varied_params_bak
            posterior_fisher = self.fisher(**values)
            #pipeline.derived = derived
            #pipeline.more_calculate = self._solve
            # flatdiff is theory - data
            x = posterior_fisher.mean()
            dx = x - posterior_fisher._center
            derivs = [()]
            for iparam1, param1 in enumerate(solved_params):
                for param2 in solved_params[iparam1:]:
                    derivs.append((param1.name, param2.name))
            indices = posterior_fisher._index(solved_params)
            indices_derivs = [], []
            for iindex1, index1 in enumerate(indices):
                for index2 in indices[iindex1:]:
                    indices_derivs[0].append(index1)
                    indices_derivs[1].append(index2)
            #indices_derivs = posterior_fisher._index([deriv[0] for deriv in derivs[1:]]), posterior_fisher._index([deriv[1] for deriv in derivs[1:]])

        sum_loglikelihood = np.zeros(len(derivs) if solved_params else None, dtype='f8')
        sum_logprior = np.zeros((), dtype='f8')
        derived = pipeline.derived

        for param, xx in zip(solved_params, x):
            sum_logprior += param.prior(xx)
            # hack to run faster than calling param.prior --- saving ~ 0.0005 s
            #sum_logprior += -0.5 * (xx - param.prior.attrs['loc'])**2 / param.prior.attrs['scale']**2 if param.prior.dist == 'norm' else 0.
            pipeline.input_values[param.name] = xx
            if derived is not None:
                derived.set(ParameterArray(xx, param=param))

        if solved_params:
            sum_logprior = np.insert(self.fisher.prior_fisher._hessian[indices_derivs], 0, sum_logprior)
        for ilikelihood, likelihood in enumerate(likelihoods):
            loglikelihood = float(likelihood.loglikelihood)
            if likelihood in solve_likelihoods:
                likelihood_fisher = self.fisher.likelihood_fishers[ilikelihood]
                # Note: priors of solved params have already been added
                loglikelihood += 1. / 2. * dx.dot(likelihood_fisher._hessian).dot(dx)
                loglikelihood += likelihood_fisher._gradient.dot(dx)
                loglikelihood = np.insert(likelihood_fisher._hessian[indices_derivs], 0, loglikelihood)
                # Set derived values
                if derived is not None:
                    derived.set(ParameterArray(loglikelihood, param=likelihood._param_loglikelihood, derivs=derivs))
            sum_loglikelihood += loglikelihood

        if indices_marg:
            sum_loglikelihood.flat[0] -= 1. / 2. * np.linalg.slogdet(- posterior_fisher._hessian[np.ix_(indices_marg, indices_marg)])[1]
            # sum_loglikelihood += 1. / 2. * len(indices_marg) * np.log(2. * np.pi)
            # Convention: in the limit of no likelihood constraint on dx, no change to the loglikelihood
            # This allows to ~ keep the interpretation in terms of -1. / 2. * chi2
            ip = self.fisher.prior_fisher._hessian[indices_marg]
            sum_loglikelihood.flat[0] += 1. / 2. * np.sum(np.log(ip[ip > 0.]))  # logdet
            # sum_loglikelihood -= 1. / 2. * len(indices_marg) * np.log(2. * np.pi)
        self.loglikelihood = sum_loglikelihood
        sum_logprior.flat[0] += self.logprior
        self.logprior = sum_logprior

        if derived is not None:
            derived.set(ParameterArray(self.loglikelihood, param=self._param_loglikelihood, derivs=derivs))
            derived.set(ParameterArray(self.logprior, param=self._param_logprior, derivs=derivs))

        return self.loglikelihood.flat[0] + self.logprior.flat[0]

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

    precision : array, default=None
        Precision matrix to be used instead of the inverse covariance.
    """
    def initialize(self, observables, covariance=None, scale_covariance=1., correct_covariance='hartlap-percival2014', precision=None, name=None, **kwargs):
        self.name = name
        if not utils.is_sequence(observables):
            observables = [observables]
        self.nobs = None
        self.observables = [obs.runtime_info.initialize() for obs in observables]
        covariance, scale_covariance, precision = (self.mpicomm.bcast(obj if self.mpicomm.rank == 0 else None, root=0) for obj in (covariance, scale_covariance, precision))
        if covariance is None:
            nmocks = [self.mpicomm.bcast(len(obs.mocks) if self.mpicomm.rank == 0 and getattr(obs, 'mocks', None) is not None else 0) for obs in self.observables]
            if any(nmocks):
                self.nobs = nmocks[0]
                if not all(nmock == nmocks[0] for nmock in nmocks):
                    raise ValueError('Provide the same number of mocks for each observable, found {}'.format(nmocks))
                if self.mpicomm.rank == 0:
                    list_y = [np.concatenate(y, axis=0) for y in zip(*[obs.mocks for obs in self.observables])]
                    covariance = np.cov(list_y, rowvar=False, ddof=1)
                covariance = self.mpicomm.bcast(covariance if self.mpicomm.rank == 0 else None, root=0)
            elif all(getattr(obs, 'covariance', None) is not None for obs in self.observables):
                covariances = [obs.covariance for obs in self.observables]
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
                return matrix
            matrix = np.atleast_2d(matrix).copy()
            if matrix.shape != (matrix.shape[0],) * 2:
                raise ValueError('{} must be a square matrix, but found shape {}'.format(name, matrix.shape))
            shape = (self.flatdata.size,) * 2
            if matrix.shape != shape:
                raise ValueError('Based on provided observables, {} expected to be a matrix of shape {}, but found {}'.format(name, shape, matrix.shape))
            return matrix

        self.precision = check_matrix(precision, 'precision')
        self.covariance = check_matrix(covariance, 'covariance')

        if self.covariance is not None:
            self.covariance *= scale_covariance
            # Set each observable's covariance (for, e.g., plots)
            start, slices = 0, []
            for obs in observables:
                stop = start + len(obs.flatdata)
                sl = slice(start, stop)
                slices.append(sl)
                obs.covariance = self.covariance[sl, sl]
                start = stop
            if self.precision is None:
                # Block-inversion is usually more numerically stable
                self.precision = utils.blockinv([[self.covariance[sl1, sl2] for sl2 in slices] for sl1 in slices])
        else:
            self.precision /= scale_covariance
        nbins = self.precision.shape[0]
        BaseLikelihood.initialize(self, **kwargs)
        self.runtime_info.requires = self.observables
        if self.nobs is not None:
            if 'hartlap' in correct_covariance:
                self.hartlap2007_factor = (self.nobs - nbins - 2.) / (self.nobs - 1.)
                if self.mpicomm.rank == 0:
                    self.log_info('Covariance matrix with {:d} points built from {:d} observations.'.format(nbins, self.nobs))
                    self.log_info('...resulting in a Hartlap 2007 factor of {:.4f}.'.format(self.hartlap2007_factor))
                self.precision *= self.hartlap2007_factor
            if 'percival' in correct_covariance:
                # eq. 8 and 18 of https://arxiv.org/pdf/1312.4841.pdf
                A = 2. / (self.nobs - nbins - 1.) / (self.nobs - nbins - 4.)
                B = (self.nobs - nbins - 2.) / (self.nobs - nbins - 1.) / (self.nobs - nbins - 4.)
                params = set()
                for obs in self.observables: params |= set(obs.all_params.names(varied=True))
                nparams = len(params)
                self.percival2014_factor = (1 + B * (nbins - nparams)) / (1 + A + B * (nparams + 1))
                if self.mpicomm.rank == 0:
                    self.log_info('Covariance matrix with {:d} points built from {:d} observations, varying {:d} parameters.'.format(nbins, self.nobs, nparams))
                    self.log_info('...resulting in a Percival 2014 factor of {:.4f}.'.format(self.percival2014_factor))
                self.precision /= self.percival2014_factor

    def calculate(self):
        self.flatdiff = self.flattheory - self.flatdata
        self.loglikelihood = -0.5 * chi2(self.flatdiff, self.precision)

    @property
    def flattheory(self):
        return jnp.concatenate([obs.flattheory for obs in self.observables], axis=0)

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