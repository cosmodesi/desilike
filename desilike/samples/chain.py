import os
import re
import glob

import numpy as np

from desilike.parameter import ParameterCollection, Parameter, ParameterPrior, ParameterArray, Samples, ParameterCovariance, _reshape, is_parameter_sequence
from desilike import LikelihoodFisher

from . import utils


def vectorize(func):
    """Vectorize input function ``func`` for input parameters."""
    from functools import wraps

    @wraps(func)
    def wrapper(self, params=None, *args, **kwargs):
        if params is None:
            params = self.params()

        def _reshape(result, pshape):

            def __reshape(array):
                try:
                    array.shape = array.shape[:array.ndim - len(pshape)] + pshape
                except AttributeError:
                    pass
                return array

            if isinstance(result, tuple):  # for :meth:`Chain.interval`
                for res in result:
                    __reshape(res)
            else:
                __reshape(result)
            return result

        if is_parameter_sequence(params):
            params = [self[param].param for param in params]
            return [_reshape(func(self, param, *args, **kwargs), param.shape) for param in params]
        return _reshape(func(self, params, *args, **kwargs), self[params].param.shape)

    return wrapper


def _get_solved_covariance(chain, params=None):
    solved_params = chain.params(solved=True)
    if params is None:
        params = solved_params
    solved_indices = [solved_params.index(param) for param in params]
    logposterior = -(chain[chain._loglikelihood] + chain[chain._logprior])
    logposterior = np.array([[logposterior[param1, param2].ravel() for param2 in solved_params] for param1 in solved_params])
    logposterior = np.moveaxis(logposterior, -1, 0).reshape(chain.shape + (len(solved_params),) * 2)
    return np.linalg.inv(logposterior)[(Ellipsis,) + np.ix_(solved_indices, solved_indices)]


class Chain(Samples):

    """Class that holds samples drawn from posterior (in practice, :class:`Samples` with a log-posterior and optional weights)."""

    _type = ParameterArray
    _attrs = Samples._attrs + ['_logposterior', '_loglikelihood', '_logprior', '_aweight', '_fweight', '_weight']

    def __init__(self, data=None, params=None, logposterior='logposterior', loglikelihood='loglikelihood', logprior='logprior', aweight='aweight', fweight='fweight', weight='weight', attrs=None):
        """
        Initialize :class:`Chain`.

        Parameters
        ----------
        data : list, dict, Samples
            Can be:

            - list of :class:`ParameterArray`, or :class:`np.ndarray` if list of parameters
              (or :class:`ParameterCollection`) is provided in ``params``
            - dictionary mapping parameter to array

        params : list, ParameterCollection
            Optionally, list of parameters.

        logposterior : str, default='logposterior'
            Name of log-posterior in ``data``.

        aweight : str, default='aweight'
            Name of sample weights (which default to 1. if not provided in ``data``).

        fweight : str, default='fweight'
            Name of sample frequency weights (which default to 1 if not provided in ``data``).

        weight : str, default='weight'
            Name of sample total weight. It is defined as the product of :attr:`aweight` and :attr:`fweight`,
            hence should not provided in ``data``.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        self._logposterior = str(logposterior)
        self._loglikelihood = str(loglikelihood)
        self._logprior = str(logprior)
        self._aweight = str(aweight)
        self._fweight = str(fweight)
        self._weight = str(weight)
        self._derived = [self._logposterior, self._loglikelihood, self._logprior, self._aweight, self._fweight, self._weight]
        super(Chain, self).__init__(data=data, params=params, attrs=attrs)
        for name in self._derived:
            if name in self:
                self[name].param.update(derived=True)

    @property
    def aweight(self):
        """Sample weights (floats)."""
        if self._aweight not in self:
            self[Parameter(self._aweight, derived=True)] = np.ones(self.shape, dtype='f8')
        return self[self._aweight]

    @property
    def fweight(self):
        """Sample frequency weights (integers)."""
        if self._fweight not in self:
            self[Parameter(self._fweight, derived=True)] = np.ones(self.shape, dtype='i8')
        return self[self._fweight]

    @property
    def logposterior(self):
        """Log-posterior."""
        if self._logposterior not in self:
            self[Parameter(self._logposterior, derived=True)] = np.zeros(self.shape, dtype='f8')
        return self[self._logposterior]

    @aweight.setter
    def aweight(self, item):
        """Set weights (floats)."""
        self[Parameter(self._aweight, derived=True)] = item

    @fweight.setter
    def fweight(self, item):
        """Set frequency weights (integers)."""
        self[Parameter(self._fweight, derived=True)] = item

    @logposterior.setter
    def logposterior(self, item):
        """Set log-posterior."""
        self[Parameter(self._logposterior, derived=True)] = item

    @property
    def weight(self):
        """Return total weight, as the product of :attr:`aweight` and :attr:`fweight`."""
        return ParameterArray(self.aweight * self.fweight, Parameter(self._weight, latex=utils.outputs_to_latex(self._weight)))

    def remove_burnin(self, burnin=0):
        """
        Return new samples with burn-in removed.

        Parameters
        ----------
        burnin : float, int
            If ``burnin`` between 0 and 1, remove that fraction of samples.
            Else, remove ``burnin`` (integer) first points.

        Returns
        -------
        samples : Chain
        """
        if 0 < burnin < 1:
            burnin = burnin * len(self)
        burnin = int(burnin + 0.5)
        return self[burnin:]

    def sample_solved(self, size=1, seed=42):
        """Sample parameters that have been analytic marginalized over (``solved``)."""
        new = self.deepcopy()
        solved_params = self.params(solved=True)
        if not solved_params: return new
        covariance = _get_solved_covariance(self, params=solved_params)
        L = np.moveaxis(np.linalg.cholesky(covariance), (-2, -1), (0, 1))
        new.data = []
        for array in self:
            new.set(array.clone(value=np.repeat(array, size, axis=self.ndim - 1)))
        rng = np.random.RandomState(seed=seed)
        noise = rng.standard_normal((len(solved_params),) + self.shape + (size,))
        values = np.sum(noise[None, ...] * L[..., None], axis=0)
        for param, value in zip(solved_params, values):
            new[param] = new[param].clone(value=new[param] + value.reshape(new.shape), param=param.clone(derived=False))
        dlogposterior = 0.
        for param in [self._loglikelihood, self._logprior]:
            icov = - np.array([[self[param][param1, param2] for param2 in solved_params] for param1 in solved_params])
            log = -0.5 * np.sum(values[None, ...] * icov[..., None] * values[:, None, ...], axis=(0, 1)).reshape(new.shape)
            new[param] = self[param].clone(value=new[param][()] + log, derivs=None)
            dlogposterior += log
        new.logposterior[...] += dlogposterior
        return new

    def get(self, name, *args, **kwargs):
        """
        Return parameter array of name ``name`` in chain.

        Parameters
        ----------
        name : Parameter, str
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.

        Returns
        -------
        array : ParameterArray
        """
        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = kwargs['default']
        try:
            return self.data[self.index(name)]
        except KeyError:
            if name == self._weight:
                return self.weight
            if has_default:
                return default
            raise KeyError('Column {} does not exist'.format(name))

    @classmethod
    def read_getdist(cls, base_fn, ichains=None, concatenate=False):
        """
        Load samples in *CosmoMC* format, i.e.:

        - '_{ichain}.txt' files for sample values
        - '.paramnames' files for parameter names / latex
        - '.ranges' for parameter ranges

        Note
        ----
        GetDist package *is not* required.

        Parameters
        ----------
        base_fn : str, Path
            Base *CosmoMC* file name. Will be appended by '_{ichain}.txt' for sample values,
            '.paramnames' for parameter names and '.ranges' for parameter ranges.

        ichains : int, tuple, list, default=None
            Chain numbers to load. Defaults to all chains matching pattern '{base_fn}*.txt'.
            If a single number is provided, return a unique chain.
            If multiple numbers are provided, or is ``None``, return a list of chains (see ``concatenate``).

        concatenate : bool, default=False
            If ``True``, concatenate all chains in one.

        Returns
        -------
        samples : list, Chain
            Chain or list of chains.
        """
        params_fn = '{}.paramnames'.format(base_fn)
        cls.log_info('Loading params file: {}.'.format(params_fn))
        params = ParameterCollection()
        with open(params_fn) as file:
            for line in file:
                line = [item.strip() for item in line.split(maxsplit=1)]
                if line:
                    name, latex = line
                    derived = name.endswith('*')
                    if derived: name = name[:-1]
                    params.set(Parameter(basename=name, latex=latex.replace('\n', ''), fixed=False, derived=derived))

            ranges_fn = '{}.ranges'.format(base_fn)
            if os.path.exists(ranges_fn):
                cls.log_info('Loading parameter ranges from {}.'.format(ranges_fn))
                with open(ranges_fn) as file:
                    for line in file:
                        name, low, high = [item.strip() for item in line.split()]
                        latex = latex.replace('\n', '')
                        limits = []
                        for lh, li in zip([low, high], [-np.inf, np.inf]):
                            if lh == 'N': lh = li
                            else: lh = float(lh)
                            limits.append(lh)
                        if name in params:
                            params[name].update(prior=ParameterPrior(limits=limits))
            else:
                cls.log_info('Parameter ranges file {} does not exist.'.format(ranges_fn))

        chain_fn = '{}_{{}}.txt'.format(base_fn)
        isscalar = False
        chain_fns = []
        if ichains is not None:
            isscalar = np.ndim(ichains) == 0
            if isscalar:
                ichains = [ichains]
            for ichain in ichains:
                chain_fns.append(chain_fn.format('{:d}'.format(ichain)))
        else:
            chain_fns = glob.glob(chain_fn.format('[0-9]*'))

        toret = []
        for chain_fn in chain_fns:
            cls.log_info('Loading chain file: {}.'.format(chain_fn))
            array = np.loadtxt(chain_fn, unpack=True)
            new = cls()
            new.fweight, new.logposterior = array[0], -array[1]
            for param, values in zip(params, array[2:]):
                new.set(ParameterArray(values, param))
            toret.append(new)
        for new in toret:
            for param in new.params(basename='chi2_*'):
                namespace = param.name[4:]
                if namespace == 'prior':
                    new_param = param.clone(basename=new._logprior)
                else:
                    new_param = param.clone(basename=new._loglikelihood, namespace=namespace)
                new[new_param] = -0.5 * new[param]
        if isscalar:
            return toret[0]
        if concatenate:
            return cls.concatenate(toret)
        return toret

    def write_getdist(self, base_fn, params=None, ichain=None, fmt='%.18e', delimiter=' ', **kwargs):
        """
        Save samples to disk in *CosmoMC* format.

        Note
        ----
        GetDist package *is not* required.

        Parameters
        ----------
        base_fn : str, Path
            Base *CosmoMC* file name. Will be prepended by '_{ichain}.txt' for sample values,
            '.paramnames' for parameter names and '.ranges' for parameter ranges.

        params : list, ParameterCollection, default=None
            Parameters to save samples of (weight and log-posterior are added anyway). Defaults to all parameters.

        ichain : int, default=None
            If not ``None``, append '_{ichain:d}' to ``base_fn``.

        fmt : str, default='%.18e'
            How to format floats.

        delimiter : str, default=' '
            String or character separating columns.

        ichain : int, default=None
            Chain number to append to file name, i.e. sample values will be saved as '{base_fn}_{ichain}.txt'.
            If ``None``, does not append any number, sample values will be saved as '{base_fn}.txt'.

        kwargs : dict
            Optional arguments for :func:`numpy.savetxt`.
        """
        if self.params(solved=True):
            self = self.sample_solved()
        if params is None: params = self.params()
        columns = list([str(param) for param in params])
        outputs_columns = [self._weight, self._logposterior]
        shape = self.shape
        outputs = [array.param.name for array in self if array.shape != shape]
        for column in outputs:
            if column in columns: del columns[columns.index(column)]
        data = self.to_array(params=outputs_columns + columns, struct=False, derivs=()).reshape(-1, self.size)
        data[1] *= -1
        data = data.T
        utils.mkdir(os.path.dirname(base_fn))
        chain_fn = '{}.txt'.format(base_fn) if ichain is None else '{}_{:d}.txt'.format(base_fn, ichain)
        self.log_info('Saving chain to {}.'.format(chain_fn))
        np.savetxt(chain_fn, data, header='', fmt=fmt, delimiter=delimiter, **kwargs)

        output = ''
        params = self.params(name=columns)
        for param in params:
            tmp = '{}* {}\n' if getattr(param, 'derived', getattr(param, 'fixed')) else '{} {}\n'
            output += tmp.format(param.name, param.latex())
        params_fn = '{}.paramnames'.format(base_fn)
        self.log_info('Saving parameter names to {}.'.format(params_fn))
        with open(params_fn, 'w') as file:
            file.write(output)

        output = ''
        for param in params:
            limits = param.prior.limits
            limits = tuple('N' if limit is None or np.abs(limit) == np.inf else limit for limit in limits)
            output += '{} {} {}\n'.format(param.name, limits[0], limits[1])
        ranges_fn = '{}.ranges'.format(base_fn)
        self.log_info('Saving parameter ranges to {}.'.format(ranges_fn))
        with open(ranges_fn, 'w') as file:
            file.write(output)

    def to_getdist(self, params=None, label=None, **kwargs):
        """
        Return GetDist hook to samples.

        Note
        ----
        GetDist package *is* required.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to save samples of (weight and log-posterior are added anyway). Defaults to all parameters.

        label : str, default=None
            Name for  GetDist to use for these samples.

        **kwargs : dict
            Optional arguments for :class:`getdist.MCSamples`.

        Returns
        -------
        samples : getdist.MCSamples
        """
        if self.params(solved=True):
            self = self.sample_solved()
        from getdist import MCSamples
        toret = None
        if params is None: params = self.params(varied=True)
        else: params = [self[param].param for param in params]
        samples = self.to_array(params=params, struct=False, derivs=()).reshape(-1, self.size)
        labels = [param.latex() for param in params]
        names = [str(param) for param in params]
        ranges = {str(param): tuple('N' if limit is None or not np.isfinite(limit) else limit for limit in param.prior.limits) for param in params}
        toret = MCSamples(samples=samples.T, weights=np.asarray(self.weight.ravel()), loglikes=-np.asarray(self.logposterior.ravel()), names=names, labels=labels, label=label, ranges=ranges, **kwargs)
        return toret

    def to_anesthetic(self, params=None, label=None, **kwargs):
        """
        Return anesthetic hook to samples.

        Note
        ----
         anesthetic package *is* required.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to save samples of (weight and log-posterior are added anyway). Defaults to all parameters.

        label : str, default=None
            Name for  anesthetic to use for these samples.

        **kwargs : dict
            Optional arguments for :class:`anesthetic.MCMCSamples`.

        Returns
        -------
        samples : anesthetic.MCMCSamples
        """
        if self.params(solved=True):
            self = self.sample_solved()
        from anesthetic import MCMCSamples
        toret = None
        if params is None: params = self.params(varied=True)
        else: params = [self[param].param for param in params]
        labels = [param.latex() for param in params]
        samples = self.to_array(params=params, struct=False, derivs=()).reshape(-1, self.size)
        names = [str(param) for param in params]
        limits = {param.name: tuple('N' if limit is None or np.abs(limit) == np.inf else limit for limit in param.prior.limits) for param in params}
        toret = MCMCSamples(samples=samples.T, columns=names, weights=np.asarray(self.weight.ravel()), logL=-np.asarray(self.logposterior.ravel()), labels=labels, label=label, logzero=-np.inf, limits=limits, **kwargs)
        return toret

    def choice(self, index='mean', params=None, return_type='dict', **kwargs):
        """
        Return parameter mean(s) or best fit(s).

        Parameters
        ----------
        index : str, default='mean'
            'argmax' to return "best fit" (as defined by the point with maximum log-posterior in the chain).
            'mean' to return mean of parameters (weighted by :attr:`weight`).

        params : list, ParameterCollection, default=None
            Parameters to compute mean / best fit for. Defaults to all parameters.

        return_type : default='dict'
            'dict' to return a dictionary mapping parameter names to mean / best fit;
            'nparray' to return an array of parameter mean / best fit;
            ``None`` to return a :class:`Chain` instance with a single value.

        **kwargs : dict
            Optional arguments passed to :meth:`params` to select params to return, e.g. ``varied=True, derived=False``.

        Returns
        -------
        toret : dict, array, Chain
        """
        if params is None:
            params = self.params(**kwargs)
        if index == 'argmax':
            index = np.argmax(self.logposterior.ravel())
            di = {str(param): _reshape(self[param], self.size)[index] for param in params}
        elif index == 'mean':
            di = {str(param): self.mean(param) for param in params}
        else:
            raise ValueError('Unknown "index" argument {}'.format(index))
        if return_type == 'dict':
            return di
        if return_type == 'nparray':
            return np.array(list(di.values()))
        toret = self.copy()
        toret.data = [ParameterArray([value], param=value.param) for value in di.values()]
        return toret

    def covariance(self, params=None, return_type='nparray', ddof=1):
        """
        Return parameter covariance computed from (weighted) samples.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to compute covariance for. Defaults to all parameters.
            If a single parameter is provided, this parameter is a scalar, and ``return_type`` is 'nparray', return a scalar.

        return_type : str, default='nparray'
            'nparray' to return matrix array;
            ``None`` to return :class:`ParameterCovariance` instance.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        covariance : array, float, ParameterCovariance
        """
        if params is None: params = self.params()
        if not is_parameter_sequence(params): params = [params]
        params = [self[param].param for param in params]
        values = [self[param].reshape(self.size, -1) for param in params]
        values = np.concatenate(values, axis=-1)
        covariance = np.atleast_2d(np.cov(values, rowvar=False, fweights=self.fweight.ravel(), aweights=self.aweight.ravel(), ddof=ddof))
        solved_params = [param for param in params if param.solved]
        if solved_params:
            solved_indices = [params.index(param) for param in solved_params]
            covariance[np.ix_(solved_indices, solved_indices)] += np.average(_get_solved_covariance(self, params=solved_params).reshape(-1, len(solved_params), len(solved_params)), weights=self.weight.ravel(), axis=0)
        return ParameterCovariance(covariance, params=params).view(return_type=return_type)

    def precision(self, params=None, return_type='nparray', ddof=1):
        """
        Return inverse parameter covariance computed from (weighted) samples.

        Parameters
        ----------
        params :  list, ParameterCollection, default=None
            Parameters to compute covariance for. Defaults to all parameters.
            If a single parameter is provided, this parameter is a scalar, and ``return_type`` is 'nparray', return a scalar.

        return_type : str, default='nparray'
            'nparray' to return matrix array.
            ``None`` to return a :class:`ParameterPrecision` instance.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        precision : array, float, ParameterPrecision
        """
        return self.covariance(params=params, ddof=ddof, return_type=None).to_precision(return_type=return_type)

    def corrcoef(self, params=None):
        """Return correlation matrix array computed from (weighted) samples (optionally restricted to input parameters)."""
        return self.covariance(params=params, return_type=None).corrcoef()

    @vectorize
    def var(self, params=None, ddof=1):
        """
        Return variance computed from (weighted) samples (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        ``ddof`` is the number of degrees of freedom.
        """
        cov = self.covariance(params, ddof=ddof, return_type='nparray')
        if np.ndim(cov) == 0: return cov  # single param
        return np.diag(cov)

    def std(self, params=None, ddof=1):
        """
        Return standard deviation computed from (weighted) samples (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        ``ddof`` is the number of degrees of freedom.
        """
        return self.var(params=params, ddof=ddof)**0.5

    @vectorize
    def mean(self, params=None):
        """
        Return mean computed from (weighted) samples (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        """
        return np.average(_reshape(self[params], self.size), weights=self.weight.ravel(), axis=0)

    @vectorize
    def argmax(self, params=None):
        """
        Return parameter values for maximum of log-posterior (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        """
        return _reshape(self[params], self.size)[np.argmax(self.logposterior.ravel())]

    def median(self, params=None, method='linear'):
        """
        Return parameter median of weighted parameter samples (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        """
        return self.quantile(params, q=0.5, method=method)

    @vectorize
    def quantile(self, params=None, q=(0.1587, 0.8413), method='linear'):
        """
        Compute the q-th quantile of the weighted parameter samples.
        If a single parameter is given as input this parameter is a scalar, and a ``q`` is a scalar, return a scalar.

        Note
        ----
        Adapted from https://github.com/minaskar/cronus/blob/master/cronus/plot.py.

        Parameters
        ----------
        params :  list, ParameterCollection, default=None
            Parameters to compute quantiles for. Defaults to all parameters.

        q : tuple, list, array
            Quantile or sequence of quantiles to compute, which must be between
            0 and 1 inclusive.

        method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default='linear'
            This optional parameter specifies the method method to
            use when the desired quantile lies between two data points
            ``i < j``:

            - linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            - lower: ``i``.
            - higher: ``j``.
            - nearest: ``i`` or ``j``, whichever is nearest.
            - midpoint: ``(i + j) / 2``.

        Returns
        -------
        quantiles : list, scalar, array
        """
        value = _reshape(self[params], self.size)
        weight = self.weight.ravel()
        weight /= np.sum(weight)

        if value.param.solved:
            from scipy import stats

            locs = value
            scales = _get_solved_covariance(self, [params])[..., 0, 0].ravel()**0.5
            cdfs = [stats.norm(loc=loc, scale=scale).cdf for loc, scale in zip(locs, scales)]

            isscalar = np.ndim(q) == 0
            q = np.atleast_1d(q)
            quantiles = np.array(q)

            for iq, qq in enumerate(q.flat):

                def cdf(x):
                    return sum(w * cdf(x) for w, cdf in zip(weight, cdfs)) - qq

                nsigmas = 100
                limits = np.min(locs - nsigmas * scales), np.max(locs + nsigmas * scales)
                if qq <= limits[0]:
                    res = limits[0]
                elif qq >= limits[1]:
                    res = limits[1]
                else:
                    x = np.linspace(*limits, num=10000)
                    cdf = cdf(x)
                    idx = np.searchsorted(cdf, 0, side='right') - 1
                    res = (x[idx + 1] - x[idx]) / (cdf[idx + 1] - cdf[idx]) * cdf[idx + 1] + x[idx]
                    #print(cdf(x[idx]), cdf(x[idx + 1]))
                    #res = optimize.bisect(cdf, x[idx], x[idx + 1], xtol=1e-6 * np.mean(scales), disp=True)
                quantiles.flat[iq] = res

            if isscalar:
                return quantiles[0]
            return quantiles

        return utils.weighted_quantile(value, q=q, weights=weight, axis=0, method=method)

    @vectorize
    def interval(self, params=None, nsigmas=1.):
        """
        Return n-sigma confidence interval(s).

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to compute confidence interval for. Defaults to all parameters.

        nsigmas : int
            Return interval for this number of sigmas.

        Returns
        -------
        interval : tuple, list
        """
        value = self[params].ravel()
        weight = self.weight.ravel()
        weight /= np.sum(weight)

        if value.param.solved:

            from scipy import stats

            locs = value
            scales = _get_solved_covariance(self, [params])[..., 0, 0].ravel()**0.5
            cdfs = [stats.norm(loc=loc, scale=scale).cdf for loc, scale in zip(locs, scales)]

            def cdf(x):
                return sum(w * cdf(x) for w, cdf in zip(weight, cdfs))

            limits = np.min(locs - 2 * nsigmas * scales), np.max(locs + 2 * nsigmas * scales)
            value = np.linspace(*limits, num=10000)
            weight = cdf(value)
            weight = np.concatenate([[weight[0]], np.diff(weight)[:-1], [1. - weight[-2]]])

        return utils.interval(value, weights=weight, nsigmas=nsigmas)

    def to_fisher(self, params=None, ddof=1, **kwargs):
        """
        Return Fisher from (weighted) samples.

        Parameters
        ----------
        params :  list, ParameterCollection, default=None
            Parameters to return Fisher for. Defaults to all parameters.

        ddof : int, default=1
            Number of degrees of freedom.

        **kwargs : dict
            Arguments for :meth:`choice`, giving the mean of the output Fisher likelihood.

        Returns
        -------
        fisher : LikelihoodFisher
        """
        precision = self.precision(params=params, ddof=ddof, return_type=None)
        params = precision._params
        mean = self.choice(params=params, return_type='nparray', **kwargs)
        return LikelihoodFisher(center=mean, params=params, offset=self.logposterior.max(), hessian=-precision.view(return_type='nparray'), with_prior=True)

    def to_stats(self, params=None, quantities=None, sigfigs=2, tablefmt='latex_raw', fn=None):
        """
        Export summary sampling quantities.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to export quantities for. Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['argmax', 'mean', 'median', 'std', 'quantile:1sigma', 'interval:1sigma']``.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : string, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.

        fn : str, default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        tab : str
            Summary table.
        """
        import tabulate
        if params is None: params = self.params(varied=True)
        else: params = [self[param].param for param in params]
        if quantities is None: quantities = ['argmax', 'mean', 'median', 'std', 'quantile:1sigma', 'interval:1sigma']
        is_latex = 'latex_raw' in tablefmt

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs, positive_sign='u')[1:]
            if is_latex: return '${{}}_{{{}}}^{{{}}}$'.format(low, up)
            return '{}/{}'.format(low, up)

        data = []
        for iparam, param in enumerate(params):
            row = []
            row.append(param.latex(inline=True) if is_latex else str(param))
            ref_center = self.mean(param)
            ref_error = self.var(param)**0.5
            for quantity in quantities:
                if quantity in ['argmax', 'mean', 'median', 'std']:
                    value = getattr(self, quantity)(param)
                    value = utils.round_measurement(value, ref_error, sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                elif quantity.startswith('quantile'):
                    nsigmas = int(re.match('quantile:(.*)sigma', quantity).group(1))
                    low, up = self.quantile(param, q=utils.nsigmas_to_quantiles_1d_sym(nsigmas))
                    row.append(round_errors(low - ref_center, up - ref_center))
                elif quantity.startswith('interval'):
                    nsigmas = int(re.match('interval:(.*)sigma', quantity).group(1))
                    low, up = self.interval(param, nsigmas=nsigmas)
                    row.append(round_errors(low - ref_center, up - ref_center))
                else:
                    raise RuntimeError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        tab = tabulate.tabulate(data, headers=quantities, tablefmt=tablefmt)
        if fn is not None:
            utils.mkdir(os.path.dirname(fn))
            self.log_info('Saving to {}.'.format(fn))
            with open(fn, 'w') as file:
                file.write(tab)
        return tab
