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


def _get_solved_covariance(chain, params=None, return_hessian=False):
    logposterior = chain[chain._loglikelihood] + chain[chain._logprior]
    if params is None:
        params = chain.params(solved=True)
    params = [str(param) for param in params]
    solved_params = []
    for param in params:
        if logposterior.isin((param, param)):
            solved_params.append(param)
    if set(solved_params) != set(params):
        import warnings
        warnings.warn('You need the covariance of analytically marginalized ("solved") parameters, but it has not been computed / saved for {}. Assuming zero covariance.'.format([param for param in params if param not in solved_params]))
    all_solved_params = [str(param) for param in chain.params(solved=True) if logposterior.isin((param, param))]
    hessian = np.array([[logposterior[param1, param2].ravel() for param2 in all_solved_params] for param1 in all_solved_params])
    hessian = np.moveaxis(hessian, -1, 0).reshape(chain.shape + (len(all_solved_params),) * 2)
    covariance = np.linalg.inv(-hessian)
    # symmetrizing helps for numerical errors
    covariance = (np.moveaxis(covariance, chain.ndim + 1, chain.ndim) + covariance) / 2.
    toret_covariance = np.zeros(chain.shape + (len(params),) * 2, dtype='f8')
    toret_hessian = toret_covariance.copy()
    for iparam1, param1 in enumerate(all_solved_params):
        if param1 not in params: continue
        index1 = params.index(param1)
        for iparam2, param2 in enumerate(all_solved_params):
            if param2 not in params: continue
            index2 = params.index(param2)
            toret_covariance[..., index1, index2] = covariance[..., iparam1, iparam2]
            toret_hessian[..., index1, index2] = hessian[..., iparam1, iparam2]
    if return_hessian:
        return toret_covariance, toret_hessian
    return toret_covariance


class Chain(Samples):
    """
    Class that holds samples drawn from posterior (in practice, :class:`Samples` with a log-posterior and optional weights).

    Parameter arrays can be accessed (and updated) as for a dictionary:

    .. code-block:: python

        chain = Chain([np.ones(100), np.zeros(100)], params=['a', 'b'])
        chain['a'] += 1.
        print(chain['a'].mean())

        chain['c'] = chain['b'] + 1
        chain['c'].param.update(latex='c')

    """

    _type = ParameterArray
    _attrs = Samples._attrs + ['_logposterior', '_loglikelihood', '_logprior', '_aweight', '_fweight', '_weight']

    def __init__(self, data=None, params=None, logposterior=None, loglikelihood=None, logprior=None, aweight=None, fweight=None, weight=None, attrs=None):
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

        loglikelihood : str, default='loglikelihood'
            Name of log-likelihood in ``data``.

        logprior : str, default='logprior'
            Name of log-prior in ``data``.

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
        super(Chain, self).__init__(data=data, params=params, attrs=attrs)
        for _name in self._attrs[-6:]:
            name = _name[1:]
            value = locals()[name]
            if getattr(self, _name, None) is None or value is not None:  # set only if not previously set, or new value are provided
                setattr(self, _name, name if value is None else str(value))
            value = getattr(self, _name)
            if value in self:
                self[value].param.update(derived=True)

    def __setstate__(self, state):
        # Backward-compatibility
        for name in ['_logposterior', '_loglikelihood', '_logprior', '_aweight', '_fweight', '_weight']:
            state.setdefault(name, name[1:])
        super(Chain, self).__setstate__(state)

    @property
    def aweight(self):
        """Sample weights (floats)."""
        if self._aweight not in self:
            self[Parameter(self._aweight, derived=True, latex=utils.outputs_to_latex(self._aweight))] = np.ones(self.shape, dtype='f8')
        return self[self._aweight]

    @property
    def fweight(self):
        """Sample frequency weights (integers)."""
        if self._fweight not in self:
            self[Parameter(self._fweight, derived=True, latex=utils.outputs_to_latex(self._fweight))] = np.ones(self.shape, dtype='i8')
        return self[self._fweight]

    @property
    def logposterior(self):
        """Log-posterior."""
        if self._logposterior not in self:
            self[Parameter(self._logposterior, derived=True, latex=utils.outputs_to_latex(self._logposterior))] = np.zeros(self.shape, dtype='f8')
        return self[self._logposterior]

    @aweight.setter
    def aweight(self, item):
        """Set weights (floats)."""
        self[Parameter(self._aweight, derived=True, latex=utils.outputs_to_latex(self._aweight))] = item

    @fweight.setter
    def fweight(self, item):
        """Set frequency weights (integers)."""
        self[Parameter(self._fweight, derived=True, latex=utils.outputs_to_latex(self._fweight))] = item

    @logposterior.setter
    def logposterior(self, item):
        """Set log-posterior."""
        self[Parameter(self._logposterior, derived=True, latex=utils.outputs_to_latex(self._logposterior))] = item

    @property
    def weight(self):
        """Return total weight, as the product of :attr:`aweight` and :attr:`fweight`."""
        return ParameterArray(self.aweight * self.fweight, Parameter(self._weight, derived=True, latex=utils.outputs_to_latex(self._weight)))

    def set_derived(self, basename, array, **kwargs):
        """
        Set derived parameter.

        Parameters
        ----------
        array : np.array
            Numpy array.

        kwargs : dict
            Arguments for :class:`Parameter`.
        """
        kwargs['basename'] = basename
        kwargs.setdefault('derived', True)
        self.set(ParameterArray(array, Parameter(kwargs)))

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
        all_solved_params = self.params(solved=True)
        solved_params = []
        for param in all_solved_params:
            if self[self._loglikelihood].isin((param, param)) and self[self._logprior].isin((param, param)):
                solved_params.append(param)
        if set(solved_params) != set(all_solved_params):
            import warnings
            warnings.warn('sample over parameters {}, derivatives for {} are not saved'.format(solved_params, [param for param in all_solved_params if param not in solved_params]))
        if not solved_params: return new
        covariance, hessian = _get_solved_covariance(self, params=solved_params, return_hessian=True)
        L = np.moveaxis(np.linalg.cholesky(covariance), (-2, -1), (0, 1))
        new.data = []
        for array in self:
            new.set(array.clone(value=np.repeat(array, size, axis=self.ndim - 1)))
        rng = np.random.RandomState(seed=seed)
        noise = rng.standard_normal((len(solved_params),) + self.shape + (size,))
        values = np.sum(noise[None, ...] * L[..., None], axis=1)
        for param, value in zip(solved_params, values):
            new[param] = new[param].clone(value=new[param] + value.reshape(new.shape), param=param.clone(derived=False))
        dlogposterior = 0.
        for param in [self._loglikelihood, self._logprior]:
            hess = np.array([[self[param][param1, param2] for param2 in solved_params] for param1 in solved_params])
            log = 1. / 2. * np.sum(values[None, ...] * hess[..., None] * values[:, None, ...], axis=(0, 1)).reshape(new.shape)
            new[param] = self[param].clone(value=new[param][()] + log, derivs=None)
            dlogposterior += log
        marg_indices = np.array([iparam for iparam, param in enumerate(solved_params) if 'auto' in param.derived or 'marg' in param.derived])
        if marg_indices.size:
            log = 1. / 2. * np.linalg.slogdet(- hessian[(Ellipsis,) + np.ix_(marg_indices, marg_indices)])[1]
            new[self._loglikelihood] += log
            dlogposterior += log
        new.logposterior[...] += dlogposterior
        return new

    def select(self, **kwargs):
        # Keep weight columns
        toret = self._select(name=[self._aweight, self._fweight])
        toret.update(self._select(**kwargs))
        return toret

    def __getitem__(self, name):
        """
        Return item corresponding to parameter ``name``.

        Parameters
        ----------
        name : Parameter, str, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
        """
        try:
            return super().__getitem__(name)
        except KeyError:
            if name == self._weight:
                return self.weight
            else:
                raise

    @classmethod
    def from_getdist(cls, samples, concatenate=None):
        """
        Turn getdist.MCSamples into a :class:`Chain` instance.

        Note
        ----
        GetDist package is required.
        """
        params = ParameterCollection()
        for param in samples.paramNames.names:
            params.set(Parameter(param.name, latex=param.label, derived=param.isDerived, fixed=False))
        param_indices = samples._getParamIndices()
        for param in params:
            limits = [samples.ranges.lower.get(param.name, -np.inf), samples.ranges.upper.get(param.name, np.inf)]
            param.update(prior=ParameterPrior(limits=limits))
        isscalar = True
        try:
            chains = samples.getSeparateChains()
            isscalar = False
        except:
            chains = [samples]
        toret = []
        for chain in chains:
            new = cls()
            fweight, new.logposterior = chain.weights, -chain.loglikes
            iweight = np.rint(fweight)
            if np.allclose(fweight, iweight, atol=0., rtol=1e-9):
                new.fweight = iweight.astype('i4')
            else:
                new.aweight = fweight
            for param in params:
                new.set(ParameterArray(chain[param_indices[param.name]], param=param))
            for param in new.params(basename='chi2_*'):
                namespace = re.match('chi2_[_]*(.*)$', param.name).groups()[0]
                if namespace == 'prior':
                    new_param = param.clone(basename=new._logprior, derived=True)
                else:
                    new_param = param.clone(basename=new._loglikelihood, namespace=namespace, derived=True)
                new[new_param] = -0.5 * new[param]
            toret.append(new)
        if isscalar:
            if concatenate or concatenate is None:
                toret = toret[0]
        elif concatenate:
            toret = cls.concatenate(toret)
        return toret

    @utils.hybridmethod
    def to_getdist(cls, chain, params=None, label=None, **kwargs):
        """
        Return GetDist hook to samples.

        Note
        ----
        GetDist package is required.

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
        from getdist import MCSamples
        isscalar = not utils.is_sequence(chain)
        if isscalar: chain = [chain]
        chains = list(chain)
        toret = None
        if params is None: params = chains[0].params(varied=True)
        else: params = [chains[0][param].param for param in params]
        if any(param.solved for param in params):
            for ichain, chain in enumerate(chains):
                chains[ichain] = chain.sample_solved()
        chain = chains[0]
        params = [param for param in params if param.name not in [chain._weight, chain._logposterior]]
        labels = [param.latex() for param in params]
        names = [str(param) for param in params]
        ranges = {str(param): tuple('N' if limit is None or not np.isfinite(limit) else limit for limit in param.prior.limits) for param in params}
        samples, weights, loglikes = [], [], []
        for chain in chains:
            samples.append(chain.to_array(params=params, struct=False, derivs=()).reshape(-1, chain.size).T)
            weights.append(chain.weight.ravel())
            loglikes.append(-np.asarray(chain.logposterior.ravel()))
        if isscalar:
            samples, weights, loglikes = samples[0], weights[0], loglikes[0]
        toret = MCSamples(samples=samples, weights=weights, loglikes=loglikes, names=names, labels=labels, label=label, ranges=ranges, **kwargs)
        return toret

    @to_getdist.instancemethod
    def to_getdist(self, *args, **kwargs):
        return self.__class__.to_getdist(self, *args, **kwargs)

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
            fweight, new.logposterior = array[0], -array[1]
            iweight = np.rint(fweight)
            if np.allclose(fweight, iweight, atol=0., rtol=1e-9):
                new.fweight = iweight.astype('i4')
            else:
                new.aweight = fweight
            for param, values in zip(params, array[2:]):
                new.set(ParameterArray(values, param))
            toret.append(new)
        for new in toret:
            for param in new.params(basename='chi2_*'):
                namespace = re.match('chi2_[_]*(.*)$', param.name).groups()[0]
                if namespace == 'prior':
                    new_param = param.clone(basename=new._logprior, derived=True)
                else:
                    new_param = param.clone(basename=new._loglikelihood, namespace=namespace, derived=True)
                new[new_param] = -0.5 * new[param]
        if isscalar:
            return toret[0]
        if concatenate:
            return cls.concatenate(toret)
        return toret

    @utils.hybridmethod
    def write_getdist(cls, chain, base_fn, params=None, ichain=None, fmt='%.18e', delimiter=' ', **kwargs):
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
            Chain number to append to file name, i.e. sample values will be saved as '{base_fn}_{ichain}.txt'.
            If ``None``, does not append any number, sample values will be saved as '{base_fn}.txt'.

        fmt : str, default='%.18e'
            How to format floats.

        delimiter : str, default=' '
            String or character separating columns.

        kwargs : dict
            Optional arguments for :func:`numpy.savetxt`.
        """
        isscalar = not utils.is_sequence(chain)
        if isscalar: chain = [chain]
        chains = list(chain)
        if params is None: params = chains[0].params()
        else: params = [chains[0][param].param for param in params]
        if any(param.solved for param in params):
            for ichain, chain in enumerate(chains):
                chains[ichain] = chain.sample_solved()

        chain = chains[0]
        columns = [str(param) for param in params]
        outputs_columns = [chain._weight, chain._logposterior]
        shape = chain.shape
        outputs = [array.param.name for array in chain if array.shape != shape]
        for column in outputs:
            if column in columns: del columns[columns.index(column)]

        if ichain is None:
            if isscalar:
                ichain = [None] * len(chains)
            else:
                ichain = list(range(len(chains)))
        if not utils.is_sequence(ichain):
            ichain = [ichain]
        assert len(ichain) == len(chains)

        utils.mkdir(os.path.dirname(base_fn))

        output = ''
        params = chain.params(name=columns)
        for param in params:
            tmp = '{}* {}\n' if getattr(param, 'derived', getattr(param, 'fixed')) else '{} {}\n'
            output += tmp.format(param.name, param.latex())
        params_fn = '{}.paramnames'.format(base_fn)
        cls.log_info('Saving parameter names to {}.'.format(params_fn))
        with open(params_fn, 'w') as file:
            file.write(output)

        output = ''
        for param in params:
            limits = param.prior.limits
            limits = tuple('N' if limit is None or np.abs(limit) == np.inf else limit for limit in limits)
            output += '{} {} {}\n'.format(param.name, limits[0], limits[1])
        ranges_fn = '{}.ranges'.format(base_fn)
        cls.log_info('Saving parameter ranges to {}.'.format(ranges_fn))
        with open(ranges_fn, 'w') as file:
            file.write(output)

        for chain, ichain in zip(chains, ichain):
            data = chain.to_array(params=outputs_columns + columns, struct=False, derivs=()).reshape(-1, chain.size)
            data[1] *= -1
            data = data.T
            chain_fn = '{}.txt'.format(base_fn) if ichain is None else '{}_{:d}.txt'.format(base_fn, ichain)
            cls.log_info('Saving chain to {}.'.format(chain_fn))
            np.savetxt(chain_fn, data, header='', fmt=fmt, delimiter=delimiter, **kwargs)

    @write_getdist.instancemethod
    def write_getdist(self, *args, **kwargs):
        return self.__class__.write_getdist(self, *args, **kwargs)

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
        from anesthetic import MCMCSamples
        toret = None
        if params is None: params = self.params(varied=True)
        else: params = [self[param].param for param in params]
        if any(param.solved for param in params):
            self = self.sample_solved()
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
        if isinstance(index, str) and index == 'mean':
            di = {str(param): self.mean(param) for param in params}
            index = (0,) # just for test below
        else:
            if isinstance(index, str) and index == 'argmax':
                index = np.unravel_index(self.logposterior.argmax(), self.shape)
            if not isinstance(index, tuple):
                index = (index,)
            di = {str(param): self[param][index] for param in params}
        if return_type == 'dict':
            return di
        if return_type == 'nparray':
            return np.array(list(di.values()))
        toret = self.copy()
        isscalar = all(np.ndim(ii) == 0 for ii in index)
        toret.data = []
        for param, value in di.items():
            value = np.asarray(value)
            toret.data.append(self[param].clone(value=value[None, ...] if isscalar else value))
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
        params = ParameterCollection([self[param].param for param in params])  # eliminates duplicates
        values = [self[param][()].reshape(self.size, -1) for param in params]  # [()] to take order 0 derivatives
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

    def var(self, params=None, ddof=1):
        """
        Return variance computed from (weighted) samples (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        ``ddof`` is the number of degrees of freedom.
        """
        isscalar = not is_parameter_sequence(params)
        cov = self.covariance(params, ddof=ddof, return_type='nparray')
        if isscalar: return cov.flat[0]  # single param
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
            if np.all(scales == 0.):
                return utils.weighted_quantile(value, q=q, weights=weight, axis=0, method=method)

            isscalar = np.ndim(q) == 0
            q = np.atleast_1d(q)
            quantiles = np.array(q)

            for iq, qq in enumerate(q.flat):

                from scipy import special

                def cdf(x):
                    toret = np.empty_like(x)
                    nx = len(x)
                    nslabs = max(nx * len(locs) // int(1e8), 1)
                    for islab in range(nslabs):
                        start, stop = islab * nx // nslabs, (islab + 1) * nx // nslabs
                        toret[start:stop] = np.sum(weight / 2. * (1. + special.erf((x[start:stop, None] - locs) / (2**0.5 * scales))), axis=-1)
                    return toret

                nsigmas = 100
                limits = np.min(locs - nsigmas * scales), np.max(locs + nsigmas * scales)
                if qq <= limits[0]:
                    res = limits[0]
                elif qq >= limits[1]:
                    res = limits[1]
                else:
                    x = np.linspace(*limits, num=10000)
                    cdf = cdf(x) - qq
                    idx = np.searchsorted(cdf, 0., side='right') - 1
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

            if not np.all(scales == 0.):

                from scipy import special

                def cdf(x):
                    toret = np.empty_like(x)
                    nx = len(x)
                    nslabs = max(nx * len(locs) // int(1e8), 1)
                    for islab in range(nslabs):
                        start, stop = islab * nx // nslabs, (islab + 1) * nx // nslabs
                        toret[start:stop] = np.sum(weight / 2. * (1. + special.erf((x[start:stop, None] - locs) / (2**0.5 * scales))), axis=-1)
                    return toret

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

        tablefmt : str, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.
            If 'list', return table as list of list of strings, and headers.
            If 'list_latex', return table as list of list of latex strings, and headers.

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
        is_latex = 'latex' in tablefmt

        def round_errors(low, up):
            low, up = utils.round_measurement(0.0, low, up, sigfigs=sigfigs, positive_sign='u')[1:]
            if is_latex: return '${{}}_{{{}}}^{{{}}}$'.format(low, up)
            return '{}/{}'.format(low, up)

        data = []
        for param in params:
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
        if 'list' in tablefmt:
            return data, quantities
        tab = tabulate.tabulate(data, headers=quantities, tablefmt=tablefmt)
        if fn is not None:
            utils.mkdir(os.path.dirname(fn))
            self.log_info('Saving to {}.'.format(fn))
            with open(fn, 'w') as file:
                file.write(tab)
        return tab