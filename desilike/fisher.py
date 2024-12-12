import os
import copy

import numpy as np

from .differentiation import Differentiation
from .parameter import ParameterPrecision, ParameterCovariance, ParameterArray, ParameterCollection, Parameter, is_parameter_sequence
from .base import BaseCalculator, _params_args_or_kwargs
from .utils import BaseClass, deep_eq
from .jax import numpy as jnp
from . import utils


class PriorCalculator(BaseCalculator):

    _initialize_with_namespace = True
    _calculate_with_namespace = True

    """Calculator that computes the logprior."""

    def calculate(self, **params):
        self.logprior = self.runtime_info.params.prior(**params)

    def get(self):
        return self.logprior


from desilike.likelihoods import BaseGaussianLikelihood


class FisherGaussianLikelihood(BaseGaussianLikelihood):

    _initialize_with_namespace = True
    _calculate_with_namespace = True

    def initialize(self, fisher):
        data = fisher.mean()
        precision = fisher.precision(return_type='nparray')
        self.offset = float(fisher._offset)
        self.params = fisher.params().deepcopy()
        self.fisher = fisher

        if fisher.with_prior:
            for param in self.params:
                param.update(prior={'limits': param.prior.limits})  # avoid prior redundancy

        self.quantities = fisher.names()
        super(FisherGaussianLikelihood, self).initialize(data=data, precision=precision)

    def calculate(self, **params):
        self.flattheory = jnp.array([params[name] for name in self.quantities])
        super(FisherGaussianLikelihood, self).calculate()
        self.loglikelihood += self.offset

    @classmethod
    def load(cls, filename):
        return LikelihoodFisher.load(filename).to_likelihood()

    def save(self, fn):
        return self.fisher.save(fn)


class LikelihoodFisher(BaseClass):

    """Class representing a Fisher representation of a likelihood."""

    def __init__(self, center, params=None, gradient=None, offset=None, hessian=None, with_prior=False, attrs=None):
        """
        Initialize :class:`LikelihoodFisher`.

        Parameters
        ----------
        center : array, list, default=None
            Point where Fisher has been computed.

        params : list, ParameterCollection
            Parameters corresponding to input ``hessian``.

        gradient : array, list, ParameterArray, default=None
            Likelihood gradient. Defaults to 0.
            If :class:`ParameterArray`, can be used to specify ``hessian``.

        offset : float, default=None
            Zero-lag log-likelihood. Defaults to 0.

        hessian : array, default=None
            Hessian, i.e. second-order derivatives of the log-likelihood.
            If ``None``, taken from :class:`ParameterArray` ``gradient``.

        with_prior : bool, default=False
            Whether input ``gradient``, ``hessian`` and ``offset`` include parameter priors.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        hessian = ParameterPrecision(hessian if hessian is not None else gradient, params=params)
        self._params = hessian.params()
        self._center = np.concatenate([np.ravel(c) for c in center])
        if self._center.size != hessian.shape[0]:
            raise ValueError('Input center and hessian matrix have different sizes: {:d} vs {:d}'.format(self._center.size, hessian.shape[0]))
        gradient_has_derivs = getattr(gradient, 'derivs', None) is not None
        if gradient is None:
            self._gradient = np.zeros_like(center)
        elif gradient_has_derivs:
            self._gradient = np.array([gradient[param] for param in self._params])
        else:
            self._gradient = np.ravel(gradient)
            if self._gradient.size != len(self._params):
                raise ValueError('Number of parameters and gradient size are different: {:d} vs {:d}'.format(len(self._params), self._gradient.size))
        if offset is None:
            if gradient_has_derivs:
                offset = gradient[()]
            else:
                offset = 0.
        self._offset = float(offset)
        self._hessian = hessian._value
        self.with_prior = bool(with_prior)
        self.attrs = dict(attrs or {})
        self._sizes

    @property
    def _sizes(self):
        # Return parameter sizes
        toret = [max(param.size, 1) for param in self._params]
        if sum(toret) != self._hessian.shape[0]:
            raise ValueError('number * size of input params must match input Hessian shape')
        return toret

    def _index(self, params):
        # Internal method to return indices corresponding to input params.""""
        idx = [self._params.index(param) for param in params]
        if idx:
            cumsizes = np.cumsum([0] + self._sizes)
            return np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx], dtype='i4')
        return np.array(idx, dtype='i4')

    def __contains__(self, name):
        """Has this parameter?"""
        return name in self._params

    def __getstate__(self):
        """Return this class' state dictionary."""
        state = {}
        for name in ['center', 'gradient', 'hessian', 'offset']: state[name] = getattr(self, '_' + name)
        for name in ['with_prior', 'attrs']: state[name] = getattr(self, name)
        state['params'] = self._params.__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class' state dictionary."""
        for name in ['center', 'gradient', 'hessian', 'offset']: setattr(self, '_' + name, state[name])
        self.with_prior = state['with_prior']
        self.attrs = state.get('attrs', {})
        self._params = ParameterCollection.from_state(state['params'])

    def deepcopy(self):
        """Deep copy."""
        return copy.deepcopy(self)

    def __repr__(self):
        """Return string representation of parameter matrix, including parameters."""
        return '{}({})'.format(self.__class__.__name__, self._params)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(deep_eq(getattr(other, name), getattr(self, name)) for name in ['_params', '_center', '_gradient', '_hessian', '_offset', 'with_prior'])

    def clone(self, center=None, params=None, gradient=None, offset=None, hessian=None, with_prior=None, attrs=None):
        """
        Clone this Fisher i.e. copy and optionally update ``center``,
        ``params``, ``gradient``, ``offset``, ``hessian``, and ``attrs``.

        Parameters
        ----------
        center : array, list, default=None
            Point where Fisher has been computed.

        params : list, ParameterCollection
            Parameters corresponding to input ``hessian``.

        gradient : array, list, default=None
            Likelihood gradient. Defaults to 0.

        offset : float, default=None
            Zero-lag log-likelihood.

        hessian : array, default=None
            Hessian, i.e. second-order derivatives of the log-likelihood.

        with_prior : bool, default=None
            Whether input ``gradient``, ``hessian`` and ``offset`` include parameter priors.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.

        Returns
        -------
        new : BaseParameterMatrix
            A new Fisher, optionally with ``center``, ``params``, ``gradient``, ``offset``, ``hessian``, and ``attrs``.
        """
        new = self.view(params=params)
        if center is not None:
            new._center[...] = center
        if gradient is not None:
            new._gradient[...] = gradient
        if offset is not None:
            new._offset = float(offset)
        if hessian is not None:
            new._hessian[...] = hessian
        if with_prior is not None:
            new.with_prior = bool(with_prior)
        if attrs is not None:
            new.attrs = dict(attrs)
        return new

    def _solve(self):
        try:
            return np.linalg.solve(self._hessian, self._gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Singular matrix: {}, for parameters {}'.format(self._hessian, self._params)) from exc
        return None

    @property
    def chi2min(self):
        r"""Minimum :math:`\chi^{2}`."""
        flatdiff = - self._solve()
        return -2. * (self._offset + self._gradient.dot(flatdiff) + 1. / 2. * flatdiff.dot(self._hessian).dot(flatdiff))

    def mean(self, params=None, return_type='nparray'):
        """
        Return likelihood mean, restricting to input ``params`` if provided.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.

        return_type : str, default='nparray'
            If 'nparray', return a numpy array.
            Else, return a dictionary mapping parameter names to mean values.

        Returns
        -------
        mean : array, dict
        """
        mean = self._center - self._solve()
        if params is None:
            params = self._params
        isscalar = not is_parameter_sequence(params)
        if isscalar:
            params = [params]
        mean = mean[self._index(params)]
        if return_type == 'nparray':
            if isscalar:
                return mean.item()
            return mean
        return {str(param): value for param, value in zip(params, mean)}

    def center(self, params=None, return_type='nparray'):
        """
        Return center, restricting to input ``params`` if provided.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.

        return_type : str, default='nparray'
            If 'nparray', return a numpy array.
            Else, return a dictionary mapping parameter names to center values.

        Returns
        -------
        center : array, dict
        """
        if params is None:
            params = self._params
        isscalar = not is_parameter_sequence(params)
        if isscalar:
            params = [params]
        center = self._center[self._index(params)]
        if return_type == 'nparray':
            if isscalar:
                return center.item()
            return center
        return {str(param): value for param, value in zip(params, center)}

    def choice(self, index='mean', params=None, return_type='dict', **kwargs):
        """
        Return mean or center.

        Parameters
        ----------
        index : str, default='mean'
            'argmax' or 'mean' to return :meth:`mean`.
            'center' to return :meth:`center`.

        params : list, ParameterCollection, default=None
            Parameters to compute mean / center for. Defaults to all parameters.

        return_type : default='dict'
            'dict' to return a dictionary mapping parameter names to mean / center;
            'nparray' to return an array of parametermean / center.

        **kwargs : dict
            Optional arguments passed to :meth:`params` to select params to return, e.g. ``varied=True, derived=False``.

        Returns
        -------
        toret : dict, array
        """
        if params is None:
            params = self.params(**kwargs)
        if index in ['argmax', 'mean']:
            toret = self.mean(params=params, return_type=return_type)
        elif index == 'center':
            toret = self.center(params=params, return_type=return_type)
        else:
            raise ValueError('Unknown "index" argument {}'.format(index))
        return toret

    def params(self, *args, **kwargs):
        """Return parameters."""
        return self._params.params(*args, **kwargs)

    def names(self, *args, **kwargs):
        """Return names of parameters."""
        return self._params.names(*args, **kwargs)

    def select(self, params=None, **kwargs):
        """
        Return a :class:`LikelihoodFisher` restricting to input ``params``.,

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Optionally, parameters to limit to.

        **kwargs : dict
            If ``params`` is ``None``, optional arguments passed to :meth:`ParameterCollection.select`
            to select parameters (e.g. ``varied=True``).

        Returns
        -------
        new : LikelihoodFisher
        """
        if params is None: params = self._params.select(**kwargs)
        return self.view(params=params)

    def precision(self, params=None, return_type='nparray'):
        """
        Return inverse covariance matrix (precision matrix) for input parameters ``params``.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.
            If a single parameter is provided, this parameter is a scalar, and ``return_type`` is 'nparray', return a scalar.

        return_type : str, default=None
            If 'nparray', return a numpy array.
            Else, return a new :class:`ParameterPrecision`.

        Returns
        -------
        new : array, float, ParameterPrecision
        """
        return ParameterPrecision(-self._hessian, params=self._params, attrs=self.attrs).view(params=params, return_type=return_type)

    def covariance(self, params=None, return_type='nparray'):
        """
        Return inverse precision matrix (covariance matrix) for input parameters ``params``.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.
            If a single parameter is provided, this parameter is a scalar, and ``return_type`` is 'nparray', return a scalar.

        return_type : str, default=None
            If 'nparray', return a numpy array.
            Else, return a new :class:`ParameterCovariance`.

        Returns
        -------
        new : array, float, ParameterCovariance
        """
        return self.precision(return_type=None).to_covariance(params=params, return_type=return_type)

    def corrcoef(self, params=None):
        """Return correlation matrix array (optionally restricted to input parameters)."""
        return self.covariance(params=params, return_type=None).corrcoef()

    def var(self, params=None):
        """
        Return variance (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        ``ddof`` is the number of degrees of freedom.
        """
        cov = self.covariance(params=params, return_type='nparray')
        if np.ndim(cov) == 0: return cov  # single param
        return np.diag(cov)

    def std(self, params=None):
        """
        Return standard deviation (optionally restricted to input parameters).
        If a single parameter is given as input and this parameter is a scalar, return a scalar.
        ``ddof`` is the number of degrees of freedom.
        """
        return self.var(params=params)**0.5

    def view(self, params=None, center=None):
        """
        Return Fisher for input parameters ``params``.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.
            If a parameter in ``params`` is not in Fisher, add it, with zero precision and gradient.

        center : str, default=None
            Center. Defaults to :meth:`center`.

        Returns
        -------
        new : LikelihoodFisher
        """
        precision = self.precision(params=params, return_type=None)
        params = precision.params()
        params_in_self = [param for param in params if param in self._params]
        ccenter, gradient = np.full(precision.shape[0], np.nan, dtype='f8'), np.zeros(precision.shape[0], dtype='f8')
        index_new, index_self = precision._index(params_in_self), self._index(params_in_self)
        gradient[index_new] = self._gradient[index_self]
        ccenter[index_new] = self._center[index_self]
        hessian = -precision._value
        offset = self._offset
        if center is None:
            center = ccenter
        center = np.asarray(center)
        flatdiff = np.zeros_like(center)
        flatdiff[index_new] = center[index_new] - ccenter[index_new]
        offset = offset + gradient.dot(flatdiff) + 1. / 2. * flatdiff.dot(hessian).dot(flatdiff)
        gradient = gradient + hessian.dot(flatdiff)
        return self.__class__(center, params=params, gradient=gradient, offset=offset, hessian=hessian, attrs=self.attrs)

    def shift(self, mean):
        """
        Shift such that new mean correspond to input ``mean``.

        Parameters
        ----------
        mean : array
            New mean.

        Returns
        -------
        fisher : LikelihoodFisher
        """
        gradient = self._hessian.dot(self._center - mean)
        return self.clone(gradient=gradient)

    @classmethod
    def sum(cls, *others):
        """Add Fisher."""
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        params = ParameterCollection.concatenate([other._params for other in others])
        names = params.names()
        centers, need_view = [], []
        for iother, other in enumerate(others):
            c = np.full(len(params), np.nan, dtype='f8')
            iparams_in_self, params_in_self = zip(*[(iparam, param) for iparam, param in enumerate(params) if param in other._params])
            c[list(iparams_in_self)] = other.center(params=params_in_self)
            centers.append(c)
            if iother == 0: center = c
            need_view.append((other._params.names() != names) or not np.all(center == c))
        if any(need_view): center = np.nanmean(centers, axis=0)
        others = [other.view(params, center=center) if need_view[iother] else other for iother, other in enumerate(others)]
        offset = gradient = hessian = 0.
        with_prior, attrs = {}, {}
        for other in others:
            offset += other._offset
            gradient += other._gradient
            hessian += other._hessian
            if other.with_prior:
                for param in other.params():
                    if param.prior.dist == 'uniform':  # if only uniform distributions, fine!
                        with_prior.setdefault(param.name, False)
                    else:
                        if with_prior.get(param.name, False):
                            import warnings
                            warnings.warn('Several input Fisher include prior information, yielding double-counting of prior information')
                        with_prior[param.name] = True
            attrs.update(other.attrs)
        return cls(center, params=params, gradient=gradient, offset=offset, hessian=hessian, with_prior=bool(with_prior), attrs=attrs)

    def __add__(self, other):
        """Sum of `self`` + ``other`` Fisher."""
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def to_likelihood(self):
        """
        Export Fisher to Gaussian likelihood.

        Note
        ----
        If :attr:`with_prior`, no prior (uniform, infinite distribution) are given to the parameters of output likelihood,
        to avoid double-counting prior information.

        Returns
        -------
        likelihood : FisherGaussianLikelihood
        """
        return FisherGaussianLikelihood(self)

    def to_stats(self, params=None, sigfigs=2, tablefmt='latex_raw', fn=None):
        """
        Export Fisher to string.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            If provided, restrict to these parameters.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : str, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.

        fn : str, default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        txt : str
            Summary table.
        """
        import tabulate
        is_latex = 'latex_raw' in tablefmt

        cov = self.covariance(params=params, return_type=None)
        headers = [param.latex(inline=True) if is_latex else str(param) for param in cov._params]

        txt = tabulate.tabulate([['FoM', '{:.2f}'.format(cov.fom())]], tablefmt=tablefmt) + '\n'
        errors = cov.std()
        data = [('mean', 'std')] + [utils.round_measurement(value, error, sigfigs=sigfigs)[:2] for value, error in zip(self.mean(params=cov._params), errors)]
        data = list(zip(*data))
        txt += tabulate.tabulate(data, headers=headers, tablefmt=tablefmt) + '\n'

        data = [[str(param)] + [utils.round_measurement(value, value, sigfigs=sigfigs)[0] for value in row] for param, row in zip(cov._params, cov._value)]
        txt += tabulate.tabulate(data, headers=headers, tablefmt=tablefmt)
        if fn is not None:
            utils.mkdir(os.path.dirname(fn))
            self.log_info('Saving to {}.'.format(fn))
            with open(fn, 'w') as file:
                file.write(txt)
        return txt

    def to_getdist(self, params=None, label=None, ignore_limits=True):
        """
        Return a GetDist Gaussian distribution, centered on :meth:`mean`, with covariance matrix :meth:`covariance`.

        Parameters
        ----------
        params : list, ParameterCollection, default=None
            Parameters to share to GetDist. Defaults to all parameters.

        label : str, default=None
            Name for GetDist to use for this distribution.

        ignore_limits : bool, default=True
            GetDist does not seem to be able to integrate over distribution if bounded;
            so drop parameter limits.

        Returns
        -------
        samples : getdist.gaussian_mixtures.MixtureND
        """
        return self.covariance(params=params, return_type=None).to_getdist(label=label, center=self.mean(params=params), ignore_limits=ignore_limits)

    @classmethod
    def read_getdist(cls, base_fn, with_prior=True, **kwargs):
        """
        Read Fisher from GetDist format.

        Parameters
        ----------
        base_fn : str
            Base *CosmoMC* file name. Will be appended by '.margestats' for marginalized parameter mean,
            '.likestats' for likelihood maximum and '.covmat' for parameter covariance matrix.

        with_prior : bool, default=True
            Whether input chains include parameter priors.

        **kwargs : dict
            If ``params`` is ``None``, optional arguments passed to :meth:`ParameterCollection.select`
            to select parameters (e.g. ``varied=True``).
            Restricting to useful parameters is relevant for the numerical accuracy of covariance inversion.

        Returns
        -------
        fisher : FisherLikelihood
        """
        mean = {}
        col = None
        stats_fn = '{}.margestats'.format(base_fn)
        cls.log_info('Loading stats file: {}.'.format(stats_fn))
        with open(stats_fn, 'r') as file:
            for line in file:
                line = [item.strip() for item in line.split()]
                if line:
                    if col is not None:
                        name, value = line[0], float(line[col])
                        mean[name] = value
                    if line[0] == 'parameter':
                        # Let's get the column col where to find the mean
                        for col, item in enumerate(line):
                            if item.strip() == 'mean': break
        stats_fn = '{}.likestats'.format(base_fn)
        cls.log_info('Loading stats file: {}.'.format(stats_fn))
        offset = 0.
        txt = 'Best fit sample -log(Like) ='
        with open(stats_fn, 'r') as file:
            for line in file:
                if line.startswith(txt):
                    offset = -float(line[len(txt):])
                    break
        covariance = ParameterCovariance.read_getdist(base_fn).select(**kwargs)
        params = covariance.params()
        mean = [mean[param.name] for param in params]
        return cls(mean, params=params, offset=offset, hessian=-covariance.to_precision(params=params, return_type='nparray'), with_prior=with_prior)


class Fisher(BaseClass):
    r"""
    Estimate Fisher matrix. If input ``likelihood`` is a :class:`BaseGaussianLikelihood` instance,
    or a :class:`SumLikelihood` of such instances, then the Fisher matrix will be computed as:

    .. math::

        F_{ij} = \frac{\partial \Delta}{\partial p_{i}} C^{-1} \frac{\partial \Delta}{\partial p_{j}}

    where :math:`\Delta` is the model (or data - model), of parameters :math:`p_{i}`, and :math:`C^{-1}`
    is the data hessian matrix.
    If input likelihood is not Gaussian, compute the second derivatives of the log-likelihood.
    """
    def __init__(self, likelihood, method=None, accuracy=2, delta_scale=1., mpicomm=None):
        """
        Initialize Fisher estimation.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        method : str, dict, default=None
            A dictionary mapping parameter name (including wildcard) to method to use to estimate derivatives,
            either 'auto' for automatic differentiation, or 'finite' for finite differentiation.
            If ``None``, 'auto' will be used if possible, else 'finite'.
            If a single value is provided, applies to all varied parameters.

        accuracy : int, dict, default=2
            A dictionary mapping parameter name (including wildcard) to derivative accuracy (number of points used to estimate it).
            If a single value is provided, applies to all varied parameters.
            Not used if ``method = 'auto'``  for this parameter.

        delta_scale : float, default=1.
            Parameter grid ranges for the estimation of finite derivatives are inferred from parameters' :attr:`Parameter.delta`.
            These values are then scaled by ``delta_scale`` (< 1. means smaller ranges).

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`.
        """
        if mpicomm is None:
            mpicomm = likelihood.mpicomm
        self.likelihood = likelihood

        solved_params = self.likelihood.all_params.select(solved=True)
        if solved_params:
            if mpicomm.rank == 0:
                import warnings
                warnings.warn('solved parameters: {}; cannot proceed with solved parameters, so we will work with likelihood.deepcopy(), varying solved parameters'.format(solved_params))
            self.likelihood = self.likelihood.deepcopy()
            for param in solved_params:
                self.likelihood.all_params[param].update(derived=False)

        self.varied_params = self.likelihood.varied_params

        prior_calculator = PriorCalculator()
        prior_calculator.params = [param for param in self.likelihood.all_params if param.depends or (not param.derived)]
        prior_simplified = all(param.prior.dist in ['norm', 'uniform'] and not param.depends for param in prior_calculator.params)
        #prior_simplified = False

        def prior_getter():
            return prior_calculator.logprior

        def prior_finalize(derivs):
            if prior_simplified:  # works but hopefully useless
                offset, gradient, hessian = 0., [], []
                for param in self.varied_params:
                    value = derivs[param.name]
                    loc, scale = getattr(param.prior, 'loc', 0.), getattr(param.prior, 'scale', np.inf)
                    prec = scale**(-2)
                    offset += - 0.5 * (value - loc)**2 * prec
                    # derivs[param]
                    gradient.append(- (value - loc) * prec)
                    hessian.append(- prec)
                hessian = np.diag(hessian)
                return {'offset': offset, 'gradient': gradient, 'hessian': hessian}
            return {'gradient': derivs}  # offset, gradient and hessian are pulled out of gradient by :class:`LikelihoodFisher`

        likelihoods = getattr(self.likelihood, 'likelihoods', [self.likelihood])
        from desilike.likelihoods import BaseGaussianLikelihood
        is_gaussian = all(isinstance(likelihood, BaseGaussianLikelihood) for likelihood in likelihoods)

        if is_gaussian:

            def getter():
                return [likelihood.flatdiff for likelihood in likelihoods]

            order = 1

            def likelihood_finalize(derivs):
                toret = []
                for likelihood, derivs in zip(likelihoods, derivs):
                    #flatderiv = np.array([derivs[param] for param in self.varied_params])
                    #flatdiff = derivs[()]
                    derivs = np.asarray(derivs)
                    flatdiff = derivs[0]
                    flatderiv = derivs[1:]  # first is zero-lag
                    precision = likelihood.precision
                    if precision.ndim == 1:
                        diffp = flatdiff * precision
                        derivp = flatderiv * precision
                    else:
                        diffp = flatdiff.dot(precision)
                        derivp = flatderiv.dot(precision)
                    offset = - diffp.dot(flatdiff.T)
                    gradient = - derivp.dot(flatdiff.T)
                    hessian = - derivp.dot(flatderiv.T)
                    toret.append({'offset': offset, 'gradient': gradient, 'hessian': hessian})
                return toret

        else:

            def getter():
                return [likelihood.loglikelihood for likelihood in likelihoods]

            order = 2

            def likelihood_finalize(derivs):
                toret = []
                for derivs in derivs:
                    offset = derivs[()]
                    gradient = np.array([derivs[param] for param in self.varied_params])
                    hessian = np.array([[derivs[param1, param2] for param2 in self.varied_params] for param1 in self.varied_params])
                    toret.append({'offset': offset, 'gradient': gradient, 'hessian': hessian})
                return toret

        if prior_simplified:
            self.prior_differentiation = None
        else:
            self.prior_differentiation = Differentiation(prior_calculator, getter=prior_getter, order=2, method=method, accuracy=accuracy, delta_scale=delta_scale, mpicomm=mpicomm)
        #self.prior_differentiation = Differentiation(prior_calculator, getter=prior_getter, order=2, method=method, accuracy=accuracy, delta_scale=delta_scale, mpicomm=mpicomm)
        self._prior_finalize = prior_finalize
        self.likelihood_differentiation = Differentiation(self.likelihood, getter=getter, order=order, method=method, accuracy=accuracy, delta_scale=delta_scale, mpicomm=mpicomm)
        self._likelihood_finalize = likelihood_finalize
        self.mpicomm = mpicomm

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm
        try:
            self.prior_differentiation.mpicomm = self._mpicomm
            self.likelihood_differentiation.mpicomm = self._mpicomm
        except AttributeError:
            pass

    def run(self, *args, **kwargs):
        params = _params_args_or_kwargs(args, kwargs)
        if self.prior_differentiation is None:
            center = diff = {**self.likelihood.runtime_info.pipeline.input_values, **params}
        else:
            diff = self.mpicomm.bcast(self.prior_differentiation(params), root=0)
            center = self.prior_differentiation.center
        #diff = self.mpicomm.bcast(self.prior_differentiation(params), root=0)
        #center = self.prior_differentiation.center
        self.prior_fisher = LikelihoodFisher(center=[center[str(param)] for param in self.varied_params], params=self.varied_params, **self._prior_finalize(diff))
        diff = self.mpicomm.bcast(self.likelihood_differentiation(**center), root=0)
        self.likelihood_fishers = [LikelihoodFisher(center=self.prior_fisher._center, params=self.varied_params, **kwargs) for kwargs in self._likelihood_finalize(diff)]

    def __call__(self, *args, **kwargs):
        """Return :class:`LikelihoodFisher` for input parameter values, as the sum of :attr:`prior_fisher` and :attr:`likelihood_fisher`."""
        self.run(*args, **kwargs)
        posterior_fisher = LikelihoodFisher.sum(self.likelihood_fishers + [self.prior_fisher])
        posterior_fisher.with_prior = True
        return posterior_fisher
