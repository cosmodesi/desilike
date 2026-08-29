"""Proposal distributions: the beta = 0 end of a tempered / bridged sampling path.

A proposal replaces the prior as the starting distribution handed to population kernels
(see :meth:`~desilike.samplers.base.BaseSampler._set_proposal`). It does not change the
inferred posterior: kernels receive the likelihood as ``log_posterior - log_proposal``, so
the tempered target ``proposal * likelihood^beta`` is the exact posterior at ``beta = 1``
for any proposal. What it changes is the *distance* the sampler has to anneal over, and
with a proposal that is already posterior-shaped -- an emulated posterior, say -- that
distance collapses from "prior to posterior" to "emulator error".

Three adapters, composable:

- :class:`SamplesProposal` -- an existing chain plus the density it was sampled under.
  The density may be unnormalized (constants cancel in ``log_posterior - log_proposal``),
  and draws are chain rows, so parameter correlations are carried over exactly.
- :class:`GaussianProposal` -- a multivariate Gaussian truncated to the hard prior box,
  typically fitted to a marginal of an existing chain and inflated.
- :class:`ProductProposal` -- a product of factors over disjoint blocks of parameters.

All three expose ``logpdf(x)`` on a flat vector in original parameter space (JAX-traceable)
and ``rvs(size, rng)``; :class:`GaussianProposal` and products of Gaussians also expose
``ppf(u)``, which nested samplers require.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np

from ..parameter import VariableCollection, _cumsize_params
from ..samples import diagnostics


logger = logging.getLogger('Proposal')


class BaseProposal(object):
    """Base class for proposal distributions.

    Subclasses implement :meth:`logpdf` and :meth:`rvs` (or :meth:`ppf`), and set
    :attr:`params` to whatever they cover. :meth:`init` is called by the sampler with the
    parameters the proposal is responsible for, in the sampler's own flat order, so a
    proposal never has to guess the parameter layout.
    """

    logger = logging.getLogger('Proposal')

    #: The parameters this proposal covers -- any container answering ``param in params``,
    #: so the object a proposal is built from (a Covariance, an MCSamples) serves directly.
    #: ``None`` covers whatever the proposal is handed.
    params = None

    def init(self, params):
        """Bind the proposal to those of *params* it covers, in the given flat order.

        :attr:`params` goes from the container the proposal was built from to the bound
        collection itself, which is a valid container in turn.
        """
        self.params = VariableCollection([param for param in params
                                          if self.params is None or param in self.params])

    @property
    def ndim(self):
        """Width of the flat parameter vector this proposal is bound to."""
        return int(_cumsize_params(self.params)[-1])

    def limits(self):
        """Return the ``(ndim, 2)`` hard prior limits of the bound parameters."""
        rows = []
        for param in self.params:
            low, high = np.asarray(param.prior.limits, dtype='f8')
            rows += [[low, high]] * param.size
        return np.array(rows)

    def logpdf(self, x):
        """Return the log-density of a flat ``(ndim,)`` point in original parameter space."""
        raise NotImplementedError

    def rvs(self, size, rng):
        """Return ``(size, ndim)`` draws in original parameter space.

        Defaults to pushing a uniform cube through :meth:`ppf`; proposals with no inverse
        CDF must override it.
        """
        ppf = getattr(self, 'ppf', None)
        if not callable(ppf):
            raise NotImplementedError(f'{type(self).__name__} exposes neither rvs nor ppf.')
        return np.asarray(jax.vmap(ppf)(rng.random((size, self.ndim))))


class PriorProposal(BaseProposal):
    """The parameters' own priors, as a proposal factor.

    Used to fill the parameters a :class:`ProductProposal` does not otherwise cover, so
    that "a fitted Gaussian for these, the prior for the rest" needs no special case.

    Like every prior density in desilike, the per-parameter logpdfs are unnormalized, so a
    product containing this factor returns an evidence offset by ``log int prior``.
    """

    def logpdf(self, x):
        cumsize = _cumsize_params(self.params)
        result = jnp.zeros(())
        for i, param in enumerate(self.params):
            if param.prior is not None:
                chunk = x[cumsize[i]:cumsize[i + 1]]
                chunk = chunk.reshape(param.shape) if param.shape else chunk[0]
                result = result + param.prior.logpdf(chunk)
        return result

    def ppf(self, u):
        cumsize = _cumsize_params(self.params)
        return jnp.concatenate([jnp.atleast_1d(param.prior.ppf(u[cumsize[i]:cumsize[i + 1]]))
                                for i, param in enumerate(self.params)])


class SamplesProposal(BaseProposal):
    """An existing chain, plus the density it was sampled under.

    This is the proposal of the emulated-to-exact bridge: *density* is the cheap
    (emulated) posterior and *samples* a converged chain under it, so
    ``log_posterior - log_proposal`` is the emulator error and nothing else. Draws are
    rows of the chain, resampled to equal weight, which carries over every parameter
    correlation the chain contains -- what a fitted Gaussian would throw away.

    The density need not be normalized: an additive constant in ``logpdf`` shifts
    ``log_posterior - log_proposal`` uniformly and cancels everywhere the samplers use it
    (weight *ratios*, tempering increments), leaving only the evidence offset by that same
    constant.
    """

    def __init__(self, density, samples):
        """
        Parameters
        ----------
        density : CompiledGraph, Calculator, or callable
            The (possibly unnormalized) log-density the samples were drawn from. A
            calculator is compiled; a compiled graph is called on the ``{name: value}``
            dict of parameters; a bare callable is called on the flat ``(ndim,)`` vector.
            Must be JAX-traceable.
        samples : MCSamples
            Chain sampled under *density*. Its weights are honoured when drawing.
        """
        self.density = density
        self.samples = self.params = samples
        self._particles = None

    def init(self, params):
        super().init(params)
        from ..base import build, CompiledGraph
        density = self.density
        if not (isinstance(density, CompiledGraph) or callable(density)):
            raise TypeError(f'density must be a CompiledGraph, Calculator or callable, got {type(density)}')
        if not isinstance(density, CompiledGraph) and hasattr(density, 'params'):
            density = build(density)
        if isinstance(density, CompiledGraph):
            missing = [param.name for param in self.params if param.name not in density.params]
            if missing:
                raise ValueError(f'density does not know about parameters {missing}.')

            cumsize = _cumsize_params(self.params)

            def logpdf(x):
                values = {param.name: (x[cumsize[i]:cumsize[i + 1]].reshape(param.shape) if param.shape
                                       else x[cumsize[i]])
                          for i, param in enumerate(self.params)}
                return density(values, return_derived=False)

            self._logpdf = logpdf
        else:
            self._logpdf = density
        columns = [np.asarray(self.samples[param.name].value) for param in self.params]
        self._particles = np.column_stack([column.reshape(len(column), -1) for column in columns])
        self._weights = np.asarray(self.samples.weight, dtype='f8').ravel()

    def logpdf(self, x):
        return self._logpdf(x)

    def rvs(self, size, rng):
        index = diagnostics.systematic_resample(self._weights, size, rng)
        return self._particles[index]


class GaussianProposal(BaseProposal):
    """A multivariate Gaussian, truncated to the hard prior box.

    Typically fitted to a marginal of an existing chain and *inflated*: an
    under-dispersed proposal is not self-healing. Measured on an analytic Gaussian
    target, a proposal 30% narrower than the posterior leaves 5-17% residual
    under-dispersion that rejuvenation does not repair, while one 1.5-2x wider recovers
    the moments to better than 0.06 sigma. Inflate.

    The Gaussian normalization is included, so a bridge starting from this proposal returns
    the actual log-evidence. The *truncation* is not renormalized, though: when the box cuts
    into the Gaussian, the returned evidence is off by the (positive) log of the retained mass.
    """

    def __init__(self, covariance, center=None, scale=1.):
        """
        Parameters
        ----------
        covariance : Covariance
            Covariance of the Gaussian, over the parameters this factor covers.
        center : array, dict, or None
            Mean. ``None`` (default) uses ``covariance.center``.
        scale : float, optional
            Multiplier on the *standard deviations* (the covariance is scaled by
            ``scale**2``). Use 1.5-2 to inflate a fitted covariance. Default is 1.
        """
        self.covariance = self.params = covariance
        self.center = center
        self.scale = float(scale)

    @classmethod
    def from_samples(cls, samples, params=None, scale=1.):
        """Fit a Gaussian to (a marginal of) *samples*.

        Parameters
        ----------
        samples : MCSamples
            Chain to fit.
        params : list, str, or None
            Parameters to keep; ``None`` keeps every varied parameter of *samples*.
            Selecting a subset gives the *marginal*, which over-covers every conditional
            slice of the full distribution -- the safe, over-dispersed side to bridge from
            when the coupling to the remaining parameters is not modelled.
        scale : float, optional
            Multiplier on the standard deviations. Default is 1.
        """
        if params is None:
            params = samples.select(derived=False).names()
        covariance = samples.covariance(params)
        return cls(covariance, center=samples.mean(params), scale=scale)

    def init(self, params):
        super().init(params)
        covariance = self.covariance.select(self.params.names())
        value = np.asarray(covariance.value, dtype='f8') * self.scale**2
        center = self.center
        if center is None:
            center = covariance.center
        elif isinstance(center, dict):
            center = np.concatenate([np.ravel(center[param.name]) for param in self.params])
        center = np.asarray(center, dtype='f8').ravel()
        if value.shape != (self.ndim, self.ndim) or center.size != self.ndim:
            raise ValueError(f'covariance / center shapes {value.shape} / {center.shape} do not match '
                             f'the {self.ndim} bound dimensions.')
        try:
            cholesky = np.linalg.cholesky(value)
        except np.linalg.LinAlgError as exc:
            raise ValueError('proposal covariance is not positive-definite.') from exc
        inverse = np.linalg.inv(cholesky)
        self._mean = jnp.asarray(center)
        self._cholesky = jnp.asarray(cholesky)
        self._precision = jnp.asarray(inverse.T @ inverse)
        # The Gaussian normalization is known, so keep it: it is what makes the evidence
        # returned by a bridge from this proposal the actual log Z, not log Z up to a constant.
        self._log_norm = float(0.5 * self.ndim * np.log(2. * np.pi) + np.sum(np.log(np.diag(cholesky))))
        # Keep the sampler strictly inside the support: a point exactly on the edge is
        # turned into a NaN by the logit rescalings some kernels apply to bounded dimensions.
        limits = self.limits()
        width = limits[:, 1] - limits[:, 0]
        margin = 1e-7 * np.where(np.isfinite(width), width, 1.)
        self._low = np.where(np.isfinite(limits[:, 0]), limits[:, 0] + margin, -np.inf)
        self._high = np.where(np.isfinite(limits[:, 1]), limits[:, 1] - margin, np.inf)

    def logpdf(self, x):
        diff = x - self._mean
        log_p = -0.5 * (diff @ self._precision @ diff) - self._log_norm
        inside = jnp.all((x >= jnp.asarray(self._low)) & (x <= jnp.asarray(self._high)))
        return jnp.where(inside, log_p, -jnp.inf)

    def ppf(self, u):
        # Clip z before the matmul: a zero row of the Cholesky factor times an infinite z
        # is a NaN, and u = 0 or 1 does happen.
        z = jnp.clip(jax.scipy.stats.norm.ppf(u), -1e38, 1e38)
        return jnp.clip(self._mean + self._cholesky @ z, jnp.asarray(self._low), jnp.asarray(self._high))

    def rvs(self, size, rng):
        mean, cholesky = np.asarray(self._mean), np.asarray(self._cholesky)
        accepted, ndraws = [], 0
        for _ in range(100):
            draws = mean + rng.standard_normal((2 * size, self.ndim)) @ cholesky.T
            mask = np.all((draws >= self._low) & (draws <= self._high), axis=1)
            accepted.append(draws[mask])
            ndraws += int(mask.sum())
            if ndraws >= size:
                return np.concatenate(accepted)[:size]
        raise RuntimeError(f'GaussianProposal.rvs: only {ndraws} of {size} draws fell inside the prior '
                           'box after 100 attempts; the Gaussian sits mostly outside the prior.')


class ProductProposal(BaseProposal):
    """A product of proposals over disjoint blocks of parameters.

    This is how an analysis grows: a joint posterior already sampled for some tracers,
    times a fitted (inflated) marginal for each tracer being added. The joint factor
    keeps the parameters shared across tracers -- cosmology -- so they appear exactly
    once, with their correlations to the nuisances intact; the new blocks contribute only
    their own nuisances. Whatever coupling this factorization misses is what the bridge
    then corrects, from the over-dispersed side.

    Parameters no factor covers keep their own prior, through an implicit
    :class:`PriorProposal` appended after the given factors. A product given as a factor is
    flattened into its own: since a product fills whatever it is handed, nesting one would
    have it swallow the parameters its siblings cover.
    """

    def __init__(self, *factors):
        """
        Parameters
        ----------
        *factors : BaseProposal
            Factors covering disjoint parameter blocks. Parameters left uncovered fall
            back to their prior.
        """
        if len(factors) == 1 and isinstance(factors[0], (list, tuple)):
            factors = tuple(factors[0])
        # A product covers whatever it is given -- it fills the gaps itself -- so `params`
        # stays None, and a nested product is flattened rather than handed a block to fill.
        self.factors = []
        for factor in factors:
            self.factors += factor.factors if isinstance(factor, ProductProposal) else [factor]

    def init(self, params):
        super().init(params)
        cumsize = _cumsize_params(self.params)
        columns = {param.name: i for i, param in enumerate(self.params)}

        def block(factor, params):
            factor.init(params)
            index = np.concatenate([np.arange(cumsize[columns[param.name]],
                                              cumsize[columns[param.name] + 1])
                                    for param in params]).astype('i8')
            return factor, index

        seen, self._blocks = set(), []
        for factor in self.factors:
            covered = [param for param in self.params
                       if factor.params is None or param in factor.params]
            if not covered:
                raise ValueError(f'factor {type(factor).__name__} covers none of the varied parameters.')
            overlap = seen.intersection(param.name for param in covered)
            if overlap:
                raise ValueError(f'factors of a ProductProposal must cover disjoint parameters; '
                                 f'{sorted(overlap)} appear in more than one factor.')
            seen.update(param.name for param in covered)
            self._blocks.append(block(factor, VariableCollection(covered)))
        # Whatever is left keeps its own prior.
        missing = [param for param in self.params if param.name not in seen]
        if missing:
            self._blocks.append(block(PriorProposal(), VariableCollection(missing)))

    def logpdf(self, x):
        result = jnp.zeros(())
        for factor, index in self._blocks:
            result = result + factor.logpdf(x[index])
        return result

    def ppf(self, u):
        missing = [type(factor).__name__ for factor, _ in self._blocks if not callable(getattr(factor, 'ppf', None))]
        if missing:
            raise NotImplementedError(f'factors {missing} expose no ppf, so the product cannot be '
                                      'inverted from the unit cube. Use a kernel that draws through rvs.')
        result = jnp.zeros(self.ndim)
        start = 0
        for factor, index in self._blocks:
            result = result.at[index].set(factor.ppf(u[start:start + index.size]))
            start += index.size
        return result

    def rvs(self, size, rng):
        result = np.empty((size, self.ndim))
        for factor, index in self._blocks:
            result[:, index] = factor.rvs(size, rng)
        return result
