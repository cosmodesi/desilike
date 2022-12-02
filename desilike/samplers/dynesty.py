import os
import sys

import numpy as np

from desilike.samples import Chain
from .base import BasePosteriorSampler, load_source, Samples, batch_iterate


class FakePool(object):

    def __init__(self, size=1):
        self.size = size

    def map(self, func, values):
        return func(values)


class BaseDynestySampler(BasePosteriorSampler):

    check = None

    def __init__(self, *args, nlive=500, bound='multi', sample='auto', update_interval=None, **kwargs):
        self.nlive = int(nlive)
        self.attrs = {'bound': bound, 'sample': sample, 'update_interval': update_interval}
        super(BaseDynestySampler, self).__init__(*args, **kwargs)
        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save dynesty state')
        self.state_fn = [os.path.splitext(fn)[0] + '.dynesty.state' for fn in self.save_fn]

    def prior_transform(self, values):
        toret = np.empty_like(values)
        for iparam, (value, param) in enumerate(zip(values.T, self.varied_params)):
            try:
                toret[..., iparam] = param.prior.ppf(value)
            except AttributeError as exc:
                raise AttributeError('{} has no attribute ppf (maybe infinite prior?). Choose proper prior for nested sampling'.format(param.prior)) from exc
        return toret

    def _prepare(self):
        self.resume = self.mpicomm.bcast(any(chain is not None for chain in self.chains), root=0)

    def _run_one(self, start, min_iterations=0, max_iterations=sys.maxsize, check_every=300, check=None, **kwargs):
        from dynesty import utils
        if check is not None and not isinstance(check, bool): kwargs.update(check)

        rstate = np.random.Generator(np.random.PCG64(self.rng.randint(0, high=0xffffffff)))

        # Instantiation already runs somes samples
        if not hasattr(self, 'sampler'):
            use_pool = {'prior_transform': True, 'loglikelihood': True, 'propose_point': False, 'update_bound': False}
            pool = FakePool(size=self.mpicomm.size)
            self._set_sampler(rstate, pool, use_pool)

        self.resume_derived, self.resume_chain = None, None
        if self.resume:
            sampler = utils.restore_sampler(self.state_fn[self._ichain])
            del sampler.loglikelihood, sampler.prior_transform, sampler.pool, sampler.M
            if type(sampler) is not type(self.sampler):
                raise ValueError('Previous run used {}, not {}.'.format(type(sampler), type(self.sampler)))
            self.sampler.__dict__.update(sampler.__dict__)
            source = load_source(self.save_fn[self._ichain])[0]
            points = {}
            for param in self.varied_params:
                points[param.name] = source.pop(param)
            self.resume_derived = [source.select(derived=True), points]

        self.sampler.rstate = rstate

        def _run_one_batch(niterations):
            it = self.sampler.it
            self._run_nested(niterations, **kwargs)
            is_converged = self.sampler.it - it < niterations
            results = self.sampler.results
            chain = [results['samples'][..., iparam] for iparam, param in enumerate(self.varied_params)]
            logprior = sum(param.prior(value) for param, value in zip(self.varied_params, chain))
            chain.append(logprior)
            chain.append(results['logl'] + logprior)
            chain.append(results['logwt'])
            chain.append(np.exp(results.logwt - results.logz[-1]))
            chain = Chain(chain, params=self.varied_params + ['logprior', 'logposterior', 'logweight', 'aweight'])

            if self.mpicomm.rank == 0:
                if self.resume_derived is not None:
                    if self.derived is not None:
                        self.derived = [Samples.concatenate([resume_derived, derived]) for resume_derived, derived in zip(self.resume_derived, self.derived)]
                    else:
                        self.derived = self.resume_derived
                chain = self._set_derived(chain)
                self.resume_chain = chain = self._set_derived(chain)
                self.resume_chain.save(self.save_fn[self._ichain])
                utils.save_sampler(self.sampler, self.state_fn[self._ichain])

            self.resume_derived = self.derived
            self.derived = None
            return is_converged

        batch_iterate(_run_one_batch, min_iterations=min_iterations, max_iterations=max_iterations, check_every=check_every)

        self.derived = self.resume_derived
        return self.resume_chain

    @classmethod
    def install(cls, config):
        config.pip('dynesty')


class StaticDynestySampler(BaseDynestySampler):

    def _set_sampler(self, rstate, pool, use_pool):
        import dynesty
        self.sampler = dynesty.NestedSampler(self.loglikelihood, self.prior_transform, len(self.varied_params), nlive=self.nlive, pool=pool, use_pool=use_pool, rstate=rstate, **self.attrs)

    def _run_nested(self, niterations, **kwargs):
        self.sampler.run_nested(maxiter=niterations, **kwargs)


class DynamicDynestySampler(BaseDynestySampler):

    def _set_sampler(self, rstate, pool, use_pool):
        import dynesty
        self.sampler = dynesty.DynamicNestedSampler(self.loglikelihood, self.prior_transform, len(self.varied_params), pool=pool, use_pool=use_pool, rstate=rstate, **self.attrs)

    def _run_nested(self, niterations, **kwargs):
        self.sampler.run_nested(nlive_init=self.nlive, maxiter=niterations + self.sampler.it, **kwargs)
