from desilike.samples import Chain
from desilike import utils
from .base import BaseBatchPosteriorSampler


class EmceeSampler(BaseBatchPosteriorSampler):

    name = 'emcee'

    def __init__(self, *args, nwalkers=None, **kwargs):
        """
        Wrapper for the affine-invariant ensemble sampler for Markov chain Monte Carlo (MCMC) proposed by Goodman & Weare (2010),
        see https://github.com/dfm/emcee.

        Parameters
        ----------
        nwalkers : int, str, default=None
            Number of walkers, defaults to 2 * max((int(2.5 * ndim) + 1) // 2, 2)
            Can be given in dimension units, e.g. '3 * ndim'.
        """
        super(EmceeSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * ndim) + 1) // 2, 2)
        self.nwalkers = utils.evaluate(nwalkers, type=int, locals={'ndim': len(self.varied_params)})
        import emcee
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.logposterior, vectorize=True)

    def _run_one(self, start, niterations=300, thin_by=1, progress=False):
        self.sampler._random = self.rng
        for _ in self.sampler.sample(initial_state=start, iterations=niterations, progress=progress, store=True, thin_by=thin_by, skip_initial_state_check=False):
            pass
        try:
            chain = self.sampler.get_chain()
        except AttributeError:
            return None
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        return Chain(data=data, params=self.varied_params + ['logposterior'])

    @classmethod
    def install(cls, config):
        config.pip('emcee')
