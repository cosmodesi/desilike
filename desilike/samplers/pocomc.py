import os

import numpy as np

from desilike.samples import Chain
from desilike import utils
from .base import BaseBatchPosteriorSampler


class PocoMCSampler(BaseBatchPosteriorSampler):

    """Wrapper for PocoMC sampler (preconditioned Monte Carlo method), see https://github.com/minaskar/pocomc."""

    name = 'pocomc'

    def __init__(self, *args, nwalkers=None, threshold=1.0, scale=True, rescale=False, diagonal=True, flow_config=None, train_config=None, **kwargs):
        """
        Initialize PocoMC sampler.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Input likelihood.

        nwalkers : int, str, default=None
            Number of walkers, defaults to 2 * max((int(2.5 * ndim) + 1) // 2, 2)
            Can be given in dimension units, e.g. '3 * ndim'.

        threshold : float, default=1.0
            The threshold value for the (normalised) proposal scale parameter below which normalising flow preconditioning (NFP) is enabled.
            Default is 1.0, meaning that NFP is used all the time.

        scale : bool, default=True
            Whether to scale the distribution of particles to have zero mean and unit variance.

        rescale : str, bool, default=False
            Whether to rescale the distribution of particles to have zero mean and unit variance in every iteration.
            Pass 'diagonal' to use a diagonal covariance matrix when rescaling instead of a full covariance.

        flow_config : dict, default=None
            Configuration of the normalizing flow.

        train_config : dict, default=None
            Configuration for training the normalizing flow.

        rng : np.random.RandomState, default=None
            Random state. If ``None``, ``seed`` is used to set random state.

        seed : int, default=None
            Random seed.

        max_tries : int, default=1000
            A :class:`ValueError` is raised after this number of likelihood (+ prior) calls without finite posterior.

        chains : str, Path, Chain
            Path to or chains to resume from.

        ref_scale : float, default=1.
            Rescale parameters' :attr:`Parameter.ref` reference distribution by this factor.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``likelihood``'s :attr:`BaseLikelihood.mpicomm`.
        """
        super(PocoMCSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * ndim) + 1) // 2, 2)
        self.nwalkers = utils.evaluate(nwalkers, type=int, locals={'ndim': ndim})
        bounds = np.array([tuple(None if np.isinf(lim) else lim for lim in param.prior.limits) for param in self.varied_params], dtype='f8')
        import pocomc
        diagonal = rescale == 'diagonal'
        rescale = bool(rescale)
        self.sampler = pocomc.Sampler(self.nwalkers, ndim, self.loglikelihood, self.logprior, bounds=bounds, threshold=threshold, scale=scale,
                                      rescale=rescale, diagonal=diagonal, flow_config=flow_config, train_config=train_config,
                                      vectorize_likelihood=True, vectorize_prior=True, infer_vectorization=False,
                                      output_dir=None, output_label=None, random_state=self.rng.randint(0, high=0xffffffff))
        if self.save_fn is None:
            raise ValueError('save_fn must be provided to save pocomc state')
        self.state_fn = [os.path.splitext(fn)[0] + '.pocomc.state' for fn in self.save_fn]

    def logprior(self, params, bounds=None):
        return super(PocoMCSampler, self).logprior(params)

    def _prepare(self):
        self.resume = self.mpicomm.bcast(any(chain is not None for chain in self.chains), root=0)

    def _run_one(self, start, niterations=300, progress=False, **kwargs):
        if self.resume:
            self.sampler.load_state(self.state_fn[self._ichain])
            #self.derived = self.sampler.derived
            #del self.sampler.derived
            from pocomc.tools import FunctionWrapper
            # Because dill is unable to cope with our loglikelihood and logprior
            self.sampler.log_likelihood = FunctionWrapper(self.loglikelihood, args=None, kwargs=None)
            self.sampler.log_prior = FunctionWrapper(self.logprior, args=None, kwargs=None)
            self.sampler.log_likelihood(self.sampler.x)  # to set derived parameters

        import torch
        np_random_state_bak, torch_random_state_bak = np.random.get_state(), torch.get_rng_state()
        self.sampler.random_state = self.rng.randint(0, high=0xffffffff)
        np.random.set_state(self.rng.get_state())  # self.rng is same for all ranks
        torch.set_rng_state(self.mpicomm.bcast(torch_random_state_bak, root=0))

        if not self.resume:
            self.sampler.run(prior_samples=start, progress=progress, **kwargs)

        self.sampler.add_samples(n=niterations)
        np.random.set_state(np_random_state_bak)
        torch.set_rng_state(torch_random_state_bak)
        try:
            result = self.sampler.results
        except ValueError:
            return None
        # This is not picklable
        del self.sampler.log_likelihood, self.sampler.log_prior
        # Clear saved quantities to save space
        for name in self.sampler.__dict__:
            if name.startswith('saved_'): setattr(self.sampler, name, [])
        # Save last parameters, which be reused in the next run
        #self.sampler.derived = [d[:-1] for d in self.derived]
        if self.mpicomm.rank == 0:
            self.sampler.save_state(self.state_fn[self._ichain])
        data = [result['samples'][..., iparam] for iparam, param in enumerate(self.varied_params)] + [result['logprior'], result['loglikelihood']]
        return Chain(data=data, params=self.varied_params + ['logprior', 'loglikelihood'])

    @classmethod
    def install(cls, config):
        config.pip('pocomc')
