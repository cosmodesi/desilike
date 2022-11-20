import logging
import random

import numpy as np

from desilike.samples import Chain
from desilike import utils
from .base import BaseBatchPosteriorSampler
from .utils import numpy_to_python_random_state


class ZeusSampler(BaseBatchPosteriorSampler):

    name = 'zeus'

    def __init__(self, *args, nwalkers=None, light_mode=False, **kwargs):
        """
        Wrapper for the zeus sampler, see https://github.com/minaskar/zeus.

        Parameters
        ----------
        nwalkers : int, str, default=None
            Number of walkers, defaults to 2 * max((int(2.5 * ndim) + 1) // 2, 2)
            Can be given in dimension units, e.g. '3 * ndim'.

        light_mode : bool, default=False
            If True (default is False) then no expansions are performed after the tuning phase.
            This can significantly reduce the number of log likelihood evaluations but works best in target distributions that are apprroximately Gaussian.
        """
        super(ZeusSampler, self).__init__(*args, **kwargs)
        ndim = len(self.varied_params)
        if nwalkers is None:
            nwalkers = 2 * max((int(2.5 * ndim) + 1) // 2, 2)
        self.nwalkers = utils.evaluate(nwalkers, type=int, locals={'ndim': ndim})
        import zeus
        handlers = logging.root.handlers.copy()
        level = logging.root.level
        self.sampler = zeus.EnsembleSampler(self.nwalkers, ndim, self.logposterior, verbose=False, light_mode=bool(light_mode), vectorize=True)
        logging.root.handlers = handlers
        logging.root.level = level

    def _run_one(self, start, niterations=300, thin_by=1, progress=False):
        py_random_state_bak, np_random_state_bak = random.getstate(), np.random.get_state()
        random.setstate(numpy_to_python_random_state(self.rng.get_state()))  # self.rng is same for all ranks
        np.random.set_state(self.rng.get_state())
        for _ in self.sampler.sample(start=start, iterations=niterations, progress=progress, thin_by=thin_by):
            pass
        chain = self.sampler.get_chain()
        data = [chain[..., iparam] for iparam, param in enumerate(self.varied_params)] + [self.sampler.get_log_prob()]
        self.sampler.reset()
        random.setstate(py_random_state_bak)
        np.random.set_state(np_random_state_bak)
        return Chain(data=data, params=self.varied_params + ['logposterior'])

    @classmethod
    def install(cls, config):
        config.pip('zeus-mcmc')
