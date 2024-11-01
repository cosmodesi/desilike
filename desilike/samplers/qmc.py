import numpy as np

from scipy.stats import qmc
from scipy.stats.qmc import Sobol, Halton, LatinHypercube

from desilike.base import vmap
from desilike.parameter import ParameterPriorError, Samples
from desilike.utils import BaseClass
from .base import RegisteredSampler


class RQuasiRandomSequence(qmc.QMCEngine):

    def __init__(self, d, seed=0.5):
        super().__init__(d=d)
        self.seed = float(seed)
        phi = 1.0
        # This is the Newton's method, solving phi**(d+1) - phi - 1 = 0
        eq_check = phi**(self.d + 1) - phi - 1
        while np.abs(eq_check) > 1e-12:
            phi -= (phi**(self.d + 1) - phi - 1) / ((self.d + 1) * phi**self.d - 1)
            eq_check = phi**(self.d + 1) - phi - 1
        self.inv_phi = [phi**(-(1 + d)) for d in range(self.d)]

    def _random(self, n=1, *, workers=1):
        toret = (self.seed + np.arange(self.num_generated + 1, self.num_generated + n + 1)[:, None] * self.inv_phi) % 1.
        self.num_generated += n
        return toret

    def reset(self):
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        self.num_generated += n
        return self


if not hasattr(qmc.QMCEngine, '_random'):  # old scipy version <= 1.8.1
    RQuasiRandomSequence.random = RQuasiRandomSequence._random
    del RQuasiRandomSequence._random


def get_qmc_engine(engine):

    return {'sobol': Sobol, 'halton': Halton, 'lhs': LatinHypercube, 'rqrs': RQuasiRandomSequence}.get(engine, engine)


class QMCSampler(BaseClass, metaclass=RegisteredSampler):

    """Quasi Monte-Carlo sequences, using :mod:`scipy.qmc` (+ RQuasiRandomSequence)."""
    name = 'qmc'

    def __init__(self, calculator, samples=None, mpicomm=None, engine='rqrs', save_fn=None, **kwargs):
        """
        Initialize QMC sampler.

        Parameters
        ----------
        calculator : BaseCalculator
            Input calculator.

        samples : str, Path, Samples
            Path to or samples to resume from.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``calculator``'s :attr:`BaseCalculator.mpicomm`.

        engine : str, default='rqrs'
            QMC engine, to choose from ['sobol', 'halton', 'lhs', 'rqrs'].

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        seed : int, default=None
            Random seed.

        **kwargs : dict
            Optional engine-specific arguments.
        """
        #self.pipeline = calculator.runtime_info.pipeline
        self.calculator = calculator
        if mpicomm is None:
            mpicomm = calculator.mpicomm
        self.mpicomm = mpicomm
        self.varied_params = self.calculator.varied_params
        self.engine = get_qmc_engine(engine)(d=len(self.varied_params), **kwargs)
        self.samples = None
        if self.mpicomm.rank == 0 and samples is not None:
            self.samples = samples if isinstance(samples, Samples) else Samples.load(samples)
        self.save_fn = save_fn

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self._mpicomm = mpicomm

    def run(self, niterations=300, offset=None):
        """
        Run sampling. Sampling can be interrupted anytime, and resumed by providing
        the path to the saved samples in ``samples`` argument of :meth:`__init__`.

        Parameters
        ----------
        niterations : int, default=300
            Number of samples to draw.
        """
        lower, upper = [], []
        for param in self.varied_params:
            try:
                lower.append(param.value - param.proposal)
                upper.append(param.value + param.proposal)
            except AttributeError as exc:
                raise ParameterPriorError('Provide parameter limits or proposal for {}'.format(param)) from exc
        if self.mpicomm.rank == 0:
            self.engine.reset()
            if offset is None:
                offset = len(self.samples) if self.samples is not None else 0
            self.engine.fast_forward(offset)
            samples = qmc.scale(self.engine.random(n=niterations), lower, upper)
            samples = Samples(samples.T, params=self.varied_params)

        vcalculate = vmap(self.calculator, backend='mpi', errors='nan', return_derived=True)
        derived = vcalculate(samples.to_dict() if self.mpicomm.rank == 0 else {}, mpicomm=self.mpicomm)[1]

        if self.mpicomm.rank == 0:
            for param in self.calculator.all_params.select(fixed=True, derived=False):
                samples[param] = np.full(samples.shape, param.value, dtype='f8')
            samples.update(derived)
            if self.samples is None:
                self.samples = samples
            else:
                self.samples = Samples.concatenate(self.samples, samples)
            if self.save_fn is not None:
                self.samples.save(self.save_fn)
        else:
            self.samples = None
        return self.samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
