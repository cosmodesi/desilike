import itertools

import numpy as np

from desilike.parameter import ParameterArray, ParameterPriorError, Samples
from desilike.utils import BaseClass
from .base import RegisteredSampler


class GridSampler(BaseClass, metaclass=RegisteredSampler):

    name = 'grid'

    def __init__(self, calculator, mpicomm=None, ngrid=3, ref_scale=1., sphere=None, save_fn=None):
        self.pipeline = calculator.runtime_info.pipeline
        if mpicomm is None:
            mpicomm = calculator.mpicomm
        self.mpicomm = mpicomm
        self.varied_params = self.pipeline.varied_params
        self.ref_scale = float(ref_scale)
        self.sphere = sphere
        for name, item in zip(['ngrid', 'sphere'], [ngrid] + ([sphere] if sphere is not None else [])):
            if not isinstance(item, dict):
                item = {'*': item}
            tmp = {str(param): None for param in self.varied_params}
            for template, value in item.items():
                for tmpname in self.varied_params.names(name=template):
                    tmp[tmpname] = int(value)
            for param, value in tmp.items():
                if value is None:
                    raise ValueError('{} not specified for parameter {}'.format(name, param))
                elif value < 1:
                    raise ValueError('{} is {:d} < 1 for parameter {}'.format(name, value, param))
            tmp = [tmp[str(param)] for param in self.varied_params]
            setattr(self, name, tmp)
        self.save_fn = save_fn

    @property
    def mpicomm(self):
        return self.pipeline.mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self.pipeline.mpicomm = mpicomm

    def run(self):
        if self.mpicomm.rank == 0:
            grid = []
            for iparam, (param, ngrid) in enumerate(zip(self.varied_params, self.ngrid)):
                ngrid = self.ngrid[iparam]
                if ngrid == 1:
                    grid.append(np.array(param.value))
                elif param.ref.is_proper():
                    grid.append(np.linspace(param.value - self.ref_scale * param.proposal, param.value + self.ref_scale * param.proposal, ngrid))
                else:
                    raise ParameterPriorError('Provide parameter limits or proposal')
            if self.sphere:
                ndim = len(grid)
                if any(len(g) % 2 == 0 for g in grid):
                    raise ValueError('Number of grid points along each axis must be odd to use sphere option')
                samples = []
                cidx = [len(g) // 2 for g in grid]
                samples.append([g[c] for g, c in zip(grid, cidx)])
                for ngrid in range(1, max(self.sphere) + 1):
                    for indices in itertools.product(range(ndim), repeat=ngrid):
                        indices = np.bincount(indices, minlength=ndim)
                        if all(i <= c for c, i in zip(cidx, indices)) and sum(indices) <= min(s for s, i in zip(self.sphere, indices) if i):
                            for signs in itertools.product(*[[-1, 1] if ind else [0] for ind in indices]):
                                samples.append([g[c + s * i] for g, c, s, i in zip(grid, cidx, signs, indices)])
                #for indices, values in zip(itertools.product(*igrid), itertools.product(*grid)):
                #    if self.sphere and sum(indices) > min([self.sphere[ii] for ii, ngrid in enumerate(self.ngrid) if indices[ii]] or [0]):
                #        continue
                #    samples.append(values)
                samples = np.array(samples).T
            else:
                samples = np.meshgrid(*grid, indexing='ij')
            samples = Samples(samples, params=self.varied_params)
            samples.attrs['ngrid'] = self.ngrid

        self.pipeline.mpicalculate(**(samples.to_dict() if self.mpicomm.rank == 0 else {}))

        if self.mpicomm.rank == 0:
            for param in self.pipeline.params.select(fixed=True, derived=False):
                samples.set(ParameterArray(np.full(samples.shape, param.value, dtype='f8'), param))
            samples.update(self.pipeline.derived)
            self.samples = samples
            if self.save_fn is not None:
                self.samples.save(self.save_fn)
        else:
            self.samples = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
