import itertools

import numpy as np

from desilike.parameter import ParameterPriorError, Samples
from desilike.utils import BaseClass, expand_dict
from .base import RegisteredSampler


class GridSampler(BaseClass, metaclass=RegisteredSampler):

    name = 'grid'

    def __init__(self, calculator, mpicomm=None, save_fn=None, **kwargs):
        self.pipeline = calculator.runtime_info.pipeline
        if mpicomm is None:
            mpicomm = calculator.mpicomm
        self.mpicomm = mpicomm
        self.varied_params = self.pipeline.varied_params
        self.save_fn = save_fn
        self.set_grid(**kwargs)

    @property
    def mpicomm(self):
        return self.pipeline.mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        self.pipeline.mpicomm = mpicomm

    def set_grid(self, grid=None, size=1, ref_scale=1., sphere=None):
        self.ref_scale = float(ref_scale)
        self.sphere = sphere
        self.grid = expand_dict(grid, self.varied_params.names())
        for name, item in zip(['size', 'sphere'], [size] + ([sphere] if sphere is not None else [])):
            tmp = expand_dict(item, self.varied_params.names())
            for param, value in tmp.items():
                if value is None:
                    if name != 'size' or self.grid[param] is None:
                        raise ValueError('{} not specified for parameter {}'.format(name, param))
                value = int(value)
                if name == 'size' and value < 1:
                    raise ValueError('{} is {} < 1 for parameter {}'.format(name, value, param))
                tmp[param] = value
            tmp = [tmp[param] for param in self.varied_params.names()]
            setattr(self, name, tmp)
        for iparam, (param, size) in enumerate(zip(self.varied_params, self.size)):
            grid = self.grid[param.name]
            if grid is None:
                if size == 1:
                    grid = [param.value]
                elif param.ref.is_limited() and not hasattr(param.ref, 'scale'):
                    limits = np.array(param.ref.limits)
                    center = param.value
                    limits = self.ref_scale * (limits - center) + center
                    low, high = np.linspace(limits[0], center, size // 2 + 1), np.linspace(center, limits[1], size // 2 + 1)
                    if size % 2:
                        grid = np.concatenate([low, high[1:]])
                    else:
                        grid = np.concatenate([low[:-1], high[1:]])
                    if not np.all(np.diff(grid) > 0):
                        raise ParameterPriorError('Parameter {} value {} is not in reference limits {}'.format(param, param.value, param.ref.limits))
                elif param.proposal:
                    grid = np.linspace(param.value - self.ref_scale * param.proposal, param.value + self.ref_scale * param.proposal, size)
                else:
                    raise ParameterPriorError('Provide parameter limited reference distribution or proposal')
            self.grid[param.name] = np.array(grid)
        self.grid = [self.grid[param] for param in self.varied_params.names()]
        self.samples = None
        if self.mpicomm.rank == 0:
            if self.sphere:
                ndim = len(self.grid)
                if any(len(g) % 2 == 0 for g in self.grid):
                    raise ValueError('Number of grid points along each axis must be odd to use sphere option')
                samples = []
                cidx = [len(g) // 2 for g in self.grid]
                for ngrid in range(1, max(self.sphere) + 1):
                    for indices in itertools.product(range(ndim), repeat=ngrid):
                        indices = np.bincount(indices, minlength=ndim)
                        if all(i <= c for c, i in zip(cidx, indices)) and sum(indices) <= min(s for s, i in zip(self.sphere, indices) if i):
                            for signs in itertools.product(*[[-1, 1] if ind else [0] for ind in indices]):
                                samples.append([g[c + s * i] for g, c, s, i in zip(self.grid, cidx, signs, indices)])
                samples.append([g[c] for g, c in zip(self.grid, cidx)])  # finish with the central point
                samples = np.array(samples).T
            else:
                samples = np.meshgrid(*self.grid, indexing='ij')
            self.samples = Samples(samples, params=self.varied_params)

    def run(self, **kwargs):
        if kwargs: self.set_grid(**kwargs)
        self.pipeline.mpicalculate(**(self.samples.to_dict(params=self.varied_params) if self.mpicomm.rank == 0 else {}))

        if self.mpicomm.rank == 0:
            for param in self.pipeline.params.select(fixed=True, derived=False):
                self.samples[param] = np.full(self.samples.shape, param.value, dtype='f8')
            self.samples.update(self.pipeline.derived)
            if self.save_fn is not None:
                self.samples.save(self.save_fn)
        else:
            self.samples = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
