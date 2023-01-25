import itertools

import numpy as np

from desilike.parameter import ParameterPriorError, Samples
from desilike.utils import BaseClass, expand_dict
from .base import RegisteredSampler


class GridSampler(BaseClass, metaclass=RegisteredSampler):

    """Evalue calculator on a grid."""
    name = 'grid'

    def __init__(self, calculator, mpicomm=None, save_fn=None, **kwargs):
        r"""
        Initialize grid.

        Parameters
        ----------
        calculator : BaseCalculator
            Input calculator.

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``calculator``'s :attr:`BaseCalculator.mpicomm`.

        save_fn : str, Path, default=None
            If not ``None``, save samples to this location.

        size : int, dict, default=1
            A dictionary mapping parameter name to grid size for this parameter.
            Can be a single value, used for all parameters.

        ref_scale : float, default=1.
            Parameter grid ranges are inferred from limits of reference distribution if bounded (and has no scale),
            else :attr:`Parameter.proposal`.
            These values are then scaled by ``ref_scale`` (< 1. means smaller ranges).

        grid : array, dict, default=None
            A dictionary mapping parameter name (including wildcard) to values.
            If provided, ``size`` and ``ref_scale`` are ignored.
        """
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

    def set_grid(self, size=1, ref_scale=1., grid=None):
        self.ref_scale = float(ref_scale)
        self.grid = expand_dict(grid, self.varied_params.names())
        self.size = expand_dict(size, self.varied_params.names())
        for param in self.varied_params:
            grid, size = self.grid[param.name], self.size[param.name]
            if grid is None:
                if size is None:
                    raise ValueError('size (and grid) not specified for parameter {}'.format(param))
                size = int(size)
                if size < 1:
                    raise ValueError('size is {} < 1 for parameter {}'.format(size, param))
                center = param.value
                limits = np.array(param.ref.limits)
                if not limits[0] <= param.value <= limits[1]:
                    raise ParameterPriorError('Parameter {} value {} is not in reference limits {}'.format(param, param.value, param.ref.limits))
                if size == 1:
                    grid = [center]
                else:
                    if param.ref.is_limited() and not hasattr(param.ref, 'scale'):
                        edges = self.ref_scale * (limits - center) + center
                    elif param.proposal:
                        edges = self.ref_scale * np.array([-param.proposal, param.proposal]) + center
                    else:
                        raise ParameterPriorError('Provide proper parameter reference distribution or proposal for {}'.format(param))
                    low, high = np.linspace(edges[0], center, size // 2 + 1), np.linspace(center, edges[1], size // 2 + 1)
                    if size % 2:
                        grid = np.concatenate([low, high[1:]])
                    else:
                        grid = np.concatenate([low[:-1], high[1:]])
                if self.mpicomm.rank == 0:
                    self.log_info('{} grid is {}.'.format(param, grid))
            else:
                grid = np.sort(np.ravel(grid))
            self.grid[param.name] = grid
        self.grid = [self.grid[param] for param in self.varied_params.names()]
        del self.size
        self.samples = None
        if self.mpicomm.rank == 0:
            samples = np.meshgrid(*self.grid, indexing='ij')
            self.samples = Samples(samples, params=self.varied_params)

    def run(self, **kwargs):
        """
        Run calculator evaluation on the grid.
        A new grid can be set by providing arguments 'size', 'ref_scale', 'grid' or 'sphere', see :meth:`__init__`.
        """
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
        return self.samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
