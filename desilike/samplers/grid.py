"""Grid sampling kernel."""

import logging

import numpy as np

from .base import StaticKernel
from ..parameter import expand_dict

logger = logging.getLogger(__name__)


class Grid(StaticKernel):
    """Evaluate the posterior on a regular grid.

    Parameters
    ----------
    None — all options are passed at run time via :meth:`get_samples`.
    """

    logger = logging.getLogger('Grid')

    def get_samples(self, varied_params, grid=11, **kwargs):
        """Return grid points in original parameter space.

        Parameters
        ----------
        varied_params : VariableCollection
        grid : dict, int, or numpy.ndarray, optional
            Per-parameter grid specification.  If a scalar, that many evenly
            spaced points are placed within the prior limits for every parameter.
            Wildcards are supported in dict form.  Default is 11.

        Returns
        -------
        numpy.ndarray, shape ``(n_samples, ndim)``
        """
        grid = expand_dict(grid, varied_params.names())
        for param in varied_params:
            if not hasattr(grid[param.name], '__len__'):
                limits = param.prior.limits if param.prior is not None else (None, None)
                if limits is None or not (np.isfinite(limits[0]) and np.isfinite(limits[1])):
                    raise ValueError(f'Provide finite limits for {param.name}.')
                grid[param.name] = np.linspace(limits[0], limits[1], grid[param.name])
                self.logger.info('Grid for %s is %s.', param.name, grid[param.name])

        grid_arrays = [grid[param] for param in varied_params.names()]
        mesh = np.meshgrid(*grid_arrays, indexing='ij')
        return np.column_stack([arr.ravel() for arr in mesh])
