"""Module implementing a generic grid sampler for low-dimensional problems."""

import numpy as np

from .base import StaticSampler
from desilike.parameter import ParameterPriorError
from desilike.utils import expand_dict


class GridSampler(StaticSampler):
    """A simple grid sampler."""

    def get_points(self, size=11, grid=None):
        """Get points on the grid.

        Parameters
        ----------
        size : dict or int, optional
            A dictionary giving the grid size along each dimension. Wildcards
            are supported. It can also be a single integer in which case
            it will be applied along all dimenions. Default is 11.

        grid : dict or None, optional
            A dictionary giving the values to sample for each parameter. If
            given for a parameter, ``size`` and ``include_default`` are
            ignored for that parameter. Default is None.

        Returns
        -------
        numpy.ndarray of shape (n_points, n_dim)
            Grid to be evaluated.
        """
        size = expand_dict(size, self.likelihood.varied_params.names())
        grid = expand_dict(grid, self.likelihood.varied_params.names())
        for param in self.likelihood.varied_params:
            if grid[param.name] is None:
                if size[param.name] is None:
                    raise ValueError("Neither size nor grid specified for "
                                     f"parameter {param.name}")
                if int(size[param.name]) < 1:
                    raise ValueError(
                        f"Size {int(size)} for parameter {param.name} is not "
                        f"positive.")
                if param.proposal is None:
                    raise ParameterPriorError(
                        f"Provide a proposal for {param.name}.")
                n_grid = size[param.name]
                grid[param.name] = np.linspace(
                    param.value - param.proposal, param.value + param.proposal,
                    n_grid)
                if size == 1:
                    grid[param.name] = param.value
                self.log_info(f"Grid for {param.name} is {grid[param.name]}.")

        grid = [grid[param] for param in self.likelihood.varied_params.names()]
        grid = np.meshgrid(*grid, indexing='ij')
        grid = np.column_stack([arr.ravel() for arr in grid])

        return grid
