"""Module implementing a generic grid sampler for low-dimensional problems."""

import numpy as np

from .base import StaticSampler
from desilike.parameter import ParameterPriorError
from desilike.utils import expand_dict


class GridSampler(StaticSampler):
    """A simple grid sampler."""

    def get_points(self, size=10, include_value=False, grid=None):
        """Get points on the grid.

        Parameters
        ----------
        size : dict or int, optional
            A dictionary giving the grid size along each dimension. Wildcards
            are supported. It can also be a single integer in which case
            it will be applied along all dimenions. Default is 10.

        include_value : bool, optional
            If True, force the default value of each parameter to be included
            in the grid. Default is False.

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
                value = param.value
                if param.ref.is_limited():
                    start, stop = param.ref.limits
                elif param.proposal is not None:
                    start = value - param.proposal
                    stop = value + param.proposal
                else:
                    raise ParameterPriorError(
                        f"Provide proper parameter reference distribution or"
                        f"proposal for {param.name}.")
                n_grid = size[param.name]
                if include_value:
                    n_grid -= 1
                grid[param.name] = np.linspace(start, stop, n_grid)
                if include_value:
                    grid[param.name] = np.sort(np.append(
                        grid[param.name], param.value))
                self.log_info(f"Grid for {param.name} is {grid[param.name]}.")

        grid = [grid[param] for param in self.likelihood.varied_params.names()]
        grid = np.meshgrid(*grid, indexing='ij')
        grid = np.column_stack([arr.ravel() for arr in grid])

        return grid
