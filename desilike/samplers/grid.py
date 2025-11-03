"""Module implementing a generic grid sampler for low-dimensional problems."""

import numpy as np

from .base import StaticSampler
from desilike.parameter import ParameterPriorError
from desilike.utils import expand_dict


class GridSampler(StaticSampler):
    """A simple grid sampler."""

    def get_points(self, grid=11):
        """Get points on the grid.

        Parameters
        ----------
        grid : dict, int, or numpy.ndarray, optional
            A dictionary giving either the grid size or the grid itself.
            If providing a number, the parameter is evenly within the prior
            limits. Wildcards are supported. If only a single value is provided
            instead of a dictionary, it is applied to all parameters. Default
            is 11.

        Returns
        -------
        numpy.ndarray of shape (n_points, n_dim)
            Grid to be evaluated.
        """
        grid = expand_dict(grid, self.likelihood.varied_params.names())
        for param in self.likelihood.varied_params:
            if not hasattr(grid[param.name], "__len__"):
                if param.limits is None:
                    raise ParameterPriorError(
                        f"Provide a limit for {param.name}.")
                grid[param.name] = np.linspace(
                    param.limits[0], param.limits[1], grid[param.name])
                self.log_info(f"Grid for {param.name} is {grid[param.name]}.")

        grid = [grid[param] for param in self.likelihood.varied_params.names()]
        grid = np.meshgrid(*grid, indexing='ij')
        grid = np.column_stack([arr.ravel() for arr in grid])

        return grid
