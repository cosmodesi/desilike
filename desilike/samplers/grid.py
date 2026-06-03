"""Module implementing a generic grid sampler for low-dimensional problems."""

import logging

import numpy as np

from .base import StaticSampler
from desilike.parameter import expand_dict

logger = logging.getLogger(__name__)


class GridSampler(StaticSampler):
    """A simple grid sampler."""

    def get_samples(self, grid=11):
        """Get samples on the grid.

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
        numpy.ndarray of shape (n_samples, n_dim)
            Grid to be evaluated.
        """
        grid = expand_dict(grid, self.varied_params.names())
        for param in self.varied_params:
            if not hasattr(grid[param.name], "__len__"):
                limits = param.prior.limits if param.prior is not None else (None, None)
                if limits is None or not (np.isfinite(limits[0]) and np.isfinite(limits[1])):
                    raise ValueError(
                        f"Provide finite limits for {param.name}.")
                grid[param.name] = np.linspace(limits[0], limits[1], grid[param.name])
                logger.info(f"Grid for {param.name} is {grid[param.name]}.")

        grid = [grid[param] for param in self.varied_params.names()]
        grid = np.meshgrid(*grid, indexing='ij')
        grid = np.column_stack([arr.ravel() for arr in grid])

        return grid
