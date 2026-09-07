"""Plotting utilities for desilike.

Provides the :func:`plotter` decorator used by observable plot methods.
"""

import os
import logging
from functools import wraps
from pathlib import Path

from . import utils


logger = logging.getLogger('Plotting')


class _FakeFigure:
    """Thin wrapper that makes an axes list look like a Figure."""

    def __init__(self, axes):
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        self.axes = list(axes)


def savefig(filename, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """Save *fig* to *filename*, creating parent directories as needed."""
    from matplotlib import pyplot as plt
    Path(filename).parent.mkdir(exist_ok=True)
    logger.info('Saving figure to {}.'.format(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)
    return fig


def plotter(*args, **kwargs):
    """Decorator that adds ``fn``, ``kw_save``, ``show`` arguments.

    Added keyword arguments
    -----------------------
    fn : str or Path, default=None
        Path where to save the figure.
    kw_save : dict, default=None
        Extra arguments forwarded to :meth:`matplotlib.figure.Figure.savefig`.
    show : bool, default=False
        Call :func:`matplotlib.pyplot.show` after returning.
    """
    use_interactive = False

    def get_wrapper(func):
        @wraps(func)
        def wrapper(*wargs, fn=None, kw_save=None, show=False, fig=None, **wkwargs):
            import matplotlib.pyplot as plt

            if fig is not None:
                if not isinstance(fig, plt.Figure):
                    fig = _FakeFigure(fig)
                elif not fig.axes:
                    fig.add_subplot(111)
                wkwargs['fig'] = fig

            fig = func(*wargs, **wkwargs)
            if fn is not None:
                savefig(fn, fig=fig, **(kw_save or {}))
            if show:
                plt.show()
            return fig
        return wrapper

    if kwargs or not args:
        if args:
            raise ValueError('unexpected positional args: {}'.format(args))
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))
    return get_wrapper(args[0])
