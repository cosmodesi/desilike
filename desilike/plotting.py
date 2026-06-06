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
    """Decorator that adds ``fn``, ``kw_save``, ``show``, and ``interactive`` arguments.

    Can be used bare (``@plotter``) or with keyword arguments to enable the
    interactive ipywidgets interface (``@plotter(interactive={...})``).

    Added keyword arguments
    -----------------------
    fn : str or Path, default=None
        Path where to save the figure.
    kw_save : dict, default=None
        Extra arguments forwarded to :meth:`matplotlib.figure.Figure.savefig`.
    show : bool, default=False
        Call :func:`matplotlib.pyplot.show` after returning.
    interactive : bool or dict, default=False
        When not False, display an ipywidgets interactive slider interface.
        Pass a dict to set default ``kw_theory`` or ``params`` overrides.
    """
    use_interactive = False

    def get_wrapper(func):
        @wraps(func)
        def wrapper(*wargs, fn=None, kw_save=None, show=False, fig=None, **wkwargs):
            from matplotlib import pyplot as plt

            if fig is not None:
                if not isinstance(fig, plt.Figure):
                    fig = _FakeFigure(fig)
                elif not fig.axes:
                    fig.add_subplot(111)
                wkwargs['fig'] = fig

            interactive = None
            if use_interactive:
                interactive = wkwargs.pop('interactive', None)

            if not interactive:
                fig = func(*wargs, **wkwargs)
                if fn is not None:
                    savefig(fn, fig=fig, **(kw_save or {}))
                if show:
                    plt.show()
                return fig
            else:
                import ipywidgets as widgets
                from IPython.display import display

                if interactive is True:
                    interactive = {}
                interactive = {**use_interactive, **interactive}
                ref_params = interactive.pop('params', None)
                ndelta = interactive.pop('ndelta', 10)

                self = wargs[0]

                def interactive_plot(**params):
                    ifig = None
                    if ref_params is not None:
                        self(**ref_params)
                        ifig = func(*wargs, **{**wkwargs, **interactive, 'fig': None})
                    self(**params)
                    func(*wargs, **{**wkwargs, 'fig': ifig})

                sliders = {}
                for param in self.all_params.select(varied=True, derived=False) + self.all_params.select(solved=True):
                    center = param.value
                    delta = param.delta
                    limits = param.prior.limits
                    if ref_params is not None and param.name in ref_params:
                        center = ref_params[param.name]
                    edges = [center - ndelta * delta[0], center + ndelta * delta[1]]
                    edges = [max(edges[0], limits[0]), min(edges[1], limits[1])]
                    sliders[param.name] = widgets.FloatSlider(
                        min=edges[0], max=edges[1],
                        step=(edges[1] - edges[0]) / 100.,
                        value=center,
                        description=param.latex(inline=True) + ' : ')
                display(widgets.interactive(interactive_plot, **sliders))

        return wrapper

    if kwargs or not args:
        if args:
            raise ValueError('unexpected positional args: {}'.format(args))
        use_interactive = kwargs.pop('interactive', False)
        if use_interactive is True:
            use_interactive = {}
        use_interactive = dict(use_interactive or {})
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))
    return get_wrapper(args[0])
