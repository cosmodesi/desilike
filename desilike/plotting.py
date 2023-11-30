"""Some plotting utilities."""

import os
import logging

from . import utils


logger = logging.getLogger('Plotting')


class FakeFigure(object):

    def __init__(self, axes):
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        self.axes = list(axes)


def savefig(filename, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """
    Save figure to ``filename``.

    Warning
    -------
    Take care to close figure at the end, ``plt.close(fig)``.

    Parameters
    ----------
    filename : string
        Path where to save figure.

    fig : matplotlib.figure.Figure, default=None
        Figure to save. Defaults to current figure.

    kwargs : dict
        Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    utils.mkdir(os.path.dirname(filename))
    logger.info('Saving figure to {}.'.format(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)
    return fig


def suplabel(axis, label, shift=0, labelpad=5, ha='center', va='center', **kwargs):
    """
    Add global x-coordinate or y-coordinate label to the figure. Similar to matplotlib.suptitle.
    Taken from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots.

    Parameters
    ----------
    axis : str
        'x' or 'y'.

    label : string
        Label string.

    shift : float, optional
        Shift along ``axis``.

    labelpad : float, optional
        Padding perpendicular to ``axis``.

    ha : str, optional
        Label horizontal alignment.

    va : str, optional
        Label vertical alignment.

    kwargs : dict
        Arguments for :func:`matplotlib.pyplot.text`.
    """
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    xmin, ymin = [], []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin, ymin = min(xmin), min(ymin)
    dpi = fig.dpi
    if axis.lower() == 'y':
        rotation = 90.
        x = xmin - float(labelpad) / dpi
        y = 0.5 + shift
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5 + shift
        y = ymin - float(labelpad) / dpi
    else:
        raise ValueError('Unexpected axis {}; chose between x and y'.format(axis))
    plt.text(x, y, label, rotation=rotation, transform=fig.transFigure, ha=ha, va=va, **kwargs)


def plotter(*args, **kwargs):

    from functools import wraps

    use_interactive = False

    def get_wrapper(func):
        """
        Return wrapper for plotting functions, that adds the following (optional) arguments to ``func``:

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        interactive : True or dict, default=False
            If not None, use interactive interface provided by ipywidgets. Interactive can be a dictionary
            with several entries:
                * ref_param : Use to display reference theory
        """
        @wraps(func)
        def wrapper(*args, fn=None, kw_save=None, show=False, fig=None, **kwargs):

            from matplotlib import pyplot as plt

            if fig is not None:

                if not isinstance(fig, plt.Figure):  # create fake figure that has axes
                    fig = FakeFigure(fig)

                elif not fig.axes:
                    fig.add_subplot(111)

                kwargs['fig'] = fig

            interactive = None
            if use_interactive:
                interactive = kwargs.pop('interactive', None)

            if (interactive is None) or (interactive is False):
                fig = func(*args, **kwargs)
                if fn is not None:
                    savefig(fn, **(kw_save or {}))
                if show: plt.show()
                return fig
            else:
                import ipywidgets as widgets
                from IPython.display import display

                if interactive is True:
                    interactive = {}
                interactive = {**use_interactive, **interactive}
                ref_params = interactive.pop('params', None)
                ndelta = interactive.pop('ndelta', 10)

                self = args[0]
                def interactive_plot(**params):
                    fig = None
                    if ref_params is not None:
                        self(**ref_params)
                        fig = func(*args, **{**kwargs, **interactive, 'fig': None})
                    self(**params)
                    func(*args, **{**kwargs, 'fig': fig})

                sliders = {}
                for param in self.all_params.select(varied=True, derived=False) + self.all_params.select(solved=True):
                    center, delta, limits = param.value, param.delta, param.prior.limits
                    if (ref_params is not None) and (param.name in ref_params): center = ref_params[param.name]
                    edges = [center - ndelta * delta[0], center + ndelta * delta[1]]
                    edges = [max(edges[0], limits[0]), min(edges[1], limits[1])]

                    sliders[param.name] = widgets.FloatSlider(min=edges[0], max=edges[1], step=(edges[1] - edges[0]) / 100.,
                                                            value=center, description=param.latex(inline=True) + ' : ')
                w = widgets.interactive(interactive_plot, **sliders)
                display(w)

        return wrapper

    if kwargs or not args:
        if args:
            raise ValueError('unexpected args: {}, {}'.format(args, kwargs))
        use_interactive = kwargs.pop('interactive', False)
        if use_interactive is True:
            use_interactive = {}
        use_interactive = dict(use_interactive or {})
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))

    return get_wrapper(args[0])