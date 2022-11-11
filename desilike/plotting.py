"""Some plotting utilities."""

import os
import logging

from matplotlib import pyplot as plt

from . import utils


logger = logging.getLogger('Plotting')


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
    fig = plt.gcf()
    xmin = []
    ymin = []
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
        raise Exception('Unexpected axis {}; chose between x and y'.format(axis))
    plt.text(x, y, label, rotation=rotation, transform=fig.transFigure, ha=ha, va=va, **kwargs)
