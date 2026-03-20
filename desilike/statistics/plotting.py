"""Module implementing plotting routines."""

from functools import wraps

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ModuleNotFoundError:
    MATPLOTLIB_INSTALLED = False


def plotter(f):
    """Add plotting arguments and check if ``matplotlib`` is installed.

    Parameters
    ----------
    filepath : str, pathlib.Path or None, optional
        If not ``None``, save the figure to that location. Default is ``None``.
    show : bool, optional
        If True, show the figure. Default is ``False``.
    save_options : dict or None, optional
        Additional options passed to the ``savefig`` function of
        ``matplotlib``. Default is ``None``.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.

    """
    @wraps(f)
    def wrapper(*args, filepath=None, show=False, save_options=None, **kwargs):

        if not MATPLOTLIB_INSTALLED:
            raise ImportError("'matplotlib' is required for plotting.")

        fig = f(*args, **kwargs)
        if show:
            plt.show()
        if filepath is not None:
            if save_options is None:
                save_options = {}
            fig.savefig(filepath, **save_options)
        return fig

    return wrapper


@plotter
def trace(chains, keys=None, colors=None, fontsize=None, plot_options=None,
          fig=None):
    """
    Make trace plot as a function of steps, with a panel for each parameter.

    Parameters
    ----------
    chains : desilike.Samples or list of desilike.Chain
        List of (or single) :class:``Chain`` instance(s).
    keys : list or None, optional
        Parameters to plot trace for. If ``None``, plot all parameters.
    colors : str, list, or None, optional
        List of (or single) color(s) for chains. Default is ``None``.
    fontsize : int or None, optional
        Label sizes. Default is None.
    plot_options : dict or None, optional
        Optional arguments for `matplotlib.axes.Axes.plot`. Default is
        ``None``.
    fig : matplotlib.figure.Figure or None, optional
        Figure to plot on. If ``None``, create a new one. Default is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plot on it.

    """
    if not isinstance(chains, list):
        chains = [chains]

    if keys is None:
        keys = chains[0].keys

    if fig is None:
        fig = plt.subplots(nrows=1, ncols=len(keys))[0]

    if len(fig.axes) < len(keys):
        raise ValueError(
            "The provided figure must have at least as many axes as keys "
            "to plot.")

    if plot_options is None:
        plot_options = {}

    if not hasattr(colors, '__len__'):
        colors = [colors] * len(chains)

    for key, ax in zip(keys, fig.axes):
        for chain, color in zip(chains, colors):
            ax.plot(chain[key], color=color, **plot_options)
        ax.set_xlabel('Step', fontsize=fontsize)
        ax.set_ylabel(chain.latex.get(key, key))

    return fig
