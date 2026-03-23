"""Module implementing plotting routines."""

from functools import wraps

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ModuleNotFoundError:
    MATPLOTLIB_INSTALLED = False
import numpy as np

from . import diagnostics


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
    chains : desilike.Samples or list of desilike.Samples
        List of (or single) :class:``Samples`` instance(s).
    keys : list or None, optional
        Parameters to plot trace for. If ``None``, plot all parameters. Default
        is ``None``.
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

    Raises
    ------
    ValueError
        If the provided figure has less axes than the chains have keys.

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

    if not isinstance(colors, list):
        colors = [colors] * len(chains)

    for key, ax in zip(keys, fig.axes):
        for chain, color in zip(chains, colors):
            ax.plot(chain[key], color=color, **plot_options)
        ax.set_xlabel('Step', fontsize=fontsize)
        ax.set_ylabel(chain.latex.get(key, key))

    return fig


@plotter
def integrated_autocorrelation_time(
        chains, keys=None, colors=None, slices=10, fontsize=None,
        plot_options=None, legend_options=None, fig=None):
    """Plot integrated autocorrelation time as a function of steps.

    Parameters
    ----------
    chains : desilike.Samples or list of desilike.Samples
        List of (or single) :class:``Samples`` instance(s).
    keys : list or None, optional
        Parameters to plot the integrated autocorrelation time for. If
        ``None``, plot all parameters. Default is ``None``.
    colors : str, list, or None, optional
        Dictionary of (or single) color(s) for parameters. Default is ``None``.
    slices : int, optional
        Number of linearly spaced steps for which to compute the integrated
        autocorrelation time. Default is 10.
    fontsize : int or None, optional
        Label sizes. Default is None.
    plot_options : dict or None, optional
        Optional arguments for `matplotlib.axes.Axes.plot`. Default is
        ``None``.
    legend_options : dict or None, optional
        Optional arguments for `matplotlib.axes.Axes.legend`. Default is
        ``None``.
    fig : matplotlib.figure.Figure or None, optional
        Figure to plot on. If ``None``, create a new one. Default is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plot on it.

    Raises
    ------
    ValueError
        If not all chains have the same length.

    """
    if not isinstance(chains, list):
        chains = [chains]

    if not len(np.unique([len(chain) for chain in chains])) == 1:
        raise ValueError('All chains must have the same length.')

    if keys is None:
        keys = chains[0].keys

    if fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        ax = fig.gca()

    if plot_options is None:
        plot_options = {}

    if legend_options is None:
        legend_options = {}

    if not isinstance(colors, dict):
        colors = {key: colors for key in keys}

    n_steps = len(chains[0])
    steps = np.linspace(0, n_steps, slices + 1)[1:].astype(int)
    tau = []
    for steps_max in steps:
        tau.append(diagnostics.integrated_autocorrelation_time(
            [chain[:steps_max] for chain in chains], keys=keys))
    tau = {key: np.array([step[key] for step in tau]) for key in keys}

    for key in keys:
        ax.plot(steps, tau[key], label=chains[0].latex.get(key, key),
                color=colors.get(key, None), **plot_options)
    ax.set_xlabel('Step', fontsize=fontsize)
    ax.set_ylabel(r'$\tau$', fontsize=fontsize)
    ax.legend(fontsize=fontsize, **legend_options)

    return fig


@plotter
def gelman_rubin(
        chains, keys=None, colors=None, n_splits=None, threshold=None,
        slices=100, offset=None, fontsize=None, plot_options=None,
        legend_options=None, fig=None):
    """Plot Gelman-Rubin statistics as a function of steps.

    Parameters
    ----------
    chains : desilike.Samples or list of desilike.Samples
        List of (or single) :class:``Samples`` instance(s).
    keys : list or None, optional
        Parameters to plot the Gelman-Rubin statistic for. If ``None``, plot
        all parameters. Default is ``None``.
    colors : str, list, or None, optional
        Dictionary of (or single) color(s) for parameters. Default is ``None``.
    n_splits : int or None, optional
        Number of splits for each chain. If ``None``, a single chain will be
        split into 2 parts. Splitting allows computation of Gelman-Rubin
        statistics even with one chain. Default is ``None``.
    threshold : float, optional
        If not ``None``, plot horizontal line at this value. Default is
        ``None``.
    slices : int, optional
        Number of linearly spaced steps for which to compute the Gelman-Rubin
        statistic. Default is 100.
    offset : float or None, optional
        Offset to apply to the Gelman-Rubin statistics, typically 0 or -1.
        Default is ``None``.
    fontsize : int or None, optional
        Label sizes. Default is None.
    plot_options : dict or None, optional
        Optional arguments for `matplotlib.axes.Axes.plot`. Default is
        ``None``.
    legend_options : dict or None, optional
        Optional arguments for `matplotlib.axes.Axes.legend`. Default is
        ``None``.
    fig : matplotlib.figure.Figure or None, optional
        Figure to plot on. If ``None``, create a new one. Default is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plot on it.

    Raises
    ------
    ValueError
        If not all chains have the same length.

    """
    if not isinstance(chains, list):
        chains = [chains]

    if not len(np.unique([len(chain) for chain in chains])) == 1:
        raise ValueError('All chains must have the same length.')

    if keys is None:
        keys = chains[0].keys

    if fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        ax = fig.gca()

    if plot_options is None:
        plot_options = {}

    if legend_options is None:
        legend_options = {}

    if not isinstance(colors, dict):
        colors = {key: colors for key in keys}

    if offset is None:
        ylabel = r'$\hat{R}$'
        offset = 0
    else:
        ylabel = rf'$\hat{{R}} {offset:+}$'

    n_steps = len(chains[0])
    steps = np.linspace(0, n_steps, slices + 1)[1:].astype(int)
    gr = []
    for steps_max in steps:
        gr.append(diagnostics.gelman_rubin(
            [chain[:steps_max] for chain in chains], n_splits=n_splits,
            keys=keys))
    gr = {key: np.array([step[key] for step in gr]) for key in keys}

    for key in keys:
        ax.plot(steps, gr[key] + offset, label=chains[0].latex.get(key, key),
                color=colors.get(key, None), **plot_options)
    ax.set_xlabel('Step', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(fontsize=fontsize, **legend_options)

    if threshold is not None:
        ax.axhline(y=threshold, linestyle='--', linewidth=1, color='k')

    return fig
