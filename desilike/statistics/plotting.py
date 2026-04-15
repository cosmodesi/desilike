"""Module implementing plotting routines."""

from functools import wraps
import warnings

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ModuleNotFoundError:
    MATPLOTLIB_INSTALLED = False
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

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
        List of (or single) :class:``desilike.Samples`` instance(s).
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
        List of (or single) :class:``desilike.Samples`` instance(s).
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
        List of (or single) :class:``desilike.Samples`` instance(s).
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


def triangle_posterior(samples, params=None, **kwargs):
    """Create a triangle posterior plot using ``getdist``.

    .. rubric:: References
    - https://getdist.readthedocs.io/en/latest/plots.html#getdist.plots.GetDistPlotter.triangle_plot

    Parameters
    ----------
    samples : desilike.Samples or list of desilike.Samples
        List of (or single) :class:`desilike.Samples` instance(s).
    params : list or None, optional
        Parameters to plot posterior for. If ``None``, plot all parameters.
        Default is ``None``.
    **kwargs
        Optional parameters for
        :meth:`getdist.plots.GetDistPlotter.triangle_plot`.

    Raises
    ------
    ImportError
        If ``getdist`` is not installed.

    """
    try:
        from getdist import plots
    except ImportError:
        raise ImportError("'getdist' is required for triangle plots.")

    if not isinstance(samples, list):
        samples = [samples]

    samples = [sample.getdist(params) for sample in samples]
    plots.get_subplot_plotter().triangle_plot(samples, **kwargs)


def one_dimensional_profile(
        samples, param, ax=None, plot=True, plot_kwargs=None, scatter=False,
        scatter_kwargs=None):
    r"""
    Add 1D profile to axes.

    Parameters
    ----------
    samples : desilike.Samples
        :class:`desilike.Samples` instance returned from a profiler.
    param : str
        Parameter to plot profile for.
    ax : matplotlib.axes.Axes, default=None
        Axes where to add profile. If ``None``, use ``plt.gca()``. Default
        is ``None``.
    plot : bool, optional
        Whether to interpolate and plot the profile. Default is ``True``.
    plot_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.plot`. Default is
        ``None``.
    scatter : bool, optional
        Whether the plot individual points. Default is ``False``.
    scatter_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.scatter`. Default is
        ``None``.

    Raises
    ------
    ValueError
        If both or neither of the posterior and likelihood are given.

    """
    if ax is None:
        ax = plt.gca()

    if plot_kwargs is None:
        plot_kwargs = {}

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if 'log_posterior' in samples.keys:
        if 'log_likelihood' in samples.keys:
            raise ValueError('Samples have both posterior and likelihood.')
        key = 'log_posterior'
    elif 'log_likelihood' in samples.keys:
        key = 'log_likelihood'
    else:
        raise ValueError('Samples have neither posterior nor likelihood.')

    use = np.isin(samples['fixed'], [param, ''])
    if np.sum(use) < 4:
        warnings.warn(f"Not enough points to plot profile for {param}.")
        return
    x = samples[param][use]
    y = samples[key][use]
    y = np.exp(y - np.amax(y))
    y = y[np.argsort(x)]
    x = np.sort(x)

    if scatter:
        ax.scatter(x, y, **scatter_kwargs)

    if plot and len(x) > 3:
        x_plot = np.linspace(np.amin(x), np.amax(x), 300)
        y_plot = np.exp(interp1d(x, np.log(y), kind='cubic')(x_plot))
        ax.plot(x_plot, y_plot, **scatter_kwargs)

    ax.set_xlabel(samples.latex.get(param, param))
    ax.set_yticks([])
    ax.set_ylim(ymin=0)


def two_dimensional_profile(
        samples, params, ax=None, levels=[-4.61, -3.00, -1.14],
        contour_kwargs=None, scatter=False, scatter_kwargs=None):
    r"""
    Add 2D profile to axes.

    Parameters
    ----------
    samples : desilike.Samples
        :class:`desilike.Samples` instance returned from a profiler.
    params : tuple of str
        Parameters to plot profile for.
    ax : matplotlib.axes.Axes, default=None
        Axes where to add profile. If ``None``, use ``plt.gca()``. Default
        is ``None``.
    levels : list, optional
        Confidence levels to plot, i.e., the values :math:`z` where
        :math:`\log \mathcal{P} = \max \log \mathcal{P} + z`. Default is
        [-4.61, -3.00, -1.14] which correspond to the 68%, 95%, and 99%
        credible intervals for a two-dimensional Gaussian.
    contour_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.contour`. Default is
        ``None``.
    scatter : bool, optional
        Whether the plot individual points. Default is ``False``.
    scatter_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.scatter`. Default is
        ``None``.

    Raises
    ------
    ValueError
        If both or neither of the posterior and likelihood are given or an
        incorrect number of parameters is given.

    """
    if ax is None:
        ax = plt.gca()

    if contour_kwargs is None:
        contour_kwargs = dict(colors='black')

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if 'log_posterior' in samples.keys:
        if 'log_likelihood' in samples.keys:
            raise ValueError('Samples have both posterior and likelihood.')
        key = 'log_posterior'
    elif 'log_likelihood' in samples.keys:
        key = 'log_likelihood'
    else:
        raise ValueError('Samples have neither posterior nor likelihood.')

    if len(params) != 2:
        raise ValueError(f"Please specify 2 parameters. Got {len(params)}.")

    use = samples['fixed'] == '/'.join(np.sort(params))
    x = samples[params[0]][use]
    y = samples[params[1]][use]
    z = samples[key][use]

    # Determine which points span a 2D grid.
    use = np.ones(len(x), dtype=bool)
    while np.any(use):
        x_unique, x_counts = np.unique(x[use], return_counts=True)
        y_unique, y_counts = np.unique(y[use], return_counts=True)
        xy = np.column_stack((x[use], y[use]))
        if (len(np.unique(xy, axis=0)) == len(x_unique) * len(y_unique)):
            break
        if np.amin(x_counts) < np.amin(y_counts):
            use = use & (x != x_unique[np.argmin(x_counts)])
        else:
            use = use & (y != y_unique[np.argmin(y_counts)])

    if not np.any(use):
        warnings.warn(
            f"Could not determine a grid for '{params[0]}' vs. '{params[1]}'.")
        return

    x = x[use]
    y = y[use]
    z = z[use]

    # Sort values.
    argsort = np.argsort(y, stable=True)
    x, y, z = x[argsort], y[argsort], z[argsort]
    argsort = np.argsort(x, stable=True)
    x, y, z = x[argsort], y[argsort], z[argsort]

    if scatter:
        ax.scatter(x, y, **scatter_kwargs)

    if len(levels) > 0:
        x, y = np.unique(x), np.unique(y)
        interp = RegularGridInterpolator(
            (x, y), z.reshape(len(x), len(y)), method='cubic')

        x_plot = np.linspace(np.amin(x), np.amax(x), 300)
        y_plot = np.linspace(np.amin(y), np.amax(y), 300)
        x_plot, y_plot = np.meshgrid(x_plot, y_plot)
        z_plot = interp(np.column_stack([x_plot.ravel(), y_plot.ravel()])
                        ).reshape(x_plot.shape)
        z_plot = z_plot - np.amax(z_plot)
        ax.contour(x_plot, y_plot, z_plot, levels=levels, **contour_kwargs)

    ax.set_xlabel(samples.latex.get(params[0], params[0]))
    ax.set_ylabel(samples.latex.get(params[1], params[1]))


@plotter
def triangle_profile(
        samples, params=None, plot=True, plot_kwargs=None,
        levels=[-4.61, -3.00, -1.14], contour_kwargs=None, scatter=False,
        scatter_kwargs=None, fig=None):
    r"""Create a triangle profile plot.

    Parameters
    ----------
    samples : desilike.Samples
        Samples for which to plot the profile for.
    params : list or None, optional
        Parameters to plot profile for. If ``None``, plot all parameters.
        Default is ``None``.
    plot : bool, optional
        Whether to interpolate and plot the one-dimensional profiles. Default
        is ``True``.
    plot_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.plot`. Default is
        ``None``.
    levels : list, optional
        Confidence levels to plot for the two-dimensional profiles, i.e., the
        values :math:`z` where
        :math:`\log \mathcal{P} = \max \log \mathcal{P} + z`. Default is
        [-4.61, -3.00, -1.14] which correspond to the 68%, 95%, and 99%
        credible intervals for a two-dimensional Gaussian.
    contour_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.contour`. Default is
        ``None``.
    scatter : bool, optional
        Whether the plot individual points. Default is ``False``.
    scatter_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.scatter`. Default is
        ``None``.
    fig : matplotlib.figure.Figure or None, optional
        Figure to plot on. If ``None``, create a new one. Default is ``None``.

    """
    if params is None:
        params = samples.params

    if fig is not None:
        axs = fig.axes
        gs = axs.get_gridspec()
        if gs.ncols != len(params) or gs.nrows != len(params):
            raise ValueError(
                f"The provided figure must have exactly {len(params)} rows "
                f"and {len(params)} columns.")
    else:
        fig, axs = plt.subplots(nrows=len(params), ncols=len(params),
                                sharex='col')

    for i, param in enumerate(params):
        one_dimensional_profile(
            samples, param, ax=axs[i, i], plot=plot, plot_kwargs=plot_kwargs,
            scatter=scatter, scatter_kwargs=scatter_kwargs)

    for i in range(len(params)):
        for k in range(i):
            two_dimensional_profile(
                samples, (params[k], params[i]), ax=axs[i, k], levels=levels,
                contour_kwargs=contour_kwargs, scatter=scatter,
                scatter_kwargs=scatter_kwargs)

    for i in range(len(params)):
        for k in range(i + 1, len(params)):
            axs[i, k].axis('off')

    # Synchronize the ranges across rows.
    for i in range(len(params)):
        xmin, xmax = axs[i, i].get_xlim()
        for k in range(i):
            xmin = min(xmin, axs[i, k].get_ylim()[0])
            xmax = max(xmax, axs[i, k].get_ylim()[1])
        axs[i, i].set_xlim(xmin, xmax)
        for k in range(i):
            axs[i, k].set_ylim(xmin, xmax)

    fig.subplots_adjust(hspace=0, wspace=0)

    return fig
