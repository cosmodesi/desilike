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

    Raises
    ------
    ValueError
        If the provided figure has less axes than the chains have keys.

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

    Raises
    ------
    ValueError
        If not all chains have the same length.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plot on it.

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

    Raises
    ------
    ValueError
        If not all chains have the same length.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plot on it.

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

    - `triangle_plot documentation  <https://getdist.readthedocs.io/en/latest/\
        plots.html#getdist.plots.GetDistPlotter.triangle_plot>`_

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
        samples, param, posterior=True, interpolate=True, plot_kwargs=None,
        scatter=True, scatter_kwargs=None, ax=None):
    r"""
    Add 1D profile to axes.

    Parameters
    ----------
    samples : desilike.Samples
        :class:`desilike.Samples` instance returned from a profiler.
    param : str
        Parameter to plot profile for.
    posterior : bool, optional
        If ``True``, profile the posterior. Otherwise, profile the
        likelihood. Default is ``True``.
    interpolate : bool, optional
        Whether to plot the interpolated profile. Default is ``True``.
    plot_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.plot`. Default is
        ``None``.
    scatter : bool, optional
        Whether the plot individual points. Default is ``True``.
    scatter_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.scatter`. Default is
        ``None``.
    ax : matplotlib.axes.Axes, default=None
        Axes where to add profile. If ``None``, use ``plt.gca()``. Default
        is ``None``.

    Raises
    ------
    ValueError
        If both or neither of the posterior and likelihood are given.

    """
    if ax is None:
        ax = plt.gca()

    use = np.ones(len(samples), dtype=bool)
    for param_other in samples.params:
        if param_other != param:
            use = use & samples['optimized'][param_other]

    if interpolate:
        interp = samples.profile_interpolator(param, posterior=posterior)
        x = np.linspace(np.amin(samples[param][use]),
                        np.amax(samples[param][use]), 1000)
        kwargs = {} if plot_kwargs is None else plot_kwargs
        ax.plot(x, np.exp(interp(x)), **kwargs)

    if scatter:
        kwargs = {} if scatter_kwargs is None else scatter_kwargs
        ax.scatter(samples[param][use], np.exp(samples[
            'log_posterior' if posterior else 'log_likelihood'][use]),
            **kwargs)

    ax.set_xlabel(samples.latex.get(param, param))
    ax.set_yticks([])
    ax.set_ylim(ymin=0)


def two_dimensional_profile(
        samples, params, posterior=True, levels=None, contour_kwargs=None,
        scatter=True, scatter_kwargs=None, ax=None):
    r"""
    Add 2D profile to axes.

    Parameters
    ----------
    samples : desilike.Samples
        :class:`desilike.Samples` instance returned from a profiler.
    params : tuple of str
        Parameters to plot profile for.
    posterior : bool, optional
        If ``True``, profile the posterior. Otherwise, profile the
        likelihood. Default is ``True``.
    levels : list or None, optional
        Confidence levels to plot, i.e., the values :math:`z` where
        :math:`\log \mathcal{P} = \max \log \mathcal{P} + z`. If ``None``,
        default to [-4.61, -3.00, -1.14] which correspond to the 68%, 95%,
        and 99% credible intervals for a two-dimensional Gaussian. Default is
        ``None``.
    contour_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.contour`. Default is
        ``None``.
    scatter : bool, optional
        Whether the plot individual points. Default is ``True``.
    scatter_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.scatter`. Default is
        ``None``.
    ax : matplotlib.axes.Axes, default=None
        Axes where to add profile. If ``None``, use ``plt.gca()``. Default
        is ``None``.

    Raises
    ------
    ValueError
        If both or neither of the posterior and likelihood are given or an
        incorrect number of parameters is given.

    """
    if levels is None:
        levels = [-4.61, -3.00, -1.14]

    if ax is None:
        ax = plt.gca()

    use = np.ones(len(samples), dtype=bool)
    for param in samples.params:
        if param in params:
            use = use & ~samples['optimized'][param]
        else:
            use = use & samples['flag_optimized'][param]

    if len(levels) > 0:
        interp = samples.profile_interpolator(params, posterior=posterior)
        x = np.linspace(np.amin(samples[params[0]][use]),
                        np.amax(samples[params[0]][use]), 1000)
        y = np.linspace(np.amin(samples[params[1]][use]),
                        np.amax(samples[params[1]][use]), 1000)
        x, y = np.meshgrid(x, y)
        z = interp(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)
        z = z - np.amax(z)
        kwargs = ({'colors': 'black'} if contour_kwargs is None else
                  contour_kwargs)
        ax.contour(x, y, z, levels=levels, **kwargs)

    if scatter:
        kwargs = {} if scatter_kwargs is None else scatter_kwargs
        ax.scatter(samples[params[0]][use], samples[params[1]][use], **kwargs)

    ax.set_xlabel(samples.latex.get(params[0], params[0]))
    ax.set_ylabel(samples.latex.get(params[1], params[1]))


@plotter
def triangle_profile(
        samples, params=None, posterior=True, interpolate=True,
        plot_kwargs=None, levels=None, contour_kwargs=None, scatter=True,
        scatter_kwargs=None, threshold=-5, fig=None):
    r"""Create a triangle profile plot.

    Parameters
    ----------
    samples : desilike.Samples
        Samples for which to plot the profile for.
    params : list or None, optional
        Parameters to plot profile for. If ``None``, plot all parameters.
        Default is ``None``.
    posterior : bool, optional
        If ``True``, profile the posterior. Otherwise, profile the
        likelihood. Default is ``True``.
    interpolate : bool, optional
        Whether to interpolate and plot the one-dimensional profiles. Default
        is ``True``.
    plot_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.plot`. Default is
        ``None``.
    levels : list or None, optional
        Confidence levels to plot, i.e., the values :math:`z` where
        :math:`\log \mathcal{P} = \max \log \mathcal{P} + z`. If ``None``,
        default to [-4.61, -3.00, -1.14] which correspond to the 68%, 95%,
        and 99% credible intervals for a two-dimensional Gaussian. Default is
        ``None``.
    contour_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.contour`. Default is
        ``None``.
    scatter : bool, optional
        Whether the plot individual points. Default is ``True``.
    scatter_kwargs : dict or None, optional
        Optional arguments for :meth:`matplotlib.axes.Axes.scatter`. Default is
        ``None``.
    threshold : float, optional
        Limit the ranges for each parameter to the corresponding intervals for
        this threshold. Default is -5.
    fig : matplotlib.figure.Figure or None, optional
        Figure to plot on. If ``None``, create a new one. Default is ``None``.

    """
    if levels is None:
        levels = [-4.61, -3.00, -1.14]

    if params is None:
        params = samples.params

    if fig is not None:
        axs = fig.axes
        gs = axs.get_gridspec()
        if gs.ncols != len(params) or gs.nrows != len(params):
            msg = (f"The provided figure must have exactly {len(params)} rows "
                   f"and {len(params)} columns.")
            raise ValueError(msg)
    else:
        fig, axs = plt.subplots(nrows=len(params), ncols=len(params),
                                sharex='col')

    for i, param in enumerate(params):
        one_dimensional_profile(
            samples, param, posterior=posterior, interpolate=interpolate,
            plot_kwargs=plot_kwargs, scatter=scatter,
            scatter_kwargs=scatter_kwargs, ax=axs[i, i])

    for i in range(len(params)):
        for k in range(i):
            two_dimensional_profile(
                samples, (params[k], params[i]), posterior=posterior,
                levels=levels, contour_kwargs=contour_kwargs, scatter=scatter,
                scatter_kwargs=scatter_kwargs, ax=axs[i, k])

    for i in range(len(params)):
        for k in range(i + 1, len(params)):
            axs[i, k].axis('off')

    # Set x-ranges.
    for i, param in enumerate(params):
        bounds = samples.interval(param, threshold, posterior=posterior)
        x_min = bounds[0][0]
        x_max = bounds[-1][1]
        # Sometimes, the grid is too small to encompas the proper bounds.
        if np.isnan(x_min) or np.isnan(x_max):
            x = samples.profile_interpolator(param, posterior=posterior).x
            x_min = np.amin(x) if np.isnan(x_min) else x_min
            x_max = np.amax(x) if np.isnan(x_max) else x_max
        axs[i, i].set_xlim(x_min, x_max)
        for k in range(i):
            axs[i, k].set_ylim(x_min, x_max)

    fig.subplots_adjust(hspace=0, wspace=0)

    return fig
