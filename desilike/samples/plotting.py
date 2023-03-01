import numpy as np
from matplotlib import gridspec, transforms

from desilike import plotting
from desilike.plotting import *
from desilike.parameter import is_parameter_sequence
from . import diagnostics, utils


def _make_list(obj, length=None, default=None):
    """
    Return list from ``obj``.

    Parameters
    ----------
    obj : object, tuple, list, array
        If tuple, list or array, cast to list.
        Else return list of ``obj`` with length ``length``.

    length : int, default=1
        Length of list to return.

    Returns
    -------
    toret : list
    """
    if is_parameter_sequence(obj):
        obj = list(obj)
        if length is not None:
            obj += [default] * (length - len(obj))
    else:
        obj = [obj]
        if length is not None:
            obj *= length
    return obj


def _get_default_chain_params(chains, params=None, varied=True, derived=False, **kwargs):
    chains = _make_list(chains)
    if params is not None:
        params = _make_list(params)
        return sum(chain.params(name=[str(param) for param in params]) for chain in chains)
    list_params = [chain.params(varied=varied, derived=derived, **kwargs) for chain in chains]
    from desilike.parameter import ParameterCollection
    return ParameterCollection([params for params in list_params[0] if all(params in lparams for lparams in list_params[1:])])


def _get_default_profiles_params(profiles, params=None, of='bestfit', varied=True, derived=False, **kwargs):
    profiles = _make_list(profiles)
    if params is not None:
        params = _make_list(params)
        list_params = [profile.get(of).params(name=[str(param) for param in params]) for profile in profiles]
    else:
        list_params = [profile.get(of).params(varied=varied, derived=derived, **kwargs) for profile in profiles]
    return [params for params in list_params[0] if all(params in lparams for lparams in list_params[1:])]


@plotting.plotter
def plot_trace(chains, params=None, figsize=None, colors=None, labelsize=None, kw_plot=None):
    """
    Make trace plot as a function of steps, with a panel for each parameter.

    Parameters
    ----------
    chains : list, default=None
        List of (or single) :class:`Chain` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot trace for.
        Defaults to varied and not derived parameters.

    figsize : float, tuple, default=None
        Figure size.

    colors : str, list
        List of (or single) color(s) for chains.

    labelsize : int, default=None
        Label sizes.

    kw_plot : dict, default=None
        Optional arguments for :meth:`matplotlib.axes.Axes.plot`.
        Defaults to ``{'alpha': 0.2}``.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    lax : array
        Array of axes.
    """
    from matplotlib import pyplot as plt
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params)
    nparams = len(params)
    colors = _make_list(colors, length=len(chains), default=None)
    kw_plot = kw_plot or {'alpha': 0.2}

    steps = 1 + np.arange(max(chain.size for chain in chains))
    figsize = figsize or (8, 1.5 * nparams)
    fig, lax = plt.subplots(nparams, sharex=True, sharey=False, figsize=figsize, squeeze=False)
    lax = lax.ravel()

    for ax, param in zip(lax, params):
        ax.grid(True)
        ax.set_ylabel(chains[0][param].param.latex(inline=True), fontsize=labelsize)
        ax.set_xlim(steps[0], steps[-1])
        for ichain, chain in enumerate(chains):
            tmp = chain[param].ravel()
            ax.plot(steps[:len(tmp)], tmp, color=colors[ichain], **kw_plot)

    lax[-1].set_xlabel('step', fontsize=labelsize)
    return lax


@plotting.plotter
def plot_gelman_rubin(chains, params=None, multivariate=False, threshold=None, slices=None, labelsize=None, ax=None, **kwargs):
    """
    Plot Gelman-Rubin statistics as a function of steps.

    Parameters
    ----------
    chains : list, default=None
        List of (or single) :class:`Chain` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot Gelman-Rubin statistics for.
        Defaults to varied and not derived parameters.

    multivariate : bool, default=False
        If ``True``, add line for maximum of eigen value of Gelman-Rubin matrix.
        See :func:`diagnostics.gelman_rubin`.

    threshold : float, default=None
        If not ``None``, plot horizontal line at this value.

    slices : list, array
        List of increasing number of steps to include in calculation of Gelman-Rubin statistics.
        Defaults to ``np.arange(100, nsteps, 500)``, where ``nsteps`` is the minimum size of input ``chains``:
        Gelman-Rubin statistics is then plotted for chain slices (0, 100), (0, 600), ...

    labelsize : int, default=None
        Label sizes.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot Gelman-Rubin statistics. If ``None``, take current axes.

    **kwargs : dict
        Optional arguments for :func:`diagnostics.gelman_rubin` ('nsplits', 'check_valid').

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from matplotlib import pyplot as plt
    params = _get_default_chain_params(chains, params=params)
    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)
    gr_multi = []
    gr = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        if multivariate: gr_multi.append(diagnostics.gelman_rubin(chains_sliced, params, method='eigen', **kwargs).max())
        for param in gr: gr[param].append(diagnostics.gelman_rubin(chains_sliced, param, method='diag', **kwargs))
    for param in gr: gr[param] = np.asarray(gr[param])

    fig = None
    if ax is None: fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ax.set_ylabel(r'$\hat{R}$', fontsize=labelsize)

    if multivariate: ax.plot(slices, gr_multi, label='multi', linestyle='-', linewidth=1, color='k')
    for param in params:
        ax.plot(slices, gr[param], label=chains[0][param].param.latex(inline=True), linestyle='--', linewidth=1)
    if threshold is not None: ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()

    return ax


@plotting.plotter
def plot_geweke(chains, params=None, threshold=None, slices=None, labelsize=None, ax=None, **kwargs):
    """
    Plot Geweke statistics.

    Parameters
    ----------
    chains : list, default=None
        List of (or single) :class:`Chain` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot Geweke statistics for.
        Defaults to varied and not derived parameters.

    threshold : float, default=None
        If not ``None``, plot horizontal line at this value.

    slices : list, array
        List of increasing number of steps to include in calculation of Geweke statistics.
        Defaults to ``np.arange(100, nsteps, 500)``, where ``nsteps`` is the minimum size of input ``chains``:
        Geweke statistics is then plotted for chain slices (0, 100), (0, 600), ...

    labelsize : int, default=None
        Label sizes.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot Geweke statistics. If ``None``, take current axes.

    **kwargs : dict
        Optional arguments for :func:`diagnostics.geweke` ('first', 'last').

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from matplotlib import pyplot as plt
    params = _get_default_chain_params(chains, params=params)
    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)
    geweke = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        for param in geweke: geweke[param].append(diagnostics.geweke(chains_sliced, param, **kwargs))
    for param in geweke: geweke[param] = np.asarray(geweke[param]).mean(axis=-1)

    fig = None
    if ax is None: fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ax.set_ylabel(r'geweke', fontsize=labelsize)

    for param in params:
        ax.plot(slices, geweke[param], label=chains[0][param].param.latex(inline=True), linestyle='-', linewidth=1)
    if threshold is not None: ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()

    return ax


@plotting.plotter
def plot_autocorrelation_time(chains, params=None, threshold=50, slices=None, labelsize=None, ax=None):
    r"""
    Plot integrated autocorrelation time.

    Parameters
    ----------
    chains : list, default=None
        List of (or single) :class:`Chain` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot autocorrelation time for.
        Defaults to varied and not derived parameters.

    threshold : float, default=50
        If not ``None``, plot :math:`y = x/\mathrm{threshold}` line.
        Integrated autocorrelation time estimation can be considered reliable when falling under this line.

    slices : list, array
        List of increasing number of steps to include in calculation of autocorrelation time.
        Defaults to ``np.arange(100, nsteps, 500)``, where ``nsteps`` is the minimum size of input ``chains``:
        Autocorrelation time is then plotted for chain slices (0, 100), (0, 600), ...

    labelsize : int, default=None
        Label sizes.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot autocorrelation time. If ``None``, take current axes.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from matplotlib import pyplot as plt
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params)
    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)
    autocorr = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        for param in autocorr:
            tmp = diagnostics.integrated_autocorrelation_time(chains_sliced, param)
            autocorr[param].append(tmp)
    for param in autocorr: autocorr[param] = np.asarray(autocorr[param])

    fig = None
    if ax is None: fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel('step $N$', fontsize=labelsize)
    ax.set_ylabel('$\tau$', fontsize=labelsize)

    for param in params:
        ax.plot(slices, autocorr[param], label=chains[0][param].param.latex(inline=True), linestyle='--', linewidth=1)
    if threshold is not None:
        ax.plot(slices, slices * 1. / threshold, label='$N/{:d}$'.format(threshold), linestyle='--', linewidth=1, color='k')
    ax.legend()

    return ax


@plotting.plotter
def plot_triangle(chains, params=None, labels=None, g=None, **kwargs):
    """
    Triangle plot.

    Note
    ----
    *GetDist* package is required.

    Parameters
    ----------
    chains : list, default=None
        List of (or single) :class:`Chain` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot distribution for.
        Defaults to varied and not derived parameters.

    labels : str, list, default=None
        Name for  *GetDist* to use for input chains.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    g : getdist subplot_plotter()
        can be created with `g = gdplt.get_subplot_plotter()` and can be modified with g.settings

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    lax : array
        Array of axes.
    """
    from getdist import plots
    if g is None: g = plots.get_subplot_plotter()
    chains = _make_list(chains)
    labels = _make_list(labels, length=len(chains), default=None)
    params = _get_default_chain_params(chains, params=params)
    chains = [chain.to_getdist(label=label, params=chain.params(name=params.names())) for chain, label in zip(chains, labels)]
    g.triangle_plot(chains, [str(param) for param in params], **kwargs)
    return g


@plotting.plotter
def plot_aligned(profiles, param, ids=None, labels=None, colors=None, truth=None, error='error',
                 labelsize=None, ticksize=None, kw_scatter=None, yband=None, kw_mean=None, kw_truth=None, kw_yband=None,
                 kw_legend=None, ax=None):
    """
    Plot best fit estimates for single parameter.

    Parameters
    ----------
    profiles : list
        List of (or single) :class:`Profiles` instance(s).

    param : Parameter, str
        Parameter name.

    ids : list, str, default=None
        Label(s) for input profiles.

    labels : list, str, default=None
        Label(s) for best fits within each :class:`Profiles` instance.

    colors : list, str, default=None
        Color(s) for best fits within each :class:`Profiles` instance.

    truth : float, bool, default=None
        Plot this truth / reference value for parameter.
        If ``True``, take :attr:`Parameter.value`.

    error : str, default='error'
        What to take as error:
        - 'error' for parabolic error
        - 'interval' for lower and upper errors corresponding to :math:`\Delta \chi^{2} = 1`.

    labelsize : int, default=None
        Label sizes.

    ticksize : int, default=None
        Tick sizes.

    kw_scatter : dict, default=None
        Optional arguments for :meth:`matplotlib.axes.Axes.scatter`.
        Defaults to ``{'marker': 'o'}``.

    yband : float, tuple, default=None
        If not ``None``, plot horizontal band.
        If tuple and last element set to ``'abs'``,
        absolute lower and upper y-coordinates of band;
        lower and upper fraction around truth.
        If float, fraction around truth.

    kw_mean : dict, default=None
        If ``None``, no mean is plotted.
        Else, optional arguments for :meth:`matplotlib.axes.Axes.errorbar`.
        Defaults to ``{'marker': 'o'}``.

    kw_truth : dict, default=None
        If ``None``, and ``truth`` not provided, no truth is plotted.
        Else, optional arguments for :meth:`matplotlib.axes.Axes.axhline`.
        Defaults to ``{'color': 'k', 'linestyle': ':', 'linewidth': 2}``.

    kw_yband : dict, default=None
        Optional arguments for :meth:`matplotlib.axes.Axes.axhspan`.

    kw_legend : dict, default=None
        Optional arguments for :meth:`matplotlib.axes.Axes.legend`.

    ax : matplotlib.axes.Axes, default=None
        Axes where to plot profiles. If ``None``, takes current axes.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from matplotlib import pyplot as plt
    profiles = _make_list(profiles)
    if truth is True or (truth is None and kw_truth is not None):
        truth = profiles[0].bestfit[param].param.value
    kw_truth = dict(kw_truth if kw_truth is not None else {'color': 'k', 'linestyle': ':', 'linewidth': 2})
    maxpoints = max(map(lambda prof: len(prof.bestfit), profiles))
    ids = _make_list(ids, length=len(profiles), default=None)
    labels = _make_list(labels, length=maxpoints, default=None)
    colors = _make_list(colors, length=maxpoints, default=None)
    add_legend = any(label is not None for label in labels)
    add_mean = kw_mean is not None
    if add_mean:
        kw_mean = kw_mean if isinstance(kw_mean, dict) else {'marker': 'o'}
    kw_scatter = dict(kw_scatter or {'marker': 'o'})
    kw_yband = dict(kw_yband or {})
    kw_legend = dict(kw_legend or {})

    xmain = np.arange(len(profiles))
    xaux = np.linspace(-0.15, 0.15, maxpoints)
    fig = None
    if ax is None: fig, ax = plt.subplots()
    for iprof, prof in enumerate(profiles):
        if param not in prof.bestfit: continue
        ibest = prof.bestfit.logposterior.argmax()
        for ipoint, point in enumerate(prof.bestfit[param]):
            yerr = None
            if error:
                try:
                    yerr = prof.get(error)[param]
                except KeyError:
                    yerr = None
                else:
                    if len(yerr) == 1:
                        yerr = yerr[0]  # only for best fit
                    else:
                        yerr = yerr[ibest]
            label = labels[ipoint] if iprof == 0 else None
            ax.errorbar(xmain[iprof] + xaux[ipoint], point, yerr=yerr, color=colors[ipoint], label=label, linestyle='none', **kw_scatter)
        if add_mean:
            ax.errorbar(xmain[iprof], prof.bestfit[param].mean(), yerr=prof.bestfit[param].std(ddof=1), linestyle='none', **kw_mean)
    if truth is not None:
        ax.axhline(y=truth, xmin=0., xmax=1., **kw_truth)
    if yband is not None:
        if np.ndim(yband) == 0:
            yband = (yband, yband)
        if yband[-1] == 'abs':
            low, up = yband[0], yband[1]
        else:
            if truth is None:
                raise ValueError('Plotting relative band requires truth value.')
            low, up = truth * (1 - yband[0]), truth * (1 + yband[1])
        ax.axhspan(low, up, **kw_yband)

    ax.set_xticks(xmain)
    ax.set_xticklabels(ids, rotation=40, ha='right', fontsize=ticksize)
    ax.grid(True, axis='y')
    ax.set_ylabel(profiles[0].bestfit[param].param.latex(inline=True), fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    if add_legend: ax.legend(**{**{'ncol': maxpoints}, **kw_legend})
    return ax


@plotting.plotter
def plot_aligned_stacked(profiles, params=None, ids=None, labels=None, truths=None, ybands=None, ylimits=None, figsize=None, **kwargs):
    """
    Plot best fits, with a panel for each parameter.

    Parameters
    ----------
    profiles : list
        List of (or single) :class:`Profiles` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot best fits for.
        Defaults to varied and not derived parameters.

    ids : list, str
        Label(s) for input profiles.

    labels : list, str
        Label(s) for best fits within each :class:`Profiles` instance.

    truths : list, default=None
        Plot these truth / reference value for each parameter.
        If ``True``, take :attr:`Parameter.value`.

    ybands : list, default=None
        If not ``None``, plot horizontal bands.
        See :func:`plot_aligned`.

    ylimits : list, default=None
        If not ``None``, limits  for y-axis.

    figsize : float, tuple, default=None
        Figure size.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    lax : array
        Array of axes.
    """
    from matplotlib import pyplot as plt
    profiles = _make_list(profiles)
    params = _get_default_profiles_params(profiles, params=params)
    truths = _make_list(truths, length=len(params), default=None)
    ybands = _make_list(ybands, length=len(params), default=None)
    ylimits = _make_list(ybands, length=len(params), default=None)
    maxpoints = max(map(lambda prof: len(prof.bestfit), profiles))

    nrows = len(params)
    ncols = len(profiles) if len(profiles) > 1 else maxpoints
    figsize = figsize or (ncols, 3. * nrows)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, 1, wspace=0.1, hspace=0.1)

    lax = []
    for iparam1, param1 in enumerate(params):
        ax = plt.subplot(gs[iparam1])
        plot_aligned(profiles, param=param1, ids=ids, labels=labels, truth=truths[iparam1], yband=ybands[iparam1], ax=ax, **kwargs)
        if (iparam1 < nrows - 1) or not ids: ax.get_xaxis().set_visible(False)
        ax.set_ylim(ylimits[iparam1])
        if iparam1 != 0:
            leg = ax.get_legend()
            if leg is not None: leg.remove()
        lax.append(ax)

    return np.array(lax)


@plotting.plotter
def plot_profile(profiles, params=None, offsets=0., nrows=1, labels=None, colors=None, linestyles=None,
                 cl=(1, 2, 3), labelsize=None, ticksize=None, kw_profile=None, kw_cl=None,
                 kw_legend=None, figsize=None):
    """
    Plot profiles, with a panel for each parameter.

    Parameters
    ----------
    profiles : list
        List of (or single) :class:`Profiles` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot profiles for.
        Defaults to varied and not derived parameters.

    offsets : list, float, default=0
        Vertical offset for each profile.

    nrows : int, default=1
        Number of rows in figure.

    labels : list, str
        Label(s) for profiles within each :class:`Profiles` instance.

    colors : list, str, default=None
        Color(s) for profiles within each :class:`Profiles` instance.

    linestyles : list, str, default=None
        Linestyle(s) for profiles within each :class:`Profiles` instance.

    cl : int, tuple, default=(1, 2, 3)
        Confidence levels to plot.

    labelsize : int, default=None
        Label sizes.

    ticksize : int, default=None
        Tick sizes.

    kw_profile : dict, default=None
        Optional arguments for :meth:`matplotlib.axes.Axes.plot`.
        Defaults to ``{'marker': 'o'}``.

    kw_cl : dict, default=None
        Optional arguments for :meth:`matplotlib.axes.Axes.axhline`.
        Defaults to ``{'color': 'k', 'linestyle': ':', 'linewidth': 2}``.

    kw_legend : dict, default=None
        Optional arguments for :meth:`matplotlib.axes.Axes.legend`.

    figsize : float, tuple, default=None
        Figure size.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    lax : array
        Array of axes.
    """
    from matplotlib import pyplot as plt
    profiles = _make_list(profiles)
    params = _get_default_profiles_params(profiles, params=params, of='profile')
    nprofiles = len(profiles)
    offsets = _make_list(offsets, length=nprofiles, default=0.)
    labels = _make_list(labels, length=nprofiles, default=None)
    colors = _make_list(colors, length=nprofiles, default=None)
    linestyles = _make_list(linestyles, length=nprofiles, default=None)
    if np.ndim(cl) == 0: cl = [cl]
    add_legend = any(label is not None for label in labels)
    kw_profile = dict(kw_profile or {})
    kw_cl = dict(kw_cl if kw_cl is not None else {'color': 'k', 'linestyle': ':', 'linewidth': 2})
    xshift_cl = kw_cl.pop('xhift', 0.9)
    kw_legend = dict(kw_legend or {})

    ncols = int(len(params) * 1. / nrows + 1.)
    figsize = figsize or (4. * ncols, 4. * nrows)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.2, hspace=0.2)

    def data_to_axis(ax, y):
        axis_to_data = ax.transAxes + ax.transData.inverted()
        return axis_to_data.inverted().transform((0, y))[1]

    for iparam1, param1 in enumerate(params):
        ax = plt.subplot(gs[iparam1])
        for ipro, pro in enumerate(profiles):
            pro = pro.profile
            if param1 not in pro: continue
            ax.plot(pro[param1][0], pro[param1][1] - offsets[ipro], color=colors[ipro], linestyle=linestyles[ipro], label=labels[ipro], **kw_profile)
        for nsigma in cl:
            y = utils.nsigmas_to_deltachi2(nsigma, ddof=1)
            ax.axhline(y=y, xmin=0., xmax=1., **kw_cl)
            ax.text(xshift_cl, y + 0.1, r'${:d}\sigma$'.format(nsigma), horizontalalignment='left', verticalalignment='bottom',
                    transform=transforms.blended_transform_factory(ax.transAxes, ax.transData), color='k', fontsize=labelsize)
        lim = ax.get_ylim()
        ax.set_ylim(0., lim[-1] + 2.)
        ax.tick_params(labelsize=ticksize)
        ax.set_xlabel(param1.latex(inline=True), fontsize=labelsize)
        if iparam1 == 0: ax.set_ylabel(r'$\Delta \chi^{2}$', fontsize=labelsize)
        if add_legend and iparam1 == 0: ax.legend(**kw_legend)

    return gs


def plot_profile_comparison(profiles, profiles_ref, params=None, labels=None, colors=None, **kwargs):
    r"""
    Plot profile comparison, wrapping :func:`plot_profile`.
    Profiles ``profiles`` and ``profiles_ref`` are both offset by ``profiles`` minimum :math:`\chi^{2}`.

    Parameters
    ----------
    profiles : list
        List of (or single) :class:`Profiles` instance(s).

    profiles_ref : list
        List of (or single) :class:`Profiles` instance(s) to compare to.

    params : list, ParameterCollection, default=None
        Parameters to plot profiles for.
        Defaults to varied and not derived parameters.

    labels : list, str
        Label(s) for profiles within each :class:`Profiles` instance.

    colors : list, str, default=None
        Color(s) for profiles within each :class:`Profiles` instance.

    **kwargs : dict
        Optional arguments for :func:`plot_profile`
        ('nrows', 'cl', 'labelsize', 'ticksize', 'kw_profile', 'kw_cl', 'kw_legend', 'figsize').

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    lax : array
        Array of axes.
    """
    profiles = _make_list(profiles)
    profiles_ref = _make_list(profiles_ref)
    if len(profiles) != len(profiles_ref):
        raise ValueError('profiles_ref must be of same length as profiles')
    nprofiles = len(profiles)
    labels = _make_list(labels, length=nprofiles, default=None)
    colors = _make_list(colors, length=nprofiles, default=None)
    # Subtract chi2_min of profiles from both profiles and profiles_ref
    offsets = [-2. * pro.bestfit.logposterior.max() for pro in profiles] * 2
    colors = colors * 2
    linestyles = ['-'] * nprofiles + ['--'] * nprofiles
    plot_profile(profiles + profiles_ref, params=params, offsets=offsets, labels=labels, colors=colors, linestyles=linestyles, **kwargs)
