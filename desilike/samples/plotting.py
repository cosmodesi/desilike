import numpy as np
from matplotlib import pyplot as plt
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

    length : int, default=None
        Length of list to return.

    Returns
    -------
    toret : list
    """
    if obj is None:
        obj = default
    if is_parameter_sequence(obj):
        obj = list(obj)
        if length is not None:
            obj += [default] * (length - len(obj))
    else:
        obj = [obj]
        if length is not None:
            obj += [default] * (length - len(obj))
    return obj


def _get_default_chain_params(chains, params=None, **kwargs):
    from desilike.parameter import ParameterCollection
    chains = _make_list(chains)
    if params is not None:
        params = _make_list(params)
        list_params = ParameterCollection()
        for param in params:
            for chain in chains[::-1]:
                list_params += chain.params(name=[str(param)])
        return list_params
    list_params = [chain.params(**kwargs) for chain in chains]
    return ParameterCollection([params for params in list_params[0] if all(params in lparams for lparams in list_params[1:])])


def _get_default_profiles_params(profiles, params=None, of='bestfit', **kwargs):
    from desilike.parameter import ParameterCollection
    profiles = _make_list(profiles)
    if not profiles:
        return ParameterCollection()
    of = _make_list(of)
    if params is not None:
        params = _make_list(params)
        list_params = ParameterCollection()
        list_params = ParameterCollection()
        for param in params:
            for profile in profiles[::-1]:
                for off in of:
                    tmp = profile.get(off, None)
                    if tmp is not None:
                        list_params += tmp.params(name=[str(param)])
        return list_params
    list_params = []
    for profile in profiles:
        lparams = []
        for off in of:
            tmp = profile.get(off, None)
            if tmp is not None:
                lparams += tmp.params(**kwargs)
        list_params.append(lparams)
    return ParameterCollection([params for params in list_params[0] if all(params in lparams for lparams in list_params[1:])])


@plotting.plotter
def plot_trace(chains, params=None, figsize=None, colors=None, labelsize=None, kw_plot=None, fig=None):
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

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least as many axes as ``params``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params, varied=True, derived=False)
    nparams = len(params)
    colors = _make_list(colors, length=len(chains), default=None)
    kw_plot = kw_plot or {'alpha': 0.2}

    steps = 1 + np.arange(max(chain.size for chain in chains))
    figsize = figsize or (8, 1.5 * nparams)
    if fig is None:
        fig, lax = plt.subplots(nparams, sharex=True, sharey=False, figsize=figsize, squeeze=False)
        lax = lax.ravel()
    else:
        lax = fig.axes

    for ax, param in zip(lax, params):
        ax.grid(True)
        ax.set_ylabel(chains[0][param].param.latex(inline=True), fontsize=labelsize)
        ax.set_xlim(steps[0], steps[-1])
        for ichain, chain in enumerate(chains):
            tmp = chain[param].ravel()
            ax.plot(steps[:len(tmp)], tmp, color=colors[ichain], **kw_plot)

    lax[-1].set_xlabel('step', fontsize=labelsize)
    return fig


@plotting.plotter
def plot_gelman_rubin(chains, params=None, multivariate=False, threshold=None, slices=None, offset=0, labelsize=None, fig=None, **kwargs):
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

    offset : float, default=0
        Offset to apply to the Gelman-Rubin statistics, typically 0 or -1.

    labelsize : int, default=None
        Label sizes.

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least 1 axis.

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
    fig : matplotlib.figure.Figure
    """
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params, varied=True, derived=False)
    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)
    gr_multi = []
    gr = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        if multivariate: gr_multi.append(diagnostics.gelman_rubin(chains_sliced, params, method='eigen', **kwargs).max())
        for param in gr: gr[param].append(diagnostics.gelman_rubin(chains_sliced, param, method='diag', **kwargs))
    gr_multi = np.asarray(gr_multi)
    for param in gr: gr[param] = np.asarray(gr[param])

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ylabel = r'$\hat{{R}} {} {}$'.format('-' if (offset < 0) else '+', abs(offset)) if offset != 0 else r'$\hat{{R}}$'
    ax.set_ylabel(ylabel, fontsize=labelsize)

    if multivariate: ax.plot(slices, gr_multi + offset, label='multi', linestyle='-', linewidth=1, color='k')
    for param in params:
        ax.plot(slices, gr[param] + offset, label=chains[0][param].param.latex(inline=True), linestyle='--', linewidth=1)
    if threshold is not None: ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()
    return fig


@plotting.plotter
def plot_geweke(chains, params=None, threshold=None, slices=None, labelsize=None, fig=None, **kwargs):
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

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least 1 axis.

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
    fig : matplotlib.figure.Figure
    """
    params = _get_default_chain_params(chains, params=params, varied=True, derived=False)
    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)
    geweke = {param: [] for param in params}
    for end in slices:
        chains_sliced = [chain.ravel()[:end] for chain in chains]
        for param in geweke: geweke[param].append(diagnostics.geweke(chains_sliced, param, **kwargs))
    for param in geweke: geweke[param] = np.asarray(geweke[param]).mean(axis=-1)

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ax.set_ylabel(r'geweke', fontsize=labelsize)

    for param in params:
        ax.plot(slices, geweke[param], label=chains[0][param].param.latex(inline=True), linestyle='-', linewidth=1)
    if threshold is not None: ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()
    return fig


@plotting.plotter
def plot_autocorrelation_time(chains, params=None, threshold=50, slices=None, labelsize=None, fig=None):
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

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least 1 axis.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params, varied=True, derived=False)
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

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    ax.grid(True)
    ax.set_xlabel('step $N$', fontsize=labelsize)
    ax.set_ylabel('$\tau$', fontsize=labelsize)

    for param in params:
        ax.plot(slices, autocorr[param], label=chains[0][param].param.latex(inline=True), linestyle='--', linewidth=1)
    if threshold is not None:
        ax.plot(slices, slices * 1. / threshold, label='$N/{:d}$'.format(threshold), linestyle='--', linewidth=1, color='k')
    ax.legend()

    return fig


def add_legend(labels, colors=None, linestyles=None, fig=None, kw_handle=None, **kwargs):
    """
    Add legend to figure.

    Parameters
    ----------
    labels : list, str
        Label(s) for profiles within each :class:`Profiles` instance.

    colors : list, str, default=None
        Color(s) for profiles within each :class:`Profiles` instance.

    linestyles : list, str, default=None
        Linestyle(s) for profiles within each :class:`Profiles` instance.

    fig : matplotlib.figure.Figure, default=None
        Optionally, figure to add legend to. Else, take ``plt.gcf()``.

    **kwargs : dict
        Other arguments for :meth:`fig.legend`.
    """
    if fig is None: fig = plt.gcf()
    labels = _make_list(labels)
    nlabels = len(labels)
    colors = _make_list(colors, length=nlabels, default=None)
    for i, color in enumerate(colors):
        if color is None: colors[i] = 'C{:d}'.format(i)
    linestyles = _make_list(linestyles, length=nlabels, default=None)
    kw_handle = dict(kw_handle or {})
    from matplotlib.lines import Line2D
    handles = [Line2D([0, 1], [0, 1], color=color, linestyle=linestyle, **kw_handle) for color, linestyle in zip(colors, linestyles)]
    kwargs.setdefault('handles', handles)
    kwargs.setdefault('labels', labels)
    fig.legend(**kwargs)


def add_1d_profile(profile, param, ax=None, **kwargs):
    """
    Add 1D profile to axes.
    Requires :attr:`Profiles.profile` (or :attr:`Profiles.bestfit` and :attr:`Profiles.error` or :attr:`Profiles.covariance` for Gaussian approximation).

    Parameters
    ----------
    profile : Profile
        :class:`Profile` instance.

    param : Parameter, str
        Parameter to plot profile for.

    ax : matplotlib.axes.Axes, default=None
        Axes where to add profile. Defaults to ``plt.gca()``.

    **kwargs : dict
        Other arguments for :meth:`plt.plot`.
    """
    if ax is None: ax = plt.gca()

    def get_gaussian_1d_profile(mean, std, nsigma=3):
        t = np.linspace(mean - nsigma * std, mean + nsigma * std, endpoint=False)
        return t, np.exp(-(t - mean)**2 / (2 * std**2))

    pro = profile.get('profile', {})
    if param in pro:
        x = pro[param][:, 0]
        pdf = np.exp(pro[param][:, 1] - pro[param][:, 1].max())
    else:
        mean = profile.get('bestfit', None)
        std = profile.get('error', None)
        is_cov = std is None
        if is_cov: std = profile.get('covariance', None)
        if mean is not None and std is not None and (param in mean.params() and param in std.params()):
            index = mean.logposterior.argmax()
            mean = mean[param][index]
            std = std.std(param) if is_cov else std[param][index]
            x, pdf = get_gaussian_1d_profile(mean, std)
        else:
            return
    ax.plot(x, pdf, **kwargs)


def add_2d_contour(profile, param1, param2, ax=None, cl=(1, 2), color='C0', filled=False, pale_factor=0.6, alpha=1., **kwargs):
    r"""
    Add 2D contour to axes.
    Requires :attr:`Profiles.contour` (or :attr:`Profiles.bestfit` and :attr:`Profiles.covariance` for Gaussian approximation).

    Parameters
    ----------
    profile : Profile
        :class:`Profile` instance.

    param1 : Parameter, str
        First parameter to plot contour for.

    param2 : Parameter, str
        Second parameter to plot contour for.

    ax : matplotlib.axes.Axes, default=None
        Axes where to add profile. Defaults to ``plt.gca()``.

    cl : int, default=2
        Plot contours up to ``cl`` :math:`\sigma`.

    color : str, default='C0'
        Color.

    filled : bool, default=False
        If ``True``, draw filled contours.

    pale_factor : float, default=0.6
        When ``filled``, lightens contour colors of increasing confidence levels by this amount.

    alpha : float, default=1.
        Opacity.

    **kwargs : dict
        Other arguments for :meth:`plt.plot`.
    """
    if ax is None: ax = plt.gca()

    def pale_colors(color, nlevels, pale_factor=pale_factor):
        """Make color paler. Same as GetDist."""
        from matplotlib.colors import colorConverter
        color = colorConverter.to_rgb(color)
        colors = [color]
        for _ in range(1, nlevels):
            colors.append([c * (1 - pale_factor) + pale_factor for c in colors[-1]])
        return colors

    def get_gaussian_2d_contour(mean, cov, nsigma):
        radius = utils.nsigmas_to_deltachi2(nsigma, ddof=2)**0.5
        t = np.linspace(0., 2. * np.pi, 1000, endpoint=False)
        ct, st = np.cos(t), np.sin(t)
        sigx2, sigy2, sigxy = cov[0, 0], cov[1, 1], cov[0, 1]
        a = radius * np.sqrt(0.5 * (sigx2 + sigy2) + np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
        b = radius * np.sqrt(0.5 * (sigx2 + sigy2) - np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
        th = 0.5 * np.arctan2(2. * sigxy, sigx2 - sigy2)
        x1 = mean[0] + a * ct * np.cos(th) - b * st * np.sin(th)
        x2 = mean[1] + a * ct * np.sin(th) + b * st * np.cos(th)
        x1, x2 = (np.concatenate([xx, xx[:1]], axis=0) for xx in (x1, x2))
        return x1, x2

    cl = _make_list(cl)
    ccolors = dict(zip(cl, pale_colors(color, len(cl), pale_factor=pale_factor)))
    for nsigma in cl[::-1]:
        contour = profile.get('contour', {}).get(nsigma, [])
        if (param1, param2) in contour:
            x1, x2 = contour[param1, param2]
        else:
            mean = profile.get('bestfit', None)
            cov = profile.get('covariance', None)
            if mean is not None and cov is not None and all(param in mean.params() and param in cov.params() for param in [param1, param2]):
                mean = mean.choice(params=[param1, param2], return_type='nparray')
                cov = cov.view(params=[param1, param2], return_type='nparray')
                x1, x2 = get_gaussian_2d_contour(mean, cov, nsigma)
            else:
                continue
        if filled:
            ax.fill(x1, x2, color=ccolors[nsigma], alpha=alpha)
        ax.plot(x1, x2, color=ccolors[cl[0]], **kwargs)


@plotting.plotter
def plot_triangle_contours(profiles, params=None, labels=None, colors=None, linestyles=None, filled=False, pale_factor=0.6, cl=2, alpha=1., truths=None,
                           kw_contour=None, kw_truth=None, labelsize=None, kw_legend=None, figsize=None, fig=None):
    r"""
    Triangle plot for likelihood profiling.
    Requires :attr:`Profiles.profile` (or :attr:`Profiles.bestfit` and :attr:`Profiles.error` or :attr:`Profiles.covariance` for Gaussian approximation)
    and :attr:`Profiles.contour` (or :attr:`Profiles.bestfit` and :attr:`Profiles.covariance` for Gaussian approximation).

    Parameters
    ----------
    profiles : list, default=None
        List of (or single) :class:`Profiles` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot distribution for.
        Defaults to varied and not derived parameters.

    labels : list, str
        Label(s) for profiles within each :class:`Profiles` instance.

    colors : list, str, default=None
        Color(s) for profiles within each :class:`Profiles` instance.

    linestyles : list, str, default=None
        Linestyle(s) for profiles within each :class:`Profiles` instance.

    filled : list, bool, default=None
        If ``True``, draw filled contours. Can be provided for each :class:`Profiles` instance.

    pale_factor : float, default=0.6
        When ``filled``, lightens contour colors of increasing confidence levels by this amount.

    cl : int, default=2
        Plot contours up to ``cl`` :math:`\sigma`.

    alpha : list, float, default=1.
        Opacity(ies). Can be provided for each :class:`Profiles` instance.

    truths : list, dict, default=None
        Plot these truth / reference value for each parameter.

    kw_contour : dict, default=None
        Other options for plots.

    kw_truth : dict, default=None
        If ``None``, and ``truth`` not provided, no truth is plotted.
        Else, optional arguments for :meth:`matplotlib.axes.Axes.axhline`.
        Defaults to ``{'color': 'k', 'linestyle': ':', 'linewidth': 2}``.

    labelsize : int, default=None
        Label sizes.

    fig : matplotlib.figure.Figure, list, array, default=None
        Optionally, figure or array / list of axes.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles = _make_list(profiles)
    params = _get_default_profiles_params(profiles, params=params, of=['bestfit', 'profile'], varied=True, derived=False)
    nprofiles = len(profiles)
    if isinstance(truths, dict): truths = [truths.get(param.name, None) for param in params]
    truths = _make_list(truths, length=len(params), default=None)
    labels = _make_list(labels, length=nprofiles, default=None)
    colors = _make_list(colors, length=nprofiles, default=None)
    for i, color in enumerate(colors):
        if color is None: colors[i] = 'C{:d}'.format(i)
    alpha = _make_list(alpha, length=nprofiles, default=1.)
    filled = _make_list(filled, length=nprofiles, default=False)
    linestyles = _make_list(linestyles, length=nprofiles, default=None)
    _add_legend = any(label is not None for label in labels)
    kw_contour = dict(kw_contour or {})
    kw_legend = dict(kw_legend or {})
    kw_truth = dict(kw_truth or {'color': 'gray', 'linestyle': '--', 'linewidth': 0.5})

    ncols = nrows = len(params)
    if fig is None:
        from matplotlib.ticker import MaxNLocator
        max_nticks = 5
        factor = 2
        pltdim = factor * nrows
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
        dim = lbdim + pltdim + trdim
        figsize = figsize or (dim, dim)
        #fig, lax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(dim, dim))
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, wspace=0., hspace=0.)
        lax = np.ndarray((nrows, ncols), dtype=object)
        lax[...] = None
        for i1, param1 in enumerate(params):
            for i2 in range(nrows - 1, i1, -1):
                param2 = params[i2]
                ax = lax[i2, i1] = fig.add_subplot(gs[i2, i1], sharex=lax[nrows - 1, i1] if i2 != nrows - 1 else None, sharey=lax[i2, 0] if i1 > 0 else None)
                if i1 > 0:
                    ax.get_yaxis().set_visible(False)
                else:
                    if i2 < nrows - 1: ax.yaxis.set_major_locator(MaxNLocator(max_nticks, prune='lower'))
                    ax.set_ylabel(param2.latex(inline=True), fontsize=labelsize)
                if i2 < nrows - 1:
                    ax.get_xaxis().set_visible(False)
                else:
                    if i1 > 0: ax.xaxis.set_major_locator(MaxNLocator(max_nticks, prune='lower'))
                    ax.set_xlabel(param1.latex(inline=True), fontsize=labelsize)
                #ax.set_aspect('equal', adjustable='box')
                #ax.set_box_aspect(1)
            #for i2 in range(i1 - 1, -1, -1):
            #    ax = lax[i2, i1]
            #    ax.set_frame_on(False)
            #    ax.set_xticks([])
            #    ax.set_yticks([])
            ax = lax[i1, i1] = fig.add_subplot(gs[i1, i1], sharex=lax[nrows - 1, i1] if i1 != nrows - 1 else None)
            #ax.set_box_aspect(1)
            ax.set_ylim(0., 1.1)
            ax.get_yaxis().set_visible(False)
            if i1 < nrows - 1:
                ax.get_xaxis().set_visible(False)
            else:
                if i1 > 0: ax.xaxis.set_major_locator(MaxNLocator(max_nticks, prune='lower'))
                ax.set_xlabel(param1.latex(inline=True), fontsize=labelsize)
            #ax.set_aspect('equal', adjustable='box')
        # Format the figure.
        lb = lbdim / dim
        tr = (lbdim + pltdim) / dim
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=0., hspace=0.)
    else:
        lax = fig.axes
    lax = np.ravel(lax)

    nsigmas = min(max([cl for pro in profiles for cl in pro.get('contour', [])] + [cl]), cl)
    nsigmas = list(range(1, 1 + nsigmas))
    for i1, param1 in enumerate(params):
        for i2 in range(nrows - 1, i1, -1):
            param2 = params[i2]
            i = i2 * nrows + i1
            for ipro, pro in enumerate(profiles):
                color = colors[ipro]
                add_2d_contour(pro, param1, param2, ax=lax[i], cl=nsigmas, color=colors[ipro], pale_factor=pale_factor,
                               filled=filled[ipro], alpha=alpha[ipro], linestyle=linestyles[ipro], **kw_contour)
            if truths[i1] is not None:
                lax[i].axvline(x=truths[i1], ymin=0., ymax=1., **kw_truth)
            if truths[i2] is not None:
                lax[i].axhline(y=truths[i2], xmin=0., xmax=1., **kw_truth)
        i = i1 * (nrows + 1)
        for ipro, pro in enumerate(profiles):
            add_1d_profile(pro, param1, ax=lax[i], color=colors[ipro], linestyle=linestyles[ipro], **kw_contour)
        if truths[i1] is not None:
            lax[i].axvline(x=truths[i1], ymin=0., ymax=1., **kw_truth)

    if _add_legend:
        add_legend(colors=colors, labels=labels, kw_handle=kw_contour, fig=fig)

    return fig

'''
Version not using profiles.

@plotting.plotter
def plot_triangle(samples, params=None, labels=None, g=None, **kwargs):
    """
    Triangle plot, specifically for chains, or Gaussian approximations.
    For likelihood profiling, use :func:`plot_triangle_contours` instead.
    Uses *GetDist* package as a backend.

    Parameters
    ----------
    samples : list, default=None
        List of (or single) :class:`Chain`, :class:`Profiles` or :class:`LikelihoodFisher` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot distribution for.
        Defaults to varied and not derived parameters.

    labels : str, list, default=None
        Name for  *GetDist* to use for input samples.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    g : getdist subplot_plotter
        can be created with `g = getdist.plots.get_subplot_plotter()` and can be modified with g.settings

    show : bool, default=False
        If ``True``, show figure.

    **kwargs : dict
        Optional parameters for :meth:`GetDistPlotter.triangle_plot`.

    Returns
    -------
    g : getdist.plots.GetDistPlotter
    """
    from getdist import plots
    if g is None: g = plots.get_subplot_plotter()
    samples = _make_list(samples)
    labels = _make_list(labels, length=len(samples), default=None)
    params = _get_default_chain_params(samples, params=params, varied=True, input=True)
    samples = [sample.to_getdist(label=label, params=sample.params(name=params.names())) for sample, label in zip(samples, labels)]
    g.triangle_plot(samples, [str(param) for param in params], **kwargs)
    return g
'''


@plotting.plotter
def plot_triangle(samples, params=None, labels=None, g=None, contour_colors=None, contour_ls=None, filled=False, legend_ncol=None, legend_loc=None, markers=None, **kwargs):
    """
    Triangle plot.
    *GetDist* package is used to plot chains.
    If :class:`Profiles` are provided, requires :attr:`Profiles.profile` (or :attr:`Profiles.bestfit` and :attr:`Profiles.error` or :attr:`Profiles.covariance` for Gaussian approximation)
    and :attr:`Profiles.contour` (or :attr:`Profiles.bestfit` and :attr:`Profiles.covariance` for Gaussian approximation).

    Parameters
    ----------
    samples : list, default=None
        List of (or single) :class:`Chain`, :class:`Profiles` or :class:`LikelihoodFisher` instance(s).

    params : list, ParameterCollection, default=None
        Parameters to plot distribution for.
        Defaults to varied and not derived parameters.

    labels : str, list, default=None
        Name for  *GetDist* to use for input samples.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    g : getdist subplot_plotter
        can be created with `g = getdist.plots.get_subplot_plotter()` and can be modified with g.settings

    show : bool, default=False
        If ``True``, show figure.

    **kwargs : dict
        Optional parameters for :meth:`GetDistPlotter.triangle_plot`.

    Returns
    -------
    g : getdist.plots.GetDistPlotter
    """
    from desilike.samples import Chain, Profiles
    from getdist import plots
    if g is None: g = plots.get_subplot_plotter()
    samples = _make_list(samples)
    nsamples = len(samples)
    labels = _make_list(labels, length=nsamples, default=None)
    contour_colors = _make_list(contour_colors, length=nsamples, default=None)
    for i, color in enumerate(contour_colors):
        if color is None: contour_colors[i] = g.settings.solid_colors[i]
    filled = _make_list(filled, length=nsamples, default=False)
    contour_ls = _make_list(contour_ls, length=nsamples, default=None)

    input_params = params
    params = _get_default_chain_params([sample for sample in samples if not isinstance(sample, Profiles)], params=input_params, varied=True, input=True)
    params += _get_default_profiles_params([sample for sample in samples if isinstance(sample, Profiles)], of=['bestfit', 'profile'], params=input_params, varied=True, input=True)

    for_getdist, getdist_contour_colors, getdist_contour_ls, getdist_filled = [], [], [], []
    profiles, profiles_colors, profiles_linestyles, profiles_filled = [], [], [], []
    # Sort between what GetDist can handle (chains) and profiles
    for idx, (sample, label) in enumerate(zip(samples, labels)):
        if isinstance(sample, Chain) or (not hasattr(sample, 'profile') and not hasattr(sample, 'contour')):
            for_getdist.append(sample.to_getdist(label=label, params=sample.params(name=params.names())))
            getdist_contour_colors.append(contour_colors[idx])
            getdist_contour_ls.append(contour_ls[idx])
            getdist_filled.append(filled[idx])
        else:
            profiles.append(sample)
            profiles_colors.append(contour_colors[idx])
            profiles_linestyles.append(contour_ls[idx])
            profiles_filled.append(filled[idx])

    if for_getdist:
        g.triangle_plot(for_getdist, [str(param) for param in params], contour_colors=getdist_contour_colors, contour_ls=getdist_contour_ls, filled=filled,
                        legend_ncol=legend_ncol, legend_loc=legend_loc, markers=markers, **kwargs)
        kwargs = {'pale_factor': g.settings.solid_contour_palefactor, 'cl': g.settings.num_plot_contours, 'alpha': g.settings.alpha_factor_contour_lines, 'truths': None}
        fig = g.subplots
    else:
        fig = None
        kwargs['truths'] = markers
    fig = plot_triangle_contours(profiles, params=params, colors=profiles_colors, linestyles=profiles_linestyles, filled=profiles_filled, fig=fig, **kwargs)
    if for_getdist and profiles:
        # From GetDist
        if not legend_loc and g.settings.figure_legend_loc == 'upper center' and len(params) < 4:
            legend_loc = 'upper right'
        else:
            legend_loc = legend_loc or g.settings.figure_legend_loc
        args = {}
        if 'upper' in legend_loc:
            args['bbox_to_anchor'] = (g.plot_col / (2 if 'center' in legend_loc else 1), 1)
            args['bbox_transform'] = g.subplots[0, 0].transAxes
            args['borderaxespad'] = 0
        profiles_lines = [dict(color=color, linestyle=linestyle) for color, linestyle in zip(profiles_colors, profiles_linestyles)]
        g.contours_added += [None] * len(profiles_lines)
        try: g.legend.remove()
        except: pass
        g.lines_added.update({len(for_getdist) + i: line for i, line in enumerate(profiles_lines)})
        g.finish_plot(labels, legend_ncol=legend_ncol or g.settings.figure_legend_ncol, legend_loc=legend_loc,
                      no_extra_legend_space=True, **args)
    elif profiles:
        add_legend(labels=labels, colors=contour_colors, linestyles=contour_ls, fig=fig)
    return g


@plotting.plotter
def plot_aligned(profiles, param, ids=None, labels=None, colors=None, truth=None, error='error',
                 labelsize=None, ticksize=None, kw_scatter=None, yband=None, kw_mean=None, kw_truth=None, kw_yband=None,
                 kw_legend=None, fig=None):
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

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least 1 axis.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles = _make_list(profiles)
    if truth is True or (truth is None and kw_truth is not None):
        truth = profiles[0].bestfit[param].param.value
    kw_truth = dict(kw_truth or {'color': 'k', 'linestyle': ':', 'linewidth': 2})
    maxpoints = max(map(lambda prof: len(prof.bestfit), profiles))
    ids = _make_list(ids, length=len(profiles), default=None)
    labels = _make_list(labels, length=maxpoints, default=None)
    colors = _make_list(colors, length=maxpoints, default=['C{:d}'.format(i) for i in range(maxpoints)])
    add_legend = any(label is not None for label in labels)
    add_mean = kw_mean is not None
    if add_mean:
        kw_mean = kw_mean if isinstance(kw_mean, dict) else {'marker': 'o'}
    kw_scatter = dict(kw_scatter or {'marker': 'o'})
    kw_yband = dict(kw_yband or {})
    kw_legend = dict(kw_legend or {})

    xmain = np.arange(len(profiles))
    xaux = np.linspace(-0.15, 0.15, maxpoints)
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
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
    return fig


@plotting.plotter
def plot_aligned_stacked(profiles, params=None, ids=None, labels=None, truths=None, ybands=None, ylimits=None, figsize=None, fig=None, **kwargs):
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

    truths : list, dict, default=None
        Plot these truth / reference value for each parameter.

    ybands : list, default=None
        If not ``None``, plot horizontal bands.
        See :func:`plot_aligned`.

    ylimits : list, default=None
        If not ``None``, limits  for y-axis.

    figsize : float, tuple, default=None
        Figure size.

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least as many axes as ``params``.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles = _make_list(profiles)
    params = _get_default_profiles_params(profiles, params=params, varied=True, derived=False)
    if isinstance(truths, dict): truths = [truths.get(param.name, None) for param in params]
    truths = _make_list(truths, length=len(params), default=None)
    ybands = _make_list(ybands, length=len(params), default=None)
    ylimits = _make_list(ybands, length=len(params), default=None)
    maxpoints = max(map(lambda prof: len(prof.bestfit), profiles))

    nrows = len(params)
    ncols = len(profiles) if len(profiles) > 1 else maxpoints
    if fig is None:
        figsize = figsize or (ncols, 3. * nrows)
        fig, lax = plt.subplots(nrows, 1, figsize=figsize, squeeze=False)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
    else:
        lax = fig.axes
    lax = np.ravel(lax)

    for iparam1, param1 in enumerate(params):
        ax = lax[iparam1]
        plot_aligned(profiles, param=param1, ids=ids, labels=labels, truth=truths[iparam1], yband=ybands[iparam1], fig=ax, **kwargs)
        if (iparam1 < nrows - 1) or not ids: ax.get_xaxis().set_visible(False)
        ax.set_ylim(ylimits[iparam1])
        if iparam1 != 0:
            leg = ax.get_legend()
            if leg is not None: leg.remove()
    return fig


@plotting.plotter
def plot_profile(profiles, params=None, offsets=0., nrows=1, labels=None, colors=None, linestyles=None,
                 cl=(1, 2, 3), labelsize=None, ticksize=None, kw_profile=None, kw_cl=None,
                 kw_legend=None, figsize=None, fig=None):
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

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least as many axes as ``params``.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles = _make_list(profiles)
    params = _get_default_profiles_params(profiles, params=params, of='profile', varied=True, derived=False)
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

    ncols = int((len(params) + nrows - 1) * 1. / nrows)

    if fig is None:
        figsize = figsize or (4. * ncols, 4. * nrows)
        fig, lax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        lax = lax.ravel()
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
    else:
        lax = fig.axes

    for iparam1, param1 in enumerate(params):
        ax = lax[iparam1]
        for ipro, pro in enumerate(profiles):
            pro = pro.profile
            if param1 not in pro: continue
            ax.plot(pro[param1][:, 0], - 2 * (pro[param1][:, 1] - offsets[ipro]), color=colors[ipro], linestyle=linestyles[ipro], label=labels[ipro], **kw_profile)
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

    return fig


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

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least as many axes as ``params``.

    fn : str, Path, default=None
        Optionally, path where to save figure.
        If not provided, figure is not saved.

    kw_save : dict, default=None
        Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

    show : bool, default=False
        If ``True``, show figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles = _make_list(profiles)
    profiles_ref = _make_list(profiles_ref)
    if len(profiles) != len(profiles_ref):
        raise ValueError('profiles_ref must be of same length as profiles')
    nprofiles = len(profiles)
    labels = _make_list(labels, length=nprofiles, default=None)
    colors = _make_list(colors, length=nprofiles, default=None)
    # Subtract chi2_min of profiles from both profiles and profiles_ref
    offsets = [pro.bestfit.logposterior.max() for pro in profiles] * 2
    colors = colors * 2
    linestyles = ['-'] * nprofiles + ['--'] * nprofiles
    return plot_profile(profiles + profiles_ref, params=params, offsets=offsets, labels=labels, colors=colors, linestyles=linestyles, **kwargs)
