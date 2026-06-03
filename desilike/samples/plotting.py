"""Plotting routines for chains and profiles.

Adapted from desilike_bak/desilike/samples/plotting.py.
"""

import functools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec, transforms

from ..parameter import Variable, VariableCollection
from .samples import _vals, _normalise_params
from . import diagnostics


# ── sigma ↔ Δχ² conversions ──────────────────────────────────────────────────

def _nsigmas_to_deltachi2(nsigmas, ddof=1):
    r"""Return :math:`\Delta\chi^2` threshold for *nsigmas* Gaussian sigmas at *ddof* degrees of freedom."""
    from scipy import stats
    if ddof == 1:
        return float(nsigmas) ** 2
    quantile = stats.norm.cdf(nsigmas) - stats.norm.cdf(-nsigmas)
    return stats.chi2.ppf(quantile, ddof)


# ── plotter decorator ─────────────────────────────────────────────────────────

def plotter(func):
    """Decorator that adds ``fn``, ``kw_save``, and ``show`` keyword arguments.

    When ``fn`` is provided the figure is saved after the decorated function
    returns.  When ``show`` is ``True`` it is displayed via
    :func:`matplotlib.pyplot.show`.
    """
    @functools.wraps(func)
    def wrapper(*args, fn=None, kw_save=None, show=False, **kwargs):
        result = func(*args, **kwargs)
        fig = result if isinstance(result, plt.Figure) else None
        if fig is not None and fn is not None:
            fig.savefig(fn, **(kw_save or {}))
        if show:
            plt.show()
        return result
    return wrapper


# ── list helpers ──────────────────────────────────────────────────────────────

def _make_list(obj, length=None, default=None):
    """Coerce *obj* to a list, padding to *length* with *default* if needed."""
    if obj is None:
        obj = default
    if isinstance(obj, (list, tuple)):
        obj = list(obj)
        if length is not None:
            obj += [default] * (length - len(obj))
    else:
        obj = [obj]
        if length is not None:
            obj += [default] * (length - len(obj))
    return obj


# ── parameter-list helpers ────────────────────────────────────────────────────

def _varied_names(chain):
    """Return non-derived variable names from *chain*."""
    return [var.name for var in chain._data if not var.derived]


def _get_default_chain_params(chains, params=None):
    """Return varied parameter names common to all *chains*.

    Parameters
    ----------
    chains : list of MCSamples
    params : list of str, optional
        Restrict to these names.  Defaults to all varied non-derived names.

    Returns
    -------
    names : list of str
    """
    chains = _make_list(chains)
    if params is not None:
        param_names = [p if isinstance(p, str) else p.name for p in _make_list(params)]
        # Keep order from request; include if present in at least one chain
        result = []
        for name in param_names:
            if any(VariableCollection.__contains__(chain, name) for chain in chains):
                result.append(name)
        return result
    # Intersection of varied params across all chains (preserving first-chain order)
    all_sets = [set(_varied_names(chain)) for chain in chains]
    common = all_sets[0].intersection(*all_sets[1:])
    return [name for name in _varied_names(chains[0]) if name in common]


def _get_default_profiles_params(profiles, params=None, of=('best', 'profile')):
    """Return parameter names common to all *profiles* for the given slot(s).

    Parameters
    ----------
    profiles : list of Profiles
    params : list of str, optional
        Restrict to these names.
    of : sequence of str
        Slot names to search; e.g. ``('best', 'profile')``.

    Returns
    -------
    names : list of str
    """
    profiles = _make_list(profiles)
    if not profiles:
        return []
    of = _make_list(of)
    if params is not None:
        param_names = [p if isinstance(p, str) else p.name for p in _make_list(params)]
        result = []
        for name in param_names:
            for prof in profiles[::-1]:
                for slot in of:
                    slot_val = prof.get(slot, None)
                    if slot_val is not None and name in slot_val:
                        if name not in result:
                            result.append(name)
        return result
    # Names in first profile common to all
    first_names = []
    for slot in of:
        slot_val = profiles[0].get(slot, None)
        if slot_val is not None:
            for name in slot_val:
                if name not in first_names and name != 'logpdf':
                    first_names.append(name)
    result = []
    for name in first_names:
        if all(
            any(
                (prof.get(slot, None) is not None and name in prof.get(slot, {}))
                for slot in of
            )
            for prof in profiles
        ):
            result.append(name)
    return result


def _param_label(name, chains=None, profiles=None):
    """Return an inline LaTeX label for parameter *name*.

    Searches *chains* then *profiles* for a ``Variable``/``Parameter`` with the
    given name; falls back to the raw name string if none is found.
    """
    for chain in (chains or []):
        if VariableCollection.__contains__(chain, name):
            return VariableCollection.__getitem__(chain, name).latex(inline=True)
    for prof in (profiles or []):
        if prof.best is not None and name in prof.best:
            # Profiles don't store Variable objects; fall through
            pass
    return '${}$'.format(name)


def add_legend(labels, colors=None, linestyles=None, fig=None, kw_handle=None, **kwargs):
    """Add a legend to *fig* (defaults to :func:`plt.gcf`).

    Parameters
    ----------
    labels : str or list of str
    colors : str or list, optional
    linestyles : str or list, optional
    fig : matplotlib.figure.Figure, optional
    kw_handle : dict, optional
        Extra kwargs for :class:`~matplotlib.lines.Line2D` handles.
    **kwargs :
        Forwarded to :meth:`fig.legend`.
    """
    if fig is None:
        fig = plt.gcf()
    labels     = _make_list(labels)
    nlabels    = len(labels)
    colors     = _make_list(colors,     length=nlabels, default=None)
    linestyles = _make_list(linestyles, length=nlabels, default=None)
    for idx, color in enumerate(colors):
        if color is None:
            colors[idx] = 'C{:d}'.format(idx)
    kw_handle = dict(kw_handle or {})
    from matplotlib.lines import Line2D
    handles = [Line2D([0, 1], [0, 1], color=color, linestyle=ls, **kw_handle)
               for color, ls in zip(colors, linestyles)]
    kwargs.setdefault('handles', handles)
    kwargs.setdefault('labels', labels)
    fig.legend(**kwargs)


# ── chain-based plots ─────────────────────────────────────────────────────────

@plotter
def plot_trace(chains, params=None, figsize=None, colors=None, labelsize=None, kw_plot=None, fig=None):
    """Trace plot (parameter value vs step), one panel per parameter.

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
    params : list of str, optional
        Defaults to all varied non-derived parameters.
    figsize : tuple, optional
    colors : str or list, optional
    labelsize : int, optional
    kw_plot : dict, optional
        Forwarded to :meth:`~matplotlib.axes.Axes.plot`.  Default: ``{'alpha': 0.2}``.
    fig : matplotlib.figure.Figure, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    chains  = _make_list(chains)
    params  = _get_default_chain_params(chains, params=params)
    nparams = len(params)
    colors  = _make_list(colors, length=len(chains), default=None)
    kw_plot = kw_plot or {'alpha': 0.2}

    steps   = 1 + np.arange(max(chain.size for chain in chains))
    figsize = figsize or (8, 1.5 * nparams)

    if fig is None:
        fig, lax = plt.subplots(nparams, sharex=True, sharey=False, figsize=figsize, squeeze=False)
        lax = lax.ravel()
    else:
        lax = fig.axes

    for ax, name in zip(lax, params):
        ax.grid(True)
        ax.set_ylabel(_param_label(name, chains=chains), fontsize=labelsize)
        ax.set_xlim(steps[0], steps[-1])
        for chain_idx, chain in enumerate(chains):
            if not VariableCollection.__contains__(chain, name):
                continue
            tmp = _vals(chain, name).ravel()
            ax.plot(steps[:len(tmp)], tmp, color=colors[chain_idx], **kw_plot)

    lax[-1].set_xlabel('step', fontsize=labelsize)
    return fig


@plotter
def plot_gelman_rubin(chains, params=None, multivariate=False, threshold=None, slices=None, offset=0, labelsize=None, fig=None, **kwargs):
    """Plot Gelman-Rubin statistics as a function of chain length.

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
    params : list of str, optional
    multivariate : bool, default=False
        If ``True``, also plot the maximum eigenvalue.
    threshold : float, optional
        Draw a horizontal reference line at this value.
    slices : array_like, optional
        Step counts at which to evaluate :math:`\\hat{R}`.
        Default: ``np.arange(100, nsteps, 500)``.
    offset : float, default=0
        Vertical offset applied to all :math:`\\hat{R}` values.
    labelsize : int, optional
    fig : matplotlib.figure.Figure, optional
    **kwargs :
        Forwarded to :func:`.diagnostics.gelman_rubin`
        (e.g. ``nsplits``, ``check_valid``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params)

    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)

    gr_multi = []
    gr = {name: [] for name in params}

    for end in slices:
        chains_sliced = [chain[:end] for chain in chains]
        if multivariate:
            gr_multi.append(diagnostics.gelman_rubin(chains_sliced, params, method='eigen', **kwargs).max())
        for name in gr:
            gr[name].append(diagnostics.gelman_rubin(chains_sliced, name, method='diag', **kwargs))

    gr_multi = np.asarray(gr_multi)
    for name in gr:
        gr[name] = np.asarray(gr[name])

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]

    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ylabel = (r'$\hat{{R}} {} {}$'.format('-' if offset < 0 else '+', abs(offset))
              if offset != 0 else r'$\hat{R}$')
    ax.set_ylabel(ylabel, fontsize=labelsize)

    if multivariate:
        ax.plot(slices, gr_multi + offset, label='multi', linestyle='-', linewidth=1, color='k')
    for name in params:
        ax.plot(slices, gr[name] + offset,
                label=_param_label(name, chains=chains), linestyle='--', linewidth=1)
    if threshold is not None:
        ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()
    return fig


@plotter
def plot_geweke(chains, params=None, threshold=None, slices=None, labelsize=None, fig=None, **kwargs):
    """Plot Geweke statistics as a function of chain length.

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
    params : list of str, optional
    threshold : float, optional
        Draw a horizontal reference line at this value.
    slices : array_like, optional
        Default: ``np.arange(100, nsteps, 500)``.
    labelsize : int, optional
    fig : matplotlib.figure.Figure, optional
    **kwargs :
        Forwarded to :func:`.diagnostics.geweke`
        (e.g. ``first``, ``last``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params)

    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)

    geweke_vals = {name: [] for name in params}
    for end in slices:
        chains_sliced = [chain[:end] for chain in chains]
        for name in geweke_vals:
            geweke_vals[name].append(diagnostics.geweke(chains_sliced, name, **kwargs))
    for name in geweke_vals:
        geweke_vals[name] = np.asarray(geweke_vals[name]).mean(axis=-1)

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]

    ax.grid(True)
    ax.set_xlabel('step', fontsize=labelsize)
    ax.set_ylabel('Geweke', fontsize=labelsize)

    for name in params:
        ax.plot(slices, geweke_vals[name],
                label=_param_label(name, chains=chains), linestyle='-', linewidth=1)
    if threshold is not None:
        ax.axhline(y=threshold, xmin=0., xmax=1., linestyle='--', linewidth=1, color='k')
    ax.legend()
    return fig


@plotter
def plot_autocorrelation_time(chains, params=None, threshold=50, slices=None, labelsize=None, fig=None):
    r"""Plot integrated autocorrelation time vs chain length.

    Parameters
    ----------
    chains : MCSamples or list of MCSamples
    params : list of str, optional
    threshold : int, default=50
        If not ``None``, overplot the :math:`N/\text{threshold}` line.
    slices : array_like, optional
        Default: ``np.arange(100, nsteps, 500)``.
    labelsize : int, optional
    fig : matplotlib.figure.Figure, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    chains = _make_list(chains)
    params = _get_default_chain_params(chains, params=params)

    if slices is None:
        nsteps = min(chain.size for chain in chains)
        slices = np.arange(100, nsteps, 500)

    autocorr = {name: [] for name in params}
    for end in slices:
        chains_sliced = [chain[:end] for chain in chains]
        for name in autocorr:
            autocorr[name].append(
                diagnostics.integrated_autocorrelation_time(chains_sliced, name, check_valid='ignore')
            )
    for name in autocorr:
        autocorr[name] = np.asarray(autocorr[name])

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]

    ax.grid(True)
    ax.set_xlabel(r'step $N$', fontsize=labelsize)
    ax.set_ylabel(r'$\tau$', fontsize=labelsize)

    for name in params:
        ax.plot(slices, autocorr[name],
                label=_param_label(name, chains=chains), linestyle='--', linewidth=1)
    if threshold is not None:
        ax.plot(slices, slices / float(threshold),
                label=r'$N/{:d}$'.format(threshold), linestyle='--', linewidth=1, color='k')
    ax.legend()
    return fig


# ── profiles-based plots ──────────────────────────────────────────────────────

def add_1d_profile(profiles, param, ax=None, **kwargs):
    """Draw a 1-D profile (or Gaussian approximation) on *ax*.

    Parameters
    ----------
    profiles : Profiles
    param : str
        Parameter name.
    ax : matplotlib.axes.Axes, optional
    **kwargs :
        Forwarded to :meth:`~matplotlib.axes.Axes.plot`.
    """
    if ax is None:
        ax = plt.gca()

    def _gaussian_1d(mean, std, nsigma=3):
        t = np.linspace(mean - nsigma * std, mean + nsigma * std, 200)
        return t, np.exp(-(t - mean) ** 2 / (2.0 * std ** 2))

    name = param if isinstance(param, str) else param.name

    pro = profiles.get('profile', None)
    if pro is not None and name in pro:
        scan_vals, lp_vals = pro[name]
        scan_vals = np.asarray(scan_vals)
        lp_vals   = np.asarray(lp_vals)
        pdf = np.exp(lp_vals - lp_vals.max())
        ax.plot(scan_vals, pdf, **kwargs)
        return

    # Fall back to Gaussian approximation
    best = profiles.get('best', None)
    if best is None:
        return
    argmax = int(np.argmax(np.asarray(best.get('logpdf', [-np.inf]))))

    if name not in best:
        return
    mean_val = float(np.asarray(best[name]).ravel()[argmax])

    # Try error first, then covariance
    err = profiles.get('error', None)
    cov = profiles.get('covariance', None)
    if err is not None and name in err:
        std_val = float(np.asarray(err[name]).ravel()[argmax])
    elif cov is not None and name in cov:
        std_val = float(cov.view([name]).std()[0])
    else:
        return

    x, pdf = _gaussian_1d(mean_val, std_val)
    ax.plot(x, pdf, **kwargs)


def add_2d_contour(profiles, param1, param2, ax=None, cl=(1, 2), color='C0', filled=False, pale_factor=0.6, alpha=1., **kwargs):
    r"""Draw 2-D contours (or Gaussian approximation) on *ax*.

    Parameters
    ----------
    profiles : Profiles
    param1, param2 : str
        Parameter names.
    ax : matplotlib.axes.Axes, optional
    cl : int or tuple, default=(1, 2)
        Confidence levels (in :math:`\sigma`) to draw.
    color : str, default='C0'
    filled : bool, default=False
    pale_factor : float, default=0.6
        Paling factor for filled contour levels.
    alpha : float, default=1.
    **kwargs :
        Forwarded to :meth:`~matplotlib.axes.Axes.plot`.
    """
    if ax is None:
        ax = plt.gca()

    name1 = param1 if isinstance(param1, str) else param1.name
    name2 = param2 if isinstance(param2, str) else param2.name

    def _pale_colors(base_color, nlevels, pale_factor=pale_factor):
        from matplotlib.colors import colorConverter
        c = list(colorConverter.to_rgb(base_color))
        cols = [c]
        for _ in range(1, nlevels):
            cols.append([x * (1.0 - pale_factor) + pale_factor for x in cols[-1]])
        return cols

    def _gaussian_2d_ellipse(mean, cov_2x2, nsigma):
        radius    = _nsigmas_to_deltachi2(nsigma, ddof=2) ** 0.5
        t         = np.linspace(0.0, 2.0 * np.pi, 1000, endpoint=False)
        ct, st    = np.cos(t), np.sin(t)
        sigx2, sigy2, sigxy = cov_2x2[0, 0], cov_2x2[1, 1], cov_2x2[0, 1]
        a   = radius * np.sqrt(0.5 * (sigx2 + sigy2) + np.sqrt(0.25 * (sigx2 - sigy2) ** 2 + sigxy ** 2))
        b   = radius * np.sqrt(0.5 * (sigx2 + sigy2) - np.sqrt(0.25 * (sigx2 - sigy2) ** 2 + sigxy ** 2))
        th  = 0.5 * np.arctan2(2.0 * sigxy, sigx2 - sigy2)
        x1  = mean[0] + a * ct * np.cos(th) - b * st * np.sin(th)
        x2  = mean[1] + a * ct * np.sin(th) + b * st * np.cos(th)
        return (np.concatenate([x1, x1[:1]]), np.concatenate([x2, x2[:1]]))

    cl_list = _make_list(cl)
    pale    = _pale_colors(color, len(cl_list), pale_factor=pale_factor)
    ccolors = dict(zip(cl_list, pale))

    for nsigma in cl_list[::-1]:
        # Try stored contours first
        contour_dict = profiles.get('contour', None)
        x1 = x2 = None
        if contour_dict is not None and nsigma in contour_dict:
            pair_dict = contour_dict[nsigma]
            pair_key  = (name1, name2)
            if pair_key in pair_dict:
                x1, x2 = pair_dict[pair_key]

        if x1 is None:
            # Gaussian approximation from best + covariance
            best = profiles.get('best', None)
            cov  = profiles.get('covariance', None)
            if best is None or cov is None:
                continue
            if name1 not in best or name2 not in best:
                continue
            if name1 not in cov or name2 not in cov:
                continue
            argmax   = int(np.argmax(np.asarray(best.get('logpdf', [-np.inf]))))
            mean_vec = np.array([
                float(np.asarray(best[name1]).ravel()[argmax]),
                float(np.asarray(best[name2]).ravel()[argmax]),
            ])
            cov_2x2  = np.asarray(cov.view([name1, name2]))
            x1, x2   = _gaussian_2d_ellipse(mean_vec, cov_2x2, nsigma)

        if filled:
            ax.fill(x1, x2, color=ccolors[nsigma], alpha=alpha)
        ax.plot(x1, x2, color=ccolors[cl_list[0]], **kwargs)


@plotter
def plot_triangle_contours(profiles, params=None, labels=None, colors=None, linestyles=None,
                           filled=False, pale_factor=0.6, cl=2, alpha=1., truths=None,
                           kw_contour=None, kw_truth=None, labelsize=None, kw_legend=None,
                           figsize=None, fig=None):
    r"""Triangle plot for likelihood profiling.

    Parameters
    ----------
    profiles : Profiles or list of Profiles
    params : list of str, optional
    labels : str or list, optional
    colors : str or list, optional
    linestyles : str or list, optional
    filled : bool or list, default=False
    pale_factor : float, default=0.6
    cl : int, default=2
        Plot contours up to this :math:`\sigma` level.
    alpha : float or list, default=1.
    truths : list or dict, optional
    kw_contour : dict, optional
    kw_truth : dict, optional
    labelsize : int, optional
    kw_legend : dict, optional
    figsize : tuple, optional
    fig : matplotlib.figure.Figure, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles   = _make_list(profiles)
    params     = _get_default_profiles_params(profiles, params=params, of=('best', 'profile'))
    nprofiles  = len(profiles)

    if isinstance(truths, dict):
        truths = [truths.get(name, None) for name in params]
    truths     = _make_list(truths,     length=len(params), default=None)
    labels     = _make_list(labels,     length=nprofiles,   default=None)
    colors     = _make_list(colors,     length=nprofiles,   default=None)
    alpha      = _make_list(alpha,      length=nprofiles,   default=1.)
    filled     = _make_list(filled,     length=nprofiles,   default=False)
    linestyles = _make_list(linestyles, length=nprofiles,   default=None)
    for idx, color in enumerate(colors):
        if color is None:
            colors[idx] = 'C{:d}'.format(idx)
    _add_legend = any(label is not None for label in labels)
    kw_contour  = dict(kw_contour or {})
    kw_legend   = dict(kw_legend  or {})
    kw_truth    = dict(kw_truth   or {'color': 'gray', 'linestyle': '--', 'linewidth': 0.5})

    nrows = ncols = len(params)

    if fig is None:
        from matplotlib.ticker import MaxNLocator
        max_nticks  = 5
        factor      = 2
        pltdim      = factor * nrows
        lbdim       = 0.5 * factor
        trdim       = 0.2 * factor
        dim         = lbdim + pltdim + trdim
        figsize     = figsize or (dim, dim)
        fig         = plt.figure(figsize=figsize)
        gs          = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig, wspace=0., hspace=0.)
        lax         = np.ndarray((nrows, ncols), dtype=object)
        lax[...]    = None

        for col_idx, name1 in enumerate(params):
            for row_idx in range(nrows - 1, col_idx, -1):
                name2   = params[row_idx]
                share_x = lax[nrows - 1, col_idx] if row_idx != nrows - 1 else None
                share_y = lax[row_idx, 0]         if col_idx > 0          else None
                ax      = lax[row_idx, col_idx] = fig.add_subplot(gs[row_idx, col_idx], sharex=share_x, sharey=share_y)
                if col_idx > 0:
                    ax.get_yaxis().set_visible(False)
                else:
                    if row_idx < nrows - 1:
                        ax.yaxis.set_major_locator(MaxNLocator(max_nticks, prune='lower'))
                    ax.set_ylabel(name2, fontsize=labelsize)
                if row_idx < nrows - 1:
                    ax.get_xaxis().set_visible(False)
                else:
                    if col_idx > 0:
                        ax.xaxis.set_major_locator(MaxNLocator(max_nticks, prune='lower'))
                    ax.set_xlabel(name1, fontsize=labelsize)

            share_x_diag = lax[nrows - 1, col_idx] if col_idx != nrows - 1 else None
            ax_diag      = lax[col_idx, col_idx] = fig.add_subplot(gs[col_idx, col_idx], sharex=share_x_diag)
            ax_diag.set_ylim(0., 1.1)
            ax_diag.get_yaxis().set_visible(False)
            if col_idx < nrows - 1:
                ax_diag.get_xaxis().set_visible(False)
            else:
                if col_idx > 0:
                    ax_diag.xaxis.set_major_locator(MaxNLocator(max_nticks, prune='lower'))
                ax_diag.set_xlabel(name1, fontsize=labelsize)

        lb = lbdim / dim
        tr = (lbdim + pltdim) / dim
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=0., hspace=0.)
    else:
        lax = fig.axes
    lax = np.ravel(lax)

    # Determine actual sigma levels to plot
    stored_cls = [
        cl_val
        for pro in profiles
        for cl_val in (pro.contour or {})
    ]
    nsigmas = list(range(1, 1 + min(max(stored_cls + [cl]), cl)))

    for col_idx, name1 in enumerate(params):
        for row_idx in range(nrows - 1, col_idx, -1):
            name2  = params[row_idx]
            ax_idx = row_idx * nrows + col_idx
            for prof_idx, prof in enumerate(profiles):
                add_2d_contour(
                    prof, name1, name2,
                    ax=lax[ax_idx], cl=nsigmas,
                    color=colors[prof_idx],
                    pale_factor=pale_factor,
                    filled=filled[prof_idx],
                    alpha=alpha[prof_idx],
                    linestyle=linestyles[prof_idx],
                    **kw_contour,
                )
            if truths[col_idx] is not None:
                lax[ax_idx].axvline(x=truths[col_idx], ymin=0., ymax=1., **kw_truth)
            if truths[row_idx] is not None:
                lax[ax_idx].axhline(y=truths[row_idx], xmin=0., xmax=1., **kw_truth)

        diag_idx = col_idx * (nrows + 1)
        for prof_idx, prof in enumerate(profiles):
            add_1d_profile(prof, name1, ax=lax[diag_idx],
                           color=colors[prof_idx], linestyle=linestyles[prof_idx],
                           **kw_contour)
        if truths[col_idx] is not None:
            lax[diag_idx].axvline(x=truths[col_idx], ymin=0., ymax=1., **kw_truth)

    if _add_legend:
        add_legend(colors=colors, labels=labels, kw_handle=kw_contour, fig=fig)

    return fig


@plotter
def plot_triangle(samples, params=None, labels=None, g=None, contour_colors=None, contour_ls=None,
                  filled=False, legend_ncol=None, legend_loc=None, markers=None, **kwargs):
    """Triangle plot using GetDist for chains, profiles for profile likelihoods.

    Parameters
    ----------
    samples : MCSamples, Profiles, or list of either
    params : list of str, optional
    labels : str or list, optional
    g : getdist.plots.GetDistPlotter, optional
    contour_colors : str or list, optional
    contour_ls : str or list, optional
    filled : bool or list, default=False
    legend_ncol : int, optional
    legend_loc : str, optional
    markers : dict, optional
    **kwargs :
        Forwarded to :meth:`~getdist.plots.GetDistPlotter.triangle_plot`.

    Returns
    -------
    g : getdist.plots.GetDistPlotter
    """
    from desilike.samples import MCSamples, Profiles
    from getdist import plots

    if g is None:
        g = plots.get_subplot_plotter()

    samples     = _make_list(samples)
    nsamples    = len(samples)
    labels      = _make_list(labels,         length=nsamples, default=None)
    contour_colors = _make_list(contour_colors, length=nsamples, default=None)
    for idx, color in enumerate(contour_colors):
        if color is None:
            contour_colors[idx] = g.settings.solid_colors[idx]
    filled      = _make_list(filled,    length=nsamples, default=False)
    contour_ls  = _make_list(contour_ls, length=nsamples, default=None)

    input_params = params
    chain_samples    = [s for s in samples if isinstance(s, MCSamples)]
    profiles_samples = [s for s in samples if isinstance(s, Profiles)]

    params = _get_default_chain_params(chain_samples, params=input_params) if chain_samples else []
    params = list(params) + [
        name for name in _get_default_profiles_params(
            profiles_samples, of=('best', 'profile'), params=input_params
        )
        if name not in params
    ]

    for_getdist, gd_colors, gd_ls, gd_filled = [], [], [], []
    prof_list, prof_colors, prof_ls, prof_filled = [], [], [], []

    for idx, (sample, label) in enumerate(zip(samples, labels)):
        if isinstance(sample, MCSamples):
            chain_params = [name for name in params if VariableCollection.__contains__(sample, name)]
            for_getdist.append(sample.to_getdist(label=label, params=chain_params))
            gd_colors.append(contour_colors[idx])
            gd_ls.append(contour_ls[idx])
            gd_filled.append(filled[idx])
        else:
            prof_list.append(sample)
            prof_colors.append(contour_colors[idx])
            prof_ls.append(contour_ls[idx])
            prof_filled.append(filled[idx])

    if for_getdist:
        g.triangle_plot(
            for_getdist, [str(name) for name in params],
            contour_colors=gd_colors, contour_ls=gd_ls, filled=filled,
            legend_ncol=legend_ncol, legend_loc=legend_loc, markers=markers,
            **kwargs,
        )
        triangle_kwargs = {
            'pale_factor': g.settings.solid_contour_palefactor,
            'cl': g.settings.num_plot_contours,
            'alpha': g.settings.alpha_factor_contour_lines,
            'truths': None,
        }
        fig = g.subplots
    else:
        fig = None
        triangle_kwargs = dict(kwargs)
        triangle_kwargs['truths'] = markers

    fig = plot_triangle_contours(
        prof_list, params=params,
        colors=prof_colors, linestyles=prof_ls, filled=prof_filled,
        fig=fig, **triangle_kwargs,
    )

    if for_getdist and prof_list:
        if not legend_loc and g.settings.figure_legend_loc == 'upper center' and len(params) < 4:
            legend_loc = 'upper right'
        else:
            legend_loc = legend_loc or g.settings.figure_legend_loc
        args = {}
        if 'upper' in legend_loc:
            args['bbox_to_anchor'] = (g.plot_col / (2 if 'center' in legend_loc else 1), 1)
            args['bbox_transform'] = g.subplots[0, 0].transAxes
            args['borderaxespad'] = 0
        prof_lines = [{'color': color, 'linestyle': ls}
                      for color, ls in zip(prof_colors, prof_ls)]
        g.contours_added += [None] * len(prof_lines)
        try:
            g.legend.remove()
        except Exception:
            pass
        g.lines_added.update({len(for_getdist) + idx: line for idx, line in enumerate(prof_lines)})
        g.finish_plot(labels, legend_ncol=legend_ncol or g.settings.figure_legend_ncol,
                      legend_loc=legend_loc, no_extra_legend_space=True, **args)
    elif prof_list:
        add_legend(labels=labels, colors=contour_colors, linestyles=contour_ls, fig=fig)

    return g


@plotter
def plot_aligned(profiles, param, ids=None, labels=None, colors=None, truth=None, error='error',
                 labelsize=None, ticksize=None, kw_scatter=None, yband=None,
                 kw_mean=None, kw_truth=None, kw_yband=None, kw_legend=None, fig=None):
    """Plot best-fit values with error bars for a single parameter across multiple profile sets.

    Parameters
    ----------
    profiles : Profiles or list of Profiles
    param : str
        Parameter name.
    ids : list of str, optional
        x-axis labels for each Profiles instance.
    labels : list of str, optional
        Legend labels for best-fit points within each Profiles.
    colors : list, optional
    truth : float, optional
        Reference value.
    error : str, default='error'
        Slot to use for error bars (``'error'`` or ``'interval'``).
    labelsize : int, optional
    ticksize : int, optional
    kw_scatter : dict, optional
    yband : float or tuple, optional
    kw_mean : dict, optional
        If not ``None``, also plot the mean.
    kw_truth : dict, optional
    kw_yband : dict, optional
    kw_legend : dict, optional
    fig : matplotlib.figure.Figure, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles = _make_list(profiles)
    name     = param if isinstance(param, str) else param.name

    if truth is None and kw_truth is not None:
        first_best = profiles[0].get('best', None)
        if first_best is not None and name in first_best:
            argmax = int(np.argmax(np.asarray(first_best.get('logpdf', [-np.inf]))))
            truth  = float(np.asarray(first_best[name]).ravel()[argmax])

    kw_truth   = dict(kw_truth   or {'color': 'k', 'linestyle': ':', 'linewidth': 2})
    kw_scatter = dict(kw_scatter or {'marker': 'o'})
    kw_yband   = dict(kw_yband   or {})
    kw_legend  = dict(kw_legend  or {})

    maxpoints = max(
        (len(np.asarray(prof.best[name])) if prof.best is not None and name in prof.best else 0)
        for prof in profiles
    )
    ids     = _make_list(ids,    length=len(profiles), default=None)
    labels  = _make_list(labels, length=maxpoints,     default=None)
    colors  = _make_list(colors, length=maxpoints,     default=['C{:d}'.format(idx) for idx in range(maxpoints)])
    add_mean   = kw_mean is not None
    if add_mean:
        kw_mean = kw_mean if isinstance(kw_mean, dict) else {'marker': 'o'}
    add_lgd = any(label is not None for label in labels)

    xmain = np.arange(len(profiles))
    xaux  = np.linspace(-0.15, 0.15, maxpoints) if maxpoints > 1 else np.zeros(1)

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]

    for prof_idx, prof in enumerate(profiles):
        if prof.best is None or name not in prof.best:
            continue
        best_vals = np.asarray(prof.best[name]).ravel()
        argmax    = int(np.argmax(np.asarray(prof.best.get('logpdf', [-np.inf]))))
        for point_idx, best_val in enumerate(best_vals):
            yerr = None
            if error:
                err_slot = prof.get(error, None)
                if err_slot is not None and name in err_slot:
                    err_arr = np.asarray(err_slot[name]).ravel()
                    yerr    = float(err_arr[0] if len(err_arr) == 1 else err_arr[argmax])
            point_label = labels[point_idx] if prof_idx == 0 else None
            ax.errorbar(xmain[prof_idx] + xaux[point_idx], float(best_val),
                        yerr=yerr, color=colors[point_idx], label=point_label,
                        linestyle='none', **kw_scatter)
        if add_mean:
            mean_val = float(best_vals.mean())
            std_val  = float(best_vals.std(ddof=1)) if len(best_vals) > 1 else 0.0
            ax.errorbar(xmain[prof_idx], mean_val, yerr=std_val,
                        linestyle='none', **kw_mean)

    if truth is not None:
        ax.axhline(y=truth, xmin=0., xmax=1., **kw_truth)

    if yband is not None:
        if np.ndim(yband) == 0:
            yband = (float(yband), float(yband))
        if yband[-1] == 'abs':
            low, up = float(yband[0]), float(yband[1])
        else:
            if truth is None:
                raise ValueError('Plotting a relative y-band requires a truth value.')
            low = truth * (1.0 - float(yband[0]))
            up  = truth * (1.0 + float(yband[1]))
        ax.axhspan(low, up, **kw_yband)

    ax.set_xticks(xmain)
    ax.set_xticklabels(ids, rotation=40, ha='right', fontsize=ticksize)
    ax.grid(True, axis='y')
    ax.set_ylabel('${}$'.format(name), fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    if add_lgd:
        ax.legend(**{**{'ncol': maxpoints}, **kw_legend})
    return fig


@plotter
def plot_aligned_stacked(profiles, params=None, ids=None, labels=None, truths=None,
                         ybands=None, ylimits=None, figsize=None, fig=None, **kwargs):
    """Stacked :func:`plot_aligned` panels, one per parameter.

    Parameters
    ----------
    profiles : Profiles or list of Profiles
    params : list of str, optional
    ids : list of str, optional
    labels : list, optional
    truths : list or dict, optional
    ybands : list, optional
    ylimits : list, optional
    figsize : tuple, optional
    fig : matplotlib.figure.Figure, optional
    **kwargs :
        Forwarded to :func:`plot_aligned`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles = _make_list(profiles)
    params   = _get_default_profiles_params(profiles, params=params)

    if isinstance(truths, dict):
        truths = [truths.get(name, None) for name in params]
    truths  = _make_list(truths,  length=len(params), default=None)
    ybands  = _make_list(ybands,  length=len(params), default=None)
    ylimits = _make_list(ylimits, length=len(params), default=None)

    maxpoints = max(
        (len(np.asarray(prof.best[params[0]])) if prof.best is not None and params else 0)
        for prof in profiles
    ) if profiles and params else 1

    nrows = len(params)
    ncols = len(profiles) if len(profiles) > 1 else maxpoints

    if fig is None:
        figsize = figsize or (ncols, 3.0 * nrows)
        fig, lax = plt.subplots(nrows, 1, figsize=figsize, squeeze=False)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
    else:
        lax = fig.axes
    lax = np.ravel(lax)

    for param_idx, name in enumerate(params):
        ax = lax[param_idx]
        plot_aligned(profiles, param=name, ids=ids, labels=labels,
                     truth=truths[param_idx], yband=ybands[param_idx],
                     fig=ax, **kwargs)
        if param_idx < nrows - 1 or not ids:
            ax.get_xaxis().set_visible(False)
        ax.set_ylim(ylimits[param_idx])
        if param_idx != 0:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    return fig


@plotter
def plot_profile(profiles, params=None, offsets=0., nrows=1, labels=None, colors=None,
                 linestyles=None, cl=(1, 2, 3), labelsize=None, ticksize=None,
                 kw_profile=None, kw_cl=None, kw_legend=None, figsize=None, fig=None):
    r"""Plot 1-D profile likelihoods, one panel per parameter.

    Parameters
    ----------
    profiles : Profiles or list of Profiles
    params : list of str, optional
    offsets : float or list, default=0.
        Vertical offset(s) for each profile.
    nrows : int, default=1
    labels : str or list, optional
    colors : str or list, optional
    linestyles : str or list, optional
    cl : int or tuple, default=(1, 2, 3)
        :math:`\sigma` confidence levels to mark.
    labelsize : int, optional
    ticksize : int, optional
    kw_profile : dict, optional
    kw_cl : dict, optional
    kw_legend : dict, optional
    figsize : tuple, optional
    fig : matplotlib.figure.Figure, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles   = _make_list(profiles)
    params     = _get_default_profiles_params(profiles, params=params, of=('profile',))
    nprofiles  = len(profiles)
    offsets    = _make_list(offsets,    length=nprofiles, default=0.)
    labels     = _make_list(labels,     length=nprofiles, default=None)
    colors     = _make_list(colors,     length=nprofiles, default=None)
    linestyles = _make_list(linestyles, length=nprofiles, default=None)
    if np.ndim(cl) == 0:
        cl = [cl]
    add_lgd    = any(label is not None for label in labels)
    kw_profile = dict(kw_profile or {})
    kw_cl_base = dict(kw_cl if kw_cl is not None else {'color': 'k', 'linestyle': ':', 'linewidth': 2})
    xshift_cl  = kw_cl_base.pop('xhift', 0.9)
    kw_legend  = dict(kw_legend or {})

    ncols = int((len(params) + nrows - 1) / nrows)

    if fig is None:
        figsize = figsize or (4.0 * ncols, 4.0 * nrows)
        fig, lax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        lax = lax.ravel()
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
    else:
        lax = fig.axes

    for param_idx, name in enumerate(params):
        ax = lax[param_idx]
        for prof_idx, prof in enumerate(profiles):
            pro_dict = prof.get('profile', None)
            if pro_dict is None or name not in pro_dict:
                continue
            scan_vals, lp_vals = pro_dict[name]
            scan_vals = np.asarray(scan_vals)
            lp_vals   = np.asarray(lp_vals)
            ax.plot(scan_vals, -2.0 * (lp_vals - float(offsets[prof_idx])),
                    color=colors[prof_idx], linestyle=linestyles[prof_idx],
                    label=labels[prof_idx], **kw_profile)
        for nsigma in cl:
            y_level = _nsigmas_to_deltachi2(nsigma, ddof=1)
            ax.axhline(y=y_level, xmin=0., xmax=1., **kw_cl_base)
            ax.text(xshift_cl, y_level + 0.1, r'${:d}\sigma$'.format(nsigma),
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    color='k', fontsize=labelsize)
        ylim = ax.get_ylim()
        ax.set_ylim(0., ylim[-1] + 2.)
        ax.tick_params(labelsize=ticksize)
        ax.set_xlabel('${}$'.format(name), fontsize=labelsize)
        if param_idx == 0:
            ax.set_ylabel(r'$\Delta \chi^{2}$', fontsize=labelsize)
        if add_lgd and param_idx == 0:
            ax.legend(**kw_legend)

    return fig


def plot_profile_comparison(profiles, profiles_ref, params=None, labels=None, colors=None, **kwargs):
    r"""Plot profile-likelihood comparison.

    Both *profiles* and *profiles_ref* are offset by the minimum :math:`\chi^2`
    of *profiles*.

    Parameters
    ----------
    profiles : Profiles or list of Profiles
    profiles_ref : Profiles or list of Profiles
    params : list of str, optional
    labels : list of str, optional
    colors : list, optional
    **kwargs :
        Forwarded to :func:`plot_profile`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    profiles     = _make_list(profiles)
    profiles_ref = _make_list(profiles_ref)
    if len(profiles) != len(profiles_ref):
        raise ValueError('profiles_ref must have the same length as profiles')
    nprofiles  = len(profiles)
    labels     = _make_list(labels, length=nprofiles, default=None)
    colors     = _make_list(colors, length=nprofiles, default=None)
    offsets    = [prof.best['logpdf'].max() for prof in profiles] * 2
    colors     = colors * 2
    linestyles = ['-'] * nprofiles + ['--'] * nprofiles
    return plot_profile(profiles + profiles_ref, params=params,
                        offsets=offsets, labels=labels,
                        colors=colors, linestyles=linestyles, **kwargs)
