"""desilike.samples — MCMC chains and profiling results."""

from .samples import Samples
from .chain import MCSamples
from .profiles import Profiles
from .covariance import Covariance, Precision
from . import diagnostics, plotting
from .diagnostics import (
    gelman_rubin,
    autocorrelation,
    integrated_autocorrelation_time,
    geweke,
)
from .plotting import (
    plotter,
    plot_trace,
    plot_gelman_rubin,
    plot_geweke,
    plot_autocorrelation_time,
    add_legend,
    add_1d_profile,
    add_2d_contour,
    plot_triangle_contours,
    plot_triangle,
    plot_aligned,
    plot_aligned_stacked,
    plot_profile,
    plot_profile_comparison,
)
