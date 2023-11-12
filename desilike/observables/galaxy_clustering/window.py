import numpy as np
from scipy import special

from desilike.jax import numpy as jnp
from desilike.jax import jit
from desilike.base import BaseCalculator
from desilike import plotting, utils


def window_matrix_bininteg(list_edges, resolution=1):
    r"""
    Build window matrix for binning, in the continuous limit, i.e. integral of :math:`\int dx x^2 f(x) / \int dx x^2` over each bin.

    Parameters
    ----------
    resolution : int, default=1
        Number of evaluation points in the integral.

    Returns
    -------
    xin : array
        Input theory coordinates.

    full_matrix : array
        Window matrix.
    """
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError('resolution must be a strictly positive integer')
    if np.ndim(list_edges[0]) == 0:
        list_edges = [list_edges]

    step = min(np.diff(edges).min() for edges in list_edges) / resolution
    start, stop = min(np.min(edges) for edges in list_edges), max(np.max(edges) for edges in list_edges)
    xin = np.arange(start + step / 2., stop, step)

    matrices = []
    for edges in list_edges:
        x, w = [], []
        for ibin, bin in enumerate(zip(edges[:-1], edges[1:])):
            x.append(np.linspace(*bin, resolution + 2)[1:-1])
            edge = np.linspace(*bin, resolution + 1)
            line = np.zeros((len(edges) - 1) * resolution, dtype='f8')
            line[ibin * resolution:(ibin + 1) * resolution] = (edge[1:]**2 - edge[:-1]**2) / (edge[-1]**2 - edge[0]**2)  # dx is constant
            w.append(line)
        matrices.append(utils.matrix_lininterp(xin, np.concatenate(x)).dot(np.column_stack(w)))  # linear interpolation * integration weights
    full_matrix = []
    for iin, matin in enumerate(matrices):
        line = []
        for i, mat in enumerate(matrices):
            if i == iin:
                line.append(mat)
            else:
                line.append(np.zeros_like(mat))
        full_matrix.append(line)
    full_matrix = np.bmat(full_matrix).A
    return xin, full_matrix


def unpack(x, flatarray):
    toret = []
    nout = 0
    for xx in x:
        sl = slice(nout, nout + len(xx))
        toret.append(flatarray[sl])
        nout = sl.stop
    return toret


class WindowedPowerSpectrumMultipoles(BaseCalculator):
    """
    Window effect on the power spectrum multipoles.

    Parameters
    ----------
    klim : dict, default=None
        Optionally, wavenumber limits: a dictionary mapping multipoles to (min separation, max separation, step (float)),
        e.g. ``{0: (0.01, 0.2, 0.01), 2: (0.01, 0.15, 0.01)}``. If ``None``, no selection is applied for the given multipole.

    k : array, default=None
        Optionally, observed wavenumbers :math:`k`, as an array or a list of such arrays (one for each multipole).
        If not specified, taken from ``wmatrix`` if provided;
        else defaults to edge centers, based on ``klim`` if provided;
        else defaults to ``np.arange(0.01, 0.205, 0.01)``.

    ells : tuple, default=None
        Observed multipoles.
        Defaults to poles in ``klim``, if provided; else ``(0, 2, 4)``.

    wmatrix : str, Path, pypower.BaseMatrix, dict, array, default=None
        Optionally, window matrix.
        Can be e.g. {'resolution': 2}, specifying the number of theory :math:`k` to integrate over per observed bin.
        If a 2D array (window matrix), output and input wavenumbers and multipoles ``k``, ``ells``, ``kin``, ``ellsin`` should be provided.

    kin : array, default=None
        If provided, linearly interpolate ``wmatrix`` along input wavenumbers to these wavenumbers,
        or if ``wmatrix`` is a 2D array, assume those are the input wavenumbers.

    kinrebin : int, default=1
        If ``wmatrix`` (which is defined for input theory wavenumbers) is provided,
        rebin theory wavenumbers by this factor.

    kinlim : tuple, default=None
        If ``wmatrix`` (which is defined for input theory wavenumbers) is provided,
        limit theory wavenumbers in the provided range.

    ellsin : tuple, default=None
        Optionally, input theory multipoles.
        If not specified, taken from ``wmatrix`` if provided, else ``(0, 2, 4)``.

    shotnoise : float, default=0.
        Shot noise (window matrix must be applied to power spectrum with shot noise).

    fiber_collisions : BaseFiberCollisionsPowerSpectrumMultipoles, default=None
        Optionally, fiber collisions.

    systematic_templates : SystematicTemplatePowerSpectrumMultipoles, default=None
        Optionally, systematic templates.

    theory : BaseTheoryPowerSpectrumMultipoles
        Theory power spectrum multipoles, defaults to :class:`KaiserTracerPowerSpectrumMultipoles`.
    """
    def initialize(self, klim=None, k=None, ells=None, wmatrix=None, kin=None, kinrebin=1, kinlim=None, ellsin=None, shotnoise=None, fiber_collisions=None, systematic_templates=None, theory=None):
        from scipy import linalg

        _default_step = 0.01

        if ells is None:
            if klim is not None:
                ells = klim.keys()
            else:
                ells = (0, 2, 4)
        self.ells = tuple(ells)
        self.kedges = None

        if klim is not None:
            klim = dict(klim)
            self.kedges = []
            for ell in self.ells:
                if klim[ell] is None:
                    self.kedges = None
                    break
                (lo, hi, *step) = klim[ell]
                if not step: step = (_default_step,)
                self.kedges.append(np.arange(lo, hi + step[0] / 2., step=step[0]))

        if self.kedges is None:
            self.kedges = np.arange(0.01 - _default_step / 2., 0.2 + _default_step, _default_step)  # gives k = np.arange(0.01, 0.2 + _default_step / 2., _default_step)

        if k is None:
            k = [(edges[:-1] + edges[1:]) / 2. for edges in self.kedges]

        if np.ndim(k[0]) == 0:
            k = [k] * len(self.ells)
        self.k = [np.array(kk, dtype='f8') for kk in k]
        if len(self.k) != len(self.ells):
            raise ValueError("Provided as many k's as ells")

        if theory is None:
            from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
            theory = KaiserTracerPowerSpectrumMultipoles()
        self.theory = theory

        self.ellsin = ellsin
        self.matrix_full, self.kmask, self.offset = None, None, None
        if wmatrix is None:
            self.kin = np.unique(np.concatenate(self.k, axis=0))
            if not all(kk.shape == self.kin.shape and np.allclose(kk, self.kin) for kk in self.k):
                self.kmask = [np.searchsorted(self.kin, kk, side='left') for kk in self.k]
                assert all(np.allclose(self.kin[kmask], kk) for kk, kmask in zip(self.k, self.kmask))
                self.kmask = np.concatenate(self.kmask, axis=0)
            self.ellsin = tuple(self.ells)
        elif isinstance(wmatrix, dict):
            self.ellsin = tuple(self.ells)
            self.kin, self.matrix_full = window_matrix_bininteg(self.kedges, **wmatrix)
            self.matrix_full = self.matrix_full.T
        elif isinstance(wmatrix, np.ndarray):
            self.ellsin = tuple(self.ells)
            self.matrix_full = np.array(wmatrix).T
            ksize = sum(len(k) for k in self.k)
            if self.matrix_full.shape[0] != ksize:
                raise ValueError('output "wmatrix" size is {:d}, but got {:d} output "k"'.format(self.matrix_full.shape[0], ksize))
            kin = np.asarray(kin).flatten()
            self.kin = kin.copy()
            if kinrebin is not None:
                self.kin = self.kin[::kinrebin]
            # print(wmatrix.xout[0], max(kk.max() for kk in self.k) * 1.2)
            if kinlim is not None:
                self.kin = self.kin[(self.kin >= kinlim[0]) & (self.kin <= kinlim[-1])]
            wmatrix_rebin = linalg.block_diag(*[utils.matrix_lininterp(self.kin, kin) for ell in self.ellsin])
            self.matrix_full = self.matrix_full.dot(wmatrix_rebin.T)
        else:
            if utils.is_path(wmatrix):
                from pypower import MeshFFTWindow, BaseMatrix
                fn = wmatrix
                wmatrix = MeshFFTWindow.load(fn)
                if hasattr(wmatrix, 'poles'):
                    wmatrix = wmatrix.poles
                else:
                    wmatrix = BaseMatrix.load(fn)
            else:
                wmatrix = wmatrix.deepcopy()
            # TODO: implement best match BaseMatrix method
            for iout, (projout, kk) in enumerate(zip(wmatrix.projsout, self.k)):
                ksize, factorout = None, 1
                if klim is not None:
                    lo, hi, *step = klim[projout.ell]
                    if step: ksize = int((hi - lo) / step[0] + 0.5)  # nearest integer
                else:
                    lo, hi, ksize = 2 * kk[0] - kk[1], 2 * kk[-1] - kk[-2], kk.size
                if ksize is not None:
                    nmk = np.sum((wmatrix.xout[iout] >= lo) & (wmatrix.xout[iout] <= hi))
                    factorout = nmk // ksize
                wmatrix.slice_x(sliceout=slice(0, len(wmatrix.xout[iout]) // factorout * factorout, factorout), projsout=projout)
                # wmatrix.slice_x(sliceout=slice(0, len(wmatrix.xout[iout]) // factorout * factorout), projsout=projout)
                # wmatrix.rebin_x(factorout=factorout, projsout=projout)
                if klim is not None:
                    istart = np.flatnonzero(wmatrix.xout[iout] >= lo)[0]
                    ksize = np.flatnonzero(wmatrix.xout[iout] <= hi)[-1] - istart + 1
                else:
                    istart = np.nanargmin(np.abs(wmatrix.xout[iout] - kk[0]))
                wmatrix.slice_x(sliceout=slice(istart, istart + ksize), projsout=projout)
                self.k[iout] = wmatrix.xout[iout]
                if klim is None and not np.allclose(wmatrix.xout[iout], kk, rtol=1e-4):
                    raise ValueError('k-coordinates {} for ell = {:d} could not be found in input matrix (rebinning = {:d})'.format(kk, projout.ell, factorout))
            if ellsin is not None:
                self.ellsin = list(ellsin)
            else:
                self.ellsin = []
                for proj in wmatrix.projsin:
                    assert proj.wa_order in (None, 0)
                    self.ellsin.append(proj.ell)
            projsin = [proj for proj in wmatrix.projsin if proj.ell in self.ellsin]
            self.ellsin = tuple(proj.ell for proj in projsin)
            wmatrix.select_proj(projsout=[(ell, None) for ell in self.ells], projsin=projsin)
            if kinrebin is not None:
                wmatrix.slice_x(slicein=slice(0, len(wmatrix.xin[0]) // kinrebin * kinrebin, kinrebin))
            # print(wmatrix.xout[0], max(kk.max() for kk in self.k) * 1.2)
            if kinlim is not None:
                wmatrix.select_x(xinlim=kinlim)
            self.kin = wmatrix.xin[0]
            self.matrix_full = wmatrix.value.T
            if kin is not None:
                self.kin = np.asarray(kin).flatten()
                wmatrix_rebin = linalg.block_diag(*[utils.matrix_lininterp(self.kin, xin) for xin in wmatrix.xin])
                self.matrix_full = self.matrix_full.dot(wmatrix_rebin.T)
            else:
                assert all(np.allclose(xin, self.kin) for xin in wmatrix.xin), 'input coordinates of "wmatrix" are not the same for all multipoles; pass an k-coordinate array to "kin"'
        if fiber_collisions is not None:
            self.theory.init.update(k=self.kin, ells=self.ellsin)  # fiber_collisions take kin, ellsin from theory
            fiber_collisions.init.update(k=self.kin, ells=self.ellsin, theory=self.theory)
            fiber_collisions = fiber_collisions.runtime_info.initialize()
            if self.matrix_full is None:
                self.matrix_full = np.bmat([list(kernel) for kernel in fiber_collisions.kernel_correlated]).A
                if fiber_collisions.with_uncorrelated: self.offset = fiber_collisions.kernel_uncorrelated.ravel()
                if self.kmask is not None:
                    self.matrix_full = self.matrix_full[..., self.kmask]
                    if fiber_collisions.with_uncorrelated: self.offset = self.offset[self.kmask]
            else:
                if fiber_collisions.with_uncorrelated: self.offset = self.matrix_full.dot(fiber_collisions.kernel_uncorrelated.ravel())
                self.matrix_full = self.matrix_full.dot(np.bmat([list(kernel) for kernel in fiber_collisions.kernel_correlated]).A)
            self.ellsin, self.kin = fiber_collisions.ellsin, fiber_collisions.kin
        if systematic_templates is not None:
            if not isinstance(systematic_templates, SystematicTemplatePowerSpectrumMultipoles):
                systematic_templates = SystematicTemplatePowerSpectrumMultipoles(systematic_templates)
            systematic_templates.init.update(k=self.k, ells=self.ells)
        self.systematic_templates = systematic_templates
        self.theory.init.update(k=self.kin, ells=self.ellsin)
        if shotnoise is None:
            shotnoise = 0.
        else:
            shotnoise = float(shotnoise)
            if 'shotnoise' in getattr(self.theory, '_default_options', {}):
                self.theory.init.update(shotnoise=shotnoise)
        self.shotnoise = np.array([shotnoise * (ell == 0) for ell in self.ellsin])
        self.flatshotnoise = np.concatenate([np.full_like(k, shotnoise * (ell == 0), dtype='f8') for ell, k in zip(self.ells, self.k)])

    @jit(static_argnums=[0])
    def _apply(self, theory):
        theory = jnp.ravel(theory)
        if self.matrix_full is not None:
            theory = jnp.dot(self.matrix_full, theory)
        if self.kmask is not None:
            theory = theory[self.kmask]
        if self.offset is not None:
            theory = theory + self.offset
        return theory

    def calculate(self):
        self.flatpower = self._apply(self.theory.power + self.shotnoise[:, None]) - self.flatshotnoise
        if self.systematic_templates is not None:
            self.flatpower += self.systematic_templates.flatpower

    def get(self):
        return self.flatpower

    @property
    def power(self):
        return unpack(self.k, self.flatpower)

    def __getstate__(self):
        state = {}
        for name in ['kin', 'ellsin', 'k', 'ells', 'kedges', 'fiducial', 'matrix_full', 'kmask', 'offset', 'flatpower', 'shotnoise', 'flatshotnoise']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot window function effect on power spectrum multipoles.

        Parameters
        ----------
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
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot([], [], linestyle='-', color='k', label='theory')
        ax.plot([], [], linestyle='--', color='k', label='window')
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            ax.plot([], [], linestyle='-', color=color, label=r'$\ell = {:d}$'.format(ell))
            k = self.k[ill]
            if ell in self.ellsin:
                illin = self.ellsin.index(ell)
                maskin = (self.kin >= k[0]) & (self.kin <= k[-1])
                ax.plot(self.kin[maskin], self.kin[maskin] * self.theory.power[illin][maskin], color=color, linestyle='-', label=None)
            ax.plot(k, k * self.power[ill], color=color, linestyle='--', label=None)
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


class WindowedCorrelationFunctionMultipoles(BaseCalculator):
    """
    Window effect (for now, none) on correlation function multipoles.

    Parameters
    ----------
    slim : dict, default=None
        Optionally, separation limits: a dictionary mapping multipoles to (min separation, max separation, (optionally) step (float)),
        e.g. ``{0: (30., 160., 5.), 2: (30., 160., 5.)}``. If ``None``, no selection is applied for the given multipole.

    s : array, default=None
        Optionally, observed separations :math:`s`, as an array or a list of such arrays (one for each multipole).
        If not specified, defaults to edge centers, based on ``slim`` if provided;
        else defaults to ``np.arange(20, 151, 5)``.

    ells : tuple, default=None
        Observed multipoles, defaults to ``(0, 2, 4)``.

    wmatrix : dict, default=None
        Can be e.g. {'resolution': 2}, specifying the number of theory :math:`s` to integrate over per observed bin.
        If a 2D array (window matrix), output and input separations and multipoles ``s``, ``ells``, ``sin``, ``ellsin`` should be provided.

    sin : array, default=None
        If ``wmatrix`` is a 2D array, assume those are the input separations.

    fiber_collisions : BaseFiberCollisionsCorrelationFunctionMultipoles, default=None
        Optionally, fiber collisions.

    systematic_templates : SystematicTemplateCorrelationFunctionMultipoles, default=None
        Optionally, systematic templates.

    theory : BaseTheoryCorrelationFunctionMultipoles
        Theory correlation function multipoles, defaults to :class:`KaiserTracerCorrelationFunctionMultipoles`.
    """
    def initialize(self, slim=None, s=None, ells=None, wmatrix=None, sin=None, sinrebin=1, sinlim=None, ellsin=None, fiber_collisions=None, systematic_templates=None, theory=None):
        from scipy import linalg
        _default_step = 5.

        if ells is None:
            if slim is not None:
                ells = slim.keys()
            else:
                ells = (0, 2, 4)
        self.ells = tuple(ells)

        self.sedges = None
        if slim is not None:
            slim = dict(slim)
            self.sedges = []
            for ell in self.ells:
                if slim[ell] is None:
                    self.sedges = None
                    break
                (lo, hi, *step) = slim[ell]
                if not step: step = (_default_step,)
                self.sedges.append(np.arange(lo, hi + step[0] / 2., step=step[0]))

        if self.sedges is None:
            self.sedges = np.arange(20. - _default_step / 2., 150 + _default_step, _default_step)  # gives s = np.arange(20, 150 + _default_step / 2., _default_step)

        if s is None:
            s = [(edges[:-1] + edges[1:]) / 2. for edges in self.sedges]

        if np.ndim(s[0]) == 0:
            s = [s] * len(self.ells)
        self.s = [np.array(ss, dtype='f8') for ss in s]
        if len(self.s) != len(self.ells):
            raise ValueError("Provided as many s's as ells")
        # No matrix for the moment
        self.ellsin = tuple(self.ells)

        if theory is None:
            from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles
            theory = KaiserTracerCorrelationFunctionMultipoles()
        self.theory = theory

        self.matrix_diag, self.matrix_full, self.smask, self.offset = None, None, None, None
        if wmatrix is None:
            self.sin = np.unique(np.concatenate(self.s, axis=0))
            if not all(np.allclose(ss, self.sin) for ss in self.s):
                self.smask = [np.searchsorted(self.sin, ss, side='left') for ss in self.s]
                assert all(smask.min() >= 0 and smask.max() < ss.size for ss, smask in zip(self.s, self.smask))
                self.smask = np.concatenate(self.smask, axis=0)
        elif isinstance(wmatrix, dict):
            wmatrix = dict(wmatrix)
            muedges = wmatrix.pop('muedges', None)
            self.ellsin = tuple(self.ells)
            self.sin, self.matrix_full = window_matrix_bininteg(self.sedges, **wmatrix)
            self.matrix_full = self.matrix_full.T
        elif isinstance(wmatrix, np.ndarray):
            self.ellsin = tuple(self.ells)
            self.matrix_full = np.array(wmatrix).T
            ssize = sum(len(s) for s in self.s)
            if self.matrix_full.shape[0] != ssize:
                raise ValueError('output "wmatrix" size is {:d}, but got {:d} output "s"'.format(self.matrix_full.shape[0], ssize))
            sin = np.asarray(sin).flatten()
            self.sin = sin.copy()
            if sinrebin is not None:
                self.sin = self.sin[::sinrebin]
            # print(wmatrix.xout[0], max(kk.max() for kk in self.k) * 1.2)
            if sinlim is not None:
                self.sin = self.sin[(self.sin >= sinlim[0]) & (self.sin <= sinlim[-1])]
            wmatrix_rebin = linalg.block_diag(*[utils.matrix_lininterp(self.sin, sin) for ell in self.ellsin])
            self.matrix_full = self.matrix_full.dot(wmatrix_rebin.T)
        else:
            raise ValueError('unrecognized wmatrix {}'.format(wmatrix))
        if fiber_collisions is not None:
            fiber_collisions.init.update(s=self.sin, ells=self.ellsin, theory=self.theory)
            fiber_collisions = fiber_collisions.runtime_info.initialize()
            self.ellsin = tuple(fiber_collisions.ellsin)
            if fiber_collisions.with_uncorrelated: self.offset = fiber_collisions.kernel_uncorrelated.ravel()
            if self.matrix_full is None:
                self.matrix_diag = fiber_collisions.kernel_correlated
                if self.smask is not None:
                    self.matrix_diag = self.matrix_diag[..., self.smask]
                    if fiber_collisions.with_uncorrelated: self.offset = self.offset[self.smask]
            else:
                # kernel_correlated is (ellout, ellin, s)
                self.matrix_full = self.matrix_full.dot(np.bmat([[np.diag(kk) for kk in kernel] for kernel in fiber_collisions.kernel_correlated]).A)
                self.ellsin, self.sin = fiber_collisions.ellsin, fiber_collisions.sin
        self.theory.init.update(s=self.sin, ells=self.ellsin)
        if systematic_templates is not None:
            if not isinstance(systematic_templates, SystematicTemplateCorrelationFunctionMultipoles):
                systematic_templates = SystematicTemplateCorrelationFunctionMultipoles(systematic_templates)
            systematic_templates.init.update(s=self.s, ells=self.ells)
        self.systematic_templates = systematic_templates

    @jit(static_argnums=[0])
    def _apply(self, theory):
        if self.matrix_diag is not None:
            theory = jnp.sum(self.matrix_diag * theory[None, ...], axis=1)
        theory = jnp.ravel(theory)
        if self.matrix_full is not None:
            theory = jnp.dot(self.matrix_full, theory)
        elif self.smask is not None:
            theory = theory[self.smask]
        if self.offset is not None:
            theory = theory + self.offset
        return theory

    def calculate(self):
        self.flatcorr = self._apply(self.theory.corr)
        if self.systematic_templates is not None:
            self.flatcorr += self.systematic_templates.flatcorr

    def get(self):
        return self.flatcorr

    @property
    def corr(self):
        return unpack(self.s, self.flatcorr)

    def __getstate__(self):
        state = {}
        for name in ['sin', 'ellsin', 's', 'ells', 'sedges', 'matrix_diag', 'matrix_full', 'smask', 'offset', 'flatcorr']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot window function effect on correlation function multipoles.

        Parameters
        ----------
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
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot([], [], linestyle='-', color='k', label='theory')
        ax.plot([], [], linestyle='--', color='k', label='window')
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            ax.plot([], [], linestyle='-', color=color, label=r'$\ell = {:d}$'.format(ell))
            s = self.s[ill]
            if ell in self.ellsin:
                illin = self.ellsin.index(ell)
                maskin = (self.sin >= s[0]) & (self.sin <= s[-1])
                ax.plot(self.sin[maskin], self.sin[maskin]**2 * self.theory.corr[illin][maskin], color=color, linestyle='-', label=None)
            ax.plot(s, s**2 * self.corr[ill], color=color, linestyle='--', label=None)
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig


class BaseFiberCollisionsPowerSpectrumMultipoles(BaseCalculator):

    def initialize(self, k=None, ells=(0, 2, 4), theory=None, with_uncorrelated=True):
        if k is None: k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)

        if theory is None:
            from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
            theory = KaiserTracerPowerSpectrumMultipoles()
        self.theory = theory
        self.theory.runtime_info.initialize()
        self.kin = np.array(self.theory.k, dtype='f8')
        self.ellsin = tuple(self.theory.ells)
        self.with_uncorrelated = bool(with_uncorrelated)

    def correlated(self, power):
        return jnp.sum(self.kernel_correlated * power[None, :, None, :], axis=(1, 3))

    def uncorrelated(self):
        if self.with_uncorrelated:
            return self.kernel_uncorrelated
        return np.zeros_like(self.kernel_uncorrelated)

    def calculate(self):
        self.power = self.correlated(self.theory.power) + self.uncorrelated()
        # self.power = self.theory.power + self.uncorrelated()

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot fiber collision effect on power spectrum multipoles.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot([], [], linestyle='-', color='k', label='theory')
        ax.plot([], [], linestyle='--', color='k', label='fiber collisions')
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            if ell in self.ellsin:
                illin = self.ellsin.index(ell)
                maskin = (self.kin >= self.k[0]) & (self.kin <= self.k[-1])
                ax.plot(self.kin[maskin], self.kin[maskin] * self.theory.power[illin][maskin], color=color, linestyle='-', label=None)
            ax.plot(self.k, self.k * self.power[ill], color=color, linestyle='--', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


def _format_kernel(sep, kernel):
    sep = np.array(sep, dtype='f8')
    kernel = np.array(kernel, dtype='f8')
    if kernel.size == 1:
        # Constant kernel for all sep
        kernel = np.full_like(sep, kernel.flat[0])
    if sep[0] > 0.:
        # Include 0
        sep = np.insert(sep, 0, 0.)
        kernel = np.insert(kernel, 0, kernel[0])
    return sep, kernel


class FiberCollisionsPowerSpectrumMultipoles(BaseFiberCollisionsPowerSpectrumMultipoles):
    r"""
    Fiber collision effect on power spectrum multipoles.
    Contrary to Hahn et al. 2016:

    - the kernel is assumed to be a sum of top-hat functions
    - no :math:`k D_{fc} \ll 1` approximation, where :math:`D_{fc}` is the fiber collision scale

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers.

    ells : tuple, default=(0, 2, 4)
        Multipoles.

    sep : array, default=None
        Transverse separation values for ``kernel``.

    kernel : array, default=None
        Window representing the number of pairs lost as a function of separation.

    theory : BaseTheoryPowerSpectrumMultipoles
        Theory power spectrum multipoles, defaults to :class:`KaiserTracerPowerSpectrumMultipoles`.

    with_uncorrelated : bool, default=True
        If ``False``, do not include the uncorrelated part (due to not correcting the selection function for missing pairs).


    Reference
    ---------
    https://arxiv.org/abs/1609.01714
    """
    def initialize(self, *args, sep=None, kernel=None, **kwargs):

        super(FiberCollisionsPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.sep, self.kernel = _format_kernel(sep=sep, kernel=kernel)

        def kernel_fourier(k):
            toret = 0.
            for isep in range(len(sep) - 1):
                x, yc = np.array(self.sep[isep:isep + 2]), np.mean(self.kernel[isep:isep + 2])
                # yc * integral(x J0(k x) dx)
                tmp = np.zeros_like(k)
                nonzero = k > 0.
                tmp[nonzero] = np.diff(yc / k[nonzero] * x[:, None] * special.j1(k[nonzero] * x[:, None]), axis=0)[0]
                tmp[~nonzero] = np.diff(yc * x[:, None]**2 / 2., axis=0)[0]
                toret += 2. * np.pi * tmp
            return toret

        self.kernel_uncorrelated = - np.array([np.pi * (2. * ellout + 1.) * special.legendre(ellout)(0.) for ellout in self.ells])[:, None] * kernel_fourier(self.k) / self.k
        # phi: angle between k_perp and q_perp
        phi = np.linspace(0., np.pi, 100)
        k_perp = np.linspace(0., self.k[-1], len(self.k))
        q_perp = np.linspace(0., self.kin[-1], len(self.kin))

        kk, qq = np.meshgrid(k_perp, q_perp, indexing='ij')
        integral_kernel = 0.
        for pp, ww in zip(phi, utils.weights_trapz(phi) / (2. * np.pi)):
            kq_perp = np.sqrt(kk**2 - 2. * kk * qq * np.cos(pp) + qq**2)
            integral_kernel += 2. * ww * kernel_fourier(kq_perp)
        from scipy.interpolate import RectBivariateSpline
        interp_kernel = RectBivariateSpline(k_perp, q_perp, integral_kernel, kx=3, ky=3, s=0)

        wq = utils.weights_trapz(self.kin)
        diag = utils.matrix_lininterp(self.kin, self.k).T
        self.kernel_correlated = []
        for ellout in self.ells:
            legout = special.legendre(ellout)
            self.kernel_correlated.append([])
            for ellin in self.ellsin:
                legin = special.legendre(ellin)
                fll = np.zeros((len(self.k), len(self.kin)), dtype='f8')
                for ik, kk in enumerate(self.k):
                    mu = np.linspace(0., 1., 50)
                    mu = mu[:, None] * np.clip(self.kin / kk, None, 1.)
                    if (ellout + ellin) % 2 == 0:
                        # Exploit mu-symmetry
                        wmu = 2. * utils.weights_trapz(mu)
                    else:
                        mu = np.concatenate([-mu[::-1], mu[1:]], axis=0)
                        wmu = utils.weights_trapz(mu)
                    k_perp = np.sqrt(1. - mu**2) * kk
                    q_perp = np.sqrt(np.clip(self.kin**2 - (kk * mu)**2, 0., None))
                    fll[ik, :] = np.sum(legout(mu) * legin(kk / self.kin * mu) * interp_kernel(k_perp, q_perp, grid=False) * wmu, axis=0)
                self.kernel_correlated[-1].append((ellin == ellout) * diag - (2. * ellout + 1) / (4. * np.pi) * fll * self.kin * wq)
        self.kernel_correlated = np.array(self.kernel_correlated)

    def to_tophat(self):
        fs = np.trapz(self.kernel, x=self.sep) / np.trapz(self.sep, x=self.sep)
        Dfc = 2. * np.trapz(self.sep * self.kernel, x=self.sep) / np.trapz(self.kernel, x=self.sep)
        return TopHatFiberCollisionsPowerSpectrumMultipoles(k=self.k, kin=self.kin, ells=self.ells, ellsin=self.ellsin, theory=self.theory, fs=fs, Dfc=Dfc)


class TopHatFiberCollisionsPowerSpectrumMultipoles(BaseFiberCollisionsPowerSpectrumMultipoles):
    r"""
    Fiber collision effect on power spectrum multipoles, exactly following Hahn et al. 2016:

    - top-hat shape is assumed for the kernel
    - :math:`k D_{fc} \ll 1` approximation, where :math:`D_{fc}` is the fiber collision scale

    Parameters
    ----------
    k : array, default=None
        Output wavenumbers.

    ells : tuple, default=(0, 2, 4)
        Multipoles.

    fs : float, default=1.
        Fraction of pairs lost below the fiber collision scale ``Dfc``.

    Dfc : float, default=0.
        Fiber collision scale (transverse separation).

    theory : BaseTheoryPowerSpectrumMultipoles
        Theory power spectrum multipoles, defaults to :class:`KaiserTracerPowerSpectrumMultipoles`.

    with_uncorrelated : bool, default=True
        If ``False``, do not include the uncorrelated part (due to not correcting the selection function for missing pairs).


    Reference
    ---------
    https://arxiv.org/abs/1609.01714
    """
    def initialize(self, *args, fs=1., Dfc=0., **kwargs):

        super(TopHatFiberCollisionsPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.fs = float(fs)
        self.Dfc = float(Dfc)

        # Appendix Hahn 2016 arXiv:1609.01714v1
        def W2D(x):
            return 2. * special.j1(x) / x

        def H(*ells):
            if ells == (2, 0):
                return lambda x: x**2 - 1.
            if ells == (4, 0):
                return lambda x: 7. / 4. * x**4 - 5. / 2. * x**2 + 3. / 4.
            if ells == (4, 2):
                return lambda x: x**4 - x**2
            if ells == (6, 0):
                return lambda x: 33. / 8. * x**6 - 63. / 8. * x**4 + 35. / 8. * x**2 - 5. / 8.
            if ells == (6, 2):
                return lambda x: 11. / 4. * x**6 - 9. / 2. * x**4 + 7. / 4. * x**2
            if ells == (6, 4):
                return lambda x: x**6 - x**4

        self.kernel_uncorrelated = - np.array([(2. * ellout + 1.) * special.legendre(ellout)(0.) for ellout in self.ells])[:, None] * self.fs * (np.pi * self.Dfc)**2 / self.k * W2D(self.k * self.Dfc)

        kk, qq = np.meshgrid(self.k, self.kin, indexing='ij')
        wq = utils.weights_trapz(self.kin)
        diag = utils.matrix_lininterp(self.kin, self.k).T
        self.kernel_correlated = []
        for ellout in self.ells:
            self.kernel_correlated.append([])
            for ellin in self.ellsin:
                if ellin == ellout:
                    tmp = qq / kk
                    tmp[tmp > 1.] = 1.
                    fll = tmp * W2D(qq * self.Dfc) * (np.min([kk, qq], axis=0) / np.max([kk, qq], axis=0))**ellout
                else:
                    tmp = qq / kk
                    tmp[tmp > 1.] = 1.
                    tmp = tmp * W2D(qq * self.Dfc) * (2. * ellout + 1.) / 2. * H(max(ellout, ellin), min(ellout, ellin))(np.min([kk, qq], axis=0) / np.max([kk, qq], axis=0))
                    fll = np.zeros_like(tmp)
                    nonzero = ((ellout >= ellin) & (kk >= qq)) | ((ellout <= ellin) & (kk <= qq))
                    fll[nonzero] = tmp[nonzero]
                self.kernel_correlated[-1].append((ellin == ellout) * diag - self.fs * self.Dfc**2 / 2. * fll * self.kin * wq)
        self.kernel_correlated = np.array(self.kernel_correlated)


class BaseFiberCollisionsCorrelationFunctionMultipoles(BaseCalculator):

    def initialize(self, s=None, ells=(0, 2, 4), theory=None, with_uncorrelated=True):
        if s is None: s = np.linspace(20., 200, 101)
        self.s = np.array(s, dtype='f8')
        self.ells = tuple(ells)

        if theory is None:
            from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles
            theory = KaiserTracerCorrelationFunctionMultipoles(s=self.s)
        self.theory = theory
        self.theory.runtime_info.initialize()
        self.ellsin = tuple(self.theory.ells)
        self.with_uncorrelated = bool(with_uncorrelated)

    @property
    def sin(self):
        return self.s

    def correlated(self, corr):
        return jnp.sum(self.kernel_correlated * corr[None, ...], axis=1)

    def uncorrelated(self):
        if self.with_uncorrelated:
            return self.kernel_uncorrelated
        return np.zeros_like(self.kernel_uncorrelated)

    def calculate(self):
        self.corr = self.correlated(self.theory.corr) + self.uncorrelated()

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot fiber collision effect on correlation function multipoles.

        Parameters
        ----------
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
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot([], [], linestyle='-', color='k', label='theory')
        ax.plot([], [], linestyle='--', color='k', label='fiber collisions')
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            ax.plot(self.s, self.s**2 * self.theory.corr[ill], color=color, linestyle='-', label=None)
            ax.plot(self.s, self.s**2 * self.corr[ill], color=color, linestyle='--', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig


def integral_cosn(n=0, range=(-np.pi, np.pi)):
    if n == 0:
        return np.diff(range, axis=0)[0]
    if n == 1:
        return np.diff(np.sin(range), axis=0)[0]
    return (np.diff(np.sin(range) * np.cos(range)**(n - 1), axis=0)[0] + (n - 1) * integral_cosn(n=n - 2, range=range)) / n


class FiberCollisionsCorrelationFunctionMultipoles(BaseFiberCollisionsCorrelationFunctionMultipoles):
    r"""
    Fiber collision effect on correlation function multipoles.
    Contrary to Hahn et al. 2016, the kernel is assumed to be a sum of top-hat functions.

    Parameters
    ----------
    s : array, default=None
        Output separations.

    ells : tuple, default=(0, 2, 4)
        Multipoles.

    sep : array, default=None
        Transverse separation values for ``kernel``.

    kernel : array, default=None
        Window representing the number of pairs lost as a function of separation.

    theory : BaseTheoryPowerSpectrumMultipoles
        Theory correlation function multipoles, defaults to :class:`KaiserTracerCorrelationFunctionMultipoles`.

    with_uncorrelated : bool, default=True
        If ``False``, do not include the uncorrelated part (due to not correcting the selection function for missing pairs).

    Reference
    ---------
    https://arxiv.org/abs/1609.01714
    """
    def initialize(self, *args, sep=None, kernel=None, **kwargs):

        super(FiberCollisionsCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)

        self.sep, self.kernel = _format_kernel(sep=sep, kernel=kernel)

        def trapz_poly(poly):
            integ = poly.integ()
            toret = 0.
            for isep in range(len(sep) - 1):
                x, yc = np.array(self.sep[isep:isep + 2]), np.mean(self.kernel[isep:isep + 2])
                mu_min = np.sqrt(np.clip(1. - (x[:, None] / self.s)**2, 0., None))
                toret += yc * np.diff(integ(1.) - integ(mu_min) + integ(-mu_min) - integ(-1.), axis=0)[0]
            return toret

        self.kernel_uncorrelated = - np.array([(2. * ellout + 1.) / 2. * trapz_poly(special.legendre(ellout)) for ellout in self.ells])
        self.kernel_correlated = []
        for ellout in self.ells:
            self.kernel_correlated.append([])
            for ellin in self.ellsin:
                poly = (special.legendre(ellout) * special.legendre(ellin))
                fll = (2. * ellout + 1.) / 2. * trapz_poly(poly)
                self.kernel_correlated[-1].append((ellin == ellout) * 1. - fll)
        self.kernel_correlated = np.array(self.kernel_correlated)

    def to_tophat(self):
        fs = np.trapz(self.kernel, x=self.sep) / np.trapz(self.sep, x=self.sep)
        Dfc = 2. * np.trapz(self.sep * self.kernel, x=self.sep) / np.trapz(self.kernel, x=self.sep)
        return TopHatFiberCollisionsCorrelationFunctionMultipoles(s=self.s, ells=self.ells, theory=self.theory, fs=fs, Dfc=Dfc)


class TopHatFiberCollisionsCorrelationFunctionMultipoles(BaseFiberCollisionsCorrelationFunctionMultipoles):
    r"""
    Fiber collision effect on correlation function multipoles, exactly following Hahn et al. 2016, assuming top-hat shape for the kernel.

    Parameters
    ----------
    s : array, default=None
        Output separations.

    ells : tuple, default=(0, 2, 4)
        Multipoles.

    fs : float, default=1.
        Fraction of pairs lost below the fiber collision scale ``Dfc``.

    Dfc : float, default=0.
        Fiber collision scale (transverse separation).

    theory : BaseTheoryPowerSpectrumMultipoles
        Theory correlation function multipoles, defaults to :class:`KaiserTracerCorrelationFunctionMultipoles`.

    with_uncorrelated : bool, default=True
        If ``False``, do not include the uncorrelated part (due to not correcting the selection function for missing pairs).

    mu_range_cut : bool, default=False
        If ``True``, normalize the Legendre integral by the uncut :math:`\mu` range (instead of ``2``):
        in case the :math:`R1R2` counts are cut by the tophat kernel in the estimation of correlation function multipoles.


    Reference
    ---------
    https://arxiv.org/abs/1609.01714
    """
    def initialize(self, *args, fs=1., Dfc=0., mu_range_cut=False, **kwargs):

        super(TopHatFiberCollisionsCorrelationFunctionMultipoles, self).initialize(*args, **kwargs)
        self.fs = float(fs)
        self.Dfc = float(Dfc)
        self.mu_range_cut = bool(mu_range_cut)

        mu_min = np.sqrt(np.clip(1. - (self.Dfc / self.s)**2, 0., None))

        def trapz_poly(poly):
            integ = poly.integ()
            toret = integ(1.) - integ(mu_min) + integ(-mu_min) - integ(-1.)
            return toret

        self.kernel_uncorrelated = - np.array([(2 * ellout + 1.) / 2. * self.fs * trapz_poly(special.legendre(ellout)) for ellout in self.ells])

        self.kernel_correlated = []
        for ellout in self.ells:
            self.kernel_correlated.append([])
            for ellin in self.ellsin:
                fll = (2 * ellout + 1.) / 2. * self.fs * trapz_poly(special.legendre(ellout) * special.legendre(ellin))
                tmp = (ellin == ellout) * 1. - fll
                if self.mu_range_cut:
                    tmp[mu_min > 0.] /= mu_min[mu_min > 0.]
                self.kernel_correlated[-1].append(tmp)
        self.kernel_correlated = np.array(self.kernel_correlated)


def get_templates(templates, ells=(0, 2, 4), x=None):
    from collections.abc import Mapping
    if not isinstance(templates, Mapping) and not utils.is_sequence(templates):
        templates = [templates]
    if not isinstance(templates, Mapping):
        templates = {'syst_{:d}'.format(i): v for i, v in enumerate(templates)}
    toret = {}
    for name, template in templates.items():
        if x is not None:
            if callable(template):
                template = np.concatenate([template(ell, xx) for ell, xx in zip(ells, x)])
            template = np.ravel(template)
            sizes = [xx.size for xx in x]
            size = sum(sizes)
            if template.size != size:
                raise ValueError('provided template is size {:d}, but expected {:d} = sum({:d})'.format(template.size, size, sizes))
        toret[name] = template
    return toret


class BaseSystematicTemplateMultipoles(BaseCalculator):

    @staticmethod
    def _params(params, templates=None):
        names = list(get_templates(templates=templates).keys())
        for iname, name in enumerate(names):
            params[name] = dict(value=0., ref=dict(limits=[-1e-3, 1e-3]), delta=0.005, latex='s_{{{:d}}}'.format(iname))
        return params

    @jit(static_argnums=[0])
    def _apply(self, **params):
        return jnp.array([params[name] for name in self.templates]).dot(jnp.array(list(self.templates.values())))


class SystematicTemplatePowerSpectrumMultipoles(BaseSystematicTemplateMultipoles):
    r"""
    Systematic templates for power spectrum multipoles.

    Parameters
    ----------
    templates : callable, list or dict, default=()
        List of templates; one parameter called 'syst_{i:d}' will be created for each template i.
        or dict of templates; the key will be used as parameter name.
        Each template can be an array for all multipoles stacked, or a callable that takes (ell, k) as input
        and returns an array of size ``k.size``.

    k : array, list, default=None
        Output wavenumbers. If list, one array for each multipole.

    ells : tuple, default=(0, 2, 4)
        Multipoles.

    """
    def initialize(self, templates=tuple(), k=None, ells=(0, 2, 4)):
        self.ells = tuple(ells)
        if k is None: k = np.linspace(0.01, 0.2, 101)
        if not isinstance(k, (tuple, list)):
            k = [k] * len(self.ells)
        self.k = tuple(k)
        self.templates = get_templates(templates, ells=self.ells, x=self.k)

    def calculate(self, **params):
        self.flatpower = self._apply(**params)

    @property
    def power(self):
        return unpack(self.k, self.flatpower)

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot systematic template multipoles.

        Parameters
        ----------
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
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            k = self.k[ill]
            ax.plot(k, k * self.power[ill], color=color, linestyle='-', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        return fig


class SystematicTemplateCorrelationFunctionMultipoles(BaseSystematicTemplateMultipoles):
    r"""
    Systematic templates for correlation function multipoles.

    Parameters
    ----------
    templates : callable, list or dict, default=()
        List of templates; one parameter called 'syst_{i:d}' will be created for each template i.
        or dict of templates; the key will be used as parameter name.
        Each template can be an array for all multipoles stacked, or a callable that takes (ell, s) as input
        and returns an array of size ``s.size``.

    s : array, list, default=None
        Output separations. If list, one array for each multipole.

    ells : tuple, default=(0, 2, 4)
        Multipoles.

    """
    def initialize(self, templates=tuple(), s=None, ells=(0, 2, 4)):
        self.ells = tuple(ells)
        if s is None: s = np.linspace(20., 200, 101)
        if not isinstance(s, (tuple, list)):
            s = [s] * len(self.ells)
        self.s = tuple(s)
        self.templates = get_templates(templates, ells=self.ells, x=self.s)

    def calculate(self, **params):
        self.flatcorr = self._apply(**params)

    @property
    def corr(self):
        return unpack(self.s, self.flatcorr)

    @plotting.plotter
    def plot(self, fig=None):
        """
        Plot systematic template multipoles.

        Parameters
        ----------
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
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            color = 'C{:d}'.format(ill)
            s = self.s[ill]
            ax.plot(s, s**2 * self.corr[ill], color=color, linestyle='--', label=r'$\ell = {:d}$'.format(ell))
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        return fig