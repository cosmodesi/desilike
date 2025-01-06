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
    #xin = np.arange(start + step / 2., stop, step)
    edges = np.arange(start, stop + step / 2., step)
    xin = 3. / 4. * (edges[1:]**4 - edges[:-1]**4) / (edges[1:]**3 - edges[:-1]**3)
    #print(xin)

    matrices = []
    for edges in list_edges:
        x, w = [], []
        for ibin, bin in enumerate(zip(edges[:-1], edges[1:])):
            edge = np.linspace(*bin, resolution + 1)
            #x.append((edge[1:] + edge[:-1]) / 2.)
            x.append(3. / 4. * (edge[1:]**4 - edge[:-1]**4) / (edge[1:]**3 - edge[:-1]**3))
            line = np.zeros((len(edges) - 1) * resolution, dtype='f8')
            tmp = edge[1:]**3 - edge[:-1]**3
            line[ibin * resolution:(ibin + 1) * resolution] = tmp / tmp.sum()
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


def window_matrix_RR(soutedges, sedges, muedges, wcounts, ellsin=(0, 2, 4), resolution=1):
    r"""
    Build window matrix for binning, in the continuous limit, i.e. integral of :math:`\int dx x^2 f(x) / \int dx x^2` over each bin.

    Parameters
    ----------
    soutedges : dict
        :math:`s`-edges, for each output :math:`\ell`.

    muedges : array
        :math:`\mu`-edges.

    wcounts : 2D array
        RR (weighted) pair counts, of shape ``(len(sedges) - 1, len(muedges) - 1)``.

    ellsin : tuple, default=(0, 2, 4)
        Input, theory, :math:`\ell`.

    resolution : int, default=1
        Number of evaluation points in the integral, in addition to the rebinning factor ``(len(sedges) - 1) / (len(list_sedges[0]) - 1)``.

    Returns
    -------
    sin : array
        Input theory coordinates.

    full_matrix : array
        Window matrix.
    """
    sin, binmatrix = window_matrix_bininteg(sedges, resolution=resolution)  # binmatrix shape (len(sin), len(sinedges) - 1)
    full_matrix, idxin = [], []
    for ellout, soutedges in soutedges.items():
        idx = np.flatnonzero(sedges == soutedges[0])
        if not idx.size:
            raise ValueError('output edges {} not found in RR s-edges {}'.format(soutedges, sedges))
        idx = idx[0]
        diffout = soutedges[1] - soutedges[0]
        diffin = sedges[idx + 1] - sedges[idx]
        factor = np.rint(diffout / diffin).astype('i4')
        if factor == 0:
            raise ValueError('s-resolution {} of RR counts is larger than required output s-binning {}'.format(sedges, soutedges))
        line = []
        for ellin in ellsin:
            integ = (special.legendre(ellout) * special.legendre(ellin)).integ()
            matrix = np.zeros((len(sedges) - 1, len(soutedges) - 1), dtype='f8')
            #print(idx)
            for iout in range(matrix.shape[1]):
                iin = idx + factor * iout
                wc = wcounts[iin:iin + factor]
                wcmu = np.sum(wc, axis=0)
                mask_nonzero = wcmu != 0.
                wcmu[~mask_nonzero] = 1.
                tmp = wc / wcmu
                # Integration over mu
                tmp = (2. * ellout + 1.) * np.sum(tmp * mask_nonzero * (integ(muedges[1:]) - integ(muedges[:-1])), axis=-1) / np.sum(mask_nonzero * (muedges[1:] - muedges[:-1]))  # normalization of mu-integral over non-empty s-rebinned RR(s, mu) bins
                #if ellout != ellin:
                #    print(ellin, ellout, mask_nonzero.all(), integ(muedges[-1]) - integ(muedges[0]), np.abs(tmp).max())
                matrix[iin:iin + factor, iout] = tmp
            matrix = binmatrix.dot(matrix)
            line.append(matrix.T)
        full_matrix.append(line)
        idxin.append(np.any([np.any(matrix != 0, axis=0) for matrix in line], axis=0))  # all sin for which wmatrix != 0
    idxin = np.any(idxin, axis=0)
    # Let's remove useless sin
    sin = sin[idxin]
    full_matrix = [[matrix[:, idxin] for matrix in line] for line in full_matrix]
    full_matrix = np.bmat(full_matrix).A
    return sin, full_matrix.T


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

    wshotnoise : float, default=None
        Optionally, response of the window to a shot noise.

    fiber_collisions : BaseFiberCollisionsPowerSpectrumMultipoles, default=None
        Optionally, fiber collisions.

    systematic_templates : SystematicTemplatePowerSpectrumMultipoles, default=None
        Optionally, systematic templates.

    theory : BaseTheoryPowerSpectrumMultipoles
        Theory power spectrum multipoles, defaults to :class:`KaiserTracerPowerSpectrumMultipoles`.
    """
    def initialize(self, klim=None, k=None, kedges=None, ells=None, wmatrix=None, kin=None, kinrebin=1, kinlim=None, ellsin=None, shotnoise=None, wshotnoise=None, fiber_collisions=None, systematic_templates=None, theory=None):
        from scipy import linalg

        _default_step = 0.01

        if ells is None:
            if klim is not None: ells = list(klim)
            else: ells = (0, 2, 4)
        self.ells = tuple(ells)

        self.k = self.kmasklim = self.kedges = None
        if k is not None:
            if np.ndim(k[0]) == 0:
                k = [k] * len(self.ells)
            self.k = [np.array(kk, dtype='f8') for kk in k]
            if len(self.k) != len(self.ells):
                raise ValueError("provide as many k's as ells")
        input_klim = klim is not None
        if kedges is not None:
            if np.ndim(kedges[0]) == 0:
                kedges = [kedges] * len(self.ells)
            self.kedges = [np.array(kk, dtype='f8') for kk in kedges]
            #print(self.kedges, self.ells)
            if len(self.kedges) != len(self.ells):
                raise ValueError("provide as many kedges as ells")
            if klim is None:
                klim = {ell: (edges[0], edges[-1], np.mean(np.diff(edges))) for ell, edges in zip(self.ells, self.kedges)}

        if input_klim:
            klim = dict(klim)
            if self.k is not None:
                k, ells, self.kmasklim = [], [], {}
                for ill, ell in enumerate(self.ells):
                    kk = self.k[ill]
                    self.kmasklim[ell] = np.zeros(len(kk), dtype='?')
                    if ell not in klim: continue  # do not take this multipole
                    self.kmasklim[ell][...] = True
                    lim = klim[ell]
                    if lim is not None:  # cut
                        (lo, hi, *step) = klim[ell]
                        kmask = (kk >= lo) & (kk <= hi)
                        kk = kk[kmask]  # scale cuts
                        self.kmasklim[ell][...] = kmask
                    if kk.size:
                        k.append(kk)
                        ells.append(ell)
                self.k, self.ells = k, tuple(ells)
            elif list(self.ells) != list(klim):
                raise ValueError('incompatible ells = {} and klim = {}; just remove ells?', self.ells, list(klim))
            kedges = []
            for ill, ell in enumerate(self.ells):
                if klim[ell] is None:
                    kedges = None
                    break
                (lo, hi, *step) = klim[ell]
                if not step:
                    if self.k is not None: step = ((hi - lo) / self.k[ill].size,)
                    else: step = (_default_step,)
                kedges.append(np.arange(lo, hi + step[0] / 2., step=step[0]))
            if self.kedges is None: self.kedges = kedges
        else:
            klim = {ell: None for ell in self.ells}

        if self.kedges is None:
            if self.k is not None:
                self.kedges = []
                for xx in self.k:
                    tmp = (xx[:-1] + xx[1:]) / 2.
                    tmp = np.concatenate([[tmp[0] - (xx[1] - xx[0])], tmp, [tmp[-1] + (xx[-1] - xx[-2])]])
                    self.kedges.append(tmp)
            else:
                self.kedges = [np.arange(0.01 - _default_step / 2., 0.2 + _default_step, _default_step)] * len(self.ells)  # gives k = np.arange(0.01, 0.2 + _default_step / 2., _default_step)

        if self.k is None:
            self.k = [(edges[:-1] + edges[1:]) / 2. for edges in self.kedges]
        self.k = [np.array(kk) for kk in self.k]

        if theory is None:
            from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles
            theory = KaiserTracerPowerSpectrumMultipoles()
        self.theory = theory

        self.matrix_full, self.kmask, self.offset = None, None, None
        if wmatrix is None:
            self.ellsin = tuple(self.ells)
            self.kin = np.unique(np.concatenate(self.k, axis=0))
            if not all(kk.shape == self.kin.shape and np.allclose(kk, self.kin) for kk in self.k):
                self.kmask = [np.searchsorted(self.kin, kk, side='left') for kk in self.k]
                assert all(np.allclose(self.kin[kmask], kk) for kk, kmask in zip(self.k, self.kmask)), self.k
                self.kmask = [self.kin.size * i + kmask for i, kmask in enumerate(self.kmask)]
                self.kmask = np.concatenate(self.kmask, axis=0)
        elif isinstance(wmatrix, dict):
            self.ellsin = tuple(self.ells)
            if kedges is None: raise ValueError('provide kedges or klim to compute the binning matrix')
            self.kin, matrix_full = window_matrix_bininteg(self.kedges, **wmatrix)
            self.matrix_full = matrix_full.T
        elif isinstance(wmatrix, np.ndarray):
            self.ellsin = tuple(ellsin or self.ells)
            matrix_full = np.array(wmatrix, dtype='f8')
            ksize = sum(len(k) for k in self.k)
            if matrix_full.shape[0] != ksize:
                raise ValueError('output "wmatrix" size is {:d}, but got {:d} output "k"'.format(matrix_full.shape[0], ksize))
            kin = np.asarray(kin).flatten()
            self.kin = kin.copy()
            if kinrebin is not None:
                self.kin = self.kin[::kinrebin]
            # print(wmatrix.xout[0], max(kk.max() for kk in self.k) * 1.2)
            if kinlim is not None:
                self.kin = self.kin[(self.kin >= kinlim[0]) & (self.kin <= kinlim[-1])]
            wmatrix_rebin = linalg.block_diag(*[utils.matrix_lininterp(self.kin, kin) for ell in self.ellsin])
            self.matrix_full = matrix_full.dot(wmatrix_rebin.T)
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
            for illout, (kk, ellout) in enumerate(zip(self.k, self.ells)):
                for iout, projout in enumerate(wmatrix.projsout):
                    if projout.ell == ellout: break
                if projout.ell != ellout:
                    raise ValueError('ell = {:d} not found in wmatrix.projsout = {}'.format(ellout, wmatrix.projsout))
                lim = klim.get(projout.ell, None)
                if lim is not None:
                    lo, hi, *step = lim
                else:
                    lo, hi = 2 * kk[0] - kk[1], 2 * kk[-1] - kk[-2]
                xwout = wmatrix.xout[iout]
                isnan = np.isnan(xwout)
                if isnan.any():
                    isnan = np.isnan(xwout)
                    kkisnan = np.interp(np.flatnonzero(isnan), np.flatnonzero(~isnan), xwout[~isnan])
                    #if self.mpicomm.rank == 0:
                    #    self.log_warning('NaN found in k: {}, replaced by {}.'.format(xwout, kkisnan))
                    xwout[isnan] = kkisnan
                wmatrix.xout[iout] = xwout

                for factorout in range(1, xwout.size // kk.size + 1):
                    wmat = wmatrix.deepcopy()
                    wmat.slice_x(sliceout=slice(0, len(wmat.xout[iout]) // factorout * factorout, factorout), projsout=projout)
                    # wmatrix.slice_x(sliceout=slice(0, len(xwout) // factorout * factorout), projsout=projout)
                    # wmatrix.rebin_x(factorout=factorout, projsout=projout)
                    xwout = wmat.xout[iout]
                    if lim is not None:
                        istart = np.flatnonzero(xwout >= lo)[0]
                        ksize = np.flatnonzero(xwout <= hi)[-1] - istart + 1
                    else:
                        istart = np.nanargmin(np.abs(xwout - kk[0]))
                        ksize = np.nanargmin(np.abs(xwout - kk[-1])) - istart + 1
                    wmat.slice_x(sliceout=slice(istart, istart + ksize), projsout=projout)
                    if ksize == kk.size:
                        wmatrix = wmat
                        break
                self.k[illout] = xwout = wmatrix.xout[iout]
                isfinite = np.isfinite(xwout) & np.isfinite(kk)
                if ksize != xwout.size or (lim is None and not np.allclose(xwout[isfinite], kk[isfinite], rtol=1e-4)):
                    raise ValueError('k-coordinates {} for ell = {:d} could not be found in input matrix (rebinning = {:d}, best guess = {})'.format(kk, projout.ell, factorout, xwout))
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
            if wshotnoise is None:
                wshotnoise = getattr(wmatrix, 'vectorout', None)
                if wshotnoise is not None: wshotnoise = np.concatenate(wshotnoise, axis=0)
            if kin is not None:
                self.kin = np.asarray(kin).flatten()
                wmatrix_rebin = linalg.block_diag(*[utils.matrix_lininterp(self.kin, xin) for xin in wmatrix.xin])
                self.matrix_full = self.matrix_full.dot(wmatrix_rebin.T)
            else:
                assert all(np.allclose(xin, self.kin) for xin in wmatrix.xin), 'input coordinates of "wmatrix" are not the same for all multipoles; pass an k-coordinate array to "kin"'
        if fiber_collisions is not None:
            self.theory.init.update(k=self.kin, ells=self.ellsin)  # fiber_collisions takes kin, ellsin from theory
            fiber_collisions.init.update(k=self.kin, ells=self.ellsin, theory=self.theory)
            fiber_collisions = fiber_collisions.runtime_info.initialize()
            if self.matrix_full is None:  # ellsin = ells
                if fiber_collisions.with_uncorrelated: self.offset = fiber_collisions.kernel_uncorrelated.ravel()
                self.matrix_full = np.bmat([list(kernel) for kernel in fiber_collisions.kernel_correlated]).A
            else:
                if fiber_collisions.with_uncorrelated: self.offset = self.matrix_full.dot(fiber_collisions.kernel_uncorrelated.ravel())
                self.matrix_full = self.matrix_full.dot(np.bmat([list(kernel) for kernel in fiber_collisions.kernel_correlated]).A)
            self.ellsin, self.kin = fiber_collisions.ellsin, fiber_collisions.kin
        if systematic_templates is not None:
            if not isinstance(systematic_templates, SystematicTemplatePowerSpectrumMultipoles):
                systematic_templates = SystematicTemplatePowerSpectrumMultipoles(templates=systematic_templates)
            systematic_templates.init.update(k=self.k, ells=self.ells)
        self.systematic_templates = systematic_templates
        self.theory.init.update(k=self.kin, ells=self.ellsin)
        if shotnoise is None:
            shotnoise = 0.
        else:
            shotnoise = float(shotnoise)
            if 'shotnoise' in getattr(self.theory, '_default_options', {}):
                self.theory.init.setdefault('shotnoise', shotnoise)
        self.shotnoisein = np.array([shotnoise * (ell == 0) for ell in self.ellsin])
        wshotnoisebase = np.concatenate([np.full_like(k, (ell == 0), dtype='f8') for ell, k in zip(self.ells, self.k)])
        self.shotnoiseout = shotnoise * wshotnoisebase
        if wshotnoise is not None:
            self.shotnoisein[...] = 0.
            self.shotnoiseout[...] = shotnoise * (wshotnoisebase - wshotnoise)
        self.wshotnoise = wshotnoise

    @jit(static_argnums=[0])
    def _apply(self, theory):
        theory = jnp.ravel(theory)
        if self.matrix_full is not None:
            theory = jnp.dot(self.matrix_full, theory)
        if self.offset is not None:
            theory = theory + self.offset
        if self.kmask is not None:
            theory = theory[self.kmask]
        return theory

    def calculate(self):
        self.flatpower = self._apply(self.theory.power + self.shotnoisein[:, None]) - self.shotnoiseout
        if self.systematic_templates is not None:
            self.flatpower += self.systematic_templates.flatpower

    def get(self):
        return self.flatpower

    @property
    def power(self):
        return unpack(self.k, self.flatpower)

    def __getstate__(self):
        state = {}
        for name in ['kin', 'ellsin', 'k', 'kedges', 'ells', 'fiducial', 'matrix_full', 'kmask', 'offset', 'flatpower', 'shotnoisein', 'shotnoiseout']:
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
        Or {'sedges': sedges, 'muedges': muedges, 'RR': RR}, specifying the tabulated RR counts.
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
    def initialize(self, slim=None, s=None, sedges=None, ells=None, wmatrix=None, sin=None, sinrebin=1, sinlim=None, ellsin=None, fiber_collisions=None, systematic_templates=None, theory=None):
        from scipy import linalg
        _default_step = 5.

        if ells is None:
            if slim is not None: ells = list(slim)
            else: ells = (0, 2, 4)
        self.ells = tuple(ells)
        self.s = self.smasklim = self.sedges = None
        if s is not None:
            if np.ndim(s[0]) == 0:
                s = [s] * len(self.ells)
            self.s = [np.array(ss, dtype='f8') for ss in s]
            if len(self.s) != len(self.ells):
                raise ValueError("provide as many s's as ells")
        if sedges is not None:
            if np.ndim(sedges[0]) == 0:
                sedges = [sedges] * len(self.ells)
            self.sedges = [np.array(ss, dtype='f8') for ss in sedges]
            if len(self.sedges) != len(self.ells):
                raise ValueError("provide as many sedges as ells")
            if slim is None:
                slim = {ell: (edges[0], edges[-1], np.mean(np.diff(edges))) for ell, edges in zip(self.ells, self.sedges)}

        if slim is not None:
            slim = dict(slim)
            if self.s is not None:
                s, ells, self.smasklim = [], [], {}
                for ill, ell in enumerate(self.ells):
                    ss = self.s[ill]
                    self.smasklim[ell] = np.zeros(len(ss), dtype='?')
                    if ell not in slim: continue  # do not take this multipole
                    self.smasklim[ell][...] = True
                    lim = slim[ell]
                    if lim is not None:  # cut
                        (lo, hi, *step) = slim[ell]
                        smask = (ss >= lo) & (ss <= hi)
                        ss = ss[smask]  # scale cuts
                        self.smasklim[ell][...] = smask
                    if ss.size:
                        s.append(ss)
                        ells.append(ell)
                self.s, self.ells = s, tuple(ells)
            elif list(self.ells) != list(slim.keys()):
                raise ValueError('incompatible ells = {} and klim = {}; just remove ells?', self.ells, list(slim))
            sedges = []
            for ill, ell in enumerate(self.ells):
                if slim[ell] is None:
                    self.sedges = None
                    break
                (lo, hi, *step) = slim[ell]
                if not step:
                    if self.s is not None: step = ((hi - lo) / self.s[ill].size,)
                    else: step = (_default_step,)
                sedges.append(np.arange(lo, hi + step[0] / 2., step=step[0]))
            if self.sedges is None: self.sedges = sedges
        else:
            slim = {ell: None for ell in self.ells}

        if self.sedges is None:
            if self.s is not None:
                self.sedges = []
                for xx in self.s:
                    tmp = (xx[:-1] + xx[1:]) / 2.
                    tmp = np.concatenate([[tmp[0] - (xx[1] - xx[0])], tmp, [tmp[-1] + (xx[-1] - xx[-2])]])
                    self.sedges.append(tmp)
            else:
                self.sedges = [np.arange(20. - _default_step / 2., 150 + _default_step, _default_step)] * len(self.ells)  # gives k = np.arange(0.01, 0.2 + _default_step / 2., _default_step)
        if self.s is None:
            self.s = [(edges[:-1] + edges[1:]) / 2. for edges in self.sedges]
        self.s = [np.array(ss) for ss in self.s]

        if theory is None:
            from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles
            theory = KaiserTracerCorrelationFunctionMultipoles()
        self.theory = theory

        self.matrix_diag, self.matrix_full, self.smask, self.offset = None, None, None, None
        if wmatrix is None:
            self.ellsin = tuple(self.ells)
            self.sin = np.unique(np.concatenate(self.s, axis=0))
            if not all(ss.shape == self.sin.shape and np.allclose(ss, self.sin) for ss in self.s):
                self.smask = [np.searchsorted(self.sin, ss, side='left') for ss in self.s]
                assert all(np.allclose(self.sin[smask], ss) for ss, smask in zip(self.s, self.smask))
                self.smask = [self.sin.size * i + smask for i, smask in enumerate(self.smask)]
                self.smask = np.concatenate(self.smask, axis=0)
        elif isinstance(wmatrix, dict):
            if 'wcounts' in wmatrix:
                self.ellsin = tuple(ellsin or self.theory.init.get('ells', None) or self.ells)
                self.sin, matrix_full = window_matrix_RR({ell: self.sedges[ill] for ill, ell in enumerate(self.ells)}, ellsin=self.ellsin, **wmatrix)
            else:
                self.ellsin = tuple(self.ells)
                self.sin, matrix_full = window_matrix_bininteg(self.sedges, **wmatrix)
            self.matrix_full = matrix_full.T
        elif isinstance(wmatrix, np.ndarray):
            self.ellsin = tuple(ellsin or self.ells)
            matrix_full = np.array(wmatrix).T
            ssize = sum(len(s) for s in self.s)
            if matrix_full.shape[0] != ssize:
                raise ValueError('output "wmatrix" size is {:d}, but got {:d} output "s"'.format(matrix_full.shape[0], ssize))
            sin = np.asarray(sin).flatten()
            self.sin = sin.copy()
            if sinrebin is not None:
                self.sin = self.sin[::sinrebin]
            # print(wmatrix.xout[0], max(kk.max() for kk in self.k) * 1.2)
            if sinlim is not None:
                self.sin = self.sin[(self.sin >= sinlim[0]) & (self.sin <= sinlim[-1])]
            wmatrix_rebin = linalg.block_diag(*[utils.matrix_lininterp(self.sin, sin) for ell in self.ellsin])
            self.matrix_full = matrix_full.dot(wmatrix_rebin.T)
        else:
            raise ValueError('unrecognized wmatrix {}'.format(wmatrix))
        if fiber_collisions is not None:
            self.theory.init.update(s=self.sin, ells=self.ellsin)  # fiber_collisions takes sin, ellsin from theory
            fiber_collisions.init.update(ells=self.ellsin, theory=self.theory)
            fiber_collisions = fiber_collisions.runtime_info.initialize()
            self.ellsin = tuple(fiber_collisions.ellsin)
            if self.matrix_full is None:
                # kernel_correlated is (ellout, ellin, s)
                matrix_diag = fiber_collisions.kernel_correlated
                if self.matrix_diag is None:  # ellsin = ells, offset ok
                    if fiber_collisions.with_uncorrelated: self.offset = fiber_collisions.kernel_uncorrelated.ravel()
                    self.matrix_diag = matrix_diag
                else:
                    if fiber_collisions.with_uncorrelated: self.offset = np.sum(self.matrix_diag * fiber_collisions.kernel_uncorrelated[None, ...], axis=1).ravel()
                    self.matrix_diag = np.sum(self.matrix_diag[:, :, None, ...] * matrix_diag[None, ...], axis=1)
            else:
                if fiber_collisions.with_uncorrelated: self.offset = self.matrix_full.dot(fiber_collisions.kernel_uncorrelated.ravel())
                self.matrix_full = self.matrix_full.dot(np.bmat([[np.diag(kk) for kk in kernel] for kernel in fiber_collisions.kernel_correlated]).A)
                self.ellsin, self.sin = fiber_collisions.ellsin, fiber_collisions.sin
        self.theory.init.update(s=self.sin, ells=self.ellsin)
        if systematic_templates is not None:
            if not isinstance(systematic_templates, SystematicTemplateCorrelationFunctionMultipoles):
                systematic_templates = SystematicTemplateCorrelationFunctionMultipoles(templates=systematic_templates)
            systematic_templates.init.update(s=self.s, ells=self.ells)
        self.systematic_templates = systematic_templates
        #diff = (self.matrix_full.dot(self.sin) - s[0]) / s[0]
        #print(diff, self.sin.shape)
        #print(np.sum(self.matrix_full[len(self.s[0]):, :len(self.sin)], axis=-1))
        #exit()

    @jit(static_argnums=[0])
    def _apply(self, theory):
        if self.matrix_diag is not None:
            theory = jnp.sum(self.matrix_diag * theory[None, ...], axis=1)
        theory = jnp.ravel(theory)
        if self.matrix_full is not None:
            theory = jnp.dot(self.matrix_full, theory)
        if self.offset is not None:
            theory = theory + self.offset
        if self.smask is not None:
            theory = theory[self.smask]
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
        self.ells = tuple(ells)

        if theory is None:
            from desilike.theories.galaxy_clustering import KaiserTracerCorrelationFunctionMultipoles
            theory = KaiserTracerCorrelationFunctionMultipoles()
        self.theory = theory
        if s is not None: self.theory.init.update(s=s)
        self.theory.runtime_info.initialize()
        self.s = np.array(self.theory.s, dtype='f8')
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
    if templates is None:
        templates = {}
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
                raise ValueError('provided template is size {:d}, but expected {:d} = sum({})'.format(template.size, size, sizes))
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