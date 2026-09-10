"""ACT DR6 + SPT-3G CMB lensing likelihood (JAX-compatible).

JAX adaptation of https://github.com/ACTCollaboration/act_dr6_spt_lenslike.
"""

import os
import contextlib
import warnings

import numpy as np
import jax.numpy as jnp

from desilike.base import GaussianLikelihood
from desilike.parameter import Parameter, Variable, VariableCollection


def _pp_to_kk(cl_pp, ells):
    """Convert lensing potential C_ell^pp to convergence C_ell^kk."""
    return cl_pp * (ells * (ells + 1.)) ** 2. / 4.


#: Only files at least this large are converted to ``.npy``. The N1 derivative matrices are
#: 225 MB of ASCII each and everything else under ``like_corrs`` is under 1.2 MB, so this
#: separates the two by a wide margin rather than by name.
_LOADTXT_CACHE_MIN_BYTES = 10 * 1024**2


def _npy_cache_name(fn):
    """Cache file name for *fn*, carrying the source's size and mtime.

    The stamp is the invalidation: a data release that replaces a ``.txt`` in place produces a
    different name, so a stale cache is never read rather than being silently preferred. Old
    entries are left behind rather than deleted -- this may be a directory shared between users,
    where deciding that another run's file is dead is not this function's call.
    """
    stat = os.stat(fn)
    stem = os.path.splitext(os.path.basename(fn))[0]
    return f'{stem}-{stat.st_size}-{int(stat.st_mtime)}.npy'


class _CachingNumpy:
    """Stand-in for the ``np`` binding inside ``act_dr6_spt_lenslike``.

    Everything but :meth:`loadtxt` is numpy itself; :meth:`loadtxt` reads a ``.npy`` conversion
    where one exists and writes it where it does not. Substituting the module's binding rather
    than patching ``numpy.loadtxt`` keeps this scoped to the one caller -- patching numpy itself
    would reach every library in the process.
    """
    def __init__(self, read_dir, write_dir):
        self._read_dir, self._write_dir = read_dir, write_dir
        self._warned = False

    def __getattr__(self, name):
        return getattr(np, name)

    def loadtxt(self, fname, **kwargs):
        # `usecols` / `unpack` change the returned array, and the cache is keyed by file alone,
        # so anything but a plain read goes straight to numpy. Only the N1 derivative matrices
        # are read plainly, and they are the whole cost.
        fn = str(fname)
        if kwargs or 'like_corrs' not in fn:
            return np.loadtxt(fname, **kwargs)
        try:
            if os.path.getsize(fn) < _LOADTXT_CACHE_MIN_BYTES:
                return np.loadtxt(fname, **kwargs)
            name = _npy_cache_name(fn)
        except OSError:
            return np.loadtxt(fname, **kwargs)
        cached = os.path.join(self._read_dir, name)
        if os.path.isfile(cached):
            try:
                return np.load(cached)
            except (OSError, ValueError):
                # A truncated or corrupt entry must not be fatal: fall through, re-parse, and
                # rewrite it below.
                pass
        toret = np.loadtxt(fname, **kwargs)
        self._save(name, toret)
        return toret

    def _save(self, name, array):
        """Write *array* to the cache, atomically, and never fatally.

        The rename is what makes this safe under ``srun -n 16``: every rank converts the same
        file to the same bytes and each publishes it in one step, so a reader sees either the
        previous entry or a complete new one, never a half-written array.
        """
        target = os.path.join(self._write_dir, name)
        tmp = '{}.tmp.{}'.format(target, os.getpid())
        try:
            os.makedirs(self._write_dir, exist_ok=True)
            np.save(tmp, array)
            # np.save appends '.npy' unless the name already ends in it
            os.replace(tmp + '.npy', target)
        except OSError as exc:
            if not self._warned:
                warnings.warn('could not cache {}: {}. Falling back to parsing the ASCII data on '
                              'every construction, which costs ~50 s each.'.format(target, exc))
                self._warned = True
            try: os.remove(tmp + '.npy')
            except OSError: pass


@contextlib.contextmanager
def _cache_loadtxt(module, section, version):
    """Give *module* a :class:`_CachingNumpy` for the duration of the block.

    ``act_dr6_spt_lenslike.load_data`` reads its N1 derivative matrices -- five for ACT and,
    for an ``actplanck``/``actspt3g`` variant, five more for Planck -- with ``np.loadtxt``, at
    225 MB of ASCII each. That is ~48 s per construction, and desilike's ``build()`` re-runs
    every ``__init__``, so one posterior pays it two or three times over. Converted to ``.npy``
    the same matrix loads in 0.02 s against 3.87 s, measured on a Perlmutter compute node.

    The cache is written to the canonical install path and read back through the ``'ro'`` alias
    (see :meth:`~desilike.install.Installer.data_dir`), which on Perlmutter serves the same
    bytes over a caching read-only DVS mount -- which is what that alias exists for.
    """
    from desilike.install import Installer
    installer = Installer()
    sub = os.path.join(version, 'like_corrs_npy')
    read_dir = os.path.join(installer.data_dir(section, ro=True), sub)
    write_dir = os.path.join(installer.data_dir(section), sub)
    original = module.np
    module.np = _CachingNumpy(read_dir, write_dir)
    try:
        yield
    finally:
        module.np = original


class ACTDR6SPTLensingLikelihood(GaussianLikelihood):
    r"""
    Python likelihood for ACT DR6 + SPT-3G CMB lensing.

    JAX-compatible wrapper around the ``act_dr6_spt_lenslike`` package, supporting all
    variants including ACT-only, ACT+Planck, and combined ACT+SPT+Planck variants.

    Reference
    ---------
    https://arxiv.org/abs/2504.20038 (ACT+SPT combined analysis)
    https://arxiv.org/abs/2304.05203 (ACT DR6 lensing)

    Parameters
    ----------
    variant : str, default='actplanck_baseline'
        Likelihood variant. One of:
        ``act_baseline``, ``act_extended``, ``actplanck_baseline``, ``actplanck_extended``,
        ``act_polonly``, ``act_cibdeproj``, ``act_cinpaint``,
        ``spt3g``, ``actspt3g_baseline``, ``actspt3g_extended``,
        ``actplanckspt3g_baseline``, ``actplanckspt3g_extended``.
    lens_only : bool, default=False
        If True, skip likelihood corrections and use the CMB-marginalized covariance.
        Automatically forced to True for the ``spt3g`` variant.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``CosmoprimoCosmology(engine='camb')``.
    data_dir : str, default=None
        Path to the ``v1.2`` data sub-directory (containing bandpowers and ``like_corrs/``).
        Defaults to the path recorded by :class:`~desilike.install.Installer`.
    params : list of Parameter, default=None
        Extra parameters to add alongside the class defaults.
    """

    installer_section = 'ACTDR6SPTLensingLikelihood'
    version = 'v1.2'
    T0_cmb = 2.7255
    trim_lmax = 2998
    nsims_act = 796
    nsims_planck = 400
    apply_hartlap = True

    @classmethod
    def propose_params(cls):
        return VariableCollection([
            Parameter('Alens', value=1., latex=r'A_{\mathrm{lens}}'),
        ])

    def __init__(self, variant='actplanck_baseline', lens_only=False, cosmo=None, data_dir=None, params=None):
        import act_dr6_spt_lenslike as alike

        if data_dir is None:
            # The copy `install` stages, not the one shipped inside act_dr6_spt_lenslike: that
            # one lives in the python environment, so it is read at whatever path the package
            # was installed to and misses the ``'ro'`` alias entirely -- on Perlmutter its
            # `like_corrs` is a symlink onto the read-write mount, which is the slow way to
            # read the same bytes. `ro=True` here is the point of putting it under `data_dir`.
            from desilike.install import Installer
            data_dir = os.path.join(Installer().data_dir(self.installer_section, ro=True), self.version)
            # A directory is not the data: `like_corrs_npy` alone creates one (see
            # `_cache_loadtxt`), and so does a partial download. Say what is missing here
            # rather than letting `load_data` fail on whichever file it happens to want first.
            if not os.path.isdir(os.path.join(data_dir, 'like_corrs')):
                raise ValueError('no ACT DR6 lensing data at {}. Run the installer for {} '
                                 '(desilike.install), or pass `data_dir` explicitly.'
                                 .format(data_dir, self.installer_section))

        only_spt = (variant == 'spt3g')
        if only_spt:
            lens_only = True
        like_corrections = not lens_only

        # The N1 derivative matrices are ASCII, and `like_corrections` is what asks for them:
        # ~2.25 GB of text parsed per construction. `_cache_loadtxt` converts them to `.npy`
        # once and reads that thereafter -- see its docstring for the numbers.
        with _cache_loadtxt(alike.act_dr6_spt_lenslike, self.installer_section, self.version):
            data = alike.load_data(
                variant, ddir=data_dir, lens_only=lens_only,
                like_corrections=like_corrections,
                apply_hartlap=self.apply_hartlap,
                nsims_act=self.nsims_act, nsims_planck=self.nsims_planck,
                trim_lmax=self.trim_lmax, version=self.version,
            )

        self._variant = variant
        self._like_corrections = like_corrections
        self._include_planck = data['include_planck']
        self._include_spt = data['include_spt'] or data['include_spt_no_planck']
        self._only_spt = data['only_spt']

        # nlen_act: size of ACT correction-matrix arrays (trim_lmax + lbuffer)
        # nlen_spt: size of SPT binning arrays (spt_trim_lmax=3100 + lbuffer=2)
        self._nlen_act = self.trim_lmax + 2   # 3000
        self._nlen_spt = 3102

        # Determine the maximum ell to request from the cosmology engine
        if self._include_spt or self._only_spt:
            self._ellmax = self._nlen_spt - 1   # 3101
        else:
            self._ellmax = self._nlen_act - 1   # 2999

        self.flatdata = Variable(f'{type(self).__name__}.flatdata', value=jnp.asarray(data['data_binned_clkk']))
        self.precision = jnp.asarray(data['cinv'])

        self._binmat_act = jnp.asarray(data['binmat_act'])
        if self._include_planck:
            self._binmat_planck = jnp.asarray(data['binmat_planck'])
        if self._include_spt:
            self._binmat_spt = jnp.asarray(data['binmat_spt'])

        if like_corrections:
            # Correction matrices for ACT (order: tt, ee, bb, te — same as upstream)
            self._cl_kk_fid = jnp.asarray(data['fiducial_cl_kk'])
            # Fiducial CMB spectra stacked (4, nlen_act): tt, ee, bb, te
            self._cl_fids = jnp.stack([jnp.asarray(data[f'fiducial_cl_{spec}'])
                                        for spec in ('tt', 'ee', 'bb', 'te')])
            # N1 lensing correction: (nlen_act, nlen_act)
            self._dN1_kk = jnp.asarray(data['dN1_kk'])
            # N1 CMB corrections stacked (4, nlen_act, nlen_act): tt, ee, bb, te
            self._dN1_cmb = jnp.stack([jnp.asarray(data[f'dN1_{spec}'])
                                        for spec in ('tt', 'ee', 'bb', 'te')])
            # Normalization correction matrix (4, nlen_act, nlen_act)
            self._dAL_dC = jnp.asarray(data['dAL_dC'])
            # Precomputed JAX-compatible normalization denominator (avoids in-place indexing)
            ls_act = jnp.arange(self._nlen_act)
            self._norm_denom = jnp.where(ls_act >= 2, jnp.asarray(data['fAL']), 1.)

            if self._include_planck:
                self._cl_kk_fid_planck = self._cl_kk_fid  # same fiducial kk
                self._cl_fids_planck = self._cl_fids
                self._dN1_kk_planck = jnp.asarray(data['dN1_kk_planck'])
                self._dN1_cmb_planck = jnp.stack([jnp.asarray(data[f'dN1_{spec}_planck'])
                                                    for spec in ('tt', 'ee', 'bb', 'te')])
                self._dAL_dC_planck = jnp.asarray(data['dAL_dC_planck'])
                ls_planck = jnp.arange(self._nlen_act)
                self._norm_denom_planck = jnp.where(ls_planck >= 2, jnp.asarray(data['fAL_planck']), 1.)

        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(engine='camb')
        self.cosmo = cosmo

        variable_collection = self.propose_params()
        if params is not None:
            variable_collection = variable_collection + VariableCollection(params)
        self.params = {param.basename: param for param in variable_collection}

    def __post_init__(self, *args, **kwargs):
        requirements = [{'ellmax': self._ellmax}]
        self.cosmo.add_requirements({'harmonic.lens_potential_cl': requirements})
        if self._like_corrections:
            self.cosmo.add_requirements({'harmonic.lensed_cl': requirements})

    def _apply_corrections(self, cl_kk, cl_tt, cl_ee, cl_bb, cl_te,
                           cl_kk_fid, cl_fids, dN1_kk, dN1_cmb, dAL_dC, norm_denom):
        """JAX-compatible lensing likelihood corrections.

        Replaces the in-place indexed version in get_corrected_clkk from the upstream package.
        All inputs are JAX arrays; no in-place mutation.
        """
        cl_specs = jnp.stack([cl_tt, cl_ee, cl_bb, cl_te])       # (4, nlen)
        cl_diffs = cl_specs - cl_fids                               # (4, nlen)

        N1_kk_corr = dN1_kk @ (cl_kk - cl_kk_fid)                # (nlen,)
        N1_cmb_corr = jnp.einsum('ijk,ik->j', dN1_cmb, cl_diffs)  # (nlen,)
        # -2 * sum_i(dAL_dC[i] @ cl_diff[i]) / fid_norm, avoiding in-place indexing
        norm_corr = -2. * jnp.einsum('ijk,ik->j', dAL_dC, cl_diffs) / norm_denom

        return cl_kk + norm_corr * cl_kk_fid + N1_kk_corr + N1_cmb_corr

    def __call__(self):
        harmonic = self.cosmo.get_harmonic()
        cl_pp_full = harmonic.lens_potential_cl(ellmax=self._ellmax)['pp']
        ells = jnp.arange(self._ellmax + 1)

        Alens = self.params['Alens'].value
        cl_kk_full = _pp_to_kk(cl_pp_full / Alens, ells)

        # Slice theory arrays to the sizes expected by the correction matrices
        cl_kk_act = cl_kk_full[:self._nlen_act]

        if self._like_corrections:
            # The like_corrs fiducial spectra are in muK^2 (they come from
            # cosmo2017_10K_acc3_lensedCls.dat, a CAMB Dl file), while cosmoprimo returns
            # dimensionless C_ell. Without this factor cl_specs - cl_fids is just -cl_fids and
            # every correction is garbage: measured -7173.4 against cobaya's -20.5 on the same
            # theory. C_ell^kk needs no such factor -- phiphi is dimensionless on both sides.
            unit = (self.T0_cmb * 1e6) ** 2
            cl_lensed = harmonic.lensed_cl(ellmax=self._ellmax)
            cl_tt = unit * cl_lensed['tt'][:self._nlen_act]
            cl_te = unit * cl_lensed['te'][:self._nlen_act]
            cl_ee = unit * cl_lensed['ee'][:self._nlen_act]
            cl_bb = unit * cl_lensed['bb'][:self._nlen_act]

            cl_kk_corr_act = self._apply_corrections(
                cl_kk_act, cl_tt, cl_ee, cl_bb, cl_te,
                self._cl_kk_fid, self._cl_fids,
                self._dN1_kk, self._dN1_cmb, self._dAL_dC, self._norm_denom,
            )
        else:
            cl_kk_corr_act = cl_kk_act

        if self._only_spt:
            # spt3g variant: ACT binmat was loaded as SPT binmat (nlen_spt columns)
            cl_kk_spt = cl_kk_full[:self._nlen_spt]
            bclkk = self._binmat_act @ cl_kk_spt
        else:
            bclkk = self._binmat_act @ cl_kk_corr_act

        if self._include_planck:
            if self._like_corrections:
                cl_kk_corr_planck = self._apply_corrections(
                    cl_kk_act, cl_tt, cl_ee, cl_bb, cl_te,
                    self._cl_kk_fid_planck, self._cl_fids_planck,
                    self._dN1_kk_planck, self._dN1_cmb_planck, self._dAL_dC_planck, self._norm_denom_planck,
                )
            else:
                cl_kk_corr_planck = cl_kk_act
            bclkk = jnp.append(bclkk, self._binmat_planck @ cl_kk_corr_planck)

        if self._include_spt:
            cl_kk_spt = cl_kk_full[:self._nlen_spt]
            bclkk = jnp.append(bclkk, self._binmat_spt @ cl_kk_spt)

        self.flattheory = bclkk
        return super().__call__()

    @classmethod
    def install(cls, installer):
        """Download ACT DR6 likelihood data and copy SPT-3G bandpowers.

        Requires the ``act_dr6_spt_lenslike`` package to be installed and importable.
        Install it with ``pip install .`` from the ``spt_act_likelihood`` repository.
        """
        from desilike.install import exists_path, download, extract

        # The pip check is a no-op if already importable; otherwise points the user to the repo
        installer.pip('act_dr6_spt_lenslike', pkgname='act_dr6_spt_lenslike')

        data_dir = installer.data_dir(cls.installer_section)
        version_dir = os.path.join(data_dir, cls.version)

        if installer.reinstall or not exists_path(os.path.join(version_dir, 'like_corrs')):
            tar_base = f'ACT_dr6_likelihood_{cls.version}.tgz'
            url = f'https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/{tar_base}'
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn)
            extract(tar_fn, data_dir)

        # Copy SPT-3G bandpowers (muse_likelihood.npz) from the installed package's data dir
        muse_target = os.path.join(version_dir, 'muse_likelihood.npz')
        if installer.reinstall or not exists_path(muse_target):
            import shutil
            import act_dr6_spt_lenslike as alike
            muse_source = os.path.join(os.path.dirname(alike.__file__), 'data', cls.version, 'muse_likelihood.npz')
            if os.path.exists(muse_source):
                shutil.copy2(muse_source, muse_target)

        installer.write({cls.installer_section: {'data_dir': data_dir}})
