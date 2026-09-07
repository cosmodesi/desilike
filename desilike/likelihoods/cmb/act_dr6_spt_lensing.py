"""ACT DR6 + SPT-3G CMB lensing likelihood (JAX-compatible).

JAX adaptation of https://github.com/ACTCollaboration/act_dr6_spt_lenslike.
"""

import os

import numpy as np
import jax.numpy as jnp

from desilike.base import GaussianLikelihood
from desilike.parameter import Parameter, VariableCollection


def _pp_to_kk(cl_pp, ells):
    """Convert lensing potential C_ell^pp to convergence C_ell^kk."""
    return cl_pp * (ells * (ells + 1.)) ** 2. / 4.


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
            from desilike.install import Installer
            data_dir = os.path.join(Installer().data_dir(self.installer_section), self.version)

        only_spt = (variant == 'spt3g')
        if only_spt:
            lens_only = True
        like_corrections = not lens_only

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

        self.flatdata = jnp.asarray(data['data_binned_clkk'])
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
            cl_lensed = harmonic.lensed_cl(ellmax=self._ellmax)
            cl_tt = cl_lensed['tt'][:self._nlen_act]
            cl_te = cl_lensed['te'][:self._nlen_act]
            cl_ee = cl_lensed['ee'][:self._nlen_act]
            cl_bb = cl_lensed['bb'][:self._nlen_act]

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
