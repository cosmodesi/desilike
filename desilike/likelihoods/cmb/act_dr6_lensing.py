"""JAX-adaptation of https://github.com/ACTCollaboration/act_dr6_lenslike/blob/main/act_dr6_lenslike/act_dr6_lenslike.py"""

import os

import numpy as np
from desilike.likelihoods.base import BaseGaussianLikelihood
from desilike.jax import numpy as jnp

from .base import ClTheory



def pp_to_kk(clpp, ell):
    return clpp * (ell * (ell + 1.))**2. / 4.


def get_corrected_clkk(data_dict,clkk,cltt,clte,clee,clbb,suff='',
                       do_norm_corr=True, do_N1kk_corr=True, do_N1cmb_corr=True,
                       act_calib=False, no_like_cmb_corrections=False):
    if no_like_cmb_corrections:
        do_norm_corr = False
        do_N1cmb_corr = False
    clkk_fid = data_dict['fiducial_cl_kk']
    cl_dict = {'tt':cltt,'te':clte,'ee':clee,'bb':clbb}
    if do_N1kk_corr:
        N1_kk_corr = data_dict[f'dN1_kk{suff}'] @ (clkk-clkk_fid)
    else:
        N1_kk_corr = 0
    dNorm = data_dict[f'dAL_dC{suff}']
    fid_norm = data_dict[f'fAL{suff}']
    N1_cmb_corr = 0.
    norm_corr = 0.

    if act_calib and not('planck' in suff):
        ocl = cl_dict['tt']
        fcl = data_dict[f'fiducial_cl_tt']
        ols = np.arange(ocl.size)
        cal_ell_min = 1000
        cal_ell_max = 2000
        sel = np.s_[np.logical_and(ols>cal_ell_min,ols<cal_ell_max)]
        cal_fact = (ocl[sel]/fcl[sel]).mean()
    else:
        cal_fact = 1.0

    for i,s in enumerate(['tt','ee','bb','te']):
        icl = cl_dict[s]
        cldiff = ((icl/cal_fact)-data_dict[f'fiducial_cl_{s}'])
        if do_N1cmb_corr:
            N1_cmb_corr = N1_cmb_corr + (data_dict[f'dN1_{s}{suff}']@cldiff)
        if do_norm_corr:
            c = - 2. * (dNorm[i] @ cldiff)
            if i==0:
                ls = np.arange(c.size)
            # CHANGE: for jax-compatibility
            c = jnp.where(ls>=2, c, fid_norm)
            norm_corr = norm_corr + c
    nclkk = clkk + norm_corr*clkk_fid + N1_kk_corr + N1_cmb_corr
    return nclkk


class ACTDR6LensingLikelihood(BaseGaussianLikelihood):

    config_fn = 'act_dr6_lensing.yaml'
    installer_section = 'ACTDR6LensingLikelihood'
    name = 'ACTDR6Lensing'
    version = 'v1.2'
    nsims_act = 792. # Number of sims used for covmat; used in Hartlap correction
    nsims_planck = 400. # Number of sims used for covmat; used in Hartlap correction
    no_like_corrections = False
    no_actlike_cmb_corrections = False
    # Any ells above this will be discarded; likelihood must at least request ells up to this
    trim_ellmax = 2998
    apply_hartlap = True
    scale_cov = None
    varying_cmb_alens = False  # Whether to divide the theory spectrum by Alens
    act_cmb_rescale = False
    act_calib = False

    def initialize(self, theory=None, cosmo=None, lens_only=False, variant='actplanck_baseline', data_dir=None):
        if lens_only: self.no_like_corrections = True
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer()[self.installer_section]['data_dir'], self.version)
        import act_dr6_lenslike as alike
        self.data = alike.load_data(variant, ddir=data_dir, lens_only=lens_only, like_corrections=not(self.no_like_corrections),
                                    apply_hartlap=self.apply_hartlap, nsims_act=self.nsims_act, nsims_planck=self.nsims_planck,
                                    trim_lmax=self.trim_ellmax, scale_cov=self.scale_cov, version=self.version,
                                    act_cmb_rescale=self.act_cmb_rescale, act_calib=self.act_calib)
        self.flatdata = self.data['data_binned_clkk']
        self.precision = self.data['cinv']
        self.ellmax = self.trim_ellmax + 1
        self.ells = np.arange(self.ellmax + 1)
        requested_cls = ['pp']
        if not(self.no_like_corrections):
            requested_cls += ['tt', 'te', 'ee', 'bb']
        requested_cls = {cl: self.ellmax for cl in requested_cls}
        if theory is None: theory = ClTheory()
        self.theory = theory
        self.theory.init.update(cls=requested_cls, lensing=True, unit='muK', T0=2.7255)
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)

    def calculate(self, Alens=1.):
        import act_dr6_lenslike as alike
        cl_pp, cl_tt, cl_te, cl_ee, cl_bb = [self.theory.cls[name] for name in ['pp', 'tt', 'te', 'ee', 'bb']]
        cl_pp = cl_pp / Alens
        cl_kk = pp_to_kk(cl_pp, self.ells)
        # jax-friendly
        clkk_act = get_corrected_clkk(self.data, cl_kk, cl_tt, cl_te, cl_ee, cl_bb,
                                      do_norm_corr=not(self.act_cmb_rescale), act_calib=self.act_calib,
                                            no_like_cmb_corrections=self.no_actlike_cmb_corrections) if self.data['likelihood_corrections'] else cl_kk
        bclkk = self.data['binmat_act'] @ clkk_act
        if self.data['include_planck']:
            clkk_planck = get_corrected_clkk(self.data, cl_kk, cl_tt, cl_te, cl_ee, cl_bb, '_planck') if self.data['likelihood_corrections'] else cl_kk
            bclkk = jnp.append(bclkk, self.data['binmat_planck'] @ clkk_planck)
        self.flattheory = bclkk
        super().calculate()

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/ACTCollaboration/act_dr6_lenslike')

        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        if installer.reinstall or not exists_path(os.path.join(data_dir, cls.version)):

            tar_base = 'ACT_dr6_likelihood_{}.tgz'.format(cls.version)
            url = 'https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/{}'.format(tar_base)
            tar_fn = os.path.join(data_dir, cls.version, tar_base)
            download(url, tar_fn)
            extract(tar_fn, data_dir)

        installer.write({cls.installer_section: {'data_dir': data_dir}})