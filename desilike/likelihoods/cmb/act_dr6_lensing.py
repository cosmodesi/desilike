"""JAX-adaptation of https://github.com/ACTCollaboration/act_dr6_lenslike/blob/main/act_dr6_lenslike/act_dr6_lenslike.py"""

import os

import numpy as np
from desilike.likelihoods.base import BaseGaussianLikelihood
from desilike.jax import numpy as jnp

from .base import ClTheory



def pp_to_kk(clpp, ell):
    return clpp * (ell * (ell + 1.))**2. / 4.


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
        clkk_act = alike.get_corrected_clkk(self.data, cl_kk, cl_tt, cl_te, cl_ee, cl_bb,
                                            do_norm_corr=not(self.act_cmb_rescale), act_calib=self.act_calib,
                                            no_like_cmb_corrections=self.no_actlike_cmb_corrections) if self.data['likelihood_corrections'] else cl_kk
        bclkk = self.data['binmat_act'] @ clkk_act
        if self.data['include_planck']:
            clkk_planck = alike.get_corrected_clkk(self.data, cl_kk, cl_tt, cl_te, cl_ee, cl_bb, '_planck') if self.data['likelihood_corrections'] else cl_kk
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
            extract(tar_fn, os.path.dirname(tar_fn))

        installer.write({cls.installer_section: {'data_dir': data_dir}})