"""JAX-adaptation of https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/base_classes/planck_2018_CamSpec_python.py."""

import os

import numpy as np

from desilike.likelihoods.base import BaseGaussianLikelihood
from desilike.jax import numpy as jnp
from desilike import utils
from .base import ClTheory, projection


class BasePlanckNPIPECamspecLikelihood(BaseGaussianLikelihood):

    config_fn = 'camspec.yaml'
    installer_section = 'PlanckNPIPECamspecLikelihood'
    all_cls = ['100x100', '143x143', '217x217', '143x217', 'TE', 'EE']

    def initialize(self, theory=None, cosmo=None, data_dir=None, **kwargs):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = os.path.join(Installer()[self.installer_section]['data_dir'], 'CamSpec_NPIPE')
        self.load_data(data_dir, **kwargs)
        requested_cls = {cl: self.ellmax for cl in ['tt', 'te', 'ee']}
        ells = np.arange(self.ellmax + 1)
        self.factor = ells * (ells + 1) / 2 / np.pi
        if theory is None: theory = ClTheory()
        self.theory = theory
        self.theory.init.update(cls=requested_cls, lensing=True, unit='muK', T0=2.7255)
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)
        super().initialize(data=self.flatdata, precision=self.precision, **kwargs)

    def load_data(self, data_dir, select_cls=None, select_ells=None, proj_order=None):
        select_cls = list(select_cls if select_cls is not None else self.all_cls[1:])  # list of spectra, among all_cls
        # select_ells may be a dictionary of cl: ell indices / masks
        input_data = np.loadtxt(os.path.join(data_dir, 'like_NPIPE_12.6_unified_spectra.txt'))
        flatdata, masks, index_ells, all_cls, index_cls = [], [], {}, [], []
        with open(os.path.join(data_dir, 'like_NPIPE_12.6_unified_data_ranges.txt'), 'r', encoding='utf-8-sig') as file:
            lines = [line for line in file if line]
            for iline, line in enumerate(lines):
                items = line.split()
                cl = items[0]
                all_cls.append(cl)
                elllim = [int(x) for x in items[1:]]
                nells = elllim[1] - elllim[0] + 1
                flatdata.append(input_data[elllim[0]: elllim[1] + 1, iline])
                tmp_ells = np.arange(elllim[0], elllim[1] + 1)
                mask = np.zeros(nells, dtype='?')
                if elllim[1] and nells:
                    if cl in select_cls:
                        if select_ells is not None:
                            mask[np.isin(tmp_ells, select_ells[cl])] = True
                        else:
                            mask[...] = True
                masks.append(mask)
                if mask.any():
                    index_ells[cl] = tmp_ells[mask]
                    index_cls.append(iline)
        assert all_cls == self.all_cls  # foregrounds, etc. are all ordered this way
        self.index_cls = np.array(index_cls, dtype='i4')
        mask = np.concatenate(masks)
        nx = len(mask)
        with open(os.path.join(data_dir, 'like_NPIPE_12.6_unified_cov.bin'), 'rb') as file:
            covariance = np.fromfile(file, dtype=np.float32)
        assert (nx ** 2 == covariance.shape[0])
        self.flatdata = np.concatenate(flatdata)[mask]
        self.covariance = covariance.reshape(nx, nx)[np.ix_(mask, mask)].astype('f8')
        fn = os.path.join(data_dir, 'precision.npy')
        # Inversion takes ~ 1 min, compute it once
        try:
            self.precision, covariance = np.load(fn)
            assert np.allclose(self.covariance, covariance)
        except:
            self.precision = utils.inv(self.covariance)
            np.save(fn, np.array([self.precision, self.covariance]))
        #self.precision = np.diag(self.precision)
        self.index_ells = index_ells
        self.ellmax = max([max(ell) for ell in self.index_ells.values()])
        self.has_foregrounds = any(cl in self.all_cls[:4] for cl in self.index_ells)
        pivot = 1500
        ells = jnp.arange(self.ellmax + 1)
        self._template_foreground_tilt = jnp.log(jnp.maximum(ells, 1) / pivot)
        self._template_foreground_amp = jnp.where(ells >= 1, jnp.ones_like(self._template_foreground_tilt), 0.)
        self.proj_order = proj_order
        if self.proj_order:
            from scipy import linalg
            proj, poly = [], []
            for icl, cl in enumerate(self.all_cls):
                if cl in self.index_ells:
                    size = self.index_ells[cl].size
                    tmp = projection(size, order=min(size, self.proj_order))
                    proj.append(tmp[0])
                    poly.append(tmp[1])
            self._proj, poly = (jnp.asarray(linalg.block_diag(*tmp)) for tmp in [proj, poly])
            self._chi2_dd = self.flatdata.dot(self.precision).dot(self.flatdata)
            chi2_dt = self.flatdata.dot(self.precision).dot(poly.T)
            self._chi2_dt = jnp.asarray(- (chi2_dt + chi2_dt.T))
            self._chi2_tt = jnp.asarray(poly.dot(self.precision).dot(poly.T))

    def get_foregrounds(self, params):
        names = ['100', '143', '217', '143x217']
        amp = jnp.array([params['amp_{}'.format(name)] for name in names])
        tilt = jnp.array([params['n_{}'.format(name)] for name in names])
        toret = amp[:, None] * self._template_foreground_amp * jnp.exp(self._template_foreground_tilt * tilt[:, None])
        return toret

    def get_cals(self, params):
        calPlanck = params.get('A_planck', 1) ** 2
        cal0 = params.get('cal0', 1)
        cal2 = params.get('cal2', 1)
        calTE = params.get('calTE', 1)
        calEE = params.get('calEE', 1)
        return jnp.array([cal0, 1, cal2, jnp.sqrt(cal2), calTE, calEE]) * calPlanck

    def compute_chi2(self, cl_tt, cl_te, cl_ee, params):
        cals = self.get_cals(params)
        if self.has_foregrounds:
            foregrounds = self.get_foregrounds(params)

        flattheory = []
        for icl, cl in enumerate(self.all_cls):
            if cl in self.index_ells:
                index = self.index_ells[cl]
                if icl <= 3:
                    tmp = cl_tt[index] + foregrounds[icl][index]
                if icl == 4:
                    tmp = cl_te[index]
                if icl == 5:
                    tmp = cl_ee[index]
                tmp = tmp / cals[icl]
                flattheory.append(tmp)
        self.flattheory = jnp.concatenate(flattheory)
        self.flatdiff = self.flatdata - self.flattheory
        if self.proj_order:
            flattheory = self._proj.dot(self.flattheory)
            chi2 = self._chi2_dd + self._chi2_dt.dot(flattheory) + flattheory.dot(self._chi2_tt).dot(flattheory)
        else:
            #chi2 = jnp.sum(self.flatdiff * self.precision * self.flatdiff, axis=0)
            chi2 = self.flatdiff.dot(self.precision).dot(self.flatdiff)
        return chi2

    def calculate(self, **params):
        cl_tt, cl_te, cl_ee = [self.factor * self.theory.cls[name] for name in ['tt', 'te', 'ee']]
        self.loglikelihood = -0.5 * self.compute_chi2(cl_tt, cl_te, cl_ee, params)

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        if installer.reinstall or not exists_path(data_dir):
            tar_base = 'CamSpec_NPIPE.zip'
            url = 'https://github.com/CobayaSampler/planck_native_data/releases/download/v1/{}'.format(tar_base)
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn)
            extract(tar_fn, data_dir)

        installer.write({cls.installer_section: {'data_dir': data_dir}})



class TTTEEEHighlPlanckNPIPECamspecLikelihood(BasePlanckNPIPECamspecLikelihood):

    name = 'TTTEEEHighlPlanck2018NPIPECamspec'

    def initialize(self, theory=None, cosmo=None, data_dir=None, **kwargs):
        super().initialize(theory=theory, cosmo=cosmo, data_dir=data_dir, select_cls=['143x143', '217x217', '143x217', 'TE', 'EE'], **kwargs)



class TTHighlPlanckNPIPECamspecLikelihood(BasePlanckNPIPECamspecLikelihood):

    name = 'TTHighlPlanck2018NPIPECamspec'

    def initialize(self, theory=None, cosmo=None, data_dir=None, **kwargs):
        super().initialize(theory=theory, cosmo=cosmo, data_dir=data_dir, select_cls=['143x143', '217x217', '143x217'], **kwargs)