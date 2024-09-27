import os

import numpy as np

from desilike.likelihoods.base import BaseLikelihood
from desilike import utils
from .base import ClTheory


class BasePlanck2018Likelihood(BaseLikelihood):
    r"""
    Base class for python likelihood of Planck's 2018 data release.

    Reference
    ---------
    https://arxiv.org/abs/1807.06209

    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    Parameters
    ----------
    theory : ClTheory, default=None
        Theory calculator for CMB :math:`C_{\ell}^{xy}`.
        If ``None``, instantiated using ``cosmo``.

    cosmo : BasePrimordialCosmology, default=None
        Optionally, cosmology calculator. Defaults to ``Cosmoprimo()``.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    config_fn = 'planck2018.yaml'
    installer_section = 'Planck2018ClikLikelihood'
    catch_clik_errors = True

    def initialize(self, theory=None, cosmo=None, elllim=None, data_dir=None):
        super().initialize()
        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer()[self.installer_section]['data_dir']
        self.data_fn = os.path.join(data_dir, self.data_basename)
        if theory is None: theory = ClTheory()
        self.theory = theory
        self.theory.init.update(lensing=True, unit='muK', T0=2.7255)
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)
        self.elllim = tuple(elllim)

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_package, exists_path, download, extract, InstallError

        if installer.reinstall or not exists_path(os.path.join(data_dir, cls.data_basename)):
            # Install data
            tar_base = cls.data_file_id
            url = 'http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID={}'.format(tar_base)
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn)
            extract(tar_fn, data_dir)

        installer.write({cls.installer_section: {'data_dir': data_dir}})


class TTLowlPlanck2018Likelihood(BasePlanck2018Likelihood):
    r"""
    Low-:math:`\ell` temperature-only plik likelihood of Planck's 2018 data release.

    Reference
    ---------
    https://arxiv.org/abs/1807.06209

    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    Parameters
    ----------
    theory : ClTheory, default=None
        Theory calculator for CMB :math:`C_{\ell}^{xy}`.
        If ``None``, instantiated using ``cosmo``.

    cosmo : BasePrimordialCosmology, default=None
        Optionally, cosmology calculator. Defaults to ``Cosmoprimo()``.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    data_file_id = 'COM_Likelihood_Data-baseline_R3.00.tar.gz'
    data_basename = 'baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik'
    name = 'TTLowlPlanck2018'

    def initialize(self, *args, elllim=(2, 29), **kwargs):
        super().initialize(*args, elllim=elllim, **kwargs)
        self.theory.init.update(cls={'tt': self.elllim[-1]})
        fn = os.path.join(self.data_fn, 'clik', 'lkl_0', '_external', 'sigma.fits')
        import fitsio
        sl = slice(self.elllim[0] - 2, self.elllim[1] - 2 + 1)
        cl2x = fitsio.read(fn, ext=0)[:, sl, :]   # (3, 249, 1000)
        self.mu = fitsio.read(fn, ext=1)[sl]   # (249,)
        self.covariance = fitsio.read(fn, ext=2)[sl, sl]
        self.mu_sigma = fitsio.read(fn, ext=3)[sl]
        self.precision = utils.inv(self.covariance)

        self._spline = []
        self._spline_derivative = []

        from scipy.interpolate import InterpolatedUnivariateSpline
        ellsize = self.elllim[1] - self.elllim[0] + 1
        self._prior = np.zeros((ellsize, 2), dtype='f8')
        for i in range(ellsize):
            j = 0
            while abs(cl2x[1, i, j] + 5) < 1e-4:
                j += 1
            self._prior[i, 0] = cl2x[0, i, j + 2]
            j = cl2x.shape[-1] - 1
            while abs(cl2x[1, i, j] - 5) < 1e-4:
                j -= 1
            self._prior[i, 1] = cl2x[0, i, j - 2]
            self._spline.append(InterpolatedUnivariateSpline(cl2x[0, i, :], cl2x[1, i, :]))
            self._spline_derivative.append(self._spline[-1].derivative())

        self._offset = self.get_loglikelihood(self.mu_sigma)
        ells = np.arange(self.elllim[0], self.elllim[1] + 1)
        self.factor = ells * (ells + 1) / 2 / np.pi

    def get_loglikelihood(self, theory):  # theory starts at ell = 2
        if any(theory < self._prior[:, 0]) or any(theory > self._prior[:, 1]):
            return -np.inf

        toret = 0.
        x = np.zeros_like(theory)
        for i, (spline, diff_spline, cl) in enumerate(zip(self._spline, self._spline_derivative, theory)):
            dxdCl = diff_spline(cl)
            if dxdCl < 0:
                return -np.inf
            toret += np.log(dxdCl)
            x[i] = spline(cl)

        self.flatdiff = x - self.mu
        toret += -0.5 * self.flatdiff.dot(self.precision).dot(self.flatdiff)
        return toret

    def calculate(self, A_planck=1.):
        theory = self.theory.cls['tt'][self.elllim[0]:] * self.factor / A_planck**2
        self.loglikelihood = self.get_loglikelihood(theory) - self._offset



class EELowlPlanck2018Likelihood(BasePlanck2018Likelihood):
    r"""
    Low-:math:`\ell` polarization plik likelihood of Planck's 2018 data release.

    Reference
    ---------
    https://arxiv.org/abs/1807.06209

    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    Parameters
    ----------
    theory : ClTheory, default=None
        Theory calculator for CMB :math:`C_{\ell}^{xy}`.
        If ``None``, instantiated using ``cosmo``.

    cosmo : BasePrimordialCosmology, default=None
        Optionally, cosmology calculator. Defaults to ``Cosmoprimo()``.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.
    """
    data_file_id = 'COM_Likelihood_Data-baseline_R3.00.tar.gz'
    data_basename = 'baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik'
    name = 'EELowlPlanck2018'

    def initialize(self, *args, elllim=(2, 29), **kwargs):
        super().initialize(*args, elllim=elllim, **kwargs)
        self.theory.init.update(cls={'ee': self.elllim[-1]})
        fn = os.path.join(self.data_fn, 'clik', 'lkl_0', 'probEE')
        import fitsio
        sl = slice(self.elllim[0] - 2, self.elllim[1] - 2 + 1)
        self._prob = fitsio.read(fn, ext=0)
        self._ncl = 3000
        self._dcl = 0.0001
        self._prob = self._prob.reshape(-1, self._ncl).T[:, sl]
        ells = np.arange(self.elllim[0], self.elllim[1] + 1)
        self.factor = ells * (ells + 1) / 2 / np.pi

    def get_loglikelihood(self, theory):  # theory starts at ell = 2
        idx = (theory / self._dcl).astype(int)
        try:
            return np.take_along_axis(self._prob, idx[None, :], 0).sum()
        except IndexError:
            return -np.inf

    def calculate(self, A_planck=1.):
        theory = self.theory.cls['ee'][self.elllim[0]:] * self.factor / A_planck**2
        self.loglikelihood = self.get_loglikelihood(theory)