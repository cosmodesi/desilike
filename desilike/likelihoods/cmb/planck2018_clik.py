import os

import numpy as np

from desilike.likelihoods.base import BaseLikelihood
from .base import ClTheory


_cache = {}


class BasePlanck2018ClikLikelihood(BaseLikelihood):
    r"""
    Base class for clik likelihood of Planck's 2018 data release.

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
    config_fn = 'planck2018_clik.yaml'
    installer_section = 'Planck2018ClikLikelihood'
    catch_clik_errors = True

    def initialize(self, theory=None, cosmo=None, data_dir=None):
        super(BasePlanck2018ClikLikelihood, self).initialize()
        import clik
        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer()[self.installer_section]['data_dir']
        data_fn = os.path.join(data_dir, self.data_basename)
        self.lensing = clik.try_lensing(data_fn)
        if data_fn in _cache:
            self.clik = _cache[data_fn]
        else:
            try:
                if self.lensing:
                    self.clik = clik.clik_lensing(data_fn)
                else:
                    self.clik = clik.clik(data_fn)
            except clik.lkl.CError as exc:
                if not os.path.exists(data_fn):
                    raise IOError('The path to the .clik file for the likelihood {} was not found at {}'.format(self.__class__.__name__, data_fn))
                else:
                    raise exc
            _cache[data_fn] = self.clik
        self.ells_max = self.clik.get_lmax()
        self.nuisance_params = list(self.clik.extra_parameter_names)

        requested_cls = ['tt', 'ee', 'bb', 'te', 'tb', 'eb']
        if self.lensing:
            has_cls = [ellmax != -1 for ellmax in self.ells_max]
            requested_cls = ['pp'] + requested_cls
        else:
            has_cls = self.clik.get_has_cl()
        self.requested_cls, sizes, cls = [], [], {}
        for cl, ellmax, has_cl in zip(requested_cls, self.ells_max, has_cls):
            if int(has_cl):
                self.requested_cls.append(cl)
                sizes.append(ellmax + 1)
                if cl not in ['tb', 'eb']: cls[cl] = ellmax

        # Placeholder for vector passed to clik
        self.cumsizes = np.insert(np.cumsum(sizes), 0, 0)
        self.vector = np.zeros(self.cumsizes[-1] + len(self.nuisance_params))
        if theory is None: theory = ClTheory()
        self.theory = theory
        self.theory.init.update(cls=cls, lensing=True, unit='muK', T0=2.7255)
        if cosmo is not None: self.theory.init.update(cosmo=cosmo)
        #basenames = [param.basename for param in self.params if param.name not in (self._param_loglikelihood.name, self._param_logprior.name) and not param.drop]
        #if set(basenames) != set(self.nuisance_params):
        #    raise ValueError('Expected nuisance parameters {}, received {}'.format(self.nuisance_params, basenames))

    def calculate(self, **params):
        self.loglikelihood = -np.inf
        for cl, start, stop in zip(self.requested_cls, self.cumsizes[:-1], self.cumsizes[1:]):
            if cl in ['tb', 'eb']: continue
            tmp = self.theory.cls[cl]
            # Check for nan's: may produce a segfault in clik
            if np.isnan(tmp).any():
                return
            self.vector[start:stop] = tmp

        # Fill with likelihood parameters
        self.vector[self.cumsizes[-1]:] = [params[p] for p in self.nuisance_params]
        import clik
        try:
            self.loglikelihood = self.clik(self.vector)[0]
        except clik.lkl.CError as exc:
            if self.catch_clik_errors:
                self.loglikelihood = -np.inf
            else:
                raise ValueError('clik failed with input vector {}'.format(self.vector)) from exc
        # "zero" of clik, and sometimes nan's returned
        if np.allclose(self.loglikelihood, -1e30) or np.isnan(self.loglikelihood):
            self.loglikelihood = -np.inf

    @property
    def size(self):
        # Theory vector size
        return self.cumsizes[-1]

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_package, exists_path, download, extract, InstallError

        if installer.reinstall or not exists_package('clik'):

            for pkg in ['cython', 'astropy']:
                installer.pip(pkg, no_deps=False, force_reinstall=False, ignore_installed=False)
            # Install clik code
            tar_base = 'COM_Likelihood_Code-v3.0_R3.10.tar.gz'
            url = 'http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID={}'.format(tar_base)
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn)
            extract(tar_fn, data_dir)

            def find_src_dir(root):
                for path, dirs, files in os.walk(root):
                    if 'waf' in files:
                        return path
                raise InstallError('clik code not found in {}'.format(root))

            src_dir = find_src_dir(os.path.join(data_dir, 'code'))
            cwd = os.getcwd()
            import sys, subprocess
            os.chdir(src_dir)
            #installer.write({'pylib_dir': os.path.join(src_dir, 'lib', 'python', 'site-packages'),
            #                 'bin_dir': os.path.join(src_dir, 'bin'),
            #                 'dylib_dir': os.path.join(src_dir, 'lib'),
            #                 'clik_path': src_dir,
            #                 'clik_data': os.path.join(src_dir, 'share', 'clik')})
            installer.write({'source': os.path.join(src_dir, 'bin', 'clik_profile.sh')})

            flags = ['--install_all_deps']
            import sysconfig
            include_dir, lib_dir = sysconfig.get_config_var('CONFINCLUDEDIR'), sysconfig.get_config_var('LIBDIR')
            for name in ['cfitsio', 'lapack']:
                if any(name in fn for fn in os.listdir(lib_dir)):
                    flags += ['--{}_include'.format(name), include_dir, '--{}_lib'.format(name), lib_dir]
                    installer.write({'dylib_dir': lib_dir})
                else:
                    flags += ['--install_{}'.format(name)]

            result = subprocess.run([sys.executable, 'waf', 'configure'] + flags, capture_output=True, text=True)
            cls.log_info(result.stdout)
            error_msg = 'cd {0}\n{1} waf configure\n### and ###\n{1} waf install\n'.format(src_dir, sys.executable)
            if 'finished successfully' not in result.stdout:
                print(result.stdout)
                print(result.stderr)
                raise InstallError('clik configuration failed, please do:\n' + error_msg)
            result = subprocess.run([sys.executable, 'waf', 'install'], capture_output=True, text=True)
            cls.log_info(result.stdout)
            if 'finished successfully' not in result.stdout:
                raise InstallError('clik installation failed, please do:\n' + error_msg)
            os.chdir(cwd)

        if installer.reinstall or not exists_path(os.path.join(data_dir, cls.data_basename)):
            # Install data
            tar_base = cls.data_file_id
            url = 'http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID={}'.format(tar_base)
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn)
            extract(tar_fn, data_dir)
            installer.write({cls.installer_section: {'data_dir': data_dir}})

    def __del__(self):
        try:
            del self.clik
        except AttributeError:
            pass


class TTHighlPlanck2018PlikLikelihood(BasePlanck2018ClikLikelihood):
    r"""
    High-:math:`\ell` temperature-only plik likelihood of Planck's 2018 data release.

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
    data_basename = 'baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik'


class TTHighlPlanck2018PlikLiteLikelihood(BasePlanck2018ClikLikelihood):
    r"""
    High-:math:`\ell` temperature-only plik likelihood of Planck's 2018 data release, marginalized over the foreground model.

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
    data_basename = 'baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik'


class TTHighlPlanck2018PlikUnbinnedLikelihood(BasePlanck2018ClikLikelihood):
    r"""
    High-:math:`\ell` temperature-only plik likelihood of Planck's 2018 data release.

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
    data_file_id = 'COM_Likelihood_Data-extra-plik-ext_R3.00.tar.gz'
    data_basename = 'extended_plik/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT_bin1.clik'


class TTTEEEHighlPlanck2018PlikLikelihood(BasePlanck2018ClikLikelihood):
    r"""
    High-:math:`\ell` temperature and polarization plik likelihood of Planck's 2018 data release.

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
    data_basename = 'baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik'


class TTTEEEHighlPlanck2018PlikLiteLikelihood(BasePlanck2018ClikLikelihood):
    r"""
    High-:math:`\ell` temperature and polarization plik likelihood of Planck's 2018 data release, marginalized over the foreground model.

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
    data_basename = 'baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik'


class TTTEEEHighlPlanck2018PlikUnbinnedLikelihood(BasePlanck2018ClikLikelihood):
    r"""
    High-:math:`\ell` temperature and polarization plik likelihood of Planck's 2018 data release.

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
    data_file_id = 'COM_Likelihood_Data-extra-plik-ext_R3.00.tar.gz'
    data_basename = 'extended_plik/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE_bin1.clik'


class LensingPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
    r"""
    Lensing likelihood of Planck's 2018 data release based on temperature+polarization map-based lensing reconstruction.

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
    data_basename = 'baseline/plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing'


class TTLowlPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
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


class EELowlPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
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
