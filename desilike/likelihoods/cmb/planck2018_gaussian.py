import os
import numpy as np

from desilike.likelihoods import BaseGaussianLikelihood
from desilike.jax import numpy as jnp
from desilike import utils


convert_planck2018_params = {'omegabh2': 'omega_b', 'omegach2': 'omega_cdm', 'omegak': 'Omega_k', 'w': 'w0_fld', 'wa': 'wa_fld', 'theta': 'theta_MC_100', 'tau': 'tau_reio',
                             'mnu': 'm_ncdm', 'logA': 'logA', 'ns': 'n_s', 'nrun': 'alpha_s', 'r': 'r', 'H0': 'H0', 'omegam': 'Omega_m'}


def planck2018_base_fn(basename, data_dir=None):
    """
    Return paths to chains and corresponding summary statistics given input base chain name,
    and data directory ``data_dir``. If ``data_dir`` is ``None``, defaults to path saved in desilike's configuration,
    as provided by :class:`Installer` if :class:`BasePlanck2018GaussianLikelihood` or :class:`FullGridPlanck2018GaussianLikelihood` have been installed.
    """
    if data_dir is None:
        installer_section = 'FullGridPlanck2018GaussianLikelihood'
        from desilike.install import Installer
        try:
            data_dir = Installer()[installer_section]['data_dir']
        except KeyError:
            if basename.startswith('base_plik'):
                installer_section = 'BasePlanck2018GaussianLikelihood'
                data_dir = Installer()[installer_section]['data_dir']
            else:
                raise
    try:
        base_dir, obs_dir = basename.split('_plikHM_')
    except ValueError as exc:
        raise ValueError('basename {0} is expected to contain "_plikHM_"; maybe you forgot to add the model name in front, e.g. base_{0}?'.format(basename)) from exc
    base_chain_fn = os.path.join(data_dir, base_dir, 'plikHM_' + obs_dir, basename)
    base_dist_fn = os.path.join(data_dir, base_dir, 'plikHM_' + obs_dir, 'dist', basename)
    return base_chain_fn, base_dist_fn


def read_planck2018_chain(basename='base_plikHM_TTTEEE_lowl_lowE_lensing', data_dir=None, weights=None, params=None):
    """
    Read Planck chains, operating basic conversion in parameters.

    Parameters
    ----------
    basename : str, default='base_plikHM_TTTEEE_lowl_lowE_lensing'
        Likelihood base name, e.g. 'base_plikHM_TT', 'base_plikHM_TTTEEE', 'base_plikHM_TTTEEE_lowl_lowE_lensing'.

     data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if :class:`BasePlanck2018GaussianLikelihood` or :class:`FullGridPlanck2018GaussianLikelihood` have been installed.

     weights : str, callable, default=None
        Callable that takes a :class:`Chain` as input and returns weights (float),
        e.g. ``weights = lambda chain: 1. / np.exp(chain['logposterior'] + 0.5 * chain['chi2_prior'] + 0.5 * chain['chi2_CMB'])``.
        If ``weights`` is 'cmb_only', the lambda function above is used to "importance unweight" the non-CMB datasets
        (useful e.g. to get an approximation of the CMB-only posterior for :math:`w_{0}` and :math:`w_{a}` extensions).

    params : list, ParameterCollection
        List of parameters to convert the chain to; e.g. ['h', 'Omega_m', 'A_s'].

    Returns
    -------
    chain : Chain
    """
    from desilike.samples import Chain
    base_chain_fn = planck2018_base_fn(basename, data_dir=data_dir)[0]
    chain = Chain.concatenate(Chain.read_getdist(base_chain_fn))

    if weights is not None:
        if isinstance(weights, str):

            if weights.lower() == 'cmb_only':

                def weights(chain):
                    loglikelihood_non_cmb = chain['logposterior'] + 0.5 * chain['chi2_prior'] + 0.5 * chain['chi2_CMB']
                    loglikelihood_non_cmb -= np.mean(loglikelihood_non_cmb)  # remove zero-lag
                    return 1. / np.exp(loglikelihood_non_cmb)

        elif not callable(weights):
            raise ValueError('weights should be a callable, found {}'.format(weights))

        chain.aweight *= weights(chain)

    if params is not None:

        inverse_convert_planck2018_params = {value: name for name, value in convert_planck2018_params.items()}

        def get_from_chain(name):
            planck2018_name = inverse_convert_planck2018_params.get(name, name)
            if planck2018_name in chain:
                return chain[planck2018_name]
            if name == 'A_s':
                return 1e-10 * np.exp(get_from_chain('logA'))
            if name == 'h':
                return get_from_chain('H0') / 100.
            if name.startswith('omega'):
                return get_from_chain('O' + name[1:]) * get_from_chain('h') ** 2
            if name in ['Omega_b', 'Omega_cdm']:
                return get_from_chain('o' + name[1:]) / get_from_chain('h') ** 2

        missing = []
        for param in params:
            name = str(param)
            array = get_from_chain(name)
            if array is None: missing.append(name)
            else: chain[param] = array
        if missing:
            raise ValueError('cannot find parameters {} from chain'.format(missing))
        chain = chain.sort([str(param) for param in params] + [chain._aweight, chain._fweight, chain._logposterior])

    # In case we needed more parameters, we could run desilike's Cosmoprimo in parallel
    return chain


class BasePlanck2018GaussianLikelihood(BaseGaussianLikelihood):
    r"""
    Gaussian approximation of "base" likelihoods of Planck's 2018 data release.

    Reference
    ---------
    https://arxiv.org/abs/1807.06209

    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    Parameters
    ----------
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.

    basename : str, default='base_plikHM_TTTEEE_lowl_lowE_lensing'
        Likelihood base name, e.g. 'base_plikHM_TT', 'base_plikHM_TTTEEE', 'base_plikHM_TTTEEE_lowl_lowE_lensing'.

    source : str, default=None
        Source, either:

        - 'covmat': use '.margestats' for mean and '.covmat' file as covariance.
        - 'chains': compute mean and covariance from chains

        Both options are very close (within precision in provided file).
        Defaults to 'chains' if ``weights`` is not ``None``, else 'covmat'.

    weights : str, callable, default=None
        Callable that takes a :class:`Chain` as input and returns weights (float),
        e.g. ``weights = lambda chain: 1. / np.exp(chain['logposterior'] + 0.5 * chain['chi2_prior'] + 0.5 * chain['chi2_CMB'])``.
        If ``weights`` is 'cmb_only', the lambda function above is used to "importance unweight" the non-CMB datasets
        (useful e.g. to get an approximation of the CMB-only posterior for :math:`w_{0}` and :math:`w_{a}` extensions).
        Only available if ``source`` is 'chains'.
    """
    config_fn = 'planck2018_gaussian.yaml'
    installer_section = 'BasePlanck2018GaussianLikelihood'
    data_file_id = 'COM_CosmoParams_base-plikHM_R3.01.zip'

    def initialize(self, cosmo=None, data_dir=None, basename='base_plikHM_TTTEEE_lowl_lowE_lensing', source=None, weights=None):
        self.name = basename
        self.base_chain_fn, self.base_dist_fn = planck2018_base_fn(basename, data_dir=data_dir)
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        basenames = ['omegabh2', 'omegach2', 'omegak', 'w', 'wa', 'theta', 'tau', 'mnu', 'logA', 'ns', 'nrun', 'r']
        if source is None:
            source = 'covmat' if weights is None else 'chains'
        if source == 'covmat':
            if weights: raise ValueError('use source = "chains" to reweight chains')
            from desilike import LikelihoodFisher
            self.fisher = LikelihoodFisher.read_getdist(self.base_dist_fn, basename=basenames)
        elif source == 'chains':
            chain = read_planck2018_chain(basename=basename, data_dir=data_dir, weights=weights)
            self.fisher = chain.select(basename=basenames).to_fisher()
        else:
            raise ValueError('source must be one of ["covmat", "chains"]')
        for param in self.fisher.params(): param.update(name=convert_planck2018_params[param.name])
        params = self.fisher.params()
        self.cosmo_quantities = params.basenames()
        super(BasePlanck2018GaussianLikelihood, self).initialize(data=self.fisher.mean(params=params), covariance=self.fisher.covariance(params=params))

    @property
    def flattheory(self):
        return jnp.array([self.cosmo[param] for param in self.cosmo_quantities])

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        if installer.force_reinstall or not exists_path(os.path.join(data_dir, 'base')):
            # Install data
            tar_base, size = cls.data_file_id, None
            if utils.is_sequence(tar_base):
                tar_base, size = cls.data_file_id
            url = 'http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID={}'.format(tar_base)
            tar_fn = os.path.join(data_dir, tar_base)
            download(url, tar_fn, size=size)
            extract(tar_fn, data_dir)
            installer.write({cls.installer_section: {'data_dir': data_dir}})


class FullGridPlanck2018GaussianLikelihood(BasePlanck2018GaussianLikelihood):
    r"""
    Gaussian approximation of the full grid of likelihoods of Planck's 2018 data release.

    Reference
    ---------
    https://arxiv.org/abs/1807.06209

    https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    Parameters
    ----------
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.

    basename : str, default='base_plikHM_TTTEEE_lowl_lowE_lensing'
        Likelihood base name, e.g. 'base_plikHM_TT', 'base_plikHM_TTTEEE', 'base_plikHM_TTTEEE_lowl_lowE_lensing', 'base_mnu_plikHM_TTTEEE_lowl_lowE_lensing'.

    source : str, default='covmat'
        Source, either:

        - 'covmat': use '.margestats' for mean and '.covmat' file as covariance.
        - 'chains': compute mean and covariance from chains

        Both options are very close (within precision in provided file).
    """
    config_fn = 'planck2018_gaussian.yaml'
    installer_section = 'FullGridPlanck2018GaussianLikelihood'
    data_file_id = ('COM_CosmoParams_fullGrid_R3.01.zip', 11e9)
