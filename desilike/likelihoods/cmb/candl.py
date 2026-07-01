"""Generic wrappers for candl-based CMB likelihoods (https://github.com/Lbalkenhol/candl),
plus concrete ACT DR6 likelihoods (data from https://github.com/Lbalkenhol/candl_data),
SPT-3G likelihoods (data from https://github.com/SouthPoleTelescope/spt_candl_data),
and Planck PR3 likelihoods via clipy.clik_candl (https://github.com/benabed/clipy)."""

import os

import numpy as np
import jax.numpy as jnp

from desilike.base import Likelihood
from desilike.parameter import Parameter, VariableCollection


class _BaseCandlLikelihood(Likelihood):
    r"""Shared setup for :class:`CandlLikelihood` (wraps ``candl.Like``) and
    :class:`CandlLensLikelihood` (wraps ``candl.LensLike``).

    candl is data-driven: a single ``.yaml`` file fully describes the data set (band
    powers, covariance, window functions, and the foreground/calibration nuisance
    model, optionally with Gaussian priors on some nuisance parameters). This wrapper
    builds the corresponding desilike nuisance :class:`~desilike.parameter.Parameter`\ s
    automatically from ``required_nuisance_parameters`` (the list of names) and
    ``priors`` (their Gaussian priors, where declared), and forwards the theory
    :math:`D_\ell` spectra computed by ``cosmo`` to ``log_like``, for every spectrum
    type in ``requirements_dict['Dl']`` (which, for :class:`CandlLensLikelihood`, may
    include both the lensing spectrum, e.g. ``'kk'``, and primary-CMB spectra needed by
    a normalization-correction transformation). ``'tt'``/``'te'``/``'ee'``/``'bb'`` are
    built from ``cosmo.get_harmonic().lensed_cl`` (raw :math:`C_\ell` converted to
    :math:`D_\ell` in :math:`\mu K^2`); ``'kk'`` is built from
    ``cosmo.get_harmonic().lens_potential_cl``'s raw :math:`C_\ell^{\phi\phi}` via
    :math:`D_\ell^{\kappa\kappa} = [\ell(\ell+1)]^2/4 \cdot C_\ell^{\phi\phi}` (the
    standard convergence/potential relation; cross-checked against candl's own Cobaya
    interface, which uses the equivalent ``Dl_kk = Dl_pp * pi/2`` on its own
    :math:`\ell`-weighted potential spectrum). Any other spectrum type is filled with
    zeros (e.g. ``'tb'``/``'eb'``, which vanish by parity, or ``'pp'``, not handled by
    this generic builder).

    Note
    ----
    By default (``split_diag_priors=False``), candl's own ``log_like`` keeps applying
    any Gaussian priors declared in the data set's ``priors`` block (which may be
    correlated across nuisance parameters); those priors are therefore *not*
    duplicated as desilike ``Parameter`` priors (which would double-count them when
    summed by :class:`~desilike.base.Prior`). The corresponding parameters are
    instead created free (``prior=None``), with ``value``/``ref`` set from the candl
    prior's central value and (diagonal) standard deviation purely to give
    samplers/profilers a sensible starting point and proposal scale.

    Set ``split_diag_priors=True`` to additionally attach the same central
    value/standard deviation as a proper desilike ``Parameter.prior`` (so it is
    visible to samplers/profilers as a real prior, e.g. in corner plots), while
    *keeping* candl's internal priors untouched (so any off-diagonal correlation
    within a joint, multi-parameter candl ``GaussianPrior`` is still handled correctly
    by candl). To avoid double-counting the now-duplicated diagonal piece,
    :meth:`__call__` subtracts ``self.params[name].prior.logpdf(value)`` for each
    split parameter from the value returned by candl's ``log_like`` — exactly
    cancelling what :class:`~desilike.base.Prior` will add back later, so the total
    posterior is numerically equivalent to the ``split_diag_priors=False`` case
    (this holds regardless of any off-diagonal correlation, since only the diagonal
    contribution is ever moved, never double-counted or dropped).

    Nuisance parameters with no declared candl prior default to ``value=1.`` (the
    common convention for candl's multiplicative calibration-like parameters) with a
    unit ``ref`` scale and no prior either way; override via ``params=`` if this is
    not appropriate.

    Some data sets declare a Gaussian prior on a parameter that is *not* a nuisance
    parameter of the data model, but an ordinary cosmological parameter under a
    different name (e.g. ACT DR6 TT/TE/EE's prior on ``'tau'``, candl's name for the
    optical depth desilike/cosmoprimo calls ``'tau_reio'``). Such parameters are
    listed in ``cosmo_params`` (candl name -> cosmoprimo name) and are read directly
    from ``cosmo`` rather than created as new desilike Parameters; candl always
    applies their prior internally (``split_diag_priors`` does not apply to them,
    since the underlying Parameter belongs to ``cosmo``, not to this Likelihood).

    Parameters
    ----------
    data_set_file : str, Path
        Path to the candl data set ``.yaml`` file (or an index file; see ``variant``).
    variant : str, default=None
        Variant to select, if ``data_set_file`` is an index file.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator. If ``None``, defaults to ``CosmoprimoCosmology(engine='camb', fiducial='DESI')``
        (a Boltzmann engine is required to compute CMB :math:`C_\ell`).
    params : Parameter, VariableCollection, dict, default=None
        Override the default nuisance parameters.
    split_diag_priors : bool, default=False
        If ``True``, expose candl's declared nuisance-parameter priors as desilike
        ``Parameter`` priors too, while still letting candl handle the full
        (possibly correlated) prior internally (see Note above).
    cosmo_params : dict, default=None
        Maps a candl prior-only parameter name to the corresponding cosmoprimo
        parameter name (see Note above).
    **kwargs
        Forwarded to the underlying candl object (e.g. additional data-set overrides).
    """
    _candl_attr = None  # 'Like' or 'LensLike', set by subclasses
    # Fixed CMB temperature used for the dimensionless-Cl -> muK^2 unit conversion
    # (matches cmb/camspec.py; not tied to the cosmology's T_cmb parameter).
    T0_cmb = 2.7255

    def __init__(self, data_set_file, variant=None, cosmo=None, params=None, split_diag_priors=False, cosmo_params=None, **kwargs):
        import candl

        if variant is not None:
            kwargs['variant'] = variant
        self.like = getattr(candl, self._candl_attr)(data_set_file, **kwargs)

        dl_requirements = self.like.requirements_dict.get('Dl', {})
        self._ellmax_standard = max([ellmax for name, ellmax in dl_requirements.items() if name.lower() != 'kk'], default=0)
        self._ellmax_potential = max([ellmax for name, ellmax in dl_requirements.items() if name.lower() == 'kk'], default=0)

        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            ellmax_cl = max(self._ellmax_standard, self._ellmax_potential)
            cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=ellmax_cl, non_linear='mead')))
        self.cosmo = cosmo
        self._cosmo_params = dict(cosmo_params or {})
        self._cosmo_prior_names = [name for name in self.like.required_prior_parameters if name not in self.like.required_nuisance_parameters]
        vc, self._split_param_names = self.propose_params(split_diag_priors=split_diag_priors)
        if params is not None:
            vc = vc + VariableCollection(params)
        self.params = {param.basename: param for param in vc}

    def __post_init__(self, *args, **kwargs):
        if self._ellmax_standard:
            self.cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': self._ellmax_standard}]})
        if self._ellmax_potential:
            self.cosmo.add_requirements({'harmonic.lens_potential_cl': [{'ellmax': self._ellmax_potential}]})
        for name in self._cosmo_prior_names:
            self.cosmo.add_requirements({'params.{}'.format(self._cosmo_params.get(name, name)): None})

    def propose_params(self, split_diag_priors=False):
        """Build one free desilike Parameter per ``self.like.required_nuisance_parameters``,
        using ``self.like.priors`` (candl's ``GaussianPrior`` list) for value/ref where
        declared. If ``split_diag_priors``, also attach those as desilike Parameter priors
        (diagonal-only; see the class Note). Returns ``(VariableCollection, split_names)``."""
        prior_info = {}
        for prior in self.like.priors:
            variance = np.diag(np.atleast_2d(np.asarray(prior.prior_covariance)))
            central_value = np.atleast_1d(np.asarray(prior.central_value))
            for name, center, var in zip(prior.par_names, central_value, variance):
                prior_info[name] = (float(center), float(var) ** 0.5)

        params, split_names = [], set()
        for name in self.like.required_nuisance_parameters:
            center, std = prior_info.get(name, (1., 1.))
            use_prior = split_diag_priors and name in prior_info
            prior = dict(dist='norm', loc=center, scale=std) if use_prior else None
            if use_prior:
                split_names.add(name)
            params.append(Parameter(name, value=center, prior=prior, ref=dict(dist='norm', loc=center, scale=std), fixed=False))
        return VariableCollection(params), split_names

    @property
    def ndata(self):
        return self.like.N_bins_total

    def _build_Dl(self):
        """Return the ``Dl`` dict candl expects: one array per ``requirements_dict['Dl']``
        spectrum type, covering ell = 2 to that spectrum's own ell_max."""
        dl_requirements = self.like.requirements_dict.get('Dl', {})
        cl_lensed = self.cosmo.get_harmonic().lensed_cl(ellmax=self._ellmax_standard) if self._ellmax_standard else {}
        cl_potential = self.cosmo.get_harmonic().lens_potential_cl(ellmax=self._ellmax_potential) if self._ellmax_potential else None

        Dl = {}
        for spec_type, ellmax in dl_requirements.items():
            key = spec_type.lower()
            if key == 'kk':
                ells = jnp.arange(self._ellmax_potential + 1)
                dl_full = (ells * (ells + 1)) ** 2 / 4 * cl_potential['pp']
            elif key in cl_lensed:
                ells = jnp.arange(self._ellmax_standard + 1)
                factor = ells * (ells + 1) / 2 / np.pi
                unit = (self.T0_cmb * 1e6) ** 2
                dl_full = factor * unit * cl_lensed[key]
            else:
                dl_full = jnp.zeros(ellmax + 1)  # e.g. 'tb', 'eb', 'pp': not handled by this generic builder
            Dl[spec_type] = dl_full[2:ellmax + 1]
        return Dl

    def __call__(self):
        params_dict = {name: param.value for name, param in self.params.items()}
        for name in self._cosmo_prior_names:
            params_dict[name] = self.cosmo[self._cosmo_params.get(name, name)]
        params_dict['Dl'] = self._build_Dl()

        logpdf = self.like.log_like(params_dict)
        for name in self._split_param_names:
            # Remove the diagonal piece now owned by desilike's Parameter.prior (added back by
            # the Prior calculator), to avoid double-counting; see the class Note for why this
            # exact cancellation holds even when the underlying candl prior is correlated.
            param = self.params[name]
            logpdf = logpdf - param.prior.logpdf(param.value)
        self.logpdf = logpdf
        return self.logpdf

    @classmethod
    def install(cls, installer):
        installer.pip('candl-like')


class CandlLikelihood(_BaseCandlLikelihood):
    """Generic wrapper around a `candl <https://github.com/Lbalkenhol/candl>`_ ``Like``
    (primary CMB power-spectrum) likelihood. See :class:`_BaseCandlLikelihood`."""
    _candl_attr = 'Like'


class CandlLensLikelihood(_BaseCandlLikelihood):
    """Generic wrapper around a `candl <https://github.com/Lbalkenhol/candl>`_ ``LensLike``
    (CMB lensing) likelihood. See :class:`_BaseCandlLikelihood`."""
    _candl_attr = 'LensLike'


class ACTDR6TTTEEELikelihood(CandlLikelihood):
    """
    ACT DR6 TT/TE/EE primary CMB likelihood.

    Data from `candl_data <https://github.com/Lbalkenhol/candl_data>`_.

    Reference
    ---------
    https://arxiv.org/abs/2503.14451
    https://arxiv.org/abs/2503.14452
    https://arxiv.org/abs/2503.14454
    """
    def __init__(self, cosmo=None, params=None, split_diag_priors=False, cosmo_params=None, **kwargs):
        import candl_data
        cosmo_params = dict(cosmo_params or {})
        cosmo_params.setdefault('tau', 'tau_reio')
        super().__init__(candl_data.ACT_DR6_TTTEEE, cosmo=cosmo, params=params,
                          split_diag_priors=split_diag_priors, cosmo_params=cosmo_params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        installer.pip('git+https://github.com/Lbalkenhol/candl_data.git')


class ACTDR6LensingLikelihood(CandlLensLikelihood):
    """
    ACT DR6 CMB lensing likelihood.

    Data from `candl_data <https://github.com/Lbalkenhol/candl_data>`_.

    Parameters
    ----------
    variant : str, default='lens_only'
        ``'lens_only'`` (lensing power spectrum alone) or ``'use_CMB'`` (adds the
        normalization correction needed when combining with a primary CMB likelihood,
        which additionally requires primary TT/TE/EE/BB theory spectra).

    Reference
    ---------
    https://arxiv.org/abs/2304.05203
    https://arxiv.org/abs/2304.05202
    https://arxiv.org/abs/2206.07773
    """
    def __init__(self, variant='lens_only', cosmo=None, params=None, split_diag_priors=False, cosmo_params=None, **kwargs):
        import candl_data
        super().__init__(candl_data.ACT_DR6_Lens, variant=variant, cosmo=cosmo, params=params,
                          split_diag_priors=split_diag_priors, cosmo_params=cosmo_params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        installer.pip('git+https://github.com/Lbalkenhol/candl_data.git')


class SPT3GD1TnELikelihood(CandlLikelihood):
    """
    SPT-3G D1 TT/TE/EE primary CMB likelihood.

    Data from `spt_candl_data <https://github.com/SouthPoleTelescope/spt_candl_data>`_.

    Parameters
    ----------
    variant : str, default=None
        ``'multifreq'`` (candl's own default; full multi-frequency likelihood with ~40
        foreground/calibration/beam nuisance parameters, including a correlated
        multi-parameter prior on 9 beam eigenmodes), ``'lite'`` (foreground-marginalised
        CMB-lite version: just ``'Tcal'``/``'Ecal'``), or any other variant declared in
        the data set's index file (e.g. a single spectrum/frequency subset such as
        ``'TT'`` or ``'TE_90x90'``).

    Reference
    ---------
    https://pole.uchicago.edu/public/Home.html
    """
    def __init__(self, variant=None, cosmo=None, params=None, split_diag_priors=False, cosmo_params=None, **kwargs):
        import spt_candl_data
        cosmo_params = dict(cosmo_params or {})
        cosmo_params.setdefault('tau', 'tau_reio')
        super().__init__(spt_candl_data.SPT3G_D1_TnE, variant=variant, cosmo=cosmo, params=params,
                          split_diag_priors=split_diag_priors, cosmo_params=cosmo_params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        installer.pip('git+https://github.com/SouthPoleTelescope/spt_candl_data.git')


# ── Planck PR3 likelihoods via clipy.clik_candl ───────────────────────────────

# PLA baseline tarball contains all clik directories under plc_3.0/:
_PLA_BASELINE_URL = ('https://pla.esac.esa.int/pla/aio/product-action'
                     '?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz')
# Paths inside the tarball for each likelihood (directory name = installer_section default key)
_CLIK_TAR_PATHS = {
    'plik_rd12_HM_v22_TT.clik':                          'baseline/plc_3.0/hi_l/plik/',
    'plik_rd12_HM_v22b_TTTEEE.clik':                     'baseline/plc_3.0/hi_l/plik/',
    'plik_lite_v22_TT.clik':                             'baseline/plc_3.0/hi_l/plik_lite_v22/',
    'plik_lite_v22_TTTEEE.clik':                         'baseline/plc_3.0/hi_l/plik_lite_v22/',
    'commander_dx12_v3_2_29.clik':                        'baseline/plc_3.0/lo_l/commander/',
    'simall_100x143_offdiag_EE_Aplanck_B.clik':          'baseline/plc_3.0/lo_l/simall/',
}


class _BaseClikCandlLikelihood(Likelihood):
    r"""Shared setup for Planck PR3 likelihoods backed by ``clipy.clik_candl``.

    ``clipy.clik_candl`` reads a Planck ``.clik`` directory produced by the
    Planck Likelihood Code (PLC) and exposes the same ``log_like`` / nuisance-
    parameter interface as ``candl.Like``, so the same :math:`D_\ell`-forwarding
    machinery used by :class:`_BaseCandlLikelihood` applies here.

    The nuisance parameters (calibration, foreground amplitudes, …) are
    discovered from ``clik_candl.required_nuisance_parameters`` at construction.
    Default values and proposal scales are seeded from the Gaussian priors
    stored in the ``.clik`` file itself (the same priors that are applied
    internally by clipy inside ``log_like``); they are *not* duplicated as
    desilike ``Parameter`` priors to avoid double-counting.

    Parameters
    ----------
    clik_file : str, Path
        Path to the ``.clik`` directory.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.  Defaults to
        ``CosmoprimoCosmology(engine='camb', fiducial='DESI')`` with
        lensing and ``ellmax_cl`` matched to the clik file's requirements.
    params : Parameter, VariableCollection, dict, default=None
        Override the auto-discovered nuisance parameters.
    **kwargs
        Forwarded verbatim to ``clipy.clik_candl`` (e.g. ``all_priors=True``,
        ``cosmomc_names=True``, ``data_selection=[...]``).
    """

    T0_cmb = 2.7255

    def __init__(self, clik_file, cosmo=None, params=None, **kwargs):
        import clipy
        self.like = clipy.clik_candl(clik_file, **kwargs)

        # Per-spectrum ell_max from the clik file's lmax array.
        self._spec_ellmax = {spec: int(self.like.lmax[self.like._dr.index(spec)])
                             for spec in self.like.unique_spec_types}
        self._ellmax_standard = max((e for s, e in self._spec_ellmax.items() if s.lower() != 'pp'), default=0)
        self._ellmax_potential = max((e for s, e in self._spec_ellmax.items() if s.lower() == 'pp'), default=0)

        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            ellmax_cl = max(self._ellmax_standard, self._ellmax_potential)
            cosmo = CosmoprimoCosmology(engine='camb', fiducial=('DESI', dict(lensing=True, ellmax_cl=ellmax_cl, non_linear='mead')))
        self.cosmo = cosmo

        vc = self.propose_params()
        if params is not None:
            vc = vc + VariableCollection(params)
        self.params = {param.basename: param for param in vc}

    def __post_init__(self, *args, **kwargs):
        if self._ellmax_standard:
            self.cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': self._ellmax_standard}]})
        if self._ellmax_potential:
            self.cosmo.add_requirements({'harmonic.lens_potential_cl': [{'ellmax': self._ellmax_potential}]})

    def propose_params(self):
        """Build one free desilike Parameter per required nuisance parameter.

        Default values and ref scales come from the single-parameter Gaussian
        priors stored in the ``.clik`` file (``self.like._prior``); parameters
        without a declared prior default to ``value=1.``.  Returns a
        ``VariableCollection``.
        """
        prior_info = {}
        for key, value in self.like._prior.items():
            if isinstance(key, str) and isinstance(value, tuple) and len(value) == 2:
                loc, var = float(value[0]), float(value[1])
                prior_info[key] = (loc, var ** 0.5)

        params = []
        for name in self.like.required_nuisance_parameters:
            center, std = prior_info.get(name, (1., 0.01))
            params.append(Parameter(name, value=center,
                                    ref=dict(dist='norm', loc=center, scale=std),
                                    fixed=False))
        return VariableCollection(params)

    def _build_Dl(self):
        """Return the ``Dl`` dict that ``clik_candl.log_like`` expects.

        Keys are the uppercase spectrum-type strings (``'TT'``, ``'TE'``,
        ``'EE'``, ...) returned by ``clik_candl.unique_spec_types``.  Each value
        is a JAX array covering ell = 2 to that spectrum's ell_max.
        """
        cl_lensed = self.cosmo.get_harmonic().lensed_cl(ellmax=self._ellmax_standard) if self._ellmax_standard else {}
        cl_potential = self.cosmo.get_harmonic().lens_potential_cl(ellmax=self._ellmax_potential) if self._ellmax_potential else None
        Dl = {}
        for spec_type, ellmax in self._spec_ellmax.items():
            key = spec_type.lower()
            if key == 'pp':
                ells = jnp.arange(self._ellmax_potential + 1)
                dl_full = (ells * (ells + 1)) ** 2 / 4 * cl_potential['pp']
            elif key in cl_lensed:
                ells = jnp.arange(self._ellmax_standard + 1)
                factor = ells * (ells + 1) / 2 / np.pi
                unit = (self.T0_cmb * 1e6) ** 2
                dl_full = factor * unit * cl_lensed[key]
            else:
                dl_full = jnp.zeros(ellmax + 1)
            Dl[spec_type] = dl_full[2:ellmax + 1]
        return Dl

    def __call__(self):
        params_dict = {name: param.value for name, param in self.params.items()}
        params_dict['Dl'] = self._build_Dl()
        self.logpdf = self.like.log_like(params_dict)
        return self.logpdf

    @classmethod
    def install(cls, installer):
        installer.pip('clipy-like')


class ClikCandlLikelihood(_BaseClikCandlLikelihood):
    r"""Generic wrapper around a ``clipy.clik_candl`` Planck ``.clik`` likelihood.

    Use this class directly when you have a ``.clik`` directory on disk and do
    not need auto-install support.  For the standard Planck PR3 high-ℓ / low-ℓ
    likelihoods, prefer the concrete subclasses below.

    Parameters
    ----------
    clik_file : str, Path
        Path to the ``.clik`` directory produced by the Planck Likelihood Code.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.  See :class:`_BaseClikCandlLikelihood`.
    params : Parameter, VariableCollection, dict, default=None
        Override the auto-discovered nuisance parameters.
    **kwargs
        Forwarded to ``clipy.clik_candl`` (e.g. ``all_priors=True``).
    """


def _install_clik(installer, installer_section, clik_basename):
    """Download the PLA baseline tarball and extract *clik_basename* to data_dir."""
    try:
        data_dir = installer[installer_section]['data_dir']
    except KeyError:
        data_dir = installer.data_dir(installer_section)

    from desilike.install import exists_path, download
    target = os.path.join(data_dir, clik_basename)
    if installer.reinstall or not exists_path(target):
        tar_path = os.path.join(data_dir, 'COM_Likelihood_Data-baseline_R3.00.tar.gz')
        download(_PLA_BASELINE_URL, tar_path)
        # Extract only the requested .clik directory from the tarball.
        import tarfile
        prefix = _CLIK_TAR_PATHS[clik_basename] + clik_basename + '/'
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = [m for m in tar.getmembers()
                       if m.name == prefix.rstrip('/') or m.name.startswith(prefix)]
            for member in members:
                member.name = os.path.relpath(member.name, _CLIK_TAR_PATHS[clik_basename])
                tar.extract(member, data_dir)
        os.remove(tar_path)

    installer.write({installer_section: {'data_dir': data_dir}})


class PlanckPR3TTLikelihood(_BaseClikCandlLikelihood):
    r"""Planck PR3 plik CMB-only high-:math:`\ell` TT likelihood.

    Wraps ``plik_rd12_HM_v22_TT.clik`` via ``clipy.clik_candl``.

    The only free nuisance parameter is ``A_planck`` (overall calibration).
    An optional Gaussian prior :math:`\mathcal{N}(1,\,0.0025^2)` on
    ``A_planck`` is available via ``all_priors=True``.

    Parameters
    ----------
    clik_file : str, Path, default=None
        Path to ``plik_rd12_HM_v22_TT.clik``.  When ``None`` the path saved
        by :class:`~desilike.install.Installer` is used.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.
    params : Parameter, VariableCollection, dict, default=None
        Override nuisance parameters.
    **kwargs
        Forwarded to ``clipy.clik_candl`` (e.g. ``all_priors=True``).

    Reference
    ---------
    Planck 2018 results V  https://arxiv.org/abs/1907.12875
    """

    installer_section = 'PlanckPR3TTLikelihood'
    _clik_basename = 'plik_rd12_HM_v22_TT.clik'

    def __init__(self, clik_file=None, cosmo=None, params=None, **kwargs):
        if clik_file is None:
            from desilike.install import Installer
            clik_file = os.path.join(Installer().data_dir(self.installer_section), self._clik_basename)
        super().__init__(clik_file, cosmo=cosmo, params=params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        _install_clik(installer, cls.installer_section, cls._clik_basename)


class PlanckPR3TTTEEELikelihood(_BaseClikCandlLikelihood):
    r"""Planck PR3 plik CMB-only high-:math:`\ell` TT/TE/EE likelihood.

    Wraps ``plik_rd12_HM_v22b_TTTEEE.clik`` via ``clipy.clik_candl``.

    Parameters
    ----------
    clik_file : str, Path, default=None
        Path to ``plik_rd12_HM_v22b_TTTEEE.clik``.  When ``None`` the path
        saved by :class:`~desilike.install.Installer` is used.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.
    params : Parameter, VariableCollection, dict, default=None
        Override nuisance parameters.
    **kwargs
        Forwarded to ``clipy.clik_candl`` (e.g. ``all_priors=True``).

    Reference
    ---------
    Planck 2018 results V  https://arxiv.org/abs/1907.12875
    """

    installer_section = 'PlanckPR3TTTEEELikelihood'
    _clik_basename = 'plik_rd12_HM_v22b_TTTEEE.clik'

    def __init__(self, clik_file=None, cosmo=None, params=None, **kwargs):
        if clik_file is None:
            from desilike.install import Installer
            clik_file = os.path.join(Installer().data_dir(self.installer_section), self._clik_basename)
        super().__init__(clik_file, cosmo=cosmo, params=params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        _install_clik(installer, cls.installer_section, cls._clik_basename)


class PlanckPR3TTTEEELiteLikelihood(_BaseClikCandlLikelihood):
    r"""Planck PR3 plik-lite foreground-marginalized high-:math:`\ell` TT/TE/EE likelihood.

    Wraps ``plik_lite_v22_TTTEEE.clik`` via ``clipy.clik_candl``.

    All foreground and beam nuisance parameters are pre-marginalized; only
    ``A_planck`` remains as a free nuisance parameter.

    Parameters
    ----------
    clik_file : str, Path, default=None
        Path to ``plik_lite_v22_TTTEEE.clik``.  When ``None`` the path saved
        by :class:`~desilike.install.Installer` is used.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.
    params : Parameter, VariableCollection, dict, default=None
        Override nuisance parameters.
    **kwargs
        Forwarded to ``clipy.clik_candl`` (e.g. ``all_priors=True``).

    Reference
    ---------
    Planck 2018 results V  https://arxiv.org/abs/1907.12875
    """

    installer_section = 'PlanckPR3TTTEEELiteLikelihood'
    _clik_basename = 'plik_lite_v22_TTTEEE.clik'

    def __init__(self, clik_file=None, cosmo=None, params=None, **kwargs):
        if clik_file is None:
            from desilike.install import Installer
            clik_file = os.path.join(Installer().data_dir(self.installer_section), self._clik_basename)
        super().__init__(clik_file, cosmo=cosmo, params=params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        _install_clik(installer, cls.installer_section, cls._clik_basename)


class PlanckPR3LowlTTLikelihood(_BaseClikCandlLikelihood):
    r"""Planck PR3 Commander low-:math:`\ell` TT likelihood (:math:`2 \le \ell \le 29`).

    Wraps ``commander_dx12_v3_2_29.clik`` via ``clipy.clik_candl``.

    Parameters
    ----------
    clik_file : str, Path, default=None
        Path to ``commander_dx12_v3_2_29.clik``.  When ``None`` the path saved
        by :class:`~desilike.install.Installer` is used.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.
    params : Parameter, VariableCollection, dict, default=None
        Override nuisance parameters.
    **kwargs
        Forwarded to ``clipy.clik_candl``.

    Reference
    ---------
    Planck 2018 results V  https://arxiv.org/abs/1907.12875
    """

    installer_section = 'PlanckPR3LowlTTLikelihood'
    _clik_basename = 'commander_dx12_v3_2_29.clik'

    def __init__(self, clik_file=None, cosmo=None, params=None, **kwargs):
        if clik_file is None:
            from desilike.install import Installer
            clik_file = os.path.join(Installer().data_dir(self.installer_section), self._clik_basename)
        super().__init__(clik_file, cosmo=cosmo, params=params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        _install_clik(installer, cls.installer_section, cls._clik_basename)


class PlanckPR3LowlEELikelihood(_BaseClikCandlLikelihood):
    r"""Planck PR3 SimAll low-:math:`\ell` EE likelihood (:math:`2 \le \ell \le 29`).

    Wraps ``simall_100x143_offdiag_EE_Aplanck_B.clik`` via ``clipy.clik_candl``.

    Parameters
    ----------
    clik_file : str, Path, default=None
        Path to ``simall_100x143_offdiag_EE_Aplanck_B.clik``.  When ``None``
        the path saved by :class:`~desilike.install.Installer` is used.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.
    params : Parameter, VariableCollection, dict, default=None
        Override nuisance parameters.
    **kwargs
        Forwarded to ``clipy.clik_candl``.

    Reference
    ---------
    Planck 2018 results V  https://arxiv.org/abs/1907.12875
    """

    installer_section = 'PlanckPR3LowlEELikelihood'
    _clik_basename = 'simall_100x143_offdiag_EE_Aplanck_B.clik'

    def __init__(self, clik_file=None, cosmo=None, params=None, **kwargs):
        if clik_file is None:
            from desilike.install import Installer
            clik_file = os.path.join(Installer().data_dir(self.installer_section), self._clik_basename)
        super().__init__(clik_file, cosmo=cosmo, params=params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        _install_clik(installer, cls.installer_section, cls._clik_basename)


class PlanckPR3LowlEESroll2Likelihood(_BaseClikCandlLikelihood):
    r"""Planck PR3 Sroll2 low-:math:`\ell` EE likelihood (:math:`2 \le \ell \le 29`).

    Wraps ``simall_100x143_sroll2_v3_EE_Aplanck.clik`` via ``clipy.clik_candl``.
    The Sroll2 reprocessing of Planck PR3 polarization data provides an alternative
    to the standard SimAll EE likelihood with improved systematic control.

    Parameters
    ----------
    clik_file : str, Path, default=None
        Path to ``simall_100x143_sroll2_v3_EE_Aplanck.clik``.  When ``None``
        the path saved by :class:`~desilike.install.Installer` is used.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator.
    params : Parameter, VariableCollection, dict, default=None
        Override nuisance parameters.
    **kwargs
        Forwarded to ``clipy.clik_candl``.

    Reference
    ---------
    Pagano et al. 2020  https://arxiv.org/abs/1908.09856
    Data               https://web.fe.infn.it/~pagano/low_ell_datasets/sroll2/
    """

    installer_section = 'PlanckPR3LowlEESroll2Likelihood'
    _clik_basename = 'simall_100x143_sroll2_v3_EE_Aplanck.clik'
    _tgz_url = 'https://web.fe.infn.it/~pagano/low_ell_datasets/sroll2/simall_100x143_sroll2_v3_EE_Aplanck.tgz'

    def __init__(self, clik_file=None, cosmo=None, params=None, **kwargs):
        if clik_file is None:
            from desilike.install import Installer
            clik_file = os.path.join(Installer().data_dir(self.installer_section), self._clik_basename)
        super().__init__(clik_file, cosmo=cosmo, params=params, **kwargs)

    @classmethod
    def install(cls, installer):
        super().install(installer)
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract
        target = os.path.join(data_dir, cls._clik_basename)
        if installer.reinstall or not exists_path(target):
            tgz_fn = os.path.join(data_dir, cls._clik_basename + '.tgz')
            download(cls._tgz_url, tgz_fn)
            extract(tgz_fn, data_dir)

        installer.write({cls.installer_section: {'data_dir': data_dir}})
