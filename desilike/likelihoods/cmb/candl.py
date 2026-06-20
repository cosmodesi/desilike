"""Generic wrappers for candl-based CMB likelihoods (https://github.com/Lbalkenhol/candl),
plus concrete ACT DR6 likelihoods (data from https://github.com/Lbalkenhol/candl_data) and
SPT-3G likelihoods (data from https://github.com/SouthPoleTelescope/spt_candl_data)."""

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
        if self._ellmax_standard:
            self.cosmo.add_requirements({'harmonic.lensed_cl': [{'ellmax': self._ellmax_standard}]})
        if self._ellmax_potential:
            self.cosmo.add_requirements({'harmonic.lens_potential_cl': [{'ellmax': self._ellmax_potential}]})

        self._cosmo_params = dict(cosmo_params or {})
        self._cosmo_prior_names = [name for name in self.like.required_prior_parameters if name not in self.like.required_nuisance_parameters]
        for name in self._cosmo_prior_names:
            self.cosmo.add_requirements({'params.{}'.format(self._cosmo_params.get(name, name)): None})

        vc, self._split_param_names = self.propose_params(split_diag_priors=split_diag_priors)
        if params is not None:
            vc = vc + VariableCollection(params)
        self.params = {param.basename: param for param in vc}

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
