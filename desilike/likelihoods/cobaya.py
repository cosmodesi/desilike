r"""Generic adapter: wrap a cobaya ``Likelihood`` as a desilike ``Likelihood``.

Rather than hand-porting each cobaya likelihood's physics to JAX (as
``desilike/likelihoods/cmb/camspec.py`` and ``desilike/likelihoods/bbn/__init__.py`` do),
:class:`CobayaLikelihood` constructs the *unmodified* cobaya class standalone, translates
its ``get_requirements()`` into calls to ``cosmo.add_requirements(...)``, and feeds it a
:class:`_Provider` shim so its own ``logp()``/``calculate()`` runs as-is against a desilike
:class:`~desilike.theories.primordial_cosmology.PrimordialCosmology`.

Because the wrapped likelihood is plain numpy/scipy cobaya code, :class:`CobayaLikelihood`
is ``_is_external = True``: it runs through desilike's ``pure_callback`` + finite-difference
path, exactly like :class:`~desilike.theories.primordial_cosmology.CosmoprimoCosmology` does
for external Boltzmann engines.

Scope
-----
The requirement translation table below mirrors what ``CosmoprimoCosmology`` (see
``desilike/theories/primordial_cosmology.py``) exposes -- itself modeled on cosmoprimo's own
cobaya binding, ``cosmoprimo/bindings/cobaya/cosmoprimo.py``. Supported: ``Cl``/``unlensed_Cl``
(optionally with lensing potential), ``Hubble``, ``angular_diameter_distance``,
``comoving_radial_distance``, ``sigma8_z``, ``fsigma8``, ``sigma_R``, z-dependent
``Omega_b``/``Omega_cdm``/``Omega_nu_massive``, linear-only ``Pk_grid``/``Pk_interpolator``
(including cross-spectra, e.g. ``('delta_tot', 'theta_cb')``), and derived/theory parameters
(``params.<name>``, e.g. ``rdrag``). Still unsupported (raises ``NotImplementedError`` naming
the exact key): ``lensed_scal_Cl`` (no scalar-only-lensed equivalent in cosmoprimo),
``source_Cl`` (arbitrary source windows), ``CAMBdata`` (engine-specific escape hatch),
``angular_diameter_distance_2`` (cross-redshift distance), and non-linear ``Pk``
(``CosmoprimoCosmology``'s ``'fourier.pk'`` requirement is linear-only today).

Naming conventions: ``CosmoprimoCosmology``'s requirement method keys and kwargs follow
cosmoprimo's own naming exactly (e.g. ``'fourier.sigma_rz'`` with kwarg ``r``, and
``'background.Omega_ncdm_tot'``, not cobaya's ``sigma_R``/``R`` or ``Omega_nu_massive``). All
cobaya-name -> cosmoprimo-name translation (``_OF_NAME_MAP``, ``_OMEGA_NAME_MAP``,
``_DEFAULT_PARAM_MAP``) therefore lives in *this* file, not in ``primordial_cosmology.py``.

Unit conventions (verified against ``cosmoprimo`` source, not assumed):

* cosmoprimo distances, ``rs_drag`` and ``sigma_rz``'s ``r`` are in Mpc/h; cobaya expects
  physical Mpc, so every such value is divided by ``h`` (multiplied by ``h`` when going the
  other way, e.g. registering a physical ``R`` as a cosmoprimo ``r``) when crossing the
  cobaya/desilike boundary.
* cosmoprimo's ``fourier.pk``/``fourier.sigma_rz`` tables are in :math:`(\mathrm{Mpc}/h)^3`
  (or unitless for :math:`\sigma_r`) with :math:`k`, :math:`r` in :math:`h/\mathrm{Mpc}`,
  :math:`\mathrm{Mpc}/h`; cobaya expects :math:`k` in :math:`1/\mathrm{Mpc}`, :math:`P(k)` in
  :math:`\mathrm{Mpc}^3` — converted with :math:`k_\mathrm{phys} = h\,k_{h/\mathrm{Mpc}}` and
  :math:`P_\mathrm{phys} = P_{(\mathrm{Mpc}/h)^3} / h^3`.
* ``background.efunc`` is the dimensionless :math:`E(z) = H(z)/H_0`; converted to
  cobaya's ``km/s/Mpc``/``1/Mpc`` units via :math:`H_0 = 100 h`. ``background.Omega_*`` are
  already dimensionless density parameters, no ``h`` rescaling needed.
* ``harmonic.lensed_cl``/``harmonic.unlensed_cl`` are raw (dimensionless, no
  :math:`\ell`-factor) :math:`C_\ell`, matching ``camspec.py``'s own usage;
  :math:`T_\mathrm{cmb} = 2.7255` K is hardcoded for unit conversion, consistent with that
  file's convention.
"""

import numpy as np
import jax.numpy as jnp

from ..base import Likelihood
from ..parameter import Parameter, VariableCollection


_C_KM_S = 299792.458  # speed of light, km/s
_T_CMB = 2.7255  # K; fixed, not tied to a sampled T_cmb parameter (matches camspec.py)

_CMB_UNIT_FACTORS = {'1': 1., 'muK2': _T_CMB * 1e6, 'K2': _T_CMB, 'FIRASmuK2': 2.7255e6, 'FIRASK2': 2.7255}

# cobaya Pk/sigma_R variable name -> cosmoprimo 'of' name, matching
# cosmoprimo/bindings/cobaya/cosmoprimo.py's `conversions_of`. Names not listed here are
# passed through unchanged (assumed already valid cosmoprimo names).
_OF_NAME_MAP = {'delta_tot': 'delta_m', 'delta_nonu': 'delta_cb',
                'v_newtonian_cdm': 'theta_cdm', 'v_newtonian_baryon': 'theta_b', 'Weyl': 'phi_plus_psi'}

# cobaya z-dependent Omega_* requirement name -> cosmoprimo Background method name.
_OMEGA_NAME_MAP = {'Omega_b': 'Omega_b', 'Omega_cdm': 'Omega_cdm', 'Omega_nu_massive': 'Omega_ncdm_tot'}

# cobaya derived/theory parameter name -> (cosmo method_key, static kwargs, needs_mpc_over_h_to_mpc).
# Extend/override per-likelihood via the `param_map` argument.
_DEFAULT_PARAM_MAP = {
    'rdrag': ('thermodynamics.rs_drag', {}, True),
    'sigma8': ('fourier.sigma8_z', {'z': 0., 'of': 'delta_m'}, False),
    'omegam': ('params.Omega_m', {}, False),
    'omegabh2': ('params.omega_b', {}, False),
    'omegach2': ('params.omega_cdm', {}, False),
    'H0': ('params.H0', {}, False),
    'ns': ('params.n_s', {}, False),
    'tau': ('params.tau_reio', {}, False),
}

_UNSUPPORTED_COSMO_KEYS = {'lensed_scal_Cl', 'source_Cl', 'CAMBdata', 'angular_diameter_distance_2'}


def _normalize_requirements(requirements):
    """Normalize cobaya's ``get_requirements()`` return value to a plain dict."""
    if not requirements:
        return {}
    if isinstance(requirements, dict):
        return dict(requirements)
    normalized = {}
    for item in requirements:
        if isinstance(item, str):
            normalized[item] = None
        else:
            name, options = item
            normalized[name] = options
    return normalized


def _convert_prior(cobaya_prior):
    """Translate a cobaya prior/ref dict into desilike ``ParameterPrior`` kwargs.

    cobaya's ``{dist: 'norm', loc:, scale:}`` already matches desilike's convention and is
    passed through as-is; cobaya's flat ``{min: a, max: b}`` (no ``dist`` key) becomes
    desilike's ``{limits: [a, b]}``.
    """
    if cobaya_prior is None:
        return None
    prior = dict(cobaya_prior)
    if 'min' in prior or 'max' in prior:
        prior['limits'] = [prior.pop('min', None), prior.pop('max', None)]
    return prior


def _params_from_cobaya(cobaya_params):
    """Build a desilike ``VariableCollection`` from a cobaya Likelihood's own ``params`` dict."""
    variables = []
    for name, spec in (cobaya_params or {}).items():
        if spec is None:
            continue
        if isinstance(spec, (int, float)):
            variables.append(Parameter(name, value=float(spec), fixed=True))
            continue
        if not isinstance(spec, dict):
            continue
        value = spec.get('value')
        if isinstance(value, str):
            # cobaya lambda-expression derived parameter: not translated in this first
            # version. Typically a pure output (not needed by logp), so just skip it here;
            # if logp genuinely needs it as an input, the missing kwarg will raise clearly.
            continue
        if 'prior' in spec:
            variables.append(Parameter(name, prior=_convert_prior(spec['prior']),
                                        ref=_convert_prior(spec.get('ref')),
                                        latex=spec.get('latex')))
        elif value is not None:
            variables.append(Parameter(name, value=float(value), fixed=True, latex=spec.get('latex')))
        # else: derived-only output (e.g. {'derived': True}), or a bare renamed alias --
        # not needed as a logp input, skip.
    return VariableCollection(variables)


def _translate_of(name):
    """Translate a single cobaya Pk/sigma_R variable name to its cosmoprimo ``of`` name."""
    return _OF_NAME_MAP.get(name, name)


def _first_vars_pair(vars_pairs):
    """Extract a single (translated) ``of`` pair from cobaya's ``vars_pairs`` list.

    cobaya's default (``vars_pairs`` empty/``None``) is the total-matter auto-spectrum;
    cross-spectra (e.g. ``('delta_tot', 'theta_cb')``) are supported, multiple pairs in a
    single requirement are not (core-subset scope).
    """
    if not vars_pairs:
        vars_pairs = [['delta_tot', 'delta_tot']]
    if len(vars_pairs) > 1:
        raise NotImplementedError(f'Multiple vars_pairs in a single requirement are not supported yet: {vars_pairs!r}.')
    var_a, var_b = vars_pairs[0]
    return (_translate_of(var_a), _translate_of(var_b))


def _apply_cl_units(cl, ells, ell_factor, units):
    """Apply cobaya's ell-factor and CMB-unit conversion to a raw (dimensionless) Cl dict."""
    result = {'ell': ells}
    for name in ('tt', 'ee', 'bb', 'te'):
        result[name] = np.asarray(cl[name])
    result['et'] = result['te']
    units_factor = _CMB_UNIT_FACTORS[units]
    ells_factor = np.zeros_like(ells, dtype=float)
    ells_factor[1:] = ells[1:] * (ells[1:] + 1)
    for name in ('tt', 'ee', 'bb', 'te', 'et'):
        if ell_factor:
            result[name] = result[name] * (ells_factor / (2 * np.pi)) * units_factor ** 2
        else:
            result[name] = result[name] * units_factor ** 2
    return result, ells_factor, units_factor


class _Provider:
    """Minimal cobaya ``provider``-like shim, backed by a desilike ``PrimordialCosmology``.

    Implements only the ``get_*`` methods a wrapped cobaya likelihood may call from its
    ``logp()``, each reading from ``wrapper.cosmo`` (a dep already computed earlier in the
    pipeline) via the requirements-API section proxies -- never ``cosmo._cosmo`` directly,
    since that is not safe to read from an external (``pure_callback``-wrapped) node.
    """

    def __init__(self, wrapper):
        self._wrapper = wrapper
        self.cosmo = wrapper.cosmo

    def _h(self):
        return float(np.asarray(self.cosmo['h']))

    def get_Cl(self, ell_factor=False, units='FIRASmuK2'):
        wrapper = self._wrapper
        ellmax = wrapper._ellmax
        ells = np.arange(ellmax + 1)
        cl = self.cosmo.get_harmonic().lensed_cl(ellmax=ellmax)
        result, ells_factor, units_factor = _apply_cl_units(cl, ells, ell_factor, units)
        if wrapper._need_lens_potential:
            cl_lens = self.cosmo.get_harmonic().lens_potential_cl(ellmax=ellmax)
            for name in ('pp', 'tp', 'ep'):
                if name in cl_lens:
                    result[name] = np.asarray(cl_lens[name])
                    result[name[::-1]] = result[name]
            if ell_factor:
                result['pp'] = result['pp'] * ells_factor ** 2 / (2 * np.pi)
            for name in ('tp', 'pt', 'ep', 'pe'):
                if name in result:
                    result[name] = result[name] * units_factor
                    if ell_factor:
                        result[name] = result[name] * ells_factor ** 1.5 / (2 * np.pi)
        return result

    def get_unlensed_Cl(self, ell_factor=False, units='FIRASmuK2'):
        wrapper = self._wrapper
        ellmax = wrapper._unlensed_ellmax
        ells = np.arange(ellmax + 1)
        cl = self.cosmo.get_harmonic().unlensed_cl(ellmax=ellmax)
        result, _, _ = _apply_cl_units(cl, ells, ell_factor, units)
        return result

    def get_Hubble(self, z, units='km/s/Mpc'):
        z = np.atleast_1d(z)
        efunc = np.asarray(self.cosmo.get_background().efunc(z))
        H = 100. * self._h() * efunc  # km/s/Mpc
        if units == 'km/s/Mpc':
            return H
        if units == '1/Mpc':
            return H / _C_KM_S
        raise ValueError(f'Unknown units {units!r} for get_Hubble.')

    def get_angular_diameter_distance(self, z):
        z = np.atleast_1d(z)
        d_m = np.asarray(self.cosmo.get_background().comoving_transverse_distance(z)) / self._h()  # Mpc
        return d_m / (1. + z)

    def get_comoving_radial_distance(self, z):
        z = np.atleast_1d(z)
        # Flat universe (desilike's cosmology requirements assume flat): comoving radial ==
        # comoving transverse.
        return np.asarray(self.cosmo.get_background().comoving_transverse_distance(z)) / self._h()  # Mpc

    def get_sigma8_z(self, z):
        z = np.atleast_1d(z)
        return np.asarray(self.cosmo.get_fourier().sigma8_z(z, of='delta_m'))

    def get_fsigma8(self, z):
        z = np.atleast_1d(z)
        growth_rate = np.asarray(self.cosmo.get_background().growth_rate(z))
        sigma8_z = np.asarray(self.cosmo.get_fourier().sigma8_z(z, of='delta_m'))
        return growth_rate * sigma8_z

    def get_Omega_b(self, z):
        return np.asarray(self.cosmo.get_background().Omega_b(np.atleast_1d(z)))

    def get_Omega_cdm(self, z):
        return np.asarray(self.cosmo.get_background().Omega_cdm(np.atleast_1d(z)))

    def get_Omega_nu_massive(self, z):
        # cosmoprimo names this Omega_ncdm_tot, not Omega_nu_massive (cobaya's name).
        return np.asarray(self.cosmo.get_background().Omega_ncdm_tot(np.atleast_1d(z)))

    def get_Pk_grid(self, var_pair=('delta_tot', 'delta_tot'), nonlinear=True):
        wrapper = self._wrapper
        of_pair = (_translate_of(var_pair[0]), _translate_of(var_pair[1]))
        h = self._h()
        rows = [np.asarray(self.cosmo.get_fourier().pk(of=of_pair, z=float(z), k=wrapper._pk_k_grid_hmpc))
                for z in wrapper._pk_zs]
        pk_hmpc3 = np.stack(rows, axis=0)  # (nz, nk), (Mpc/h)^3
        k_mpc = wrapper._pk_k_grid_hmpc * h  # 1/Mpc
        pk_mpc3 = pk_hmpc3 / h ** 3  # Mpc^3
        return k_mpc, np.asarray(wrapper._pk_zs), pk_mpc3

    def get_Pk_interpolator(self, var_pair=('delta_tot', 'delta_tot'), nonlinear=True,
                             extrap_kmin=None, extrap_kmax=None):
        from cobaya.theories.cosmo.boltzmannbase import PowerSpectrumInterpolator
        k, z, pk = self.get_Pk_grid(var_pair=var_pair, nonlinear=nonlinear)
        log_p, sign = True, 1
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
        pk_for_interp = np.log(sign * pk) if log_p else pk
        return PowerSpectrumInterpolator(z, k, pk_for_interp, logP=log_p, logsign=sign,
                                          extrap_kmin=extrap_kmin, extrap_kmax=extrap_kmax)

    def get_sigma_R(self, var_pair=('delta_tot', 'delta_tot')):
        # cobaya's own provider interface method name; cosmoprimo's matching method is
        # Fourier.sigma_rz(r, z, of=...), registered here as 'fourier.sigma_rz'.
        wrapper = self._wrapper
        of_pair = (_translate_of(var_pair[0]), _translate_of(var_pair[1]))
        h = self._h()
        rows = [np.asarray(self.cosmo.get_fourier().sigma_rz(z=float(z), r=wrapper._sigma_r_grid_hmpc, of=of_pair))
                for z in wrapper._sigma_r_zs]
        sigma_r = np.stack(rows, axis=0)  # (nz, nR)
        R_mpc = wrapper._sigma_r_grid_hmpc / h  # Mpc
        return np.asarray(wrapper._sigma_r_zs), R_mpc, sigma_r

    def get_param(self, param):
        if not isinstance(param, str):
            return [self.get_param(p) for p in param]
        method_key, static_kwargs, needs_mpc_over_h_to_mpc = self._wrapper._derived_param_reqs[param]
        value = float(np.asarray(self.cosmo.get(method_key, **static_kwargs)))
        if needs_mpc_over_h_to_mpc:
            value = value / self._h()
        return value


class CobayaLikelihood(Likelihood):
    """Wrap a cobaya ``Likelihood`` class as a desilike ``Likelihood``.

    See the module docstring for the supported subset of cobaya's requirements and the
    unit conventions applied when translating cosmoprimo quantities to cobaya's convention.

    Parameters
    ----------
    cobaya_cls : type
        A cobaya ``Likelihood`` subclass (not an instance).
    info : dict, default=None
        Options passed to ``cobaya_cls(info, standalone=True)`` (overrides its ``.yaml``
        defaults), e.g. ``{'data_dir': ...}``.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator. If ``None``, defaults to ``CosmoprimoCosmology(engine='camb',
        fiducial='DESI')`` (a Boltzmann engine is required for most cobaya likelihoods).
    params : Parameter, VariableCollection, dict, default=None
        Additional nuisance parameters to add on top of (or override) the ones parsed from
        ``cobaya_cls``'s own ``params:`` block.
    param_map : dict, default=None
        Extends/overrides ``_DEFAULT_PARAM_MAP`` for translating cobaya derived/theory
        parameter names (requested via ``get_requirements()`` with a ``None`` value, e.g.
        ``rdrag``) to desilike cosmo requirement keys. Maps
        ``name -> (method_key, static_kwargs, needs_mpc_over_h_to_mpc)``.
    packages_path : str, default=None
        Forwarded to the cobaya constructor (for ``InstallableLikelihood`` subclasses).
    """

    _is_external = True

    def __init__(self, cobaya_cls, info=None, cosmo=None, params=None, param_map=None, packages_path=None):
        self._cobaya_like = cobaya_cls(info or {}, packages_path=packages_path, standalone=True)
        if cosmo is None:
            from ..theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(engine='camb', fiducial='DESI')
        self.cosmo = cosmo  # Calculator dep; build_graph discovers it from __dict__
        self._param_map = {**_DEFAULT_PARAM_MAP, **(param_map or {})}
        vc = _params_from_cobaya(getattr(self._cobaya_like, 'params', None))
        if params is not None:
            vc = vc + VariableCollection(params)
        self.params = {param.basename: param for param in vc}

    def __post_init__(self, cobaya_cls, info=None, cosmo=None, params=None, param_map=None, packages_path=None):
        requirements = _normalize_requirements(self._cobaya_like.get_requirements())
        own_nuisance_names = set(getattr(self._cobaya_like, 'params', None) or {})

        cosmo_requirements = {}
        self._ellmax = None
        self._unlensed_ellmax = None
        self._need_lens_potential = False
        self._derived_param_reqs = {}
        pk_pairs, pk_zs, pk_kmax_invmpc = set(), np.array([]), 0.
        sigma_r_pairs, sigma_r_zs, sigma_r_Rs = set(), np.array([]), np.array([])

        for name, value in requirements.items():
            if name in own_nuisance_names:
                continue  # already exposed as a desilike Parameter; passed straight into logp

            if name in ('Cl', 'unlensed_Cl'):
                ellmax = max(value.values())
                if name == 'Cl':
                    self._ellmax = max(self._ellmax or 0, ellmax)
                    cosmo_requirements.setdefault('harmonic.lensed_cl', []).append({'ellmax': self._ellmax})
                    if set(spec.lower() for spec in value).intersection({'pp', 'tp', 'ep', 'pt', 'pe'}):
                        self._need_lens_potential = True
                        cosmo_requirements.setdefault('harmonic.lens_potential_cl', []).append({'ellmax': self._ellmax})
                else:
                    self._unlensed_ellmax = max(self._unlensed_ellmax or 0, ellmax)
                    cosmo_requirements.setdefault('harmonic.unlensed_cl', []).append({'ellmax': self._unlensed_ellmax})
                continue

            if name == 'Hubble':
                cosmo_requirements.setdefault('background.efunc', []).append({'z': np.asarray(value['z'])})
                continue

            if name in ('angular_diameter_distance', 'comoving_radial_distance'):
                cosmo_requirements.setdefault('background.comoving_transverse_distance', []).append({'z': np.asarray(value['z'])})
                continue

            if name == 'sigma8_z':
                cosmo_requirements.setdefault('fourier.sigma8_z', []).append({'z': np.asarray(value['z']), 'of': 'delta_m'})
                continue

            if name == 'fsigma8':
                z = np.asarray(value['z'])
                cosmo_requirements.setdefault('fourier.sigma8_z', []).append({'z': z, 'of': 'delta_m'})
                cosmo_requirements.setdefault('background.growth_rate', []).append({'z': z})
                continue

            if name in _OMEGA_NAME_MAP:
                method_key = f'background.{_OMEGA_NAME_MAP[name]}'
                cosmo_requirements.setdefault(method_key, []).append({'z': np.asarray(value['z'])})
                continue

            if name in ('Pk_interpolator', 'Pk_grid'):
                nonlinear = value.get('nonlinear', True)
                nonlinear_list = list(nonlinear) if isinstance(nonlinear, (list, tuple)) else [nonlinear]
                if any(nonlinear_list):
                    raise NotImplementedError(
                        f"cobaya requirement {name!r} asked for a non-linear Pk, but desilike's "
                        "'fourier.pk' requirement only exposes the linear matter power spectrum; "
                        "not supported by CobayaLikelihood yet.")
                pk_pairs.add(_first_vars_pair(value.get('vars_pairs')))
                pk_zs = np.unique(np.concatenate([pk_zs, np.atleast_1d(value['z'])]))
                pk_kmax_invmpc = max(pk_kmax_invmpc, float(value['k_max']))
                continue

            if name == 'sigma_R':
                sigma_r_pairs.add(_first_vars_pair(value.get('vars_pairs')))
                sigma_r_zs = np.unique(np.concatenate([sigma_r_zs, np.atleast_1d(value['z'])]))
                sigma_r_Rs = np.concatenate([sigma_r_Rs, np.atleast_1d(value['R'])])
                continue

            if name in _UNSUPPORTED_COSMO_KEYS:
                raise NotImplementedError(f'cobaya requirement {name!r} is not supported by CobayaLikelihood yet.')

            if value is None:
                method_key, static_kwargs, needs_mpc_over_h_to_mpc = self._param_map.get(name, (f'params.{name}', {}, False))
                self._derived_param_reqs[name] = (method_key, static_kwargs, needs_mpc_over_h_to_mpc)
                cosmo_requirements.setdefault(method_key, []).append(dict(static_kwargs))
                continue

            raise NotImplementedError(f'cobaya requirement {name!r} (value={value!r}) is not supported by CobayaLikelihood yet.')

        if pk_pairs:
            # cobaya only gives k_max (physical, 1/Mpc); cosmoprimo's 'fourier.pk' needs a
            # k grid in h/Mpc up front. h isn't known at __post_init__ time, so use a
            # generous fixed safety margin (h > 0.4 for any realistic cosmology) rather
            # than the exact (unknown) fiducial h.
            self._pk_k_grid_hmpc = np.geomspace(1e-5, max(pk_kmax_invmpc, 1e-4) / 0.4, 300)
            self._pk_zs = pk_zs
            for of_pair in pk_pairs:
                cosmo_requirements.setdefault('fourier.pk', []).append(
                    {'of': of_pair, 'z': self._pk_zs, 'k': self._pk_k_grid_hmpc})

        if sigma_r_pairs:
            # Same generous-grid trick as Pk above, but in the opposite direction (R, not
            # k, scales *up* with h): span [0.3, 1.1] x the requested physical R range.
            # As with Pk_grid, the returned grid *covers* the requested R's, but (unlike a
            # cobaya-native provider) does not necessarily *contain* them exactly -- fine
            # for a downstream consumer that interpolates, not for one that exact-indexes.
            self._sigma_r_grid_hmpc = np.geomspace(0.3 * sigma_r_Rs.min(), 1.1 * sigma_r_Rs.max(), 60)
            self._sigma_r_zs = sigma_r_zs
            for of_pair in sigma_r_pairs:
                cosmo_requirements.setdefault('fourier.sigma_rz', []).append(
                    {'of': of_pair, 'z': self._sigma_r_zs, 'r': self._sigma_r_grid_hmpc})

        self.cosmo.add_requirements(cosmo_requirements)

    def __call__(self):
        params_values = {name: float(np.asarray(param.value)) for name, param in self.params.items()}
        for cobaya_name, (method_key, static_kwargs, needs_mpc_over_h_to_mpc) in self._derived_param_reqs.items():
            value = float(np.asarray(self.cosmo.get(method_key, **static_kwargs)))
            if needs_mpc_over_h_to_mpc:
                value = value / float(np.asarray(self.cosmo['h']))
            params_values[cobaya_name] = value
        self._cobaya_like.provider = _Provider(self)
        logp = self._cobaya_like.logp(**params_values)
        self.logpdf = jnp.asarray(float(logp))
        return self.logpdf


def wrap_cobaya_likelihood(cobaya_cls, info=None, cosmo=None, params=None, param_map=None, packages_path=None):
    """Build a desilike :class:`~desilike.base.Likelihood` that wraps a cobaya Likelihood class.

    See :class:`CobayaLikelihood` for the full parameter documentation and the supported
    subset of cobaya's requirements.
    """
    return CobayaLikelihood(cobaya_cls, info=info, cosmo=cosmo, params=params,
                             param_map=param_map, packages_path=packages_path)
