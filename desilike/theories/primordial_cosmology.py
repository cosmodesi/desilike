"""
Primordial cosmology calculators.

Classes
-------
PrimordialCosmology
    Abstract base class implementing the requirements API: downstream calculators declare
    what cosmological quantities they need; the provider computes them on merged k/z grids
    and exposes them as JAX leaves through ``tree_flatten`` / ``tree_unflatten``.
CosmoprimoCosmology
    Concrete implementation backed by :mod:`cosmoprimo`.  Supports JAX-native engines
    (``'eisenstein_hu'``) and external Boltzmann codes (``'camb'``, ``'class'``, …).
"""

import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from cosmoprimo import CosmologyInputError, CosmologyComputationError

from ..base import Calculator
from ..parameter import Parameter, VariableCollection
from ..install import Installer


_COORDS = ['z', 'k', 'r']


def _normalize_static(static):
    """Return static with 'of' canonicalized to a 2-tuple, matching add_requirements's spec_key."""
    if 'of' in static:
        static = dict(static)
        static['of'] = (static['of'],) * 2 if isinstance(static['of'], str) else tuple(static['of'])
    return static


class _Section:
    """Read-side proxy: cosmo.get_fourier().pk(**kw) forwards to cosmo.get('fourier.pk', **kw)."""

    def __init__(self, cosmo, name):
        self._cosmo = cosmo
        self._name = name

    def __getattr__(self, name):
        method_key = f'{self._name}.{name}'
        # z-independent quantities (e.g. N_eff, rs_drag) are registered with no kwargs at
        # all, so a bare no-arg get() already resolves them: mirror cosmoprimo's own API by
        # returning the value directly, like a property. A KeyError means the requirement
        # needs args (z, of, k, ...), so fall back to returning a callable instead.
        try:
            return self._cosmo.get(method_key)
        except KeyError:
            pass

        def method(*args, **kwargs):
            # Mirror cosmoprimo's calling convention: z is the first positional argument
            # for every method in the table (efunc, comoving_transverse_distance, pk, ...).
            if args:
                if len(args) > 1:
                    raise TypeError(f'{method_key} only supports a single positional argument (z)')
                kwargs = {'z': args[0], **kwargs}
            return self._cosmo.get(method_key, **kwargs)
        return method


class PrimordialCosmology(Calculator):
    """Abstract base class for primordial cosmology calculators.

    Implements the **requirements API** shared by all cosmology providers:

    * Downstream calculators call :meth:`add_requirements` in their ``__post_init__``
      to declare which cosmological quantities they need (power spectra, growth rates,
      distances) and on which z/k grids.  Multiple downstreams sharing the same instance
      have their grids merged automatically.
    * At every pipeline call, the concrete subclass computes those quantities (in its
      ``__call__``) and stores them in ``self._results``.
    * :meth:`get` retrieves a pre-computed result for a given method key and kwargs,
      selecting the relevant z/k slice with ``searchsorted``.
    * :meth:`tree_flatten` / :meth:`tree_unflatten` expose the results as JAX leaves so
      that downstream calculators see pure-JAX arrays and are themselves differentiable.

    Subclass contract
    -----------------
    Concrete subclasses must:

    * Initialize ``self.params`` (a ``dict[str, Parameter]``), ``self._requirements = {}``,
      and ``self._results = {}`` in their ``__init__``.
    * Set ``self._engine`` (a string identifier for the provider) in ``__post_init__``.
    * Override :meth:`propose_params` to return the provider's cosmological parameters.
    * Implement ``__call__`` to build the cosmology, loop over ``self._requirements``, and
      populate ``self._results[spec_key]`` for every registered spec.
    """

    @classmethod
    def propose_params(cls):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this provider.

        The base implementation returns an empty collection.  Subclasses should override
        this to return the parameters appropriate for their cosmology provider.

        Returns
        -------
        VariableCollection
        """
        return VariableCollection()

    def __init__(self, *args, params=None, fiducial=None, **kwargs):
        # Per-instance flag: JAX-traceable engines run as pure JAX, others as external.
        if params is None:
            # Forward fiducial only when given, so that an omitted fiducial falls through
            # to propose_params' own default (e.g. 'DESI') instead of being clobbered by None.
            if fiducial is not None:
                kwargs['fiducial'] = fiducial
            params = self.propose_params(*args, **kwargs)
        elif not isinstance(params, VariableCollection):
            params = VariableCollection(params)
        self.derived_params = params.select(derived=True)
        self.params = params - self.derived_params
        # Requirement registry: filled by downstream calculators via add_requirements().
        # _requirements: spec_key → {'static': dict, 'z': np.array, 'k': np.array|None}
        # _results:   spec_key → jnp.array   (populated in __call__)
        self._requirements = {}
        self._results = {}
        self._param_values = {}
        # Default engine identifier; overridden by concrete subclasses in __post_init__.
        self._engine = None
        self._get_derived = {}
        for param in self.derived_params:
            if param.basename in ['sigma8_m']:
                req = ('fourier.sigma8_z', {'z': 0., 'of': 'delta_m'})
            elif param.basename in ['sigma8_cb']:
                req = ('fourier.sigma8_z', {'z': 0., 'of': 'delta_cb'})
            elif param.basename in ['rs_drag']:
                req = ('thermodynamics.rs_drag', {'of': 'delta_cb'})
            elif param.basename in ['age']:
                req = ('background.age', {})
            else:
                req = (f'params.{param.basename}', {})
            self._get_derived[param.name] = req
            self.add_requirements({req[0]: req[1]})

    # ── requirements API ──────────────────────────────────────────────────────

    def add_requirements(self, requirements):
        """Register quantities that a downstream calculator will need from this cosmology.

        Called in the downstream calculator's ``__post_init__``.  Multiple callers sharing the
        same cosmology instance are supported: z and k grids are union-merged so only one
        combined evaluation is needed at runtime.

        Parameters
        ----------
        requirements : dict
            Mapping ``{method_key: [kwargs_dict, ...]}`` where each kwargs dict carries
            the static call arguments plus ``z`` (float, required) and optionally ``k``
            (array).  ``z`` and ``k`` are dynamic — they are merged across callers.
            All other kwargs are static and form part of the spec identity, so two calls
            with different static kwargs (e.g. ``of='delta_cb'`` vs ``of='theta_cb'``)
            are tracked and computed separately.  The recognised method keys are
            provider-specific; see the concrete subclass for the full list.
            ``None`` is shorthand for ``[{}]``, i.e. a single registration with no kwargs
            (e.g. for ``'params.<name>'`` keys, which need neither ``z`` nor ``k``).

        Examples
        --------
        >>> cosmo.add_requirements({
        ...     'fourier.pk': [
        ...         {'of': 'delta_cb', 'z': 1., 'k': k_array},
        ...         {'of': 'theta_cb', 'z': 1., 'k': k_array},
        ...     ],
        ...     'background.efunc': [{'z': 1.}],
        ...     'params.m_ncdm_tot': None,
        ... })
        """
        for method_key, kwargs_list in requirements.items():
            if kwargs_list is None:
                kwargs_list = [{}]
            if not isinstance(kwargs_list, (tuple, list)):
                kwargs_list = [kwargs_list]
            for kwargs in kwargs_list:
                static = {key: val for key, val in kwargs.items() if key not in _COORDS}
                static = _normalize_static(static)
                spec_key = (method_key, tuple(sorted(static.items())))
                if spec_key not in self._requirements:
                    spec = self._requirements[spec_key] = {}
                    spec['static'] = static
                    for coord in _COORDS:
                        if coord in kwargs:
                            spec[coord] = np.sort(np.atleast_1d(kwargs[coord]))
                else:
                    spec = self._requirements[spec_key]
                    for coord in _COORDS:
                        if coord in kwargs:
                            spec[coord] = np.unique(np.concatenate([spec[coord], np.atleast_1d(kwargs[coord])]))

    def __getitem__(self, name):
        # Return parameter value. Free params are already live in _param_values (jit-safe).
        # For anything else (e.g. a derived quantity like 'm_ncdm_tot'), fall back to the
        # 'params.<name>' requirement if it was registered (jit-safe -- threaded through
        # tree_flatten/pure_callback like any other requirement), else read self._cosmo
        # directly (fine for same-trace, non-external consumers; stale under jit otherwise).
        if name in self._param_values:
            return self._param_values[name]
        try:
            return self.get(f'params.{name}')
        except KeyError:
            return self._cosmo[name]

    def get(self, method_key, **kwargs):
        """Return a pre-computed requirement result, selecting from the merged grid.

        Parameters
        ----------
        method_key : str
            Same key as registered with :meth:`add_requirements`.
        **kwargs
            Same kwargs as registration.  ``z`` (float) is looked up with
            ``searchsorted`` in the merged z grid; ``k`` (array) similarly.
            Results are plain JAX arrays — no interpolation, only index selection.

        Returns
        -------
        jnp.array
            Scalar, 1-D, or 2-D depending on the method and whether z/k were provided.
        """
        static = {key: val for key, val in kwargs.items() if key not in _COORDS}
        static = _normalize_static(static)
        spec_key = (method_key, tuple(sorted(static.items())))
        result = self._results[spec_key]
        spec   = self._requirements[spec_key]
        for coord in _COORDS:
            if coord in spec:
                idx = np.searchsorted(spec[coord], kwargs[coord])
                result = result[idx]
        return result

    def get_fourier(self):
        """Return a Fourier-section proxy: cosmo.get_fourier().pk(...) == cosmo.get('fourier.pk', ...)."""
        return _Section(self, 'fourier')

    def get_background(self):
        """Return a Background-section proxy: cosmo.get_background().efunc(...) == cosmo.get('background.efunc', ...)."""
        return _Section(self, 'background')

    def get_thermodynamics(self):
        """Return a Thermodynamics-section proxy: cosmo.get_thermodynamics().rs_drag(...) == cosmo.get('thermodynamics.rs_drag', ...)."""
        return _Section(self, 'thermodynamics')

    def get_harmonic(self):
        """Return a Harmonic-section proxy: cosmo.get_harmonic().lensed_cl(...) == cosmo.get('harmonic.lensed_cl', ...)."""
        return _Section(self, 'harmonic')

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __call__(self):
        """Proxy implementation: populate _results with zero placeholders if not already set.

        Concrete subclasses (e.g. CosmoprimoCosmology) override this with a real solver.
        When used as a pre-loaded proxy (results injected externally via compile's ``input``
        callable before the graph runs), this is a no-op because _results is already populated.
        """
        params = {param.basename: param.value for param in self.params}
        self._param_values = params
        for spec_key, spec in self._requirements.items():
            if spec_key not in self._results:
                shape = tuple(spec[coord].size for coord in _COORDS)
                self._results[spec_key] = jnp.zeros(shape)
        # Here set derived_params
        for param, getter in self._get_derived.items():
            self.derived_params[param].value = jnp.reshape(self.get(getter[0], **getter[1]), self.derived_params[param].shape)

    def tree_flatten(self):
        ordered = list(self._requirements.items())
        leaves = []
        leaves.append(self._param_values)
        for spec_key, spec in ordered:
            if spec_key in self._results:
                leaves.append(self._results[spec_key])
            else:
                # Placeholder of correct shape for compile-time structure inference.
                shape = tuple(spec[coord].size for coord in _COORDS)
                leaves.append(jnp.zeros(shape))
        # Derived param values as leaves so they propagate as JAX Tracers through the
        # external (pure_callback) path and appear correctly in derived_dict.
        for param in self.derived_params:
            v = param._value
            leaves.append(jnp.asarray(v) if v is not None else jnp.zeros(param.shape or ()))
        return leaves, {'engine': self._engine, 'ordered_specs': ordered, 'params': self.params, 'get_derived': self._get_derived, 'derived_params': self.derived_params}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj._engine = aux['engine']
        obj.params = aux['params']
        obj._get_derived = aux['get_derived']
        obj._requirements = {sk: spec for sk, spec in aux['ordered_specs']}
        n_results = len(aux['ordered_specs'])
        obj._param_values = children[0]
        obj._results = {sk: arr for (sk, _), arr in zip(aux['ordered_specs'], children[1:1 + n_results])}
        obj.derived_params = aux['derived_params']
        # Restore derived param _value from leaves so they flow as JAX Tracers
        # when this is called inside _run_graph (external node path).
        for param, val in zip(list(obj.derived_params), children[1 + n_results:]):
            param._value = val
        return obj




# Engines that produce JAX-traceable outputs through cosmoprimo.Cosmology.clone.
_JAX_ENGINES = frozenset({'eisenstein_hu'})

# Parameter name conversion: desilike name → cosmoprimo clone kwarg.
_CONVERSIONS = {}

# cosmoprimo pk_interpolator extrapolation kwargs shared by CosmoprimoCosmology and template.py.
_kw_pk = dict(extrap_kmin=1e-7, extrap_kmax=1e2)


def _get_cosmoprimo_fiducial(fiducial):
    """Return a cosmoprimo Cosmology from a name string, (name, kwargs) tuple, dict, or Cosmology."""
    import cosmoprimo
    import cosmoprimo.fiducial  # noqa: ensure submodule is accessible as cosmoprimo.fiducial
    if fiducial is None:
        raise ValueError('fiducial cosmology is required')
    if hasattr(fiducial, 'get_fourier'):
        return fiducial
    if isinstance(fiducial, str):
        fiducial = (fiducial, {})
    if isinstance(fiducial, tuple):
        name, kw = fiducial
        return getattr(cosmoprimo.fiducial, name)(**kw)
    if isinstance(fiducial, dict):
        return cosmoprimo.Cosmology(**fiducial)
    raise ValueError(f'Cannot parse fiducial cosmology: {fiducial!r}')


def _get_fiducial(fiducial, calculator=None):
    """Return a cosmoprimo Cosmology, or (if calculator is given) the fiducial computed
    through calculator's own pipeline.

    Fiducial cosmology computed with input calculator: re-runs calculator's pipeline at the
    resolved fiducial's parameter values and returns calculator itself, so its
    get_background()/get_fourier()/etc. proxies reflect the fiducial point. Parameters not
    recognized by the resolved fiducial (e.g. emulator-specific nuisance inputs like 'mu1',
    'Sigma1', ...) keep their current value on calculator.
    """
    import cosmoprimo
    fiducial = _get_cosmoprimo_fiducial(fiducial)
    if calculator is not None:
        from desilike.base import compile
        params = {}
        for param in calculator.params:
            try:
                params[param.name] = fiducial[param.basename]
            except cosmoprimo.CosmologyError:
                params[param.name] = param.value
        pipe = compile(calculator, output=lambda: calculator)
        return pipe(params)
    return fiducial


# Engine settings derived from the registered 'harmonic.*' requirements (see __call__):
# 'ellmax_cl' is always raised to the largest requested ellmax; any lensed-Cl /
# lens-potential-Cl requirement turns on 'lensing' with a non-linear matter power model
# (default engine settings under-resolve lensing otherwise); the lensing potential
# additionally gets the reconstruction accuracy boost below (both the non-linear matter
# power feeding it and the ell reach/margin around it).
# 'non_linear'/'ellmax_cl' are cosmoprimo calculation parameters (set like any other
# cosmological parameter); the rest are raw engine precision knobs forwarded via
# cosmoprimo's ``extra_params``.
_LENSING_CALC_PARAMS = {
    'camb': dict(non_linear='mead2016'),
    'class': dict(non_linear='hmcode'),
}
_LENS_POTENTIAL_CL_EXTRA_PARAMS = {
    'camb': dict(lens_margin=1250, lens_potential_accuracy=4,
                AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1),
    'class': dict(nonlinear_min_k_max=20, accurate_lensing=1, delta_l_max=800),
}
# CAMB needs enough ell reach internally (beyond the requested ellmax) for lens_margin to
# have room to work with; CLASS's 'delta_l_max' above already provides that margin relative
# to whatever ellmax_cl already is, so it needs no equivalent floor here.
_LENS_POTENTIAL_CL_MIN_ELLMAX_CL = {'camb': 4000}


def _build_cosmoprimo(fiducial, params, lensing=False, calc_params=None, extra_params=None):
    """Clone *fiducial* with the given *params* dict (desilike names → values).

    Values are passed as-is so JAX tracers are preserved for JAX-native engines;
    external engines (camb, class) always receive plain floats.

    *lensing* forwards cosmoprimo's ``lensing`` calculation parameter (default
    ``False``): without it, external engines (camb, class) never compute lensed
    Cl/lens-potential Cl, so ``get_harmonic().lensed_cl()`` /
    ``.lens_potential_cl()`` raise even though the requirement was registered.

    *calc_params* / *extra_params* carry the lensing-reconstruction accuracy overrides
    (see ``_LENS_POTENTIAL_CL_*`` above); *extra_params* is merged on top of any
    precision params the fiducial's engine already carries, rather than replacing them.
    """
    kw = {_CONVERSIONS.get(name, name): value for name, value in params.items()}
    # ``h`` and ``theta_MC_100`` are mutually exclusive inputs to cosmoprimo; when both
    # are present ``h`` takes precedence (see primordial_cosmology.yaml).
    if 'h' in kw and 'theta_MC_100' in kw:
        kw.pop('theta_MC_100')
    if lensing:
        kw['lensing'] = True
    if calc_params:
        kw.update(calc_params)
    if extra_params:
        merged_extra_params = dict(getattr(getattr(fiducial, 'engine', None), '_extra_params', None) or {})
        merged_extra_params.update(extra_params)
        return fiducial.clone(base='input', extra_params=merged_extra_params, **kw)
    return fiducial.clone(base='input', **kw)


def _nan_like(result):
    """Replace array leaves in *result* (array, or dict of arrays) with same-shape NaNs."""
    if isinstance(result, dict):
        return {key: jnp.full(jnp.shape(value), jnp.nan) for key, value in result.items()}
    return jnp.full(jnp.shape(result), jnp.nan)


class CosmoprimoCosmology(PrimordialCosmology):
    r"""
    :class:`PrimordialCosmology` backed by :mod:`cosmoprimo`.

    The ``_is_external`` flag is set per instance from *engine*: JAX-native engines
    (``'eisenstein_hu'``) run as pure JAX (``grad``/``jit``/``vmap``); external Boltzmann
    codes (``'camb'``, ``'class'``, …) run via ``pure_callback`` + finite-difference
    derivatives.  ``self._cosmo`` holds the current :class:`cosmoprimo.Cosmology` after
    each call; engine state is cached via ``Cosmology.clone(base='input', ...)``.

    Recognised method keys for :meth:`~PrimordialCosmology.add_requirements`:

    * ``'fourier.pk'``                              — kwargs: ``of``, ``z``, ``k``
    * ``'fourier.pk_now'``                          — kwargs: ``of``, ``engine``, ``z``, ``k``
    * ``'fourier.sigma8_z'``                        — kwargs: ``of``, ``z``
    * ``'background.efunc'``                        — kwargs: ``z``
    * ``'background.comoving_transverse_distance'`` — kwargs: ``z``
    * ``'background.luminosity_distance'``          — kwargs: ``z``
    * ``'background.growth_factor'``                — kwargs: ``z``
    * ``'primordial.pk'``                           — kwargs: ``k``;
      the primordial scalar power spectrum :math:`P_R(k)` on the registered k grid.
    * ``'harmonic.lensed_cl'``                       — kwargs: ``ellmax``; returns a dict
      keyed by ``'tt', 'ee', 'bb', 'te'`` of raw (dimensionless) :math:`C_\ell`.
    * ``'harmonic.unlensed_cl'``                     — kwargs: ``ellmax``; same as
      ``'harmonic.lensed_cl'`` but for the unlensed spectra.
    * ``'harmonic.lens_potential_cl'``               — kwargs: ``ellmax``; returns a dict
      keyed by ``'pp', 'tp', 'ep'`` of raw (dimensionless) :math:`C_\ell`.
    * ``'fourier.sigma_rz'``                         — kwargs: ``of``, ``z``, ``r``;
      :math:`\sigma_r(z)` (RMS of ``of`` perturbations in a sphere of radius ``r``,
      in :math:`\mathrm{Mpc}/h`), shaped ``(z, r)``. Matches cosmoprimo's own
      ``Fourier.sigma_rz`` naming.
    * ``'background.Omega_b'``, ``'background.Omega_cdm'``, ``'background.Omega_ncdm_tot'``
      — kwargs: ``z``; density parameters (unitless, no ``h`` rescaling needed). Matches
      cosmoprimo's own ``Background`` method names.
    * ``'thermodynamics.rs_drag'``                  —
    * ``'params.N_eff'``                            — effective number of relativistic species :math:`N_\mathrm{eff}`.
    * ``'params.<name>'``                           — .
      A free parameter or derived quantity (e.g. ``'params.m_ncdm_tot'``), exposed as a
      tree_flatten leaf. Register this for any name an **external** (``_is_external=True``)
      downstream calculator reads through ``cosmo[name]`` -- without it, that read is
      live under eager execution but goes stale under ``jax.jit`` (see ``__getitem__``).

    Parameters can be accessed through cosmo[name]; free params are always jit-safe,
    derived quantities are jit-safe only once registered as a ``'params.<name>'`` requirement.

    Parameters
    ----------
    engine : str, default='class'
        Boltzmann solver.  JAX-native: ``'eisenstein_hu'``.
        External: ``'camb'``, ``'class'``, etc.
    params : VariableCollection, optional
        Cosmological parameters.  When ``None`` built via :meth:`propose_params`.
        Parameter names: ``h``, ``theta_MC_100``, ``omega_cdm``, ``omega_b``, ``logA``,
        ``n_s``, ``tau_reio``, ``m_ncdm``, ``N_eff``, ``w0_fld``, ``wa_fld``, ``Omega_k``.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default=None
        Fiducial cosmology — seeds parameter default values and is the base for ``clone``.
        ``None`` falls back to a default ``cosmoprimo.Cosmology(engine=engine)``.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls, *args, fiducial='DESI', **kwargs):
        r"""Return a proposed :class:`~desilike.parameter.VariableCollection` of cosmological Parameters.

        The default values are seeded from *fiducial* (``'DESI'`` when ``None``, matching
        :meth:`__post_init__`'s own default).
        The returned collection can be edited and passed back to :meth:`__init__` via ``params=...``.

        Parameters
        ----------
        fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
            Fiducial cosmology used to seed the default parameter values.

        Returns
        -------
        VariableCollection
        """
        fiducial = _get_fiducial(fiducial)
        params = VariableCollection()
        # Planck2018 (TT,TE,EE+lowE+lensing) priors, mirroring the historical
        # primordial_cosmology.yaml.  Extra cosmological parameters (theta_MC_100, tau_reio,
        # N_eff, w0_fld, wa_fld, Omega_k) are fixed by default; free them as needed.
        params.set(Parameter('h', value=fiducial['h'],
                             prior=dict(limits=[0.1, 10.]),
                             ref=dict(dist='norm', loc=fiducial['h'], scale=0.005),
                             fd_eps=0.03, latex='h'))
        params.set(Parameter('omega_cdm', value=fiducial['omega_cdm'],
                             prior=dict(limits=[0.01, 0.99]),
                             ref=dict(dist='norm', loc=fiducial['omega_cdm'], scale=0.0012),
                             fd_eps=0.007, latex=r'\omega_{\mathrm{cdm}}'))
        params.set(Parameter('omega_b', value=fiducial['omega_b'],
                             prior=dict(limits=[0.005, 0.1]),
                             ref=dict(dist='norm', loc=fiducial['omega_b'], scale=0.00015),
                             fd_eps=0.0015, latex=r'\omega_b'))
        params.set(Parameter('logA', value=fiducial['logA'],
                             prior=dict(limits=[1.61, 3.91]),
                             ref=dict(dist='norm', loc=fiducial['logA'], scale=0.014),
                             fd_eps=0.05, latex=r'\ln(10^{10} A_s)'))
        params.set(Parameter('n_s', value=fiducial['n_s'],
                             prior=dict(limits=[0.8, 1.2]),
                             ref=dict(dist='norm', loc=fiducial['n_s'], scale=0.0042),
                             fd_eps=0.005, latex=r'n_s'))
        params.set(Parameter('tau_reio', value=fiducial['tau_reio'], fixed=True,
                             prior=dict(limits=[0.01, 0.8]),
                             ref=dict(dist='norm', loc=fiducial['tau_reio'], scale=0.01),
                             fd_eps=0.01, latex=r'\tau'))
        params.set(Parameter('m_ncdm', value=fiducial['m_ncdm_tot'], fixed=True,
                             prior=dict(limits=[0., 5.]),
                             ref=dict(dist='norm', loc=fiducial['m_ncdm_tot'], scale=0.12, limits=[0., 10.]),
                             fd_eps=(0.31, 0.15, 0.15), latex=r'm_{\mathrm{ncdm}}'))
        params.set(Parameter('N_eff', value=fiducial['N_eff'], fixed=True,
                             prior=dict(limits=[0.01, 10.]),
                             ref=dict(dist='norm', loc=fiducial['N_eff'], scale=0.16),
                             fd_eps=0.2, latex=r'N_{\mathrm{eff}}'))
        params.set(Parameter('w0_fld', value=fiducial['w0_fld'], fixed=True,
                             prior=dict(limits=[-3., 1.]),
                             ref=dict(dist='norm', loc=fiducial['w0_fld'], scale=0.08),
                             fd_eps=0.1, latex=r'w_0'))
        params.set(Parameter('wa_fld', value=fiducial['wa_fld'], fixed=True,
                             prior=dict(limits=[-3., 2.]),
                             ref=dict(dist='norm', loc=fiducial['wa_fld'], scale=0.3),
                             fd_eps=0.3, latex=r'w_a'))
        params.set(Parameter('Omega_k', value=fiducial['Omega_k'], fixed=True,
                             prior=dict(limits=[-0.3, 0.3]),
                             ref=dict(dist='norm', loc=fiducial['Omega_k'], scale=0.0065),
                             fd_eps=0.05, latex=r'\Omega_k'))
        return params

    def __post_init__(self, *args, engine='class', params=None, fiducial='DESI', **kwargs):
        self._engine = str(engine)
        self._is_external = self._engine not in _JAX_ENGINES
        # Build (or resolve) the fiducial once, forcing ``engine`` so that subsequent
        # per-call ``.clone(base='input', ...)`` use the requested engine (not the
        # fiducial's default, e.g. CLASS for the named 'DESI'/'Planck2018' fiducials).
        self._fiducial = _get_fiducial(fiducial).clone(engine=self._engine)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __call__(self):
        # JAX engines: keep tracers (clone is differentiable). External engines: plain floats.
        params = {param.basename: np.asarray(param.value).reshape(-1)[0].item() if self._is_external else param.value
                  for param in self.params}
        self._param_values = params
        # Lensed Cl / lens-potential Cl are opt-in on external engines (camb, class): without
        # requesting 'lensing' at build time, get_harmonic().lensed_cl()/.lens_potential_cl()
        # below raise even though the requirement was registered via add_requirements().
        lens_potential_cl = any(spec_key[0] == 'harmonic.lens_potential_cl' for spec_key in self._requirements)
        lensing = lens_potential_cl or any(spec_key[0] == 'harmonic.lensed_cl' for spec_key in self._requirements)
        # Engine settings derived from the registered harmonic requirements (see
        # _LENSING_CALC_PARAMS / _LENS_POTENTIAL_CL_* above): 'ellmax_cl' always covers the
        # largest requested ellmax; lensed Cl also turn on the non-linear matter power; the
        # lensing potential additionally gets the reconstruction accuracy boost.
        calc_params, extra_params = {}, None
        requested_ellmax = max([0] + [spec['static']['ellmax'] for spec_key, spec in self._requirements.items()
                                      if spec_key[0].startswith('harmonic.') and 'ellmax' in spec['static']])
        if lensing:
            calc_params.update(_LENSING_CALC_PARAMS.get(self._engine, {}))
        if lens_potential_cl:
            extra_params = _LENS_POTENTIAL_CL_EXTRA_PARAMS.get(self._engine)
            requested_ellmax = max(requested_ellmax, _LENS_POTENTIAL_CL_MIN_ELLMAX_CL.get(self._engine, 0))
        if requested_ellmax:
            # Only ever raise ellmax_cl: an explicit (larger) fiducial setting is an accuracy
            # choice that must not be undercut by a smaller likelihood-requested ellmax.
            calc_params['ellmax_cl'] = max(requested_ellmax, self._fiducial['ellmax_cl'])
        if self._is_external:
            try:
                self._cosmo = _build_cosmoprimo(self._fiducial, params, lensing=lensing,
                                                calc_params=calc_params, extra_params=extra_params)
                self._run_requirements(params)
            except (CosmologyInputError, CosmologyComputationError):
                # Unphysical or numerically-pathological input (e.g. omega_cdm < 0, or a
                # solver failure raised lazily from cosmo.get_fourier()/get_background()
                # below): external engines run through pure_callback with concrete
                # (non-Tracer) values, so cosmoprimo's usual "raise outside jax tracing,
                # NaN inside" fallback (exception_or_nan) always raises here, even under
                # jax.jit -- it can never see a real Tracer inside the callback. Mirror
                # that same eager-raise / traced-NaN contract explicitly: re-raise unless
                # the *enclosing* graph execution is jax-traced (node._is_tracing, set by
                # base.py's _run_graph right before dispatching this node's pure_callback,
                # since that is the only place able to observe the outer trace status).
                if not getattr(self, '_is_tracing', False):
                    raise
                # valid; used only for correctly-shaped placeholders below, so must still
                # support 'lensing'/accuracy overrides or _run_requirements' lensed_cl/
                # lens_potential_cl call raises instead of the NaN fallback taking effect.
                self._cosmo = (_build_cosmoprimo(self._fiducial, {}, lensing=lensing,
                                                 calc_params=calc_params, extra_params=extra_params)
                               if lensing else self._fiducial)
                self._run_requirements(params)
                for spec_key in self._requirements:
                    self._results[spec_key] = _nan_like(self._results[spec_key])
                for param in self.derived_params:
                    param.value = _nan_like(param.value)
        else:
            # JAX-native: tracers survive end-to-end (no pure_callback boundary), so
            # cosmoprimo's own exception_or_nan already raises in eager / NaNs under
            # jax.jit-grad-vmap tracing without any extra handling needed here.
            self._cosmo = _build_cosmoprimo(self._fiducial, params, lensing=lensing,
                                            calc_params=calc_params, extra_params=extra_params)
            self._run_requirements(params)

    def _run_requirements(self, params):
        """Populate ``self._results`` / ``self.derived_params`` from ``self._cosmo``."""
        cosmo = self._cosmo
        for spec_key, spec in self._requirements.items():
            method_key = spec_key[0]
            static = spec['static']
            _kw_coords = {coord: spec[coord] for coord in _COORDS if coord in spec}
            if method_key == 'fourier.pk':
                fo = cosmo.get_fourier()
                result = fo.pk_interpolator(of=static['of'], **_kw_pk)(**_kw_coords).T
            elif method_key == 'fourier.pk_now':
                from cosmoprimo import PowerSpectrumBAOFilter
                fo = cosmo.get_fourier()
                pk_interp = fo.pk_interpolator(of=static['of'], **_kw_pk).to_1d(z=_kw_coords['z'])
                bao = PowerSpectrumBAOFilter(pk_interp, engine=static['engine'],
                                             cosmo=cosmo, cosmo_fid=self._fiducial)
                result = bao.smooth_pk_interpolator()(_kw_coords['k']).T
            elif method_key == 'fourier.sigma8_z':
                fo = cosmo.get_fourier()
                result = fo.sigma8_z(**_kw_coords, of=static['of'])
            elif method_key == 'background.efunc':
                result = cosmo.get_background().efunc(**_kw_coords)
            elif method_key == 'background.comoving_transverse_distance':
                result = cosmo.get_background().comoving_transverse_distance(**_kw_coords)
            elif method_key == 'background.luminosity_distance':
                result = cosmo.get_background().luminosity_distance(**_kw_coords)
            elif method_key == 'background.growth_factor':
                result = cosmo.get_background().growth_factor(**_kw_coords)
            elif method_key == 'primordial.pk':
                result = cosmo.get_primordial(mode='scalar').pk_interpolator()(_kw_coords['k'])
            elif method_key == 'background.growth_rate':
                result = cosmo.get_background().growth_rate(**_kw_coords)
            elif method_key == 'harmonic.lensed_cl':
                # Raw (dimensionless) Cl, indexed by ell from 0 to ellmax; unit conversion
                # (e.g. to muK^2) is left to the consumer, matching e.g. background.* above.
                cl = cosmo.get_harmonic().lensed_cl(ellmax=static['ellmax'])
                result = {name: cl[name] for name in ['tt', 'ee', 'bb', 'te']}
            elif method_key == 'harmonic.unlensed_cl':
                cl = cosmo.get_harmonic().unlensed_cl(ellmax=static['ellmax'])
                result = {name: cl[name] for name in ['tt', 'ee', 'bb', 'te']}
            elif method_key == 'harmonic.lens_potential_cl':
                cl = cosmo.get_harmonic().lens_potential_cl(ellmax=static['ellmax'])
                result = {name: cl[name] for name in ['pp', 'tp', 'ep']}
            elif method_key == 'fourier.sigma_rz':
                # cosmoprimo's sigma_rz(r, z) returns shape (r, z); transpose to the (z, r)
                # convention used elsewhere (e.g. 'fourier.pk' returns (z, k)).
                result = cosmo.get_fourier().sigma_rz(spec['r'], spec['z'], of=static['of']).T
            elif method_key in ('background.Omega_b', 'background.Omega_cdm', 'background.Omega_ncdm_tot'):
                result = getattr(cosmo.get_background(), method_key.split('.')[1])(spec['z'])
            elif method_key == 'thermodynamics.rs_drag':
                result = cosmo.get_thermodynamics().rs_drag
                if 'z' in spec:
                    # z-independent; broadcast to the registered z grid so get()'s
                    # per-z searchsorted indexing below still applies cleanly.
                    result = jnp.full(spec['z'].shape, result)
            elif method_key == 'background.age':
                result = cosmo.get_background().age
            elif method_key.startswith('params.'):
                # Raw parameter/derived-quantity value, exposed as a tree_flatten leaf so
                # external (pure_callback) consumers see the live, per-call value instead
                # of a stale read off self._cosmo (see __getitem__).
                name = method_key[len('params.'):]
                if name in params:
                    result = jnp.asarray(params[name])
                else:
                    result = jnp.asarray(cosmo[name])
            else:
                raise ValueError(f'Unknown requirement method key: {method_key!r}')
            self._results[spec_key] = result
        # Here set derived_params
        for param, getter in self._get_derived.items():
            self.derived_params[param].value = jnp.reshape(self.get(getter[0], **getter[1]), self.derived_params[param].shape)
    # tree_flatten/tree_unflatten: inherited as-is from PrimordialCosmology.
    # self._cosmo (the live cosmoprimo.Cosmology) is deliberately *not* exposed as a
    # leaf: it is itself a huge, cache-dependent pytree (its leaf count can change with
    # internal caching state), and one external (pure_callback) consumer needing a
    # *fixed* leaf count per node would silently misalign on a leaf-count mismatch.
    # Same-trace (non-external) consumers needing a derived quantity not in params
    # (e.g. ``self.cosmo['m_ncdm_tot']``) still get it via the __getitem__ fallback below:
    # dep.__dict__.update(proxy.__dict__) only *adds/overwrites* keys present on the
    # proxy, so the live ``_cosmo`` set by this node's own __call__ earlier in the same
    # trace is left untouched.



_CONVERSION_JAXACE = {'ln10As': 'logA', 'ns': 'n_s', 'h': 'h', 'omega_b': 'omega_b', 'omega_c': 'omega_cdm',
                         'm_nu': 'm_ncdm', 'w0': 'w0_fld', 'wa': 'wa_fld'}


# ── Packaged trained emulators ────────────────────────────────────────────────
# Trained emulators shipped with the jaxace / jaxmapse / jaxcapse packages (downloaded on
# demand from Zenodo through each package's artifact registry).  Their nn_setup.json only
# carries free-text descriptions, so the desilike-facing metadata is declared here:
# 'inputs' are the network inputs in desilike/cosmoprimo parameter names (resolved through
# get_param in ACECosmology.__call__, so e.g. 'H0' works whether h or theta_MC_100 is
# sampled), and 'outputs' are the requirement method keys the emulator serves.
_PACKAGED_EMULATORS = {
    # jaxace ACE emulator (trained on CLASS).  Network outputs, in order:
    # (sigma8, sigma8_z, rs_drag [Mpc], H_z [km/s/Mpc], r_z [Mpc], D_z, f_z).
    # sigma8_z is total-matter; it is also served for of='delta_cb' as an approximation
    # (0.5% low at the DESI fiducial with m_ncdm = 0.06 eV).
    'ACE_mnuw0wacdm_ln10As_basis': dict(
        kind='jaxace',
        inputs=['z', 'logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'm_ncdm', 'w0_fld', 'wa_fld'],
        outputs=['fourier.sigma8_z.delta_m.delta_m', 'fourier.sigma8_z.delta_cb.delta_cb',
                 'fourier.sigma8_z.theta_cb.theta_cb',
                 'thermodynamics.rs_drag', 'thermodynamics.rs_drag.delta_cb.delta_cb'],
        # Training ranges (from the network's in_minmax), used by the out-of-range guard in
        # __call__: inputs are clipped to these before evaluation (so downstream spline /
        # linear solves never see NaN) and all results are masked to NaN outside them.
        ranges={'logA': (2.0, 3.7), 'n_s': (0.8, 1.1), 'H0': (50., 90.), 'omega_b': (0.02, 0.025),
                'omega_cdm': (0.08, 0.18), 'm_ncdm': (0., 0.5), 'w0_fld': (-3., 0.5), 'wa_fld': (-3., 2.)},
    ),
    # jaxmapse linear power spectrum emulator (trained on CLASS, in Mpc units: k_grid in
    # 1/Mpc, pk in Mpc^3 -- converted to h/Mpc and (Mpc/h)^3 in __call__).  of='theta_cb'
    # is served as f_z^2 * pk_cb with f_z from the packaged jaxace emulator above
    # (scale-independent growth), so that sigma8_z(theta_cb) = f_z * sigma8_z(delta_cb).
    'mnuw0wacdm_class': dict(
        kind='jaxmapse',
        inputs=['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'm_ncdm', 'w0_fld', 'wa_fld'],
        outputs=['fourier.pk.delta_cb.delta_cb', 'fourier.pk.delta_m.delta_m', 'fourier.pk.theta_cb.theta_cb',
                 'fourier.pk_now.delta_cb.delta_cb', 'fourier.pk_now.delta_m.delta_m'],
        # Linear-pk networks consume (z, H0, ombh2, omch2, mnu, w0, wa); logA / n_s enter
        # analytically through the postprocessing, hence no range on them here.
        ranges={'H0': (50., 90.), 'omega_b': (0.02, 0.025), 'omega_cdm': (0.08, 0.18),
                'm_ncdm': (0., 0.5), 'w0_fld': (-3., 0.5), 'wa_fld': (-3., 2.)},
    ),
    # jaxcapse CMB Cl emulator (trained on CAMB, LCDM only).  Networks output, for
    # ell = 2..5000, Dl = ell (ell + 1) / (2 pi) Cl in muK^2 (TT / TE / EE) and
    # ell^2 (ell + 1)^2 / (2 pi) Cl^phiphi (PP) -- converted to the raw dimensionless Cl
    # convention of CosmoprimoCosmology in __call__.
    'camb_lcdm': dict(
        kind='jaxcapse',
        inputs=['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio'],
        outputs=['harmonic.lensed_cl', 'harmonic.lens_potential_cl'],
        ellmax=5000,
        ranges={'logA': (2.5, 3.5), 'n_s': (0.88, 1.05), 'H0': (40., 100.), 'omega_b': (0.0193, 0.0253),
                'omega_cdm': (0.08, 0.2), 'tau_reio': (0.02, 0.12)},
    ),
}

# Default packaged-emulator selection, for ACECosmology(engine='ace').
_PACKAGED_DEFAULT_ENGINE = {'background': 'ACE_mnuw0wacdm_ln10As_basis', 'fourier': 'mnuw0wacdm_class', 'harmonic': 'camb_lcdm'}

# Cosmological parameters that, when varied, must be inputs of a matched packaged emulator;
# varying one that is not leaves the emulated quantity blind to it (see the warning in
# ACECosmology._warn_uncovered_params).
_PACKAGED_COSMO_PARAMS = frozenset(['h', 'theta_MC_100', 'omega_cdm', 'omega_b', 'logA', 'n_s', 'tau_reio',
                                    'm_ncdm', 'N_eff', 'N_ur', 'w0_fld', 'wa_fld', 'Omega_k'])

# Free-text parameter tokens found in Capse-style nn_setup.json 'parameters' descriptions
# (normalized to lowercase with spaces and trailing periods stripped), mapped to desilike
# parameter names; used by _find_capse_metadata below.
_CONVERSION_CAPSE = {'ln10^10as': 'logA', 'ln10as': 'logA', 'loga': 'logA',
                     'ns': 'n_s', 'h0': 'H0',
                     'omega_b': 'omega_b', 'ombh2': 'omega_b', 'ωb': 'omega_b', 'wb': 'omega_b',
                     'omega_c': 'omega_cdm', 'omch2': 'omega_cdm', 'ωc': 'omega_cdm', 'wc': 'omega_cdm',
                     'tau': 'tau_reio', 'τ': 'tau_reio',
                     'mnu': 'm_ncdm', 'mν': 'm_ncdm',
                     'w0': 'w0_fld', 'wa': 'wa_fld'}


def _find_capse_metadata(emulator_dir):
    """Introspect a Capse-style Cl emulator directory: per-spectrum network subdirectories
    ('TT', 'TE', 'EE', and optionally 'BB', 'PP'), each holding nn_setup.json / weights.npy /
    inminmax.npy / outminmax.npy / postprocessing.py, as produced by the CosmologicalEmulators
    training pipeline (e.g. the local 'capse_mnuw0wacdm_250001' set).  Network inputs are read
    from an explicit desilike-style 'input' list in nn_setup.json's emulator_description when
    present, else parsed from its free-text 'parameters' description via _CONVERSION_CAPSE;
    training ranges come from inminmax.npy and the maximum multipole from the outminmax.npy
    row count (outputs cover ell = 2..ellmax).  Cl conventions are assumed identical to the
    packaged 'camb_lcdm' set: Dl in muK^2 (TT/TE/EE), ell^2 (ell+1)^2 / (2 pi) Cl^phiphi (PP)."""
    import json
    spectra = [name for name in ['TT', 'TE', 'EE', 'BB', 'PP'] if (emulator_dir / name / 'nn_setup.json').is_file()]
    with open(emulator_dir / spectra[0] / 'nn_setup.json') as file:
        description = json.load(file).get('emulator_description', {})
    inputs = description.get('input', None)
    if inputs is None:
        inputs = []
        for token in str(description.get('parameters', '')).split(','):
            normalized = token.strip().strip('.').replace(' ', '').lower()
            if normalized not in _CONVERSION_CAPSE:
                raise ValueError(f"cannot map parameter token {token.strip()!r} of Capse-style emulator {emulator_dir} to a desilike name; "
                                 f"recognized tokens: {sorted(_CONVERSION_CAPSE)}; alternatively, provide an explicit 'input' list "
                                 "(desilike parameter names) in nn_setup.json's emulator_description")
            inputs.append(_CONVERSION_CAPSE[normalized])
    in_minmax = np.load(emulator_dir / spectra[0] / 'inminmax.npy')
    if len(inputs) != len(in_minmax):
        raise ValueError(f'Capse-style emulator {emulator_dir}: {len(inputs)} parameter names for {len(in_minmax)} network inputs')
    ranges = {name: (float(low), float(high)) for name, (low, high) in zip(inputs, in_minmax)}
    ellmax = np.load(emulator_dir / spectra[0] / 'outminmax.npy').shape[0] + 1
    outputs = ['harmonic.lensed_cl'] + (['harmonic.lens_potential_cl'] if 'PP' in spectra else [])
    return dict(kind='jaxcapse', inputs=list(inputs), outputs=outputs, ranges=ranges, ellmax=int(ellmax), spectra=spectra)


def _interp_loglog(k_query, k_knots, pk_knots):
    """Cubic spline interpolation in log10(k) space."""
    import interpax
    shape = jnp.shape(k_query)
    flat = jnp.ravel(k_query)
    result = interpax.interp1d(jnp.log10(flat), jnp.log10(k_knots), pk_knots, method='cubic', extrap=True)
    # Preserve pk_knots's trailing axes (e.g. the z dimension); only the k_query axis is reshaped.
    return jnp.reshape(result, shape + jnp.shape(pk_knots)[1:])


class ACECosmology(PrimordialCosmology):
    r"""
    :class:`PrimordialCosmology` backed by neural-network emulators (pure JAX end-to-end).

    Background quantities (``background.efunc``, ``background.comoving_transverse_distance``,
    ``background.growth_factor`` / ``growth_rate``, ...) are computed analytically with
    :mod:`jaxace`'s ``w0waCDMCosmology``; everything else is served by trained emulators,
    selected through *engine*:

    * a directory name under *base_dir* (custom emulators, one subdirectory per network, each
      with an ``nn_setup.json`` declaring desilike-style ``input`` / ``output`` metadata),
    * a Capse-style Cl emulator directory under *base_dir* (per-spectrum ``TT / TE / EE
      [/ BB] [/ PP]`` network subdirectories, e.g. ``'capse_mnuw0wacdm_250001'``): inputs,
      training ranges and ellmax are introspected from the networks' own metadata
      (see :func:`_find_capse_metadata`), or
    * the name of a packaged trained emulator shipped by jaxace / jaxmapse / jaxcapse
      (downloaded on demand from Zenodo); see ``_PACKAGED_EMULATORS`` for the registry.

    ``engine='ace'`` selects the default packaged set::

        engine={'background': 'ACE_mnuw0wacdm_ln10As_basis',   # sigma8_z, fsigma8 = f_z sigma8_z, rs_drag
                'fourier': 'mnuw0wacdm_class',                 # linear pk (delta_cb, delta_m, theta_cb = f_z^2 pk_cb)
                'harmonic': 'camb_lcdm'}                       # lensed TT/TE/EE + lensing potential Cl (LCDM only)

    which serves :class:`~desilike.theories.galaxy_clustering.template.DirectSpectrum2Template`
    (``fourier.pk`` of ``delta_cb`` / ``theta_cb``, ``fourier.sigma8_z``, background quantities)
    and the candl / clik CMB likelihoods (``harmonic.lensed_cl``, ``harmonic.lens_potential_cl``
    up to ellmax = 5000), plus the derived ``sigma8_m`` and ``rs_drag`` (included by default
    in :meth:`propose_params` when *engine* has a packaged jaxace emulator).  Notes: the ACE
    ``sigma8_z`` is
    total-matter (served for ``of='delta_cb'`` as an approximation, 0.5% low at the DESI
    fiducial); ``bb`` is returned as zeros; the packaged ``camb_lcdm`` Cl emulator is
    LCDM-only (a warning is emitted when a varied parameter is not an emulator input).
    """

    @classmethod
    def propose_params(cls, *args, engine='isitgr', fiducial='DESI', **kwargs):
        r"""Return a proposed :class:`~desilike.parameter.VariableCollection` of cosmological Parameters.

        The default values are seeded from *fiducial* (``'DESI'`` when ``None``, matching
        :meth:`__post_init__`'s own default).
        The returned collection can be edited and passed back to :meth:`__init__` via ``params=...``.

        Parameters
        ----------
        engine : str or dict, default='isitgr'
            Same as :meth:`__post_init__`'s *engine*.  When it includes a packaged jaxace
            emulator (e.g. ``engine='ace'``), the derived parameters ``sigma8_m`` and
            ``rs_drag`` are included (custom emulator directories do not necessarily serve
            the corresponding requirements, so they are left out otherwise).
        fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
            Fiducial cosmology used to seed the default parameter values.

        Returns
        -------
        VariableCollection
        """
        params = CosmoprimoCosmology.propose_params(*args, fiducial=fiducial, **kwargs)
        engine_names = list(engine.values()) if isinstance(engine, dict) else [engine]
        if 'ace' in engine_names or any(_PACKAGED_EMULATORS.get(name, {}).get('kind') == 'jaxace' for name in engine_names):
            params.set(Parameter('sigma8_m', value=0., derived=True, latex=r'\sigma_8'))
            params.set(Parameter('rs_drag', value=0., derived=True, latex=r'r_{\mathrm{drag}}'))
        return params

    @classmethod
    def training_ranges(cls, engine='ace', base_dir=None, basis='cosmo'):
        r"""Return the training ranges of the emulators selected by *engine*.

        These are the ranges enforced by :meth:`__call__`'s out-of-range guard: inputs are
        clipped to them before evaluation and every emulated result is NaN-masked when a
        parameter falls outside.

        Parameters
        ----------
        engine : str or dict, default='ace'
            Same as :meth:`__post_init__`'s *engine*.  Packaged emulator names and
            Capse-style directories contribute their training ranges; custom desilike-style
            emulator directories declare none.
        base_dir : str, Path, optional
            Same as :meth:`__post_init__`'s *base_dir*.
        basis : str, default='cosmo'
            ``'cosmo'``: desilike cosmological parameter names (the ``'H0'`` range is
            reported as ``'h'``, scaled by 1/100).  ``'emulator'``: the networks' native
            input names (``'H0'`` as such).

        Returns
        -------
        dict
            ``{parameter name: (low, high)}``, intersected across the selected emulators.
        """
        if basis not in ('cosmo', 'emulator'):
            raise ValueError(f"basis must be 'cosmo' or 'emulator', got {basis!r}")
        base_emulator_dir = Path(base_dir) if base_dir is not None else Path(Installer().install_dir) / 'ace-emulators'
        if isinstance(engine, str):
            engine = dict(_PACKAGED_DEFAULT_ENGINE) if engine == 'ace' else {section: engine for section in ['harmonic', 'fourier', 'background']}
        training_ranges = {}
        for engine_name in set(engine.values()):
            if engine_name is None:
                continue
            emulator_dir = base_emulator_dir / engine_name
            if (emulator_dir / 'TT' / 'nn_setup.json').is_file():
                # Capse-style Cl emulator directory: introspect the networks' training ranges.
                emulator_ranges = _find_capse_metadata(emulator_dir)['ranges']
            else:
                emulator_ranges = _PACKAGED_EMULATORS.get(engine_name, {}).get('ranges', {})
            for name, (low, high) in emulator_ranges.items():
                previous_low, previous_high = training_ranges.get(name, (-np.inf, np.inf))
                training_ranges[name] = (max(low, previous_low), min(high, previous_high))
        if basis == 'cosmo' and 'H0' in training_ranges:
            low, high = training_ranges.pop('H0')
            previous_low, previous_high = training_ranges.get('h', (-np.inf, np.inf))
            training_ranges['h'] = (max(low / 100., previous_low), min(high / 100., previous_high))
        return training_ranges

    @classmethod
    def truncate_priors(cls, params, engine='ace', base_dir=None):
        r"""Intersect each parameter's prior in *params* with the emulators' training ranges.

        Outside the training ranges (see :meth:`training_ranges`) :meth:`__call__` NaN-masks
        every emulated result (which :class:`~desilike.base.Posterior` maps to ``-inf``) — an
        effective prior truncation regardless; this makes it explicit, so prior draws (e.g.
        the initial particles of nested / SMC samplers) always land at a finite
        log-likelihood.

        Parameters
        ----------
        params : VariableCollection
            Parameters whose priors to truncate (e.g. from :meth:`propose_params`); the
            matching non-derived Parameters are updated in place.
        engine : str or dict, default='ace'
            Same as :meth:`__post_init__`'s *engine*.
        base_dir : str, Path, optional
            Same as :meth:`__post_init__`'s *base_dir*.

        Returns
        -------
        VariableCollection
            *params*, with each prior's limits intersected with the training ranges.
        """
        from ..parameter import truncate_priors as truncate_priors_to_ranges
        return truncate_priors_to_ranges(params, cls.training_ranges(engine=engine, base_dir=base_dir, basis='cosmo'))

    def __post_init__(self, *args, engine='isitgr', base_dir=None, conversion='cosmoprimo', params=None, fiducial='DESI', **kwargs):
        self._engine = str(engine)
        if base_dir is not None:
            base_emulator_dir = Path(base_dir)
        else:
            base_emulator_dir = Path(Installer().install_dir) / 'ace-emulators'
        _SECTIONS = ['harmonic', 'fourier', 'background']
        if isinstance(engine, str):
            engine = dict(_PACKAGED_DEFAULT_ENGINE) if engine == 'ace' else {section: engine for section in _SECTIONS}

        def _find_inputs_outputs(emulator_dir):
            import json
            with open(emulator_dir / "nn_setup.json") as f:
                nn_dict = json.load(f)
            description = nn_dict.get('emulator_description', {})
            inputs = description.get('input')
            outputs = description.get('output')
            return list(inputs), list(outputs)

        # Per-emulator metadata: {'inputs', 'outputs', 'kind' (None for desilike-style custom
        # directories), and for packaged / Capse-style emulators 'ranges', 'ellmax', 'spectra'}.
        self._emulator_metadata = {}
        seen_emulator_dirs = set()
        for section in _SECTIONS:
            engine_name = engine.get(section, None)
            if engine_name is None:
                continue
            emulator_dir = base_emulator_dir / engine_name
            if emulator_dir.is_dir():
                if emulator_dir in seen_emulator_dirs:
                    continue
                seen_emulator_dirs.add(emulator_dir)
                if (emulator_dir / 'TT' / 'nn_setup.json').is_file():
                    # Capse-style Cl emulator directory (per-spectrum network subdirs).
                    self._emulator_metadata[str(emulator_dir)] = _find_capse_metadata(emulator_dir)
                else:
                    # Iterate on all (leaf) emulators in emulator_dir
                    for leaf_dir in sorted(path for path in emulator_dir.iterdir() if path.is_dir()):
                        inputs, outputs = _find_inputs_outputs(leaf_dir)
                        self._emulator_metadata[str(leaf_dir)] = dict(kind=None, inputs=inputs, outputs=outputs)
            elif engine_name in _PACKAGED_EMULATORS:
                self._emulator_metadata[engine_name] = dict(_PACKAGED_EMULATORS[engine_name])
        # Packaged jaxace (ACE) emulator, if any: beyond its own matched requirements, it also
        # provides f_z for fourier.pk of='theta_cb' (see _load_emulators_for_new_requirements).
        self._ace_emulator_key = next((key for key, metadata in self._emulator_metadata.items() if metadata.get('kind') == 'jaxace'), None)
        self._loaded_emulators = {}
        self._method_emulator_matching = {}
        # Load emulators for requirements already registered before __post_init__ (e.g. derived params).
        self._load_emulators_for_new_requirements()
        self._conversion = conversion
        if self._conversion == 'cosmoprimo':
            # Build (or resolve) the fiducial once, forcing ``engine`` so that subsequent
            # per-call ``.clone(base='input', ...)`` use the requested engine (not the
            # fiducial's default, e.g. CLASS for the named 'DESI'/'Planck2018' fiducials).
            self._fiducial = _get_fiducial(fiducial).clone(engine='eisenstein_hu')
            self._cosmoprimo_params = frozenset(self._fiducial.get_default_params(include_conflicts=True))

    def add_requirements(self, requirements):
        super().add_requirements(requirements)
        if hasattr(self, '_emulator_metadata'):
            self._load_emulators_for_new_requirements()

    def _load_emulator(self, emulator_key):
        metadata = self._emulator_metadata[emulator_key]
        kind = metadata.get('kind', None)
        if kind == 'jaxace':
            import jaxace
            return jaxace.get_emulator(emulator_key)
        if kind == 'jaxmapse':
            import jaxmapse
            # Dict of per-component TransferFunctionEmulators, keyed like of= (mirrors the
            # jaxcapse per-spectrum dict below); pre/postprocessing resolve from the files
            # shipped inside each artifact component directory.
            root = Path(jaxmapse.artifact_path(emulator_key))
            return {'delta_m': jaxmapse.load_emulator(str(root / 'Pk_lin_mm')),
                    'delta_cb': jaxmapse.load_emulator(str(root / 'Pk_lin_cb'))}
        if kind == 'jaxcapse':
            import jaxcapse
            if 'spectra' in metadata:
                # Local Capse-style directory: emulator_key is the directory path,
                # holding one network subdirectory per spectrum.
                return {name: jaxcapse.load_emulator(str(Path(emulator_key) / name)) for name in metadata['spectra']}
            # Packaged set: dict of per-spectrum MLPs ('TT', 'TE', 'EE', 'PP'), auto-loaded at
            # import unless JAXCAPSE_NO_AUTO_DOWNLOAD is set, in which case entries are None.
            emulators = jaxcapse.trained_emulators.get(emulator_key, {})
            if not emulators or any(mlp is None for mlp in emulators.values()):
                emulators = jaxcapse.reload_emulators(emulator_key)[emulator_key]
            return emulators
        # Custom desilike-style emulator directory: emulator_key is the directory path.
        outputs = metadata['outputs']
        if any(output.startswith('fourier.pk') for output in outputs):
            import jaxmapse
            emulator = jaxmapse.load_emulator(emulator_key)
        elif any(output.startswith('harmonic.') for output in outputs):
            import jaxcapse
            emulator = jaxcapse.load_emulator(emulator_key)
        else:
            import jaxace
            emulator = jaxace.load_trained_emulator(emulator_key)
        return emulator

    def _warn_uncovered_params(self, emulator_key, method_key):
        """Warn when a varied cosmological parameter is not an input of a matched packaged / Capse-style emulator."""
        metadata = self._emulator_metadata[emulator_key]
        if metadata.get('kind', None) is None:
            return
        covered = set(metadata['inputs'])
        if 'H0' in covered:
            covered |= {'h', 'theta_MC_100'}
        relevant = set(_PACKAGED_COSMO_PARAMS)
        if metadata['kind'] != 'jaxcapse':
            relevant.discard('tau_reio')  # tau_reio only affects the CMB spectra
        for param in self.params:
            if not param.fixed and param.basename in relevant and param.basename not in covered:
                warnings.warn(f'parameter {param.basename!r} is varied but is not an input of emulator {emulator_key!r}: '
                              f'{method_key} will not respond to it')

    def _load_emulators_for_new_requirements(self):
        for spec_key, spec in self._requirements.items():
            method_key = spec_key[0]
            if 'of' in spec['static']:
                method_key = f'{method_key}.' + '.'.join(spec['static']['of'])
            if method_key not in self._method_emulator_matching:
                found = False
                for emulator_key, metadata in self._emulator_metadata.items():
                    if method_key in metadata['outputs']:
                        if emulator_key not in self._loaded_emulators:
                            self._loaded_emulators[emulator_key] = self._load_emulator(emulator_key)
                        self._method_emulator_matching[method_key] = emulator_key
                        self._warn_uncovered_params(emulator_key, method_key)
                        found = True
                        break
                if not found:
                    if method_key.split('.')[0] not in ('background', 'params', 'primordial'):
                        raise NotImplementedError(f"could not find {method_key} in emulators' products")
                    continue
            # Per-spec validation for packaged / Capse-style emulators (runs also when the method
            # was already matched, since e.g. a new spec may request a larger ellmax for the same method).
            emulator_key = self._method_emulator_matching[method_key]
            metadata = self._emulator_metadata[emulator_key]
            if metadata.get('kind', None) is None:
                continue
            if metadata['kind'] == 'jaxcapse' and spec['static'].get('ellmax', 0) > metadata['ellmax']:
                raise ValueError(f"requested ellmax={spec['static']['ellmax']} for {method_key} exceeds "
                                 f"emulator {emulator_key!r} training range (ellmax={metadata['ellmax']})")
            if metadata['kind'] == 'jaxmapse' and method_key == 'fourier.pk.theta_cb.theta_cb':
                # pk_tt = f_z^2 pk_cb needs f_z from the packaged jaxace emulator; load it now.
                if self._ace_emulator_key is None:
                    raise ValueError("fourier.pk with of='theta_cb' requires a packaged jaxace emulator "
                                     "(engine['background'], e.g. 'ACE_mnuw0wacdm_ln10As_basis') providing f_z")
                if self._ace_emulator_key not in self._loaded_emulators:
                    self._loaded_emulators[self._ace_emulator_key] = self._load_emulator(self._ace_emulator_key)
        self._rebuild_param_clip_ranges()

    def _rebuild_param_clip_ranges(self):
        """Intersect the training ranges of all loaded packaged / Capse-style emulators, keyed by
        desilike parameter name.  __call__ clips its inputs to these ranges before evaluation and
        masks every result to NaN when any parameter falls outside (graceful rejection instead of
        a non-finite crash in downstream spline / linear solves)."""
        self._param_clip_ranges = {}
        for emulator_key in self._loaded_emulators:
            for name, (low, high) in self._emulator_metadata[emulator_key].get('ranges', {}).items():
                if name in self._param_clip_ranges:
                    prev_low, prev_high = self._param_clip_ranges[name]
                    self._param_clip_ranges[name] = (max(low, prev_low), min(high, prev_high))
                else:
                    self._param_clip_ranges[name] = (low, high)
        if 'H0' in self._param_clip_ranges:
            low, high = self._param_clip_ranges['H0']
            self._param_clip_ranges.setdefault('h', (low / 100., high / 100.))
        # One-time warning per parameter whose prior extends beyond the emulator training range:
        # such samples yield NaN results, i.e. the prior is effectively truncated to the range.
        warned = getattr(self, '_warned_prior_ranges', set())
        for param in self.params:
            name = param.basename
            if name in warned or name not in self._param_clip_ranges or param.fixed:
                continue
            limits = getattr(param.prior, 'limits', None)
            if limits is None:
                continue
            low, high = self._param_clip_ranges[name]
            if limits[0] < low or limits[1] > high:
                warnings.warn(f'parameter {name!r} prior range {tuple(limits)} extends beyond the packaged emulator '
                              f'training range ({low}, {high}): samples outside yield NaN (effective prior truncation)')
                warned.add(name)
        self._warned_prior_ranges = warned

    def __call__(self):
        import jaxace
        self._param_values = params = {param.basename: param.value for param in self.params}
        if self._conversion == 'cosmoprimo':
            # Only forward the standard cosmological parameters to cosmoprimo: extra,
            # emulator-specific nuisance inputs (e.g. 'mu1', 'Sigma1', ...) are unknown to
            # it and must be read directly from self._param_values (see get_param below).
            cosmo_params = {name: value for name, value in self._param_values.items() if name in self._cosmoprimo_params}
            cosmoprimo_cosmo = _build_cosmoprimo(self._fiducial, cosmo_params)

            def get_param(name):
                if name in self._param_values and name not in self._cosmoprimo_params:
                    return self._param_values[name]
                if name == 'm_ncdm':
                    name = 'm_ncdm_tot'
                return cosmoprimo_cosmo[name]

        else:

            # Basic conversion
            def get_param(name):
                if name in self._param_values:
                    return self._param_values[name]
                if name == 'H0':
                    return 100. * self._param_values['h']
                if name == 'Omega_m':
                    omega_m = self._param_values.get('omega_cdm', 0.) + self._param_values.get('omega_b', 0.) + self._param_values.get('m_ncdm', 0.) / 93.14
                    return omega_m / self._param_values['h'] ** 2
                raise KeyError(f'cannot resolve parameter {name!r}')

        # Out-of-range guard for packaged emulators: clip parameter values to the training
        # ranges so every internal evaluation (networks, splines, BAO filter) stays finite,
        # record per-parameter validity, and mask all results to NaN below when invalid.
        clip_ranges = getattr(self, '_param_clip_ranges', {})
        params_in_range = {}
        if clip_ranges:
            unclipped_get_param = get_param

            def get_param(name):
                value = unclipped_get_param(name)
                if name in clip_ranges:
                    low, high = clip_ranges[name]
                    params_in_range[name] = (value >= low) & (value <= high)
                    value = jnp.clip(value, low, high)
                return value

        jaxace_cosmo = {}
        for jaxace_name, name in _CONVERSION_JAXACE.items():
            jaxace_cosmo[jaxace_name] = get_param(name)
        jaxace_cosmo = jaxace.w0waCDMCosmology(**jaxace_cosmo)

        def run_ace(z):
            """Run the packaged jaxace (ACE) network on the z grid; returns outputs of shape
            (nz, 7), in order (sigma8, sigma8_z, rs_drag [Mpc], H_z, r_z, D_z, f_z)."""
            emulator = self._loaded_emulators[self._ace_emulator_key]
            input_names = self._emulator_metadata[self._ace_emulator_key]['inputs']
            z = jnp.atleast_1d(jnp.asarray(z))
            emulator_input = jnp.stack([z if name == 'z' else jnp.full(z.shape, get_param(name)) for name in input_names], axis=-1)
            return emulator.run_emulator(emulator_input)

        for spec_key, spec in self._requirements.items():
            method_key = spec_key[0]
            if 'of' in spec['static']:
                method_key = f'{method_key}.' + '.'.join(spec['static']['of'])
            _kw_coords = {coord: spec[coord] for coord in _COORDS if coord in spec}
            emulator_key = self._method_emulator_matching.get(method_key, None)
            emulator = self._loaded_emulators[emulator_key] if emulator_key is not None else None
            input_names = self._emulator_metadata[emulator_key]['inputs'] if emulator_key is not None else None
            kind = self._emulator_metadata[emulator_key].get('kind', None) if emulator_key is not None else None
            if emulator is None:
                if method_key == 'background.efunc':
                    result = jaxace_cosmo.E_z(**_kw_coords)
                elif method_key == 'background.comoving_transverse_distance':
                    result = jaxace_cosmo.dM_z(**_kw_coords) * jaxace_cosmo.h
                elif method_key == 'background.luminosity_distance':
                    result = jaxace_cosmo.dL_z(**_kw_coords) * jaxace_cosmo.h
                elif method_key == 'background.growth_factor':
                    result = jaxace_cosmo.D_z(**_kw_coords)
                elif method_key == 'background.growth_rate':
                    result = jaxace_cosmo.f_z(**_kw_coords)
                elif method_key == 'background.age':
                    if self._conversion != 'cosmoprimo':
                        raise NotImplementedError("background.age requires conversion='cosmoprimo'")
                    # Background-only quantity, exact whatever the (JAX-traceable) transfer engine.
                    result = jnp.asarray(cosmoprimo_cosmo.get_background().age)
                elif method_key == 'primordial.pk':
                    k_arr = _kw_coords['k']
                    n_s = get_param('n_s')
                    logA = get_param('logA')
                    A_s = jnp.exp(logA) * 1e-10
                    h = get_param('h')
                    k_piv_hMpc = 0.05 / h
                    lnkkp = jnp.log(k_arr / k_piv_hMpc)
                    alpha_s = self._param_values.get('alpha_s', 0.)
                    beta_s = self._param_values.get('beta_s', 0.)
                    result = h**3 * A_s * (k_arr / k_piv_hMpc) ** (n_s - 1. + alpha_s / 2. * lnkkp + beta_s / 6. * lnkkp**2)
                elif method_key.startswith('params.'):
                    name = method_key[len('params.'):]
                    result = get_param(name)
                else:
                    raise NotImplementedError(f'no background formula for {method_key!r}')
            elif kind == 'jaxace':
                ace_output = run_ace(spec['z'] if 'z' in spec else 0.)
                if method_key.startswith('fourier.sigma8_z'):
                    # sigma8_z is total-matter; of='delta_cb' is served with the same value
                    # (see _PACKAGED_EMULATORS).  For theta: fsigma8(z) = f_z * sigma8_z.
                    result = ace_output[:, 1]
                    if spec['static']['of'][0].startswith('theta'):
                        result = ace_output[:, 6] * result
                else:
                    # thermodynamics.rs_drag: z-independent; ACE output in Mpc, convert to Mpc/h.
                    rs_drag = ace_output[:, 2] * get_param('h')
                    result = rs_drag if 'z' in spec else rs_drag[0]
            elif kind == 'jaxmapse':
                # Linear pk; the packaged networks are trained in Mpc units (k_grid in 1/Mpc,
                # pk in Mpc^3), converted below to desilike's k in h/Mpc, pk in (Mpc/h)^3.
                emulator_params = jnp.array([get_param(name) for name in input_names])
                z = spec['z']
                growth = jaxace_cosmo.D_z(z)
                of = spec['static']['of'][0]
                # theta_cb is served from the delta_cb network (times f_z^2 below)
                component = emulator['delta_m' if of == 'delta_m' else 'delta_cb']
                pk, k_grid = component.get_Pk(emulator_params, z, growth), component.k_grid
                if of.startswith('theta'):
                    # pk_tt = f_z^2 pk_cb (scale-independent growth), with f_z from the packaged
                    # jaxace emulator so that sigma8_z(theta_cb) = f_z * sigma8_z(delta_cb) exactly.
                    pk = run_ace(z)[:, 6, None]**2 * pk
                h = get_param('h')
                if method_key.startswith('fourier.pk_now'):
                    # No-wiggle pk: same cosmoprimo BAO filter as CosmoprimoCosmology, applied to
                    # the emulated pk (JAX-traceable, like the eisenstein_hu engine path).  The
                    # cosmoprimo interpolator needs concrete k knots, so first resample the pk
                    # (whose emulator k grid divided by traced h is itself traced) onto a fixed
                    # h/Mpc grid covering the emulator range for any reasonable h.
                    from cosmoprimo import PowerSpectrumBAOFilter, PowerSpectrumInterpolator1D
                    k_fixed = np.geomspace(1e-5, 50., 300)
                    pk_fixed = _interp_loglog(k_fixed, k_grid / h, (pk * h**3).T)
                    pk_interp = PowerSpectrumInterpolator1D(k_fixed, pk_fixed, **_kw_pk)
                    filter_cosmo = cosmoprimo_cosmo if self._conversion == 'cosmoprimo' else None
                    bao = PowerSpectrumBAOFilter(pk_interp, engine=spec['static']['engine'], cosmo=filter_cosmo,
                                                 cosmo_fid=self._fiducial if self._conversion == 'cosmoprimo' else None)
                    result = bao.smooth_pk_interpolator()(spec['k']).T
                else:
                    result = _interp_loglog(spec['k'], k_grid / h, (pk * h**3).T).T
            elif kind == 'jaxcapse':
                emulator_params = jnp.array([get_param(name) for name in input_names])
                ellmax = spec['static']['ellmax']
                ells = jnp.arange(2, ellmax + 1)
                if method_key == 'harmonic.lens_potential_cl':
                    # Network outputs ell^2 (ell + 1)^2 / (2 pi) Cl^phiphi for ell = 2..ellmax;
                    # convert to raw Cl^phiphi (CosmoprimoCosmology convention).
                    cl_pp = emulator['PP'].get_Cl(emulator_params)[:ellmax - 1] * (2 * jnp.pi) / (ells * (ells + 1))**2
                    result = {'pp': jnp.concatenate([jnp.zeros(2), cl_pp]), 'tp': jnp.zeros(ellmax + 1), 'ep': jnp.zeros(ellmax + 1)}
                else:
                    # harmonic.lensed_cl.  Networks output Dl = ell (ell + 1) / (2 pi) Cl in muK^2
                    # for ell = 2..ellmax; convert to raw dimensionless Cl.  'bb' is emulated when
                    # the set provides a BB network, zeros otherwise.
                    try:
                        T_cmb = get_param('T_cmb')
                    except KeyError:
                        T_cmb = 2.7255
                    to_cl = (2 * jnp.pi) / (ells * (ells + 1)) / (T_cmb * 1e6)**2
                    cl = {name: jnp.concatenate([jnp.zeros(2), emulator[name.upper()].get_Cl(emulator_params)[:ellmax - 1] * to_cl])
                          for name in ['tt', 'ee', 'te'] + (['bb'] if 'BB' in emulator else [])}
                    result = {'tt': cl['tt'], 'ee': cl['ee'], 'bb': cl.get('bb', jnp.zeros(ellmax + 1)), 'te': cl['te']}
            elif method_key.startswith('fourier.pk'):
                emulator_params = jnp.array([get_param(name) for name in input_names if name != 'z'])
                z = spec['z']
                result = emulator.get_Pk(emulator_params, z, jaxace_cosmo.D_z(z))
                # result is (nz, nk_emulator); interpax interpolates along the leading axis,
                # so transpose to (nk_emulator, nz), interpolate onto spec['k'], then transpose
                # back to the (nz, nk) convention used elsewhere (e.g. CosmoprimoCosmology.__call__).
                result = _interp_loglog(spec['k'], emulator.k_grid, result.T).T
            else:
                # e.g. sigma8_z (scalar)
                shape = jnp.shape(spec['z'])
                emulator_params = jnp.stack([spec['z'] if name == 'z' else jnp.full(shape, get_param(name)) for name in input_names])
                result = emulator.run_emulator(emulator_params)
            self._results[spec_key] = result
        if params_in_range:
            # Out-of-range guard: every result was computed from clipped (finite) inputs;
            # mask them all to NaN when any parameter fell outside its training range.
            valid = jnp.all(jnp.array(list(params_in_range.values())))
            for spec_key in self._requirements:
                self._results[spec_key] = jax.tree.map(lambda arr: jnp.where(valid, arr, jnp.nan), self._results[spec_key])
        # Here set derived_params
        for param, getter in self._get_derived.items():
            self.derived_params[param].value = jnp.reshape(self.get(getter[0], **getter[1]), self.derived_params[param].shape)
