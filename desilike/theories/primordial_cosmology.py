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

from pathlib import Path

import numpy as np
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
            params = self.propose_params(*args, fiducial=fiducial, **kwargs)
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


def _build_cosmoprimo(fiducial, params):
    """Clone *fiducial* with the given *params* dict (desilike names → values).

    Values are passed as-is so JAX tracers are preserved for JAX-native engines;
    external engines (camb, class) always receive plain floats.
    """
    kw = {_CONVERSIONS.get(name, name): value for name, value in params.items()}
    # ``h`` and ``theta_MC_100`` are mutually exclusive inputs to cosmoprimo; when both
    # are present ``h`` takes precedence (see primordial_cosmology.yaml).
    if 'h' in kw and 'theta_MC_100' in kw:
        kw.pop('theta_MC_100')
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
                             fd_eps=0.007, latex=r'\omega_\mathrm{cdm}'))
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
                             fd_eps=(0.31, 0.15, 0.15), latex=r'm_\mathrm{ncdm}'))
        params.set(Parameter('N_eff', value=fiducial['N_eff'], fixed=True,
                             prior=dict(limits=[0.01, 10.]),
                             ref=dict(dist='norm', loc=fiducial['N_eff'], scale=0.16),
                             fd_eps=0.2, latex=r'N_\mathrm{eff}'))
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
        if self._is_external:
            try:
                self._cosmo = _build_cosmoprimo(self._fiducial, params)
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
                self._cosmo = self._fiducial  # valid; used only for correctly-shaped placeholders
                self._run_requirements(params)
                for spec_key in self._requirements:
                    self._results[spec_key] = _nan_like(self._results[spec_key])
                for param in self.derived_params:
                    param.value = _nan_like(param.value)
        else:
            # JAX-native: tracers survive end-to-end (no pure_callback boundary), so
            # cosmoprimo's own exception_or_nan already raises in eager / NaNs under
            # jax.jit-grad-vmap tracing without any extra handling needed here.
            self._cosmo = _build_cosmoprimo(self._fiducial, params)
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


def _interp_loglog(k_query, k_knots, pk_knots):
    """Cubic spline interpolation in log10(k) space."""
    import interpax
    shape = jnp.shape(k_query)
    flat = jnp.ravel(k_query)
    result = interpax.interp1d(jnp.log10(flat), jnp.log10(k_knots), pk_knots, method='cubic', extrap=True)
    # Preserve pk_knots's trailing axes (e.g. the z dimension); only the k_query axis is reshaped.
    return jnp.reshape(result, shape + jnp.shape(pk_knots)[1:])


class ACECosmology(PrimordialCosmology):

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
        return CosmoprimoCosmology.propose_params(*args, fiducial=fiducial, **kwargs)

    def __post_init__(self, *args, engine='isitgr', base_dir=None, conversion='cosmoprimo', params=None, fiducial='DESI', **kwargs):
        self._engine = str(engine)
        if base_dir is not None:
            base_emulator_dir = Path(base_dir)
        else:
            base_emulator_dir = Path(Installer().install_dir) / 'ace-emulators'
        _SECTIONS = ['harmonic', 'fourier']
        if isinstance(engine, str):
            engine = {section: engine for section in _SECTIONS}

        def _find_inputs_outputs(emulator_dir):
            import json
            with open(emulator_dir / "nn_setup.json") as f:
                nn_dict = json.load(f)
            description = nn_dict.get('emulator_description', {})
            inputs = description.get('input')
            outputs = description.get('output')
            return list(inputs), list(outputs)

        self._input_outputs = {}
        seen_emulator_dirs = set()
        for section in _SECTIONS:
            emulator_dir = base_emulator_dir / engine[section]
            if not emulator_dir.is_dir() or emulator_dir in seen_emulator_dirs:
                continue
            seen_emulator_dirs.add(emulator_dir)
            # Iterate on all (leaf) emulators in emulator_dir
            for leaf_dir in sorted(path for path in emulator_dir.iterdir() if path.is_dir()):
                inputs, outputs = _find_inputs_outputs(leaf_dir)
                self._input_outputs[str(leaf_dir)] = (inputs, outputs)
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
        if hasattr(self, '_input_outputs'):
            self._load_emulators_for_new_requirements()

    def _load_emulator(self, emu_dir):
        inputs, outputs = self._input_outputs[emu_dir]
        if any(o.startswith('fourier.pk') for o in outputs):
            import jaxmapse
            emulator = jaxmapse.load_emulator(emu_dir)
        elif any(o.startswith('harmonic.') for o in outputs):
            import jaxcapse
            emulator = jaxcapse.load_emulator(emu_dir)
        else:
            import jaxace
            emulator = jaxace.load_trained_emulator(emu_dir)
        emulator.inputs = inputs
        return emulator

    def _load_emulators_for_new_requirements(self):
        for spec_key, spec in self._requirements.items():
            method_key = spec_key[0]
            if 'of' in spec['static']:
                method_key = f'{method_key}.' + '.'.join(spec['static']['of'])
            if method_key in self._method_emulator_matching:
                continue
            found = False
            for emu_dir, (inputs, outputs) in self._input_outputs.items():
                if method_key in outputs:
                    if emu_dir not in self._loaded_emulators:
                        self._loaded_emulators[emu_dir] = self._load_emulator(emu_dir)
                    self._method_emulator_matching[method_key] = self._loaded_emulators[emu_dir]
                    found = True
                    break
            if not found and method_key.split('.')[0] not in ('background', 'params', 'primordial'):
                raise NotImplementedError(f"could not find {method_key} in emulators' products")

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

            jaxace_cosmo = {}
            for jaxace_name, name in _CONVERSION_JAXACE.items():
                jaxace_cosmo[jaxace_name] = get_param(name)

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

            jaxace_cosmo = {}
            for jaxace_name, name in _CONVERSION_JAXACE.items():
                jaxace_cosmo[jaxace_name] = get_param(name)

        jaxace_cosmo = jaxace.w0waCDMCosmology(**jaxace_cosmo)
        for spec_key, spec in self._requirements.items():
            method_key = spec_key[0]
            if 'of' in spec['static']:
                method_key = f'{method_key}.' + '.'.join(spec['static']['of'])
            _kw_coords = {coord: spec[coord] for coord in _COORDS if coord in spec}
            emulator = self._method_emulator_matching.get(method_key, None)
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
            elif method_key.startswith('fourier.pk'):
                emulator_params = jnp.array([get_param(param) for param in emulator.inputs if param != 'z'])
                z = spec['z']
                result = emulator.get_Pk(emulator_params, z, jaxace_cosmo.D_z(z))
                # result is (nz, nk_emulator); interpax interpolates along the leading axis,
                # so transpose to (nk_emulator, nz), interpolate onto spec['k'], then transpose
                # back to the (nz, nk) convention used elsewhere (e.g. CosmoprimoCosmology.__call__).
                result = _interp_loglog(spec['k'], emulator.k_grid, result.T).T
            else:
                # e.g. sigma8_z (scalar)
                shape = jnp.shape(spec['z'])
                emulator_params = jnp.stack([spec['z'] if param == 'z' else jnp.full(shape, get_param(param)) for param in emulator.inputs])
                result = emulator.run_emulator(emulator_params)
            self._results[spec_key] = result
        # Here set derived_params
        for param, getter in self._get_derived.items():
            self.derived_params[param].value = jnp.reshape(self.get(getter[0], **getter[1]), self.derived_params[param].shape)
