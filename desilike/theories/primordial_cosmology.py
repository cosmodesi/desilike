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

import numpy as np
import jax.numpy as jnp

from ..base import Calculator
from ..parameter import Parameter, VariableCollection



class PrimordialCosmology(Calculator):
    """Abstract base class for primordial cosmology calculators.

    Implements the **requirements API** shared by all cosmology providers:

    * Downstream calculators call :meth:`add_requirements` in their ``__init__``
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

    * Initialize ``self.params`` (a ``dict[str, Parameter]``), ``self._req_specs = {}``,
      and ``self._results = {}`` in their ``__init__``.
    * Set ``self._engine`` (a string identifier for the provider) in ``__post_init__``.
    * Override :meth:`propose_params` to return the provider's cosmological parameters.
    * Implement ``__call__`` to build the cosmology, loop over ``self._req_specs``, and
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

    # ── requirements API ──────────────────────────────────────────────────────

    def add_requirements(self, requirements):
        """Register quantities that a downstream calculator will need from this cosmology.

        Called in the downstream calculator's ``__init__``.  Multiple callers sharing the
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

        Examples
        --------
        >>> cosmo.add_requirements({
        ...     'fourier.pk': [
        ...         {'of': 'delta_cb', 'z': 1., 'k': k_array},
        ...         {'of': 'theta_cb', 'z': 1., 'k': k_array},
        ...     ],
        ...     'background.efunc': [{'z': 1.}],
        ... })
        """
        for method_key, kwargs_list in requirements.items():
            for kwargs in kwargs_list:
                z = float(kwargs['z'])
                k = kwargs.get('k')
                static = {key: val for key, val in kwargs.items() if key not in ('z', 'k')}
                spec_key = (method_key, tuple(sorted(static.items())))

                if spec_key not in self._req_specs:
                    self._req_specs[spec_key] = {
                        'static': static,
                        'z': np.array([z]),
                        'k': np.sort(np.asarray(k, dtype='f8')) if k is not None else None,
                    }
                else:
                    spec = self._req_specs[spec_key]
                    spec['z'] = np.unique(np.append(spec['z'], z))
                    if k is not None:
                        spec['k'] = np.unique(np.concatenate([spec['k'], np.asarray(k, dtype='f8')]))

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
        z = kwargs.get('z', None)
        k = kwargs.get('k', None)
        static = {key: val for key, val in kwargs.items() if key not in ('z', 'k')}
        spec_key = (method_key, tuple(sorted(static.items())))
        result = self._results[spec_key]
        spec   = self._req_specs[spec_key]
        if z is not None:
            iz = int(np.searchsorted(spec['z'], z))
            result = result[iz]       # (nk,) for pk, scalar for sigma8/efunc/DA
        if k is not None:
            ik = np.searchsorted(spec['k'], np.asarray(k, dtype='f8'))
            result = result[ik]       # subset, same ordering as k
        return result

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def tree_flatten(self):
        param_marker = jnp.concatenate([jnp.ravel(jnp.asarray(param.value)) for param in self.params.values()])
        ordered = list(self._req_specs.items())
        leaves = [param_marker]
        for spec_key, spec in ordered:
            if spec_key in self._results:
                leaves.append(self._results[spec_key])
            else:
                # Placeholder of correct shape for compile-time structure inference.
                nz = len(spec['z'])
                nk = len(spec['k']) if spec['k'] is not None else 0
                leaves.append(jnp.zeros((nz, nk) if nk else (nz,)))
        return leaves, {'engine': self._engine, 'ordered_specs': ordered}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj._engine = aux['engine']
        obj._param_vector = children[0]
        obj._req_specs = {sk: spec for sk, spec in aux['ordered_specs']}
        obj._results   = {sk: arr  for (sk, _), arr in zip(aux['ordered_specs'], children[1:])}
        return obj




# Engines that produce JAX-traceable outputs through cosmoprimo.Cosmology.clone.
_JAX_ENGINES = frozenset({'eisenstein_hu'})

# Parameter name conversion: desilike name → cosmoprimo clone kwarg.
_CONVERSIONS = {'logA': 'ln10^10A_s'}

# cosmoprimo pk_interpolator extrapolation kwargs shared by CosmoprimoCosmology and template.py.
_kw_pk = dict(extrap_kmin=1e-7, extrap_kmax=1e2)


def _get_fiducial(fiducial):
    """Return a cosmoprimo Cosmology from a name string, (name, kwargs) tuple, dict, or Cosmology."""
    import cosmoprimo
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


def _make_cosmo_parameters(fiducial=None):
    """Return the default parameter values dict from *fiducial* (or Planck-2018 priors)."""
    defaults = dict(h=0.6736, theta_MC_100=1.04092, omega_cdm=0.1200, omega_b=0.02237,
                    logA=3.044, n_s=0.9649, tau_reio=0.0544, m_ncdm=0.06, N_eff=3.046,
                    w0_fld=-1., wa_fld=0., Omega_k=0.)
    if fiducial is not None:
        fid = _get_fiducial(fiducial)
        defaults = {name: fid.get(_CONVERSIONS.get(name, name)) for name in defaults}
    return defaults


def _build_cosmo(fiducial, params):
    """Clone *fiducial* with the given *params* dict (desilike names → values).

    Values are passed as-is so JAX tracers are preserved for JAX-native engines;
    external engines (camb, class) always receive plain floats via _current_params().
    """
    kw = {_CONVERSIONS.get(name, name): value for name, value in params.items()}
    # ``h`` and ``theta_MC_100`` are mutually exclusive inputs to cosmoprimo; when both
    # are present ``h`` takes precedence (see primordial_cosmology.yaml).
    if 'h' in kw and 'theta_MC_100' in kw:
        kw.pop('theta_MC_100')
    return fiducial.clone(base='input', **kw)


class CosmoprimoCosmology(PrimordialCosmology):
    r"""
    :class:`PrimordialCosmology` backed by :mod:`cosmoprimo`.

    The ``_is_external`` flag is set per instance from *engine*: JAX-native engines
    (``'eisenstein_hu'``) run as pure JAX (``grad``/``jit``/``vmap``); external Boltzmann
    codes (``'camb'``, ``'class'``, …) run via ``pure_callback`` + finite-difference
    derivatives.  ``self.cosmo`` holds the current :class:`cosmoprimo.Cosmology` after
    each call; engine state is cached via ``Cosmology.clone(base='input', ...)``.

    Recognised method keys for :meth:`~PrimordialCosmology.add_requirements`:

    * ``'fourier.pk'``                              — kwargs: ``of``, ``z``, ``k``
    * ``'fourier.pk_now'``                          — kwargs: ``of``, ``engine``, ``z``, ``k``
    * ``'fourier.sigma8_z'``                        — kwargs: ``of``, ``z``
    * ``'background.efunc'``                        — kwargs: ``z``
    * ``'background.transverse_comoving_distance'`` — kwargs: ``z``
    * ``'thermodynamics.rs_drag'``                  — kwargs: ``z`` (dummy; result is z-independent)
    * ``'background.N_eff'``                        — kwargs: ``z`` (dummy; result is z-independent)

    Parameters
    ----------
    engine : str, default='camb'
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
    def propose_params(cls, *args, fiducial=None, **kwargs):
        r"""Return a proposed :class:`~desilike.parameter.VariableCollection` of cosmological Parameters.

        The default values are seeded from *fiducial* (or Planck-2018 priors when ``None``).
        The returned collection can be edited and passed back to :meth:`__init__` via ``params=...``.

        Parameters
        ----------
        engine : str, default='camb'
            Boltzmann engine (kept for signature symmetry with :meth:`__init__`; does
            not affect the proposed parameters).
        fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default=None
            Fiducial cosmology used to seed the default parameter values.

        Returns
        -------
        VariableCollection
        """
        defaults = _make_cosmo_parameters(fiducial)
        params = VariableCollection()
        # Planck2018 (TT,TE,EE+lowE+lensing) priors, mirroring the historical
        # primordial_cosmology.yaml.  Extra cosmological parameters (theta_MC_100, tau_reio,
        # N_eff, w0_fld, wa_fld, Omega_k) are fixed by default; free them as needed.
        params.set(Parameter('h', value=defaults['h'],
                             prior=dict(limits=[0.1, 10.]),
                             ref=dict(dist='norm', loc=0.6736, scale=0.005),
                             fd_eps=0.03, latex='h'))
        params.set(Parameter('omega_cdm', value=defaults['omega_cdm'],
                             prior=dict(limits=[0.01, 0.99]),
                             ref=dict(dist='norm', loc=0.12, scale=0.0012),
                             fd_eps=0.007, latex=r'\omega_\mathrm{cdm}'))
        params.set(Parameter('omega_b', value=defaults['omega_b'],
                             prior=dict(limits=[0.005, 0.1]),
                             ref=dict(dist='norm', loc=0.02237, scale=0.00015),
                             fd_eps=0.0015, latex=r'\omega_b'))
        params.set(Parameter('logA', value=defaults['logA'],
                             prior=dict(limits=[1.61, 3.91]),
                             ref=dict(dist='norm', loc=3.036394, scale=0.014),
                             fd_eps=0.05, latex=r'\ln(10^{10} A_s)'))
        params.set(Parameter('n_s', value=defaults['n_s'],
                             prior=dict(limits=[0.8, 1.2]),
                             ref=dict(dist='norm', loc=0.9649, scale=0.0042),
                             fd_eps=0.005, latex=r'n_s'))
        params.set(Parameter('tau_reio', value=defaults['tau_reio'], fixed=True,
                             prior=dict(limits=[0.01, 0.8]),
                             ref=dict(dist='norm', loc=0.0544, scale=0.01),
                             fd_eps=0.01, latex=r'\tau'))
        params.set(Parameter('m_ncdm', value=defaults['m_ncdm'], fixed=True,
                             prior=dict(limits=[0., 5.]),
                             ref=dict(dist='norm', loc=0.06, scale=0.12, limits=[0., 10.]),
                             fd_eps=(0.31, 0.15, 0.15), latex=r'm_\mathrm{ncdm}'))
        params.set(Parameter('N_eff', value=defaults['N_eff'], fixed=True,
                             prior=dict(limits=[0.01, 10.]),
                             ref=dict(dist='norm', loc=3.046, scale=0.16),
                             fd_eps=0.2, latex=r'N_\mathrm{eff}'))
        params.set(Parameter('w0_fld', value=defaults['w0_fld'], fixed=True,
                             prior=dict(limits=[-3., 1.]),
                             ref=dict(dist='norm', loc=-1., scale=0.08),
                             fd_eps=0.1, latex=r'w_0'))
        params.set(Parameter('wa_fld', value=defaults['wa_fld'], fixed=True,
                             prior=dict(limits=[-3., 2.]),
                             ref=dict(dist='norm', loc=0., scale=0.3),
                             fd_eps=0.3, latex=r'w_a'))
        params.set(Parameter('Omega_k', value=defaults['Omega_k'], fixed=True,
                             prior=dict(limits=[-0.3, 0.3]),
                             ref=dict(dist='norm', loc=0., scale=0.0065),
                             fd_eps=0.05, latex=r'\Omega_k'))
        return params

    def __init__(self, *args, engine='camb', params=None, fiducial=None, **kwargs):
        # Per-instance flag: JAX-traceable engines run as pure JAX, others as external.
        self._is_external = str(engine) not in _JAX_ENGINES
        if params is None:
            params = self.propose_params(*args, engine=engine, fiducial=fiducial, **kwargs)
        elif not isinstance(params, VariableCollection):
            params = VariableCollection(params)
        # Stored as a public attribute so build_graph discovers the Parameters as
        # dependencies (it descends into VariableCollection).
        self.params = {param.basename: param for param in params}
        # Requirement registry: filled by downstream calculators via add_requirements().
        # _req_specs: spec_key → {'static': dict, 'z': np.array, 'k': np.array|None}
        # _results:   spec_key → jnp.array   (populated in __call__)
        self._req_specs = {}
        self._results = {}

    def __post_init__(self, *args, engine='camb', params=None, fiducial=None, **kwargs):
        self._engine = str(engine)
        # Build (or resolve) the fiducial once, forcing ``engine`` so that subsequent
        # per-call ``.clone(base='input', ...)`` use the requested engine (not the
        # fiducial's default, e.g. CLASS for the named 'DESI'/'Planck2018' fiducials).
        import cosmoprimo
        if fiducial is None:
            self._fiducial = cosmoprimo.Cosmology(engine=self._engine)
        else:
            self._fiducial = _get_fiducial(fiducial).clone(engine=self._engine)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __call__(self):
        # JAX engines: keep tracers (clone is differentiable). External engines: plain floats.
        params = {basename: np.asarray(param.value).reshape(-1)[0].item() if self._is_external else param.value
                  for basename, param in self.params.items()}
        self.cosmo = _build_cosmo(self._fiducial, params)
        cosmo = self.cosmo
        for spec_key, spec in self._req_specs.items():
            method_key = spec_key[0]
            static = spec['static']
            z_grid = spec['z']
            k_grid = spec['k']
            if method_key == 'fourier.pk':
                fo = cosmo.get_fourier()
                result = fo.pk_interpolator(of=static['of'], **_kw_pk)(k_grid, z=z_grid).T
            elif method_key == 'fourier.pk_now':
                from cosmoprimo import PowerSpectrumBAOFilter
                fo = cosmo.get_fourier()
                pk_interp = fo.pk_interpolator(of=static['of'], **_kw_pk).to_1d(z=z_grid)
                bao = PowerSpectrumBAOFilter(pk_interp, engine=static['engine'],
                                             cosmo=cosmo, cosmo_fid=self._fiducial)
                result = bao.smooth_pk_interpolator()(k_grid).T
            elif method_key == 'fourier.sigma8_z':
                fo = cosmo.get_fourier()
                result = fo.sigma8_z(z_grid, of=static['of'])
            elif method_key == 'background.efunc':
                result = cosmo.get_background().efunc(z_grid)
            elif method_key == 'background.transverse_comoving_distance':
                result = cosmo.get_background().comoving_transverse_distance(z_grid)
            elif method_key == 'thermodynamics.rs_drag':
                result = jnp.asarray([cosmo.rs_drag])   # z_grid is a dummy; shape (1,)
            elif method_key == 'background.N_eff':
                result = jnp.asarray([cosmo.N_eff])     # z_grid is a dummy; shape (1,)
            else:
                raise ValueError(f'Unknown requirement method key: {method_key!r}')
            self._results[spec_key] = result
        # Return None: cosmo is a Python object exposed via self.cosmo, read directly by
        # downstream calculators; returning it would break the JAX pipeline output (and, for
        # the external path, call_kind='none' avoids the dtype-object crash).