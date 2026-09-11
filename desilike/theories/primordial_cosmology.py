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
from cosmoprimo.cosmology import Cosmology

from ..base import Calculator
from ..emulators.api import CalculatorEmulator, DERIVED
from cosmoprimo.emulators.analytic import (AMPLITUDES, amplitude, harmonic_scaling, fourier_analytic_scales,
                                           theta_analytic, solve_theta_analytic,
                                           eisenstein_hu_scales, nonzero,
                                           resample_dilated as _resample_dilated, dilate as _dilate)
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

    def __init__(self, *args, params=None, fiducial=None, requirements=None, **kwargs):
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
        # Out-of-range guard state, here rather than in `__post_init__` so that it exists on
        # every instance -- a caller may ask an uncompiled cosmology what it enforces -- and so
        # that the warn-once latch is per object: `__post_init__` runs at every build, and
        # resetting it there would re-emit the same warning on each one.
        self._param_clip_ranges = {}
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
        # Requirements the caller declares are an input, like the parameters. A build re-runs
        # `__init__` -- that is what clears the ones downstream calculators register at their
        # `__post_init__` -- so a requirement registered by hand on a bare cosmology would not
        # survive it; one passed here does.
        if requirements is not None:
            self.add_requirements(requirements)

    # ── requirements API ──────────────────────────────────────────────────────

    def add_requirements(self, requirements):
        """Register quantities that a downstream calculator will need from this cosmology.

        Called in the downstream calculator's ``__post_init__``.  Multiple callers sharing the
        same cosmology instance are supported: z and k grids are union-merged so only one
        combined evaluation is needed at runtime.  A build re-runs ``__init__`` and so starts
        from an empty registry, which is what keeps repeated builds from accumulating; a
        requirement declared by hand therefore goes to the constructor (``requirements=``).

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

    def get_requirements(self):
        """The registry in the form :meth:`add_requirements` and ``requirements=`` take.

        The inverse of :meth:`add_requirements`: static kwargs and the merged coordinate grids,
        one entry per spec. For a caller that has to carry the registry somewhere a rebuild
        cannot follow -- ``cosmo.update(requirements=cosmo.get_requirements())`` pins what the
        consumers registered into ``_init``, so a build of this cosmology on its own replays it
        instead of starting empty. Training an emulator is such a build: the consumers are not
        part of it, so nothing re-registers, and the aux the deployed object is rebuilt from
        would otherwise carry no specs at all.
        """
        requirements = {}
        for (method_key, _), spec in self._requirements.items():
            kwargs = dict(spec['static'])
            for coord in _COORDS:
                if coord in spec:
                    kwargs[coord] = spec[coord]
            requirements.setdefault(method_key, []).append(kwargs)
        return requirements

    def get_emulator_cls(self):
        """The emulator this cosmology's requirements call for.

Answered for every cosmology that flattens to its
        registered requirements, which is what the routing reads -- an :class:`ACECosmology`
        included, and there the point is not the Boltzmann call (ACE is already fast) but the
        routing and the leaf naming, plus training nodes cheap enough to check a box in seconds.

        One sector's emulator when they all belong to one sector (see :func:`_sector`):
        :class:`HarmonicEmulator` for the Cl (their amplitude and optical depth are analytic,
        so they are divided out rather than expanded, and the expansion runs in the theta
        basis), :class:`FourierEmulator`, :class:`BackgroundEmulator` or
        :class:`ThermodynamicsEmulator` otherwise (the amplitude, tilt, dilation and analytic
        background are divided out, and what the leaves allow leaves the grid); a
        :class:`CosmologyEmulator`, one emulator per sector, when they span several. A plain
        method, so ``Emulator(cosmo, space)`` asks the instance after its consumers have
        registered; pass ``cls=CalculatorEmulator`` to force the generic expansion.
        """
        if getattr(self, '_conversion', 'cosmoprimo') != 'cosmoprimo':
            # The routed sections anchor their analytic cores on a cosmoprimo fiducial (see
            # `_RoutedSectionEmulator.set_ref_fiducial`), which an ACE cosmology only carries
            # when it converts through one. Anything else gets the generic expansion.
            # Not a test on `_fiducial` itself: that is set at build, and this is asked of
            # cosmologies that have only been constructed.
            return None
        sectors = {_sector(spec_key[0]) for spec_key in self._requirements}
        if len(sectors) > 1:
            return CosmologyEmulator
        return CosmologyEmulator.sectors[sectors.pop()] if sectors else FourierEmulator

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
        When used as a pre-loaded proxy (results injected externally via build's ``input``
        callable before the graph runs), this is a no-op because _results is already populated.
        """
        params = {param.basename: param.value for param in self.params}
        self._param_values = params
        for spec_key, spec in self._requirements.items():
            if spec_key not in self._results:
                shape = tuple(spec[coord].size for coord in _COORDS if coord in spec)  # only the coords this spec was registered with
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
                # Placeholder of correct shape for build-time structure inference.
                shape = tuple(spec[coord].size for coord in _COORDS if coord in spec)  # only the coords this spec was registered with
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


def find_conflicts(name, names):
    """Those of *names* that conflict with *name* -- i.e. set the same physical quantity.

    cosmoprimo's own word: ``Cosmology._conflict_parameters`` groups ``('h', 'H0')``,
    ``('A_s', 'logA', 'sigma8')``, ``('Omega_cdm', 'omega_cdm', 'Omega_m', ...)`` precisely
    because setting two of a group is contradictory, and it raises when you do.  So a
    well-formed pipeline gives at most one and the caller can unpack it; more than one is a
    pipeline worth complaining about rather than choosing between.

    This answers WHICH NAME, and nothing else.  Conflicting parameters do not share a VALUE --
    H0 is 100 h, and Omega_m differs from omega_cdm by h^2 and by the neutrino and baryon
    content -- so it must never be used to read one under another's name.  Values come from the
    cosmology (``CosmoprimoCosmology.__getitem__`` converts) or from the scalar provider.

        find_conflicts('h', ['H0', 'A_s'])       # ['H0']
        find_conflicts('w0_fld', ['H0', 'A_s'])  # []
    """
    from cosmoprimo import Cosmology

    group = next((conflicts for conflicts in Cosmology._conflict_parameters
                  if name in conflicts), (name,))
    return [other for other in names if other in group]


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
        from desilike.base import build
        params = {}
        for param in calculator.params:
            try:
                params[param.name] = fiducial[param.basename]
            except cosmoprimo.CosmologyError:
                params[param.name] = param.value
        pipe = build(calculator, output=lambda: calculator)
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

    # theta_MC_100 is not an input cosmoprimo's `clone` accepts -- it is a DERIVED quantity
    # (rs * h / D_M(z*), see Cosmology._get_params). Sampling it means solving for the h that
    # reproduces it, which `Cosmology.solve` does, with a shortcut for this very target.
    #
    # Worth the root-find whenever the CMB sets the absolute scale: h, w0 and wa act on the Cl
    # almost entirely through theta, so a box in h contains cosmologies whose spectra are
    # translated along ell by tens of multipoles, and no low-order interpolant survives that
    # (measured: a budget-3 Chebyshev over a 3.5-sigma (h, w0, wa) box gave
    # sigma(logP_emulated - logP_exact) = 2e16, wrong at every draw). In the theta basis the
    # peaks move by less than one multipole across the same box.
    theta_MC_100 = kw.pop('theta_MC_100', None)
    if theta_MC_100 is not None:
        # Solve on a LIGHT clone and hand the resulting h to the real build below, rather than
        # solving on the finished cosmology: `solve` re-clones with base='input', which drops the
        # lensing / precision settings attached here, so a cosmology solved that way then raises
        # on `lensed_cl` -- and inside pure_callback that raise becomes a silent all-NaN node
        # (measured, at the very centre of the box, every output including H0 non-finite).
        # It is also much cheaper: theta_cosmomc needs only the background, not lensed spectra.
        # limits: not the solver's default (0.6, 0.8) -- a 3.5-sigma DESI w0waCDM box reaches
        # h ~ 0.573 -- but not the (0.1, 2.) of cosmoprimo's test either, whose lower end is an
        # unphysical cosmology at any realistic omega_cdm.
        kw['h'] = float(fiducial.clone(base='input', **kw).solve(
            'h', 'theta_MC_100', target=float(theta_MC_100),
            limits=[0.4, 1.0], xtol=1e-6, maxiter=60)['h'])
    if lensing:
        kw['lensing'] = True
    if calc_params:
        kw.update(calc_params)
    if extra_params:
        merged_extra_params = dict(getattr(getattr(fiducial, 'engine', None), '_extra_params', None) or {})
        merged_extra_params.update(extra_params)
        cosmo = fiducial.clone(base='input', extra_params=merged_extra_params, **kw)
    else:
        cosmo = fiducial.clone(base='input', **kw)
    return cosmo




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
    * ``'background.h1'``, ``'background.h3'``, ``'background.h5'`` — kwargs: ``z``; the
      EFT-of-dark-energy functions h_i(eta = -ln(1 + z)) of an EFT-of-DE engine's Background.
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
                             fd=dict(eps=0.03), latex='h'))
        params.set(Parameter('omega_cdm', value=fiducial['omega_cdm'],
                             prior=dict(limits=[0.01, 0.99]),
                             ref=dict(dist='norm', loc=fiducial['omega_cdm'], scale=0.0012),
                             fd=dict(eps=0.007), latex=r'\omega_{\mathrm{cdm}}'))
        params.set(Parameter('omega_b', value=fiducial['omega_b'],
                             prior=dict(limits=[0.005, 0.1]),
                             ref=dict(dist='norm', loc=fiducial['omega_b'], scale=0.00015),
                             fd=dict(eps=0.0015), latex=r'\omega_b'))
        params.set(Parameter('logA', value=fiducial['logA'],
                             prior=dict(limits=[1.61, 3.91]),
                             ref=dict(dist='norm', loc=fiducial['logA'], scale=0.014),
                             fd=dict(eps=0.05), latex=r'\ln(10^{10} A_s)'))
        params.set(Parameter('n_s', value=fiducial['n_s'],
                             prior=dict(limits=[0.8, 1.2]),
                             ref=dict(dist='norm', loc=fiducial['n_s'], scale=0.0042),
                             fd=dict(eps=0.005), latex=r'n_s'))
        params.set(Parameter('tau_reio', value=fiducial['tau_reio'], fixed=True,
                             prior=dict(limits=[0.01, 0.8]),
                             ref=dict(dist='norm', loc=fiducial['tau_reio'], scale=0.01),
                             fd=dict(eps=0.01), latex=r'\tau'))
        params.set(Parameter('m_ncdm', value=fiducial['m_ncdm_tot'], fixed=True,
                             prior=dict(limits=[0., 5.]),
                             ref=dict(dist='norm', loc=fiducial['m_ncdm_tot'], scale=0.12, limits=[0., 10.]),
                             # sqrt(m) is the natural expansion variable (free-streaming
                             # scale ~ sqrt(m)); Chebyshev collocation over the DESI prior
                             # range: the order-n fit is the degree-n interpolant in sqrt(m)
                             # (raw dchi2 <= 4e-4 over m in [0, 0.4] at order 4).
                             fd=dict(limits=(0., 0.45), transform='sqrt'),
                             latex=r'm_{\mathrm{ncdm}}'))
        params.set(Parameter('N_eff', value=fiducial['N_eff'], fixed=True,
                             prior=dict(limits=[0.01, 10.]),
                             ref=dict(dist='norm', loc=fiducial['N_eff'], scale=0.16),
                             fd=dict(eps=0.2), latex=r'N_{\mathrm{eff}}'))
        params.set(Parameter('w0_fld', value=fiducial['w0_fld'], fixed=True,
                             prior=dict(limits=[-3., 1.]),
                             ref=dict(dist='norm', loc=fiducial['w0_fld'], scale=0.08),
                             fd=dict(eps=0.1), latex=r'w_0'))
        params.set(Parameter('wa_fld', value=fiducial['wa_fld'], fixed=True,
                             prior=dict(limits=[-3., 2.]),
                             ref=dict(dist='norm', loc=fiducial['wa_fld'], scale=0.3),
                             fd=dict(eps=0.3), latex=r'w_a'))
        params.set(Parameter('Omega_k', value=fiducial['Omega_k'], fixed=True,
                             prior=dict(limits=[-0.3, 0.3]),
                             ref=dict(dist='norm', loc=fiducial['Omega_k'], scale=0.0065),
                             fd=dict(eps=0.05), latex=r'\Omega_k'))
        return params

    def __post_init__(self, *args, engine='class', params=None, fiducial='DESI', precision=None, **kwargs):
        # Accuracy overrides on top of the settings derived from the harmonic requirements
        # (_LENSING_CALC_PARAMS / _LENS_POTENTIAL_CL_EXTRA_PARAMS): a dict with optional
        # 'calc_params' (cosmoprimo calculation parameters, e.g. non_linear, ellmax_cl) and
        # 'extra_params' (raw engine precision knobs) keys.  The defaults are tuned for
        # production sampling; emulator TRAINING wants stricter ones, since the node values
        # are the truth the fit is only as good as.
        self._precision = {key: dict((precision or {}).get(key, {}) or {})
                           for key in ('calc_params', 'extra_params')}
        # ``engine`` may be a cosmoprimo engine CLASS as well as a name -- notably
        # ``EmulatedEngine.read(fn)``, the documented way to use a trained cosmoprimo
        # emulator (e.g. an emulated harmonic section shared by CMB / FS / SN likelihoods).
        # str() would turn such a class into "<class '...'>" and cosmoprimo would then fail
        # with 'Unknown engine'.
        self._engine = str(engine) if isinstance(engine, str) else engine
        # A non-string engine is a JAX-traceable emulator unless it says otherwise; named
        # engines are looked up in the JAX list as before.
        self._is_external = isinstance(self._engine, str) and self._engine not in _JAX_ENGINES
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
        # Caller-supplied accuracy overrides win over the derived defaults, except that
        # 'ellmax_cl' still only ever goes up (same rule as just above).
        if self._precision['calc_params']:
            override_ellmax = self._precision['calc_params'].get('ellmax_cl', None)
            calc_params.update(self._precision['calc_params'])
            if override_ellmax is not None and 'ellmax_cl' in calc_params:
                calc_params['ellmax_cl'] = max(override_ellmax, requested_ellmax,
                                               self._fiducial['ellmax_cl'])
        if self._precision['extra_params']:
            extra_params = dict(extra_params or {})
            extra_params.update(self._precision['extra_params'])
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
                    self._results[spec_key] = jax.tree_util.tree_map(
                        lambda leaf: jnp.full(jnp.shape(leaf), jnp.nan),
                        self._results[spec_key])
                for param in self.derived_params:
                    param.value = jax.tree_util.tree_map(
                        lambda leaf: jnp.full(jnp.shape(leaf), jnp.nan), param.value)
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
            elif method_key in ('background.h1', 'background.h3', 'background.h5'):
                # The EFT-of-dark-energy functions of arXiv:1902.06978 (eqs. 64, 66, 68), which
                # cosmoprimo's 'mochiclass' / 'heftcamb' engines and the propto_omega emulated
                # engine expose on their Background as h1(eta), h3(eta), h5(eta), eta = ln a.
                # Registered on the z grid (eta = -ln(1 + z)) so that they travel as leaves:
                # the FKPT EFT_DE node is an external (numpy) calculator, and inside its
                # callback a JAX-native cosmology's `_cosmo` holds tracers -- reading the
                # functions off it there is what broke a jitted chain on an emulated engine.
                name = method_key.split('.')[1]
                background = cosmo.get_background()
                if not hasattr(background, name):
                    raise ValueError(f"the engine's Background has no {name}(eta); {method_key} needs an EFT-of-DE engine "
                                     "('mochiclass', 'heftcamb', or the propto_omega emulator)")
                eta = -jnp.log1p(jnp.asarray(spec['z']))
                result = jnp.asarray(getattr(background, name)(eta))
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
    # Its sigma8_z is TOTAL-MATTER, so it is declared only for of='delta_m'.  delta_cb and
    # theta_cb are not listed: they fall through to the linear-pk emulator, which integrates
    # its own P_cb in a top-hat at R = 8 (see the sigma8_z branch in __call__).  Serving
    # sigma8_M as sigma8_cb was a 0.45% bias at the DESI fiducial with m_ncdm = 0.06 eV, and
    # it did NOT cancel downstream: DirectSpectrum2Template takes sigma8_fid from a
    # cosmoprimo fiducial whatever the engine, so the whole bias landed in the physical-basis
    # rescaling A = sigma8 / sigma8_fid.  Measured on LRG3 P+B, that alone accounted for
    # sigma = 2.71 of the ACE-vs-CLASS log-posterior tilt, against 1.55 for the total.
    'ACE_mnuw0wacdm_ln10As_basis': dict(
        kind='jaxace',
        inputs=['z', 'logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'm_ncdm', 'w0_fld', 'wa_fld'],
        outputs=['fourier.sigma8_z.delta_m.delta_m',
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
        # Verified against the artifact's own `inminmax.npy` (identical for Pk_lin_mm and
        # Pk_lin_cb, 7 rows in network-input order); the values below round each edge OUTWARD
        # by at most 8e-5 relative:
        #   z          [1.0000e-05,  4.99999    ]   <- not a `ranges` entry: the z grid is the
        #                                             emulator's own axis, not a parameter
        #   H0         [50.000080  , 89.99992   ]   -> h [0.5000008, 0.8999992]
        #   omega_b    [ 0.02000001,  0.02499999]
        #   omega_cdm  [ 0.08000020,  0.1799998 ]
        #   m_ncdm     [ 1.0000e-06,  0.499997  ]
        #   w0_fld     [-2.999993  ,  0.499993  ]
        #   wa_fld     [-2.999970  ,  1.999990  ]
        # logA and n_s carry NO range here at all -- `preprocessing.py` in the artifact is
        # `drop_primordial_parameters`, which slices them off before the network sees anything.
        # So the n_s / logA edges of any box built from ACECosmology.training_ranges('ace') come
        # entirely from 'camb_lcdm' below, an emulator a full-shape fit never loads -- which is
        # what `section=` is for: name the sections a run consumes and those edges do not apply.
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


def _find_capse_ellmax(spectrum_dir, nout):
    """Maximum multipole a Capse-style network predicts, from its l.npy training grid.

    jaxcapse interpolates the network's *nout* outputs onto the dense integer grid spanned by
    l.npy, so ellmax is that grid's upper end -- NOT nout + 1.  The two coincide only when the
    network was trained on a dense grid (the 'capse_mnuw0wacdm_250001' set, 2998 outputs for
    ell = 2..2999); a set trained on Chebyshev-Lobatto nodes (256 nodes over ell = 2..9500)
    would otherwise be reported as ellmax = 257, silently reading Lobatto node values as if
    they were the first 256 multipoles.  Legacy artifacts stored the full CAMB 0..N grid in
    l.npy while the network covers ell = 2..nout + 1; that case is resolved as in
    jaxcapse.load_emulator.  Without l.npy (dense grid assumed) we fall back to nout + 1."""
    path = spectrum_dir / 'l.npy'
    if not path.is_file():
        return nout + 1
    ell = np.load(path)
    if len(ell) != nout:
        if len(ell) >= nout + 2 and ell[0] == 0 and np.all(np.diff(ell) == 1):
            ell = ell[2:nout + 2]
        else:
            raise ValueError(f'Capse-style emulator {spectrum_dir}: multipole grid length ({len(ell)}) '
                             f'does not match the network output length ({nout})')
    return int(round(float(np.max(ell))))


def _intersect_ranges(ranges_per_emulator):
    """Intersect ``{name: (low, high)}`` mappings: the tightest range each name is valid over.

    One implementation, because there are two consumers that must not drift apart --
    :meth:`ACECosmology.training_ranges` over the declared engine spec, and
    :meth:`ACECosmology._load_emulators_for_new_requirements` over the emulators actually loaded.
    """
    intersected = {}
    for ranges in ranges_per_emulator:
        for name, (low, high) in ranges.items():
            previous_low, previous_high = intersected.get(name, (-np.inf, np.inf))
            intersected[name] = (max(low, previous_low), min(high, previous_high))
    return intersected


def _warn_priors_beyond_ranges(params, ranges):
    """Warn for each varied parameter whose prior reaches outside the emulators' training ranges.

    Outside them every emulated result is NaN-masked, which a posterior turns into ``-inf``: the
    prior is effectively truncated whether or not anybody said so.  :meth:`ACECosmology.truncate_priors`
    makes it explicit; this is the warning for those who have not called it.

    No de-duplication state is kept: the message names the parameter, so it differs per parameter,
    and :mod:`warnings` already suppresses repeats of an identical message from one location.
    """
    for param in params:
        name = param.basename
        if param.fixed or name not in ranges:
            continue
        limits = param.prior.limits
        low, high = ranges[name]
        if limits[0] < low or limits[1] > high:
            warnings.warn(f'parameter {name!r} prior range {tuple(limits)} extends beyond the packaged emulator '
                          f'training range ({low}, {high}): samples outside yield NaN (effective prior truncation)')


def _add_h_from_H0(ranges, replace):
    """Report an ``H0`` range as ``h`` as well, scaled by 1/100.

    ``replace`` says whether ``H0`` itself survives, and the two consumers genuinely differ:
    prior truncation works in desilike names, so ``H0`` is popped; the run-time guard clips by
    whatever name it is asked for -- the networks' native ``H0`` and desilike's ``h`` both reach
    it -- so there it must keep both.
    """
    if 'H0' not in ranges:
        return ranges
    low, high = ranges.pop('H0') if replace else ranges['H0']
    previous_low, previous_high = ranges.get('h', (-np.inf, np.inf))
    ranges['h'] = (max(low / 100., previous_low), min(high / 100., previous_high))
    return ranges


def _find_capse_metadata(emulator_dir):
    """Introspect a Capse-style Cl emulator directory: per-spectrum network subdirectories
    ('TT', 'TE', 'EE', and optionally 'BB', 'PP'), each holding nn_setup.json / weights.npy /
    inminmax.npy / outminmax.npy / l.npy / postprocessing.py, as produced by the
    CosmologicalEmulators training pipeline (e.g. the local 'capse_mnuw0wacdm_250001' and
    'capse_mnuw0wacdm_20k_hybrid_ee_20260831' sets).  Network inputs are
    read from an explicit desilike-style 'input' list in nn_setup.json's emulator_description when
    present, else parsed from its free-text 'parameters' description via _CONVERSION_CAPSE;
    training ranges come from inminmax.npy and the maximum multipole from l.npy (see
    :func:`_find_capse_ellmax`; outputs cover ell = 2..ellmax).  Cl conventions are assumed
    identical to the packaged 'camb_lcdm' set: Dl in muK^2 (TT/TE/EE), ell^2 (ell+1)^2 / (2 pi)
    Cl^phiphi (PP).  A non-box training domain -- e.g. the 'w0 + wa < -0.5' cut of the
    mnuw0wacdm sets, declared as emulator_description['constraint'] -- is NOT enforced by the
    out-of-range guard, which only clips per-parameter ranges, so it is warned about here."""
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
    nout = np.load(emulator_dir / spectra[0] / 'outminmax.npy').shape[0]
    ellmax = _find_capse_ellmax(emulator_dir / spectra[0], nout)
    constraint = description.get('constraint', None)
    if constraint:
        warnings.warn(f'Capse-style emulator {emulator_dir} declares the training-domain constraint {constraint!r}, '
                      "which the out-of-range guard does not enforce (it only clips per-parameter ranges): "
                      'parameters satisfying every range but violating it are extrapolations, not NaN-masked')
    outputs = ['harmonic.lensed_cl'] + (['harmonic.lens_potential_cl'] if 'PP' in spectra else [])
    return dict(kind='jaxcapse', inputs=list(inputs), outputs=outputs, ranges=ranges,
                ellmax=int(ellmax), spectra=spectra)


def _ace_background(method_key, z, backend='cosmoprimo', cosmoprimo_cosmo=None, jaxace_cosmo=None):
    """Background quantities for the ACE engine, from cosmoprimo or from jaxace.

    ACE emulates the transfer functions; the background is analytic either way, so which library
    integrates it is a free choice.  ``backend='cosmoprimo'`` (default) takes E(z), the distances
    and the growth from :class:`cosmoprimo.cosmology.DefaultBackground`; ``'jaxace'`` keeps
    jaxace's own.

    The default is cosmoprimo because jaxace's growth is a trap under ``jax.vmap``: it integrates
    with ``diffrax.Tsit5`` under an adaptive ``PIDController`` (rtol 1e-6, atol 1e-8,
    ``max_steps=10000``), and a batched adaptive ``while_loop`` runs until the last lane
    converges -- so one draw whose w0waCDM background is unphysical, routine when a sampler
    proposes from the whole prior box, charges every lane the full 10000 steps.  Measured on the
    w0waCDM CMB-SPA + DESI DR2 BAO posterior, that made the ``sigma8_cb`` derived parameter
    53.67 ms/point against 0.025 ms/point for the likelihood it decorates (2234x), and it shows
    up as a step in batch size -- 70 ms at 1, 29.6 s at 8, 29.5 s at 512 -- rather than a slope.
    cosmoprimo uses a fixed 200-step RK4 whose cost cannot depend on the data.

    The growth factor is served in cosmoprimo's normalisation, D(0) = 1, whichever backend
    produces it, so that the two agree with each other and with
    :class:`CosmoprimoCosmology`.  Left alone they would not: both are "D ~ a in matter
    domination", but cosmoprimo integrates from ``eta = -6`` (z ~ 402) with D = a there, where
    radiation is still ~12% of matter, while jaxace's growth ODE carries no radiation term and
    normalises to D -> a asymptotically -- a constant 3.3% apart at every z (D(0) = 0.745 against
    0.771 at the DESI fiducial).  Once each is divided by its own D(0) they agree to 1e-5 below
    z = 2.  This is not the growth that scales the emulated pk: that one is an amplitude, needs
    jaxace's early-time normalisation, and comes from the ACE network in
    :meth:`ACECosmology.__call__`.

    Conventions, which differ between the two and are silent if got wrong:

    - growth is taken from ``DefaultBackground`` explicitly rather than through the engine
      attribute, as :func:`~desilike.theories.galaxy_clustering.template.fourier_analytic_scales`
      does, because an engine may override it with a fitting formula;
    - ``mass='cb'``, because jaxace's growth ODE sources on Omega_cb and that is what the
      linear-pk emulator's ``D`` argument means; ``mass='m'`` would put massive neutrinos in the
      source;
    - the growth rate is a log-derivative, so no normalisation enters it at all and the two
      agree to 1e-5 below z = 2;
    - jaxace returns distances in Mpc (hence the explicit ``* h`` below), cosmoprimo in Mpc/h.
    """
    if backend == 'jaxace':
        if method_key == 'background.efunc':
            return jaxace_cosmo.E_z(z)
        if method_key == 'background.comoving_transverse_distance':
            return jaxace_cosmo.dM_z(z) * jaxace_cosmo.h
        if method_key == 'background.luminosity_distance':
            return jaxace_cosmo.dL_z(z) * jaxace_cosmo.h
        if method_key == 'background.growth_factor':
            # normalised to D(0) = 1, cosmoprimo's convention; jaxace's own D is the early-time
            # one, 3.3% away.  z = 0 rides along in the same solve.
            growth = jaxace_cosmo.D_z(jnp.append(jnp.atleast_1d(jnp.asarray(z)), 0.))
            return jnp.reshape(growth[:-1], jnp.shape(z)) / growth[-1]
        return jaxace_cosmo.f_z(z)
    from cosmoprimo.cosmology import DefaultBackground
    ba = cosmoprimo_cosmo.get_background()
    if method_key == 'background.efunc':
        return ba.efunc(z)
    if method_key == 'background.comoving_transverse_distance':
        return ba.comoving_transverse_distance(z)
    if method_key == 'background.luminosity_distance':
        return ba.luminosity_distance(z)
    if method_key == 'background.growth_factor':
        # no znorm: cosmoprimo's default divides by D(0), which is the convention served here
        return DefaultBackground.growth_factor(ba, z, mass='cb')
    # growth_rate reads the interpolant growth_factor builds, and reaches for it when the cache
    # is cold -- through the engine's growth_factor, which takes no `mass` and raises.  Prime it
    # here rather than rely on a growth_factor requirement having been served first.
    DefaultBackground.growth_factor(ba, 0., mass='cb')
    return DefaultBackground.growth_rate(ba, z, mass='cb')


def _interp_loglog(k_query, k_knots, pk_knots):
    """Cubic spline interpolation in log10(k) space."""
    import interpax
    shape = jnp.shape(k_query)
    flat = jnp.ravel(k_query)
    result = interpax.interp1d(jnp.log10(flat), jnp.log10(k_knots), pk_knots, method='cubic', extrap=True)
    # Preserve pk_knots's trailing axes (e.g. the z dimension); only the k_query axis is reshaped.
    return jnp.reshape(result, shape + jnp.shape(pk_knots)[1:])


def _sigma_tophat(k, pk, radius):
    r"""Top-hat :math:`\sigma_R` of a linear power spectrum, trapezoid in :math:`\ln k`.

    .. math:: \sigma_R^2 = \int \mathrm{d}\ln k \; \frac{k^3 P(k)}{2 \pi^2} W(kR)^2, \quad
              W(x) = \frac{3 (\sin x - x \cos x)}{x^3}

    The rule wants a log-spaced *k*, which every grid this is called on is.  Units cancel as
    long as they agree: *k* in h/Mpc with *radius* in Mpc/h, or both in Mpc.

    Parameters
    ----------
    k : array
        Wavenumbers, shape ``(nk,)``.
    pk : array
        Power spectrum on *k*, shape ``(..., nk)`` -- the leading axes ride through untouched,
        which is how a ``(nz, nk)`` input gives one sigma per redshift.
    radius : float, array
        Smoothing radius, scalar or shape ``(nr,)``.

    Returns
    -------
    sigma : array
        Shape ``(..., nr)``, or ``(...)`` for a scalar *radius* -- so a scalar radius against a
        1D *pk* gives a scalar, and ``sigma8`` needs no squeeze at the call site.
    """
    scalar = jnp.ndim(radius) == 0
    x = k * jnp.atleast_1d(radius)[:, None]                                 # (nr, nk)
    window = 3. * (jnp.sin(x) - x * jnp.cos(x)) / x**3
    integrand = k**3 * pk[..., None, :] * window**2 / (2. * jnp.pi**2)      # (..., nr, nk)
    sigma = jnp.sqrt(jnp.trapezoid(integrand, x=jnp.log(k), axis=-1))       # (..., nr)
    return sigma[..., 0] if scalar else sigma


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

    background_engine : {'cosmoprimo', 'jaxace'}, default='cosmoprimo'
        Which library provides the background sector -- ``background.efunc``, the distances,
        ``background.growth_factor`` / ``growth_rate``.  ACE emulates the transfer functions, so
        the background is analytic either way and the choice is free; the growth factor is
        served in cosmoprimo's normalisation, D(0) = 1, whichever backend produces it.  It does
        not reach the growth that scales the emulated linear pk behind ``sigma8_z`` of
        ``delta_cb`` / ``theta_cb``: that one is an amplitude rather than a ratio, so it comes
        from the packaged ACE network (jaxace's ODE when there is none), in the early-time
        normalisation those networks were trained with.  See :func:`_ace_background`.

        The default is not jaxace because its growth solver is a trap under ``jax.vmap``:
        ``diffrax.Tsit5`` under an adaptive ``PIDController`` (``max_steps=10000``), and a batched
        adaptive ``while_loop`` runs until the last lane converges, so one draw whose w0waCDM
        background is unphysical charges every lane the full 10000 steps.  Measured on the
        w0waCDM CMB-SPA + DESI DR2 BAO posterior that made the derived block 53.67 ms/point
        against 0.025 ms/point for the likelihood it decorates, and it appears as a step in batch
        size (70 ms at 1, 29.6 s at 8, 29.5 s at 512) rather than a slope.  cosmoprimo integrates
        the same equation with a fixed 200-step RK4 whose cost cannot depend on the data.
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
    def training_ranges(cls, engine='ace', base_dir=None, basis='cosmo', section=None):
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
        section : str or list, default=None
            Which sections to intersect over: ``'harmonic'``, ``'fourier'``, ``'background'``,
            or a list of them.  ``None`` takes all of them, which is what the out-of-range guard
            enforces on a cosmology built with every section.

            Ask for a subset when only part of the cosmology is consumed, because the sections
            do not agree and the intersection is the tightest of them.  A full-shape fit reads
            the fourier and background sections and never evaluates the harmonic one, yet with
            ``engine='ace'`` it inherits ``camb_lcdm``'s logA (2.5, 3.5) and n_s (0.88, 1.05)
            against the (2.0, 3.7) and (0.8, 1.1) the other two allow -- measured on LRG3
            w0waCDM, that clipped the posterior at 2.1 to 2.3 sigma on the low side of both.

        Returns
        -------
        dict
            ``{parameter name: (low, high)}``, intersected across the selected emulators.
        """
        if basis not in ('cosmo', 'emulator'):
            raise ValueError(f"basis must be 'cosmo' or 'emulator', got {basis!r}")
        if isinstance(engine, PrimordialCosmology):
            # A cosmology may be handed over instead of a spec, so that a caller need not test how
            # far it has got. Resolved first, because what it yields is an engine spec like any
            # other and has to go through the normalisation below.
            #
            # Only an ACECosmology has packaged emulators; anything else declares no ranges.
            if not isinstance(engine, ACECosmology):
                return {}
            cosmology = engine
            # `_param_clip_ranges` is filled by `_load_emulators_for_new_requirements`, at build.
            # Once it is, it is what the out-of-range guard actually enforces; before that, the
            # engine spec the instance was constructed with is all there is.
            if cosmology._param_clip_ranges:
                ranges = dict(cosmology._param_clip_ranges)
                if basis == 'cosmo':
                    # The guard keeps both bases, since it is asked for whichever name the caller
                    # uses; 'cosmo' reports desilike names only.
                    ranges.pop('H0', None)
                return ranges
            engine = cosmology._init[1].get('engine', 'ace')
            if base_dir is None:
                base_dir = cosmology._init[1].get('base_dir')
        base_emulator_dir = Path(base_dir) if base_dir is not None else Path(Installer().install_dir) / 'ace-emulators'
        if isinstance(engine, str):
            engine = dict(_PACKAGED_DEFAULT_ENGINE) if engine == 'ace' else {section_: engine for section_ in ['harmonic', 'fourier', 'background']}
        if section is not None:
            sections = [section] if isinstance(section, str) else list(section)
            unknown = [name for name in sections if name not in engine]
            if unknown:
                raise ValueError(f'unknown section(s) {unknown}; engine has {sorted(engine)}')
            engine = {name: engine[name] for name in sections}
        ranges_per_emulator = []
        for engine_name in set(engine.values()):
            if engine_name is None:
                continue
            emulator_dir = base_emulator_dir / engine_name
            if (emulator_dir / 'TT' / 'nn_setup.json').is_file():
                # Capse-style Cl emulator directory: introspect the networks' training ranges.
                ranges_per_emulator.append(_find_capse_metadata(emulator_dir)['ranges'])
            else:
                ranges_per_emulator.append(_PACKAGED_EMULATORS.get(engine_name, {}).get('ranges', {}))
        training_ranges = _intersect_ranges(ranges_per_emulator)
        if basis == 'cosmo':
            training_ranges = _add_h_from_H0(training_ranges, replace=True)
        return training_ranges

    @classmethod
    def truncate_priors(cls, params, engine='ace', base_dir=None, section=None):
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
        section : str or list, default=None
            Passed to :meth:`training_ranges`: which sections the ranges come from.  Name the
            ones a run actually consumes, or the tightest section truncates the priors of a
            calculator that never evaluates it.

        Returns
        -------
        VariableCollection
            *params*, with each prior's limits intersected with the training ranges.
        """
        from ..parameter import truncate_priors as truncate_priors_to_ranges
        return truncate_priors_to_ranges(params, cls.training_ranges(engine=engine, base_dir=base_dir,
                                                                     basis='cosmo', section=section))

    def __post_init__(self, *args, engine='isitgr', base_dir=None, conversion='cosmoprimo', params=None, fiducial='DESI', background_engine='cosmoprimo', **kwargs):
        # Which library integrates the background sector; see _ace_background for why the
        # default is not jaxace (its adaptive growth ODE runs to max_steps under vmap whenever
        # one draw in the batch fails to converge -- 47x on a w0waCDM CMB+BAO posterior) and for
        # why the growth that scales the emulated pk does not come from either of them.
        if background_engine not in ('cosmoprimo', 'jaxace'):
            raise ValueError("background_engine must be 'cosmoprimo' or 'jaxace', not "
                             f'{background_engine!r}')
        self._background_engine = background_engine
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
                for prefix in ('fourier.sigma_rz.', 'fourier.sigma8_z.'):
                    # Derived products of the linear pk: sigma_rz is its top-hat integral, and
                    # sigma8_z is that integral at R = 8 Mpc/h.  Any emulator providing the
                    # corresponding fourier.pk serves them (see __call__).  For sigma8_z of
                    # delta_cb / theta_cb this is not just a fallback but the CORRECT source --
                    # the jaxace network's own sigma8_z is total-matter (see _PACKAGED_EMULATORS).
                    if found or not method_key.startswith(prefix):
                        continue
                    pk_key = method_key.replace(prefix, 'fourier.pk.', 1)
                    for emulator_key, metadata in self._emulator_metadata.items():
                        if pk_key in metadata['outputs']:
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
            if metadata['kind'] == 'jaxmapse':
                # The packaged jaxace emulator serves this one twice over: D_z scales every
                # emulated pk (see _ace_background on why that normalisation is not free), and
                # f_z turns pk_cb into pk_tt.  Load it now; only theta_cb cannot do without.
                if method_key == 'fourier.pk.theta_cb.theta_cb' and self._ace_emulator_key is None:
                    raise ValueError("fourier.pk with of='theta_cb' requires a packaged jaxace emulator "
                                     "(engine['background'], e.g. 'ACE_mnuw0wacdm_ln10As_basis') providing f_z")
                if self._ace_emulator_key is not None and self._ace_emulator_key not in self._loaded_emulators:
                    self._loaded_emulators[self._ace_emulator_key] = self._load_emulator(self._ace_emulator_key)
        # The ranges the out-of-range guard enforces: intersected over the emulators actually
        # loaded, keyed by parameter name.  `replace=False` keeps `H0` alongside `h`, since the
        # guard is asked for whichever name the caller uses.
        self._param_clip_ranges = _add_h_from_H0(
            _intersect_ranges(self._emulator_metadata[emulator_key].get('ranges', {})
                              for emulator_key in self._loaded_emulators), replace=False)
        _warn_priors_beyond_ranges(self.params, self._param_clip_ranges)

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
        clip_ranges = self._param_clip_ranges
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

        # The growth the pk networks were trained against, when there is a packaged jaxace
        # network to ask; None falls the pk scaling below back on jaxace's ODE.
        ace_growth = None
        if self._ace_emulator_key is not None and self._ace_emulator_key in self._loaded_emulators:
            ace_growth = lambda z: jnp.reshape(run_ace(z)[:, 5], jnp.shape(z))

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
                if method_key in ('background.efunc', 'background.comoving_transverse_distance',
                                  'background.luminosity_distance', 'background.growth_factor',
                                  'background.growth_rate'):
                    result = _ace_background(method_key, _kw_coords['z'], self._background_engine,
                                             cosmoprimo_cosmo=cosmoprimo_cosmo,
                                             jaxace_cosmo=jaxace_cosmo)
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
                # Not the growth _ace_background serves: jaxmapse builds
                # pk = output * T(k)^2 * D^2 * P_prim, so the absolute normalisation of D is an
                # amplitude, and the conventions on offer are a constant 3.3% apart (see
                # _ace_background).  Measured end to end against engine='class' at the DESI
                # fiducial, sigma8_cb over z = 0, 0.5, 1, 2: the ACE network's D_z and jaxace's
                # ODE both 3e-4, cosmoprimo's cb growth 3.4% low at every z (pk 6.7% low).  The
                # network's is what these networks were trained against, and it costs nothing --
                # that forward pass is made anyway for sigma8_z / rs_drag / f_z, and it stays
                # clear of the vmap trap.
                growth = jaxace_cosmo.D_z(z) if ace_growth is None else ace_growth(z)
                of = spec['static']['of'][0]
                # theta_cb is served from the delta_cb network (times f_z^2 below)
                component = emulator['delta_m' if of == 'delta_m' else 'delta_cb']
                pk, k_grid = component.get_Pk(emulator_params, z, growth), component.k_grid
                if of.startswith('theta'):
                    # pk_tt = f_z^2 pk_cb (scale-independent growth), with f_z from the packaged
                    # jaxace emulator so that sigma8_z(theta_cb) = f_z * sigma8_z(delta_cb) exactly.
                    pk = run_ace(z)[:, 6, None]**2 * pk
                h = get_param('h')
                if method_key.startswith('fourier.sigma8_z'):
                    # sigma8 is sigma_R at R = 8 Mpc/h of THIS pk -- the same top-hat integral
                    # as sigma_rz below, at a single radius.  Taking it from the emulated P_cb
                    # rather than the jaxace total-matter output is what makes sigma8_cb
                    # actually cb (measured: 3e-5 against CLASS, where the jaxace output was
                    # 4.5e-3 low).  of='theta_cb' needs no special case: pk already carries the
                    # f_z^2 factor above, so the integral returns f_z * sigma8_cb exactly, and
                    # the growth rate f = fsigma8 / sigma8 stays exact.
                    result = _sigma_tophat(k_grid / h, pk * h**3, 8.)                 # (nz,)
                elif method_key.startswith('fourier.sigma_rz'):
                    if of.startswith('theta'):
                        raise NotImplementedError('sigma_rz of theta is not served (velocity '
                                                  'variance in a top-hat is not what you want)')
                    # Top-hat sigma_R(z) from the emulated linear pk, over the network's own
                    # k grid, with k in h/Mpc and R in Mpc/h so the h factors cancel.  Comes
                    # back as (nz, nr), the (z, r) layout CosmoprimoCosmology uses for the
                    # same key.
                    result = _sigma_tophat(k_grid / h, pk * h**3, jnp.asarray(spec['r']))
                elif method_key.startswith('fourier.pk_now'):
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


# ── emulating a cosmology: the leaves, named by requirement ──────────────────
#
# A :class:`CosmoprimoCosmology` flattens to its parameter values, one leaf per registered
# requirement (a dict of spectra for the harmonic ones), then its derived parameters. The
# generic :class:`~desilike.emulators.CalculatorEmulator` names those leaves by pytree path,
# which for an array result is a bare position: ``'1'``, ``'2'``. A position says nothing an
# emulator can route on, and it moves whenever a requirement is added -- the same
# ``fourier.pk`` is leaf 1 in a cosmology that serves one theory and leaf 3 in one that also
# serves a CMB likelihood. So every emulator below names each leaf by the requirement that
# produced it (``'fourier.pk|of=delta_cb,delta_cb'``, ``'harmonic.lensed_cl|ellmax=2500.tt'``),
# the input parameters by ``'input.<name>'`` and the derived parameters by
# ``'derived_params.<name>'``. Two cosmologies that share a requirement then share its leaf
# name, which is what lets :class:`CosmologyEmulator` stitch a harmonic and a Fourier emulator
# trained on two different calculators into one.


def _spec_name(spec_key):
    """The leaf name of a requirement: its method key and static arguments.

    ``'/'`` is HDF5's group separator and these names are keys of a saved emulator, so one is
    refused rather than written into a file that cannot be read back.
    """
    method_key, static = spec_key
    parts = []
    for key, value in static:
        if isinstance(value, (tuple, list)):
            value = ','.join(str(item) for item in value)
        parts.append(f'{key}={value}')
    name = method_key + ''.join(f'|{part}' for part in parts)
    if '/' in name:
        raise ValueError(f'cannot name the requirement {spec_key!r}: "/" in {name!r}')
    return name


def _spec_key(method_key, kwargs):
    """The key :meth:`PrimordialCosmology.add_requirements` registers *kwargs* under."""
    static = _normalize_static({key: value for key, value in kwargs.items() if key not in _COORDS})
    return (method_key, tuple(sorted(static.items())))


def _amplitude_like(name):
    """Whether a bare ``params.<name>`` requirement could depend on the amplitude."""
    return name in AMPLITUDES + ('sigma8',) or name.startswith(('sigma8', 'sigma_', 'S8'))


#: ``params.<name>`` requirements cosmoprimo derives from the input parameters alone -- no
#: Boltzmann call, no perturbations -- so a routed sector can divide the exact value out and let
#: every parameter leave its grid. Deliberately a list rather than a rule: anything not on it
#: keeps the conservative treatment, and getting it wrong here is an emulator that interpolates
#: nothing and reports no error. The amplitude spellings are handled before this
#: (:func:`_amplitude_like`), since ``sigma8`` needs the Fourier section and is not derivable
#: this way.
_INPUT_DERIVED = frozenset((
    'h', 'H0', 'Omega_m', 'omega_m', 'Omega_cdm', 'omega_cdm', 'Omega_b', 'omega_b',
    'Omega_k', 'omega_k', 'Omega_ncdm_tot', 'omega_ncdm_tot', 'm_ncdm_tot', 'm_ncdm',
    'N_eff', 'N_ur', 'N_ncdm', 'T_cmb', 'w0_fld', 'wa_fld', 'n_s', 'alpha_s', 'k_pivot',
    'tau_reio', 'Omega_de', 'Omega_Lambda', 'Omega_g', 'Omega_ur', 'Omega_r'))


def _sector(method_key):
    """The sector a requirement is emulated in: the cosmoprimo section it is read from.

    The primordial spectrum goes with the Fourier one, and a bare ``params.<name>`` with the
    background unless the name is an amplitude. Each sector gets an emulator of its own in
    :class:`CosmologyEmulator`, because each depends on a different subset of the parameters:
    ``rs_drag`` on the densities alone, the background on those and ``h``, ``w0``, ``wa``, the
    spectra on everything but ``tau_reio`` -- and with one node set for all, a single ``rs_drag``
    leaf would keep ``h`` on the power spectrum's grid.
    """
    section = method_key.split('.', 1)[0]
    if section in ('fourier', 'primordial'):
        return 'fourier'
    if section == 'params':
        return 'fourier' if _amplitude_like(method_key[len('params.'):]) else 'background'
    return section


#: What a requirement is, for routing: the method key -> a kind whose dependence on the
#: cosmological parameters the emulators below know.
_LEAF_KINDS = {'fourier.pk': 'pk', 'fourier.pk_now': 'pk',
               'fourier.sigma8_z': 'sigma', 'fourier.sigma_rz': 'sigma',
               'primordial.pk': 'primordial',
               'background.efunc': 'efunc',
               'background.comoving_transverse_distance': 'distance',
               'background.luminosity_distance': 'distance',
               'background.growth_factor': 'growth', 'background.growth_rate': 'rate',
               'thermodynamics.rs_drag': 'rs_drag', 'background.age': 'age',
               'background.Omega_b': 'omega', 'background.Omega_cdm': 'omega',
               'background.Omega_ncdm_tot': 'omega'}

#: The canonical names a :class:`FourierEmulator` can take off the grid, and per kind of leaf
#: which of them it lets go: those whose dependence the transform pair divides out (exactly, or
#: through the analytic growth: ``A_s``, ``n_s``, ``h``, ``w0_fld``, ``wa_fld`` for a
#: spectrum) and those it does not depend on at all (``tau_reio`` for anything Fourier, the
#: amplitude for a background quantity). A parameter leaves the grid only when every leaf lets
#: it -- an opaque ``params.<name>`` leaf keeps everything but what it provably ignores.
_EXACT_NAMES = ('A_s', 'n_s', 'h', 'tau_reio', 'w0_fld', 'wa_fld')
_OFF_GRID = {'pk': _EXACT_NAMES, 'primordial': _EXACT_NAMES, 'input': _EXACT_NAMES,
             'sigma': ('A_s', 'tau_reio', 'w0_fld', 'wa_fld'),
             'efunc': ('A_s', 'n_s', 'h', 'tau_reio', 'w0_fld', 'wa_fld'),
             'distance': ('A_s', 'n_s', 'h', 'tau_reio', 'w0_fld', 'wa_fld'),
             'growth': ('A_s', 'n_s', 'h', 'tau_reio', 'w0_fld', 'wa_fld'),
             'rate': ('A_s', 'n_s', 'h', 'tau_reio', 'w0_fld', 'wa_fld'),
             'rs_drag': ('A_s', 'n_s', 'h', 'tau_reio', 'w0_fld', 'wa_fld'),
             'age': ('A_s', 'n_s', 'tau_reio'), 'omega': ('A_s', 'n_s', 'tau_reio'),
             'opaque': ('A_s', 'n_s', 'tau_reio'), 'amplitude': ('tau_reio',), 'cl': (),
             # divided by its own exact value, so nothing about it is interpolated
             'cosmo_param': _EXACT_NAMES}


class _SectionEmulator(CalculatorEmulator):
    """What every emulator of a :class:`CosmoprimoCosmology` shares.

    * leaves named by requirement (see the section comment above), so the routing keys off
      what a leaf is, and two calculators serving the same requirement agree on its name;
    * a description of each leaf -- its kind, its ``of`` pair, its ``z`` and ``k`` grids -- read off
      the calculator's own requirement registry, which every routing decision below is made
      from;
    * the transform pair, written once: each subclass says what to divide out
      (:meth:`routing`) and this class applies it, both ways, and rebuilds the input leaves of
      the parameters that left the grid. A parameter handled exactly is held at the box centre
      while the nodes are evaluated, so its ``input.<name>`` leaf would otherwise be predicted
      at the centre for ever.
    """

    def set_children_leafnames(self):
        aux, calculator = self.aux, self.calculator
        # in the order `tree_flatten` produces: a dict child flattens in sorted key order
        names = [f'input.{name}' for name in sorted(calculator._param_values)]
        for spec_key, spec in aux['ordered_specs']:
            base = _spec_name(spec_key)
            result = calculator._results.get(spec_key)
            if isinstance(result, dict):
                names += [f'{base}.{key}' for key in sorted(result)]
            else:
                names.append(base)
        names += [f'derived_params.{param.name}' for param in aux['derived_params']]
        self.children_leafnames = names

    def _leaf_info(self):
        """``{leaf name: description}`` for every leaf and derived output.

        A description is ``kind`` (see ``_LEAF_KINDS``; ``'input'`` for a parameter leaf,
        ``'cl'`` for a spectrum, ``'opaque'`` for a ``params.<name>`` requirement whose
        dependence is unknown), ``of`` (the pair of perturbed quantities), the ``z``/``k``/``r`` grids the
        requirement was registered on, and ``name`` for a parameter leaf.
        """
        cached = getattr(self, '_leaf_info_cache', None)
        if cached is not None:
            return cached
        aux = self.aux
        specs = {spec_key: spec for spec_key, spec in aux['ordered_specs']}
        inputs = {param.basename for param in aux['params']}

        def describe(spec_key):
            method_key, static = spec_key
            spec, static = specs[spec_key], dict(static)
            info = {'kind': _LEAF_KINDS.get(method_key), 'method': method_key,
                    'of': tuple(static.get('of', ())), 'name': None,
                    'z': spec.get('z'), 'k': spec.get('k'), 'r': spec.get('r')}
            if method_key.startswith('params.'):
                name = method_key[len('params.'):]
                info['name'] = name
                info['kind'] = ('input' if name in inputs
                                else 'amplitude' if _amplitude_like(name)
                                else 'cosmo_param' if name in _INPUT_DERIVED else 'opaque')
            elif method_key.startswith('harmonic.'):
                info['kind'] = 'cl'
            elif info['kind'] is None:
                info['kind'] = 'opaque'
            return info

        # longest base first: a dict result's leaves are `base + '.' + key`, and the prefix
        # test must not let a shorter base claim them
        bases = sorted(((_spec_name(spec_key), spec_key) for spec_key in specs),
                       key=lambda item: -len(item[0]))
        info = {}
        for leaf in self.children_leafnames:
            if leaf.startswith('input.'):
                info[leaf] = {'kind': 'input', 'method': 'input', 'of': (),
                              'name': leaf[len('input.'):], 'z': None, 'k': None, 'r': None}
                continue
            if leaf.startswith('derived_params.'):
                continue
            for base, spec_key in bases:
                if leaf == base or leaf.startswith(base + '.'):
                    info[leaf] = describe(spec_key)
                    break
            else:
                raise ValueError(f'the leaf {leaf!r} matches no registered requirement')
        # a derived parameter is both a child (`derived_params.`) and a graph output (`derived.`)
        for name, getter in aux['get_derived'].items():
            # the getter's own coordinates, not the requirement's merged grid. A derived sigma8
            # is one number at one redshift, while `fourier.sigma8_z` may serve several -- a
            # template at z = 0.8 and this at z = 0 -- and a factor built over the merged grid
            # broadcasts the scalar it divides into a vector. Measured: two derived sigma8 leaves
            # turned a 12-wide derived row into a 14-wide one, and emcee died on the mismatch
            # several hundred steps in, where the walkers' blobs no longer lined up.
            method_key, kwargs = getter
            described = dict(describe(_spec_key(*getter)))
            described.update({coord: np.atleast_1d(kwargs[coord])
                              for coord in _COORDS if coord in kwargs})
            info[f'derived_params.{name}'] = info[f'{DERIVED}{name}'] = described
        self._leaf_info_cache = info
        return info

    def _exact_input_leaves(self):
        """``{leaf: parameter name}`` for the input leaves of the parameters handled exactly."""
        out = {}
        for name in self.exact_params:
            basename = self.graph_params[name].basename if name in self.graph_params else name
            leaf = f'input.{basename}'
            if leaf in self.children_leafnames:
                out[leaf] = name
        return out

    def _transforms(self):
        """Expansion variables needing a transform, as ``{name: spec}``, in the MAPPED names.

        Declared here rather than inline in :meth:`training_space` so a subclass adds its own by
        calling ``super()`` -- and because a transform may only be declared for a name the
        mapping introduces, which is the one rule tying the two together.
        """
        names = set(getattr(self.space, 'params', []))
        # `w0 + wa` is bounded above -- CAMB's PPF refuses w0 + wa > 0 ("giving w>0 at high
        # redshift"), stricter than cosmoprimo's own 1/3 radiation-domination check, and 0 is the
        # hard prior the analysis applies. It is the SUM that is bounded, so no per-axis limit
        # expresses it; a logit onto (-5, 0) makes the bound unreachable rather than an edge to
        # cut back from, and a Smolyak grid is unisolvent, so one node past it is not a smaller
        # problem but a singular one.
        return {'w0pwa': 'logit_w0pwa'} if {'w0_fld', 'wa_fld'} <= names else {}

    def routing(self, params):
        """``(factors, dilations)``: ``{leaf: factor}`` divided out of the leaf before the fit
        and multiplied back at prediction, and ``{leaf: s}`` for the leaves read at ``k / s``
        (in the reference frame) before the fit and at ``k s`` after it. Nothing, by default."""
        return {}, {}

    def transform(self, values, params):
        factors, dilations = self.routing(params)
        info, exact = self._leaf_info(), self._exact_input_leaves()
        out = {}
        for name, value in values.items():
            if name in exact:
                continue
            if name in factors:
                # in the leaf's shape when the sizes agree: a derived scalar is stored with
                # shape () while its factor comes from a z grid of one
                factor = factors[name]
                if jnp.size(factor) == jnp.size(value):
                    factor = jnp.reshape(factor, jnp.shape(value))
                value = value / factor
            if name in dilations:
                # c(k) = P(k / s) / s^3: the leaf in the reference frame, a smooth function of
                # h rather than the BAO wiggles sliding through the k grid
                scale = dilations[name]
                value = _dilate(info[name]['k'], value, 1. / scale) / scale**3
            out[name] = value
        return out

    def inverse_transform(self, values, params):
        factors, dilations = self.routing(params)
        info, exact = self._leaf_info(), self._exact_input_leaves()
        out = {}
        for name, value in values.items():
            if name in dilations:
                scale = dilations[name]
                value = scale**3 * _dilate(info[name]['k'], value, scale)
            if name in factors:
                factor = factors[name]
                if jnp.size(factor) == jnp.size(value):
                    factor = jnp.reshape(factor, jnp.shape(value))
                value = value * factor
            out[name] = value
        if exact:
            user = self.from_training(dict(params))
            for leaf, name in exact.items():
                out[leaf] = jnp.asarray(user[name])
        return out


# ── emulating a cosmology for its Cl ──────────────────────────────────────────
#
# Emulating the CMB spectra a likelihood asks a cosmology for.
#
# The cut is the cosmology, not the likelihood: :class:`~desilike.theories.primordial_cosmology.CosmoprimoCosmology`
# flattens to its registered requirement results, so emulating it replaces exactly the Boltzmann
# call and leaves every foreground, calibration and window downstream untouched. Those nuisance
# parameters are cheap and numerous -- emulating them would be paying to interpolate arithmetic.
#
# What this class adds over the generic :class:`~desilike.emulators.CalculatorEmulator` is the two
# things a :math:`C_\ell` is analytic in, divided out before the fit and put back at prediction:
#
# - the amplitude, :math:`C_\ell \propto A_s`. Exact for the primary anisotropies; not once lensing
#   is applied, since the deflection power is itself proportional to it (measured residual 1.0e-3
#   over a 10% amplitude change). So it flattens the dependence but the parameter stays on the
#   grid -- which is the honest outcome, not a compromise: what remains is a much smaller thing to
#   interpolate.
# - the optical depth, one :math:`e^{-\tau}` per screened leg: ``tt`` and ``ee`` carry
#   :math:`e^{-2\tau}`, ``tp`` and ``ep`` one factor, ``pp`` none. Also a flattening rather than a
#   removal -- below :math:`\ell \sim 30` reionization puts power back, which no prefactor
#   describes.
#
# Getting the per-leg count wrong is a silent factor of :math:`e^{\tau}`, which is why the legs are
# counted from the spectrum's own name rather than listed.


class HarmonicEmulator(_SectionEmulator):
    r"""A cosmology emulated for its :math:`C_\ell`, with the amplitude and optical depth routed.

    Declared by :meth:`CosmoprimoCosmology.get_emulator_cls` for a cosmology whose requirements
    are all harmonic, and used for the harmonic sector of a :class:`CosmologyEmulator`. A
    Fourier leaf on the same cosmology gets its amplitude power divided out too, so a joint
    cosmology forced onto this class is flattened in :math:`A_s` throughout; it is the
    :math:`\theta_\mathrm{MC}` basis and the optical depth that are specific to the spectra.
    """

    def spectra(self):
        """``{leaf key: spectrum name}`` for the leaves that are spectra.

        A leaf whose name ends in one of these is a spectrum; any other leaf (a parameter value,
        a derived quantity) is passed through untouched.
        """
        spectra = ('tt', 'ee', 'bb', 'te', 'pp', 'tp', 'ep')
        return {name: name.rsplit('.', 1)[-1]
                for name in getattr(self, 'children_leafnames', [])
                if name.rsplit('.', 1)[-1] in spectra}

    def routing(self, params):
        r"""``{leaf key: factor}`` divided out at training and multiplied back at prediction,
        and no dilation."""
        value = amplitude(params)
        factors = harmonic_scaling(self.spectra(), value, params.get('tau_reio', None))
        if value is not None:
            # the Fourier leaves of a joint cosmology: exact powers of the amplitude
            for key, info in self._leaf_info().items():
                power = {'pk': 1., 'sigma': 0.5, 'primordial': 1.}.get(info['kind'], 0.)
                if power:
                    factors[key] = value ** power
        return factors, {}

    def _theta_args(self, params):
        r"""What :func:`~cosmoprimo.emulators.analytic.theta_analytic` needs besides the
        densities: :math:`w_0`, :math:`w_a`, and the radiation content, read off the cosmology
        this emulator holds whenever the space does not vary them.

        ``w_a`` is reconstructed from ``w0pwa``, since by the time this is read the expansion
        variable is in hand rather than ``wa_fld`` itself. ``m_ncdm`` goes through as an array so
        that the compiled ``theta_analytic`` sees one argument structure rather than retracing
        between a tuple of masses and an array of them.
        """
        fiducial = getattr(getattr(self, 'calculator', None), '_fiducial', None)
        w0 = params.get('w0_fld', -1.)
        return {'w0': w0, 'wa': params.get('w0pwa', -1.) - w0,
                'm_ncdm': jnp.atleast_1d(params.get('m_ncdm', fiducial['m_ncdm'])),
                'N_ur': params.get('N_ur', fiducial['N_ur']),
                'T_cmb': params.get('T_cmb', fiducial['T_cmb'])}

    def to_training(self, params):
        r""":math:`(h, w_0, w_a) \rightarrow (\theta_\mathrm{MC}, w_0, w_0 + w_a)`.

        The expansion basis belongs to the EMULATOR, not to the cosmology: the pipeline keeps
        sampling ``h`` and ``wa_fld``, priors and refs stay in them, chains come out in them, and
        nothing downstream has to learn about ``theta_MC_100`` or ``w0pwa``.

        Both maps are cheap and traceable, which is the whole reason they can run here: this is
        applied at EVERY prediction, inside the jit, where the exact ``theta`` cannot be computed
        at all -- it sits behind a ``pure_callback``. :func:`theta_analytic` is 0.0205
        ms/point under ``vmap``, against the emulator's own 0.0176.

        Its ~1.6 sigma(theta) bias does not matter HERE, and that is worth being precise about:
        ``theta`` is only an internal relabelling of ``h``. The training box is built by mapping
        points through this same function (:meth:`training_space`), the nodes are evaluated by
        inverting it (:meth:`from_training`), and predictions enter through it again -- so the
        composition is the identity and the bias cancels exactly. It would only matter if a
        ``theta`` from somewhere else (a chain, a published box) were fed in, which is the trap
        that put an earlier box 5.3 sigma off.
        """
        params = dict(params)
        if 'wa_fld' in params and 'w0_fld' in params:
            params['w0pwa'] = params.pop('wa_fld') + params['w0_fld']
        if 'h' in params:
            params['theta_MC_100'] = 100. * theta_analytic(
                params.pop('h'), params['omega_b'], params['omega_cdm'],
                **self._theta_args(params))
        return params

    def from_training(self, params):
        r"""The inverse, to call the cosmology in ITS parameters: neither ``w0pwa`` nor
        ``theta_MC_100`` is a cosmoprimo input.

        ``h`` comes back through :func:`solve_theta_analytic`, the bisection on the same closed
        form -- not through ``Cosmology.solve``, which runs a background per iteration. Using the
        SAME function in both directions is what makes the round trip exact.
        """
        params = dict(params)
        if 'w0pwa' in params:
            params['wa_fld'] = params.pop('w0pwa') - params['w0_fld']
        if 'theta_MC_100' in params:
            # `wa_fld` is back by now, so `w0pwa` is rebuilt for `_theta_args`, which reads it:
            # the pair must be the one `to_training` used or the round trip is not the identity
            params['h'] = solve_theta_analytic(
                params.pop('theta_MC_100'), params['omega_b'], params['omega_cdm'],
                **self._theta_args({**params, 'w0pwa': params.get('w0_fld', -1.) + params.get('wa_fld', 0.)}))
        return params

    def training_space(self):
        """The user's space, re-expressed in the expansion basis by mapping its points.

        Paired with :meth:`to_training`, as :class:`Space` requires. ``Space.map`` transforms
        points rather than propagating a Jacobian, and applies the declared transform to those
        points itself, so mean, covariance and limits all end up in the expansion variable without
        the caller arranging it.

        A plain box, not an ellipsoid on the principal axes. Whitened, the nodes fill a band across
        the box and the box's off-diagonal corners hold none, so
        :meth:`~cosmoprimo.emulators.tools.engines.BaseEngine.outside` refuses points that are
        inside every one of their own limits. Measured on DESI DR2 BAO + CMB-SPA in w0waCDM, the
        whitened sector answered 66.68% of that posterior -- the missing third refused as off the
        node cloud, not outside the box -- and the truncation reached the chain as a hard wall at
        `w0 = -0.16`, costing 13.3% of the published w0 marginal while Gelman-Rubin read 1.00.
        Uncorrelated it answers 99.87%.

        The whitening is not what buys the accuracy here, which is why this costs nothing. Both
        arms trained at budget 3 (330 nodes), `|dchi2|` against exact CLASS on 96 draws of the
        reference chain: whitened 1.442 but finite on only 61 of them, uncorrelated 1.466 finite
        on all 96, and better at the 95th percentile (3.266 against 3.304). The 350x that whitening
        is worth elsewhere is a Chebyshev-grid measurement, where the rotation is what orients the
        grid onto the posterior; with a regression engine at ``measure='samples'`` the candidate
        pool is already the chain, so the nodes are posterior-shaped either way and the rotation
        adds only the mask. Reconsider this if the engine changes.
        """
        names = getattr(self.space, 'params', [])
        # Nothing to map: the basis change is `h -> theta_MC_100` and the dark-energy pair, and a
        # space with neither varies nothing this expansion re-expresses.
        if not any(name in names for name in ('wa_fld', 'h')):
            space = self.space
        else:
            space = self.space.map(self.to_training, transforms=self._transforms())
        return space.uncorrelated()


# ── emulating a cosmology for everything that is not a Cl ─────────────────────
#
# The linear power spectrum a full-shape theory asks a cosmology for -- and the growth, the
# distances and the sound horizon a BAO or supernova likelihood asks for -- is analytic in more
# of its parameters than a :math:`C_\ell` is, and pays for none of the harmonic sector's
# difficulties: no optical depth, no acoustic-peak variable, no lensing. So what is divided
# out here is divided out exactly, and the parameter leaves the grid:
#
# - the amplitude, :math:`P \propto A_s`, :math:`\sigma_8 \propto A_s^{1/2}`;
# - the tilt, :math:`P \propto (k h / k_\mathrm{pivot})^{n_s - 1}` through the primordial
#   spectrum alone, the transfer function knowing nothing of it;
# - ``h``, through the dilation :math:`P_h(k) = s^3 P_\mathrm{fid}(k s)`, :math:`s = h /
#   h_\mathrm{fid}`, for a spectrum in :math:`(\mathrm{Mpc}/h)^3` on a grid in
#   :math:`h/\mathrm{Mpc}`: at fixed physical densities the transfer function in
#   :math:`\mathrm{Mpc}^{-1}` does not move with ``h``, and only the late-time growth does --
#   which the next item carries;
# - the growth, :math:`D(z)` per density leg and :math:`f(z) D(z)` per velocity leg, from the
#   same analytic w0waCDM core :class:`~desilike.theories.galaxy_clustering.template.ScalingScalars`
#   is built on (:func:`~cosmoprimo.emulators.analytic.fourier_analytic_scales`).
#   That one is a preconditioner rather than an identity -- neutrinos count as matter, radiation
#   is in the background but not the growth source -- and it is what carries ``w0_fld`` and
#   ``wa_fld``, and ``h``'s effect on the growth, off the grid.
#
# Whether a parameter actually leaves the grid is decided per cosmology, from what it serves
# (``_OFF_GRID`` above): a ``sigma8_z`` leaf keeps ``h`` and ``n_s`` on it,
# because its top-hat window moves with ``h`` and its tilt dependence is an integral. Whatever
# stays on the grid is still divided out, so the expansion carries only the residual -- which is
# the FOLPSD emulator's ``precondition`` mechanism, applied one level down.


class _RoutedSectionEmulator(_SectionEmulator):
    r"""The routing above, for one non-harmonic sector of a cosmology.

    :class:`FourierEmulator`, :class:`BackgroundEmulator` and :class:`ThermodynamicsEmulator`
    are this, named for the section they serve: the routing is decided leaf by leaf, so the
    same code does the right thing on a power spectrum, a distance or a sound horizon, and
    what differs between the sections is only which parameters their leaves let off the grid.

    Identity basis: a ``k`` grid is in :math:`h/\mathrm{Mpc}`, so ``h`` enters directly, and
    the :math:`\theta_\mathrm{MC}` route would put ``omega_b`` on the grid for nothing.
    ``exact`` lists, in canonical names, what may leave the grid; :meth:`select_params` grants
    each one only when every leaf either routes it or does not depend on it. ``tau_reio`` never
    reaches any of these quantities and always leaves.
    """
    exact = _EXACT_NAMES

    #: What the analytic core moves: the fiducial is cloned with these, in canonical spellings,
    #: and everything else stays at its fiducial value.
    _ref_update_names = ('h', 'omega_b', 'omega_cdm', 'm_ncdm', 'N_eff', 'Omega_k', 'w0_fld',
                         'wa_fld')

    def __init__(self, calculator, space, exact=None, **options):
        """As the base, plus the anchors the analytic core is written against, so a saved
        emulator predicts with no calculator at hand.

        *exact* overrides the class's list of what may leave the grid, in canonical names:
        ``exact=('A_s', 'n_s', 'tau_reio')`` keeps ``h``, ``w0_fld`` and ``wa_fld`` expanded
        (still divided out, so the expansion carries the residual only). The reason to: the
        analytic growth is a preconditioner, and against CLASS its residual on a ``delta_cb``
        spectrum is below 1e-4 at every k for a 10% change in ``h`` but reaches 7e-3 at
        k = 1e-3 h/Mpc for ``wa_fld`` moved by 0.5 -- below 1.4e-4 from k = 3e-3 up -- which
        an analysis reaching such scales may not want to leave uninterpolated.
        """
        if exact is not None:
            self.exact = tuple(exact)
        super().__init__(calculator, space, **options)
        self._anchors = {
            # the eisenstein_hu clone, not the fiducial itself: its state names an engine, and
            # a saved emulator should not need the Boltzmann code the fiducial was built with
            # just to rebuild the analytic core
            'fiducial': self.calculator._fiducial.clone(engine='eisenstein_hu').__getstate__(),
            'defaults': {param.basename: float(np.sum(np.atleast_1d(param.value)))
                         for param in self.calculator.params},
            'zs': np.array(sorted({float(z) for info in self._leaf_info().values()
                                   if info['z'] is not None for z in np.atleast_1d(info['z'])}))}
        self.set_ref_fiducial()

    def set_ref_fiducial(self):
        """Rebuild the fiducial and its analytic scalars from the anchors, once.

        Recomputed rather than stored: the growth ratios are numerator over denominator from the
        same function, and an emulator trained before a change to it must be retrained, not
        redeployed with a stale denominator.
        """

        self._fiducial = Cosmology.from_state(self._anchors['fiducial'])
        scalars = [fourier_analytic_scales(float(z), self._fiducial) for z in self._anchors['zs']]
        self._ref = {name: np.array([float(item[name]) for item in scalars])
                     for name in ('invE', 'DM', 'D', 'f')}
        self._ref_rs_drag = float(eisenstein_hu_scales(self._fiducial)['rs_drag'])

    def select_params(self, names):
        """Everything but what the leaves let the transform pair carry exactly.

        A candidate is granted only when every leaf either routes it (its dependence is divided
        out) or is independent of it: one ``sigma8_z`` leaf keeps ``h`` on the grid for the
        whole emulator, since its window moves with ``h`` and nothing here follows that.
        """
        # The spellings the routing can convert. `sigma8` is deliberately not one for the
        # amplitude: the analytic core reads `A_s` off an eisenstein_hu clone, and the `A_s` that
        # engine derives from a sampled `sigma8` is not the Boltzmann code's.
        spellings = {'A_s': AMPLITUDES, 'h': ('h', 'H0')}
        kinds, off_grid = {info['kind'] for info in self._leaf_info().values()}, self._off_grid()
        return [name for name in names
                if not any(name in spellings.get(canonical, (canonical,))
                           and all(canonical in off_grid[kind] for kind in kinds)
                           for canonical in self.exact)]

    def _off_grid(self):
        """Per kind of leaf, what it lets off the grid: ``_OFF_GRID``, unless a sector's
        expansion basis makes more of its leaves independent of a parameter."""
        return _OFF_GRID

    def _exact_input_leaves(self):
        """As the base, for every parameter of the user's space that is not expanded -- not only
        those handled exactly, but also those the training basis replaced: ``omega_b`` is not a
        training parameter of a sector expanding in ``Omega_b``, and its leaf is rebuilt from
        :meth:`from_training` like the others."""
        out = {}
        for name in self.space.params:
            if name in self.params:
                continue
            basename = self.graph_params[name].basename if name in self.graph_params else name
            leaf = f'input.{basename}'
            if leaf in self.children_leafnames:
                out[leaf] = name
        return out

    def routing(self, params):
        """*params* are in the user's spelling -- ``H0``, ``logA``, whatever the pipeline
        samples -- so the fiducial is cloned with them and the canonical names are read off the
        clone, exactly as :class:`~desilike.theories.galaxy_clustering.template.ScalingScalarsEmulator`
        does. Traceable: eisenstein_hu is a JAX engine."""

        # `params` are in the training basis, which a sector may have changed
        params = self.from_training(dict(params))
        given = {self.graph_params[name].basename if name in self.graph_params else name: value
                 for name, value in params.items()}
        fid, zs = self._fiducial, self._anchors['zs']
        cosmo = fid.clone(**{name: given.get(name, value) for name, value in self._anchors['defaults'].items()})
        # the analytic core, at the canonical values the clone resolved. `m_ncdm` goes back as the
        # TOTAL: `cosmo['m_ncdm']` is per-species -- three entries under a degenerate hierarchy --
        # while a fiducial that declares one takes the sum and re-splits it, and handing it the
        # list raises "neutrino_hierarchy cannot be passed with a list for m_ncdm, only with a
        # sum". Summing is not a workaround for that check: the total is the quantity the
        # hierarchy is a statement about, and it is what the fiducial was cloned from. With no
        # hierarchy declared the list has one entry and the sum is that entry, unchanged.
        update = {name: cosmo[name] for name in self._ref_update_names}
        if 'm_ncdm' in update:
            update['m_ncdm'] = jnp.sum(jnp.atleast_1d(update['m_ncdm']))
        scalars = [fourier_analytic_scales(float(z), fid.clone(**update)) for z in zs]
        ratios = {}
        for name, ref in self._ref.items():
            values = jnp.array([item[name] for item in scalars]) if scalars else jnp.zeros(0)
            # DM is 0 at z = 0: a ratio of nothing to nothing is 1
            ratios[name] = jnp.where(ref != 0., values / np.where(ref != 0., ref, 1.), 1.)
        amplitude_ratio = cosmo['A_s'] / fid['A_s']
        h, n_s, k_pivot = cosmo['h'], cosmo['n_s'], fid['k_pivot']
        scale = h / fid['h']

        def at(name, z):
            return ratios[name][np.searchsorted(zs, np.atleast_1d(z))]

        def growth(z, of):
            factor = 1.
            for name in of:
                factor = factor * at('D', z)
                if str(name).startswith('theta'):
                    factor = factor * at('f', z)
            return factor

        factors, dilations = {}, {}
        for leaf, info in self._leaf_info().items():
            kind = info['kind']
            if kind == 'pk':
                k = np.asarray(info['k'])
                # the tilt on the leaf's own grid, at the live h: after the dilation reads this
                # at k / s the primordial factor is (k h_fid / k_pivot)^(n_s - 1), h-free
                tilt = (k * h / k_pivot) ** (n_s - fid['n_s'])
                factors[leaf] = amplitude_ratio * growth(info['z'], info['of'])[:, None] * tilt[None, :]
                dilations[leaf] = scale
            elif kind == 'sigma':
                factor = amplitude_ratio ** 0.5 * growth(info['z'], info['of'][:1])
                factors[leaf] = factor[:, None] if info['r'] is not None else factor
            elif kind == 'primordial':
                # cosmoprimo's primordial spectrum: h^3 A_s (k h / k_pivot)^(n_s - 1) on a grid
                # in h/Mpc -- measured, not read: the h^3 is the (Mpc/h)^3 volume
                k = np.asarray(info['k'])
                factors[leaf] = (amplitude_ratio * scale ** 3
                                 * (k * h / k_pivot) ** (n_s - 1.)
                                 / (k * fid['h'] / k_pivot) ** (fid['n_s'] - 1.))
            elif kind == 'efunc':
                factors[leaf] = 1. / at('invE', info['z'])
            elif kind == 'distance':
                factors[leaf] = at('DM', info['z'])
            elif kind == 'growth':
                factors[leaf] = at('D', info['z'])
            elif kind == 'rate':
                factors[leaf] = at('f', info['z'])
            elif kind == 'rs_drag':
                # the Eisenstein & Hu fitting formula, in Mpc/h like the leaf: it carries the
                # unit's h exactly and the dependence on the densities to a few per cent, so what
                # stays on the grid is the Boltzmann code's correction to a fitting formula
                factors[leaf] = eisenstein_hu_scales(cosmo)['rs_drag'] / self._ref_rs_drag
            elif kind == 'cosmo_param':
                # the exact value, off the same clone the analytic core is built on: the leaf is
                # a function of the sampled parameters and of nothing else, so dividing by it
                # leaves the constant 1 for the interpolant and every parameter can leave this
                # sector's grid. `nonzero` because a fixed Omega_k is exactly 0, and a factor
                # that is divided out and multiplied back cancels whatever it is.
                factors[leaf] = nonzero(cosmo[info['name']])
            elif kind == 'age':
                # in Gyr: 1/H0 times a function of the density fractions and the dark energy
                factors[leaf] = 1. / scale
        return factors, dilations

    def __getstate__(self):
        state = super().__getstate__()
        state['anchors'] = self._anchors
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._anchors = state['anchors']
        self.set_ref_fiducial()


class FourierEmulator(_RoutedSectionEmulator):
    r"""A cosmology emulated for its power spectra (``fourier.*`` and ``primordial.*``).

    Declared by :meth:`CosmoprimoCosmology.get_emulator_cls` when those are all it serves, and
    the Fourier sector of a :class:`CosmologyEmulator` otherwise. Everything in ``exact`` leaves
    the grid for a set of spectra; a ``sigma8_z`` leaf keeps ``h`` and ``n_s`` on it.
    """


class BackgroundEmulator(_RoutedSectionEmulator):
    r"""A cosmology emulated for its background (``background.*`` and a bare ``params.<name>``).

    The background is a function of today's density fractions: :math:`E(z)^2 = \sum_i \Omega_i
    g_i(z)`, and a distance in Mpc/h, the growth factor and rate, the age in units of
    :math:`1/H_0` and :math:`\Omega_i(z)` all follow from it, so with :math:`\Omega_b`,
    :math:`\Omega_{cdm}`, :math:`\Omega_k` and :math:`(w_0, w_a)` fixed, ``h`` has dropped
    out -- up to the radiation fraction :math:`\omega_\gamma / h^2`, set by ``T_cmb``, and the
    neutrino transition, set by the physical mass: a few 1e-5 at the redshifts a likelihood
    asks for. So this sector expands in :math:`(\Omega_b, \Omega_{cdm})` rather than in the
    physical densities (:meth:`training_space`), ``h`` leaves the grid because nothing depends
    on it, and the two leaves whose unit carries ``h`` -- ``rs_drag`` is not here, ``age`` is --
    have that power divided out. ``w0_fld`` and ``wa_fld`` leave through the analytic w0waCDM
    core, which also flattens what stays. A BAO or supernova likelihood's cosmology is emulated
    with this alone.

    The basis change applies when the pipeline samples ``h`` with ``omega_b`` / ``omega_cdm``;
    with ``H0`` or ``Omega_m`` sampled it is not needed or not attempted, and ``age`` and
    :math:`\Omega_i(z)` then keep ``h`` on the grid.
    """
    _DENSITIES = {'omega_b': 'Omega_b', 'omega_cdm': 'Omega_cdm'}

    def _omega_basis(self, params):
        """Whether *params* (the user's) are the ones the basis change converts."""
        return 'h' in params and any(name in params for name in self._DENSITIES)

    def to_training(self, params):
        r""":math:`(h, \omega_b, \omega_{cdm}) \rightarrow (h, \Omega_b, \Omega_{cdm})`, and
        :math:`(w_0, w_a) \rightarrow (w_0, w_0 + w_a)`; ``h`` stays a training parameter, one
        that then leaves the grid.

        The dark-energy pair is remapped for the same reason
        :meth:`HarmonicEmulator.to_training` remaps it: ``w0 + wa`` is bounded above, and it is
        the sum, not either parameter alone, that is bounded, so no per-axis limit expresses
        it. Naming the sum lets
        :meth:`training_space` put a logit on it and make the bound unreachable. Routing does not
        exempt this sector -- ``efunc`` and the distances are exact in ``w0_fld`` and ``wa_fld``,
        but exact means divided by the analytic w0waCDM core, and that core is what diverges
        where ``w0 + wa > 0``, so an unphysical node poisons the scaling rather than escaping it.
        """
        params = dict(params)
        if 'wa_fld' in params and 'w0_fld' in params:
            params['w0pwa'] = params.pop('wa_fld') + params['w0_fld']
        return self._to_densities(params)

    def _to_densities(self, params):
        """The density half of the basis change, on its own -- :meth:`training_space` needs to
        apply the two halves in separate `Space.map` calls, see the note there."""
        if not self._omega_basis(params):
            return params
        return {self._DENSITIES.get(name, name): value / params['h'] ** 2 if name in self._DENSITIES else value
                for name, value in params.items()}

    def from_training(self, params):
        params = dict(params)
        if 'w0pwa' in params:
            params['wa_fld'] = params.pop('w0pwa') - params['w0_fld']
        if not self._omega_basis(self.space.params):
            return params
        fractions = {fraction: density for density, fraction in self._DENSITIES.items()
                     if density in self.space.params}
        return {fractions.get(name, name): value * params['h'] ** 2 if name in fractions else value
                for name, value in params.items()}

    def _transforms(self):
        """The dark-energy logit, plus a log on each density fraction.

        Strictly positive quantities on a box that is a plain rectangle (see
        :meth:`training_space`), so a wide fraction crosses zero: measured at nsigma 3.75 on the
        SPA box, ``Omega_cdm`` reached -0.037 and ``Omega_b`` -0.0065, and a negative density is a
        non-finite background rather than a slightly wrong one -- ``params.H0``, ``Omega_m``,
        ``Omega_Lambda``, ``Omega_k``, ``background.age`` and ``efunc`` all came back nan
        together. In the log the bound is unreachable instead of an edge to trim.
        """
        transforms = super()._transforms()
        if self._omega_basis(self.space.params):
            transforms.update({name: 'log' for name in self._DENSITIES.values()})
        return transforms

    def training_space(self):
        """A plain rectangle over the image, rather than an ellipsoid on its principal axes.

        ``Omega_b`` and ``Omega_cdm`` are both ``omega / h^2``, so their image is a narrow band
        (correlation 0.85, and -0.96 and -0.84 against ``h``) whose shape says nothing about the
        posterior: it is the basis change talking. A whitened grid follows that band, and a point
        that moves ``omega_cdm`` alone at fixed ``h`` steps across it and is refused as off the
        node cloud -- measured, ``omega_cdm = 0.1511`` inside a box reaching 0.1599, which cut the
        tail off a chain. Every corner of this box is a valid background, since ``h`` is exact
        here and nothing else constrains the pair, so filling it costs only nodes, and a
        background node is 0.04 s.

        :meth:`~cosmoprimo.emulators.tools.space.Space.uncorrelated` keeps the samples while
        dropping the covariance, which is what ``measure='samples'`` needs to draw candidates from.

        Mapped here rather than through ``super()``: the base returns the user's space untouched,
        which is right for the sectors that expand in it (fourier, thermodynamics) and would leave
        this one unmapped.
        """
        return self.space.map(self.to_training, transforms=self._transforms()).uncorrelated()

    def _off_grid(self):
        # in the fraction basis the age (its 1/h divided out) and Omega_i(z) are h-free too
        if not self._omega_basis(self.space.params):
            return _OFF_GRID
        return {**_OFF_GRID, 'age': _OFF_GRID['age'] + ('h',), 'omega': _OFF_GRID['omega'] + ('h',)}


class ThermodynamicsEmulator(_RoutedSectionEmulator):
    r"""A cosmology emulated for its sound horizon (``thermodynamics.*``).

    ``rs_drag`` is set before recombination, so at fixed physical densities it depends on
    neither ``h`` nor the dark energy -- cosmoprimo returns it in Mpc/h, and that ``h`` is the
    unit -- and only ``omega_b``, ``omega_cdm`` (and a varied neutrino content) stay on the grid.
    The Eisenstein & Hu (1998) fitting formula
    (:func:`cosmoprimo.emulators.analytic.eisenstein_hu_scales`) is divided out, unit
    included, so the expansion carries only the Boltzmann code's correction to it.
    """


# ── one cosmology, one emulator per sector ────────────────────────────────────


class CosmologyEmulator(_SectionEmulator):
    r"""A cosmology serving several sectors, emulated as one emulator per sector.

    The sectors do not want the same expansion. The spectra want the
    :math:`\theta_\mathrm{MC}` basis and the optical depth on the grid; the Fourier leaves want
    ``h`` directly, never depend on ``tau_reio``, and are exactly linear in :math:`A_s`; the
    background and the sound horizon depend on fewer parameters still. One node set over the
    union of parameters would spend a ``tau`` axis on outputs that ignore it, keep ``h`` on
    the power spectrum's grid for the sake of one ``rs_drag`` leaf, and put the Cl on an ``h``
    grid no low-order polynomial survives (measured, a budget-3 box in ``h`` gave dchi2 of
    2e16).

    So each sector (see :func:`_sector`) gets an emulator of its own on a calculator of its
    own -- a clone of the cosmology registered with that sector's requirements only, so a
    background node never pays for a lensed Cl. Their leaves are named by requirement, so
    :meth:`predict` is a merge and :meth:`to_calculator` needs nothing the base does not
    already do.

    ``budget``, ``checkpoint`` and the other training options may be given per sector as a
    dict keyed by the sector names; a plain value goes to all.
    """
    sectors = {'harmonic': HarmonicEmulator, 'fourier': FourierEmulator,
               'background': BackgroundEmulator, 'thermodynamics': ThermodynamicsEmulator}

    def __init__(self, calculator, space, exact=None, **options):
        super().__init__(calculator, space, **options)
        root = self.calculator
        self._sectors = {}
        for name, cls in self.sectors.items():
            specs = [(spec_key, spec) for spec_key, spec in self.aux['ordered_specs']
                     if _sector(spec_key[0]) == name]
            if not specs:
                continue
            # A cosmology like the emulated one, declared with this sector's requirements only
            # -- through the constructor, since a build starts a cosmology from its declaration.
            # Fresh parameter nodes, not the pipeline's: training sets values on them, and the
            # pipeline's should not move. A derived parameter goes with the sector its getter is
            # read from.
            params = [param.clone() for param in root.params]
            params += [param.clone() for param in root.derived_params
                       if _sector(root._get_derived[param.name][0]) == name]
            requirements = {}
            for spec_key, spec in specs:
                requirements.setdefault(spec_key[0], []).append(
                    {**spec['static'], **{coord: spec[coord] for coord in _COORDS if coord in spec}})
            args, kwargs = root._init
            sub = type(root)(*args, **{**kwargs, 'params': VariableCollection(params), 'requirements': requirements})
            # `exact` belongs to the routed sectors (see `_RoutedSectionEmulator.__init__`)
            self._sectors[name] = cls(sub, space, **(options if name == 'harmonic' else dict(options, exact=exact)))
        if len(self._sectors) < 2:
            raise ValueError(f'{type(self).__name__} is for a cosmology serving several sectors; '
                             f'this one serves {list(self._sectors)} -- use that sector\'s '
                             f'emulator directly')

    def _per_sector(self, value, name):
        if isinstance(value, dict) and value and set(value) <= set(self._sectors):
            return value.get(name)
        return value

    @property
    def trained(self):
        return bool(self._sectors) and all(sub.trained for sub in self._sectors.values())

    def nodes(self, budget=None, **kwargs):
        """``{sector: nodes}``: each sector sizes its own run."""
        return {name: sub.nodes(budget=self._per_sector(budget, name),
                                **{key: self._per_sector(value, name) for key, value in kwargs.items()})
                for name, sub in self._sectors.items()}

    def train(self, budget=None, checkpoint=None, **kwargs):
        for name, sub in self._sectors.items():
            path = self._per_sector(checkpoint, name)
            if path is not None and not isinstance(checkpoint, dict):
                path = Path(path)
                path = path.with_name(f'{path.stem}_{name}{path.suffix}')
            sub.train(budget=self._per_sector(budget, name), checkpoint=path,
                      **{key: self._per_sector(value, name) for key, value in kwargs.items()})
        return self

    def predict(self, **params):
        out = {}
        # every sector predicts the input leaves, each correctly (see `_SectionEmulator`), and
        # each derived parameter is predicted by the one sector that holds it
        for sub in self._sectors.values():
            out.update(sub.predict(**params))
        missing = [name for name in self.children_leafnames if name not in out]
        if missing:
            raise RuntimeError(f'no sector predicts the leaves {missing}')
        return out

    def to_calculator(self, calculator=None, center=True):
        # The sectors get their calculator BEFORE the deploy, not after. The base runs no
        # constructor, so it gives the deployed object its state by predicting at the space
        # centre -- and that prediction goes through each sector's `to_training`, which reads the
        # fiducial's neutrino content (`m_ncdm`, `N_ur`, `T_cmb`) off the sector's own calculator.
        # A sector read back from a file has none, and handing them one afterwards is too late:
        # `TypeError: 'NoneType' object is not subscriptable` out of `_theta_args`.
        if calculator is not None:
            self.calculator = calculator
        source = getattr(self, 'calculator', None)
        # `_fiducial` is latched in `__post_init__`, i.e. at a build, so a calculator that has
        # only been constructed carries none -- and the harmonic sector is the one sector with no
        # fiducial of its own (the routed ones rebuild theirs from their anchors), so it reads
        # the neutrino content off this calculator and gets None. Run that setup here rather than
        # requiring the caller to have built what they hand over.
        if source is not None and getattr(source, '_fiducial', None) is None:
            args, kwargs = source._init
            source.__post_init__(*args, **kwargs)
        for sub in self._sectors.values():
            if getattr(sub, 'calculator', None) is None:
                sub.calculator = source
        return super().to_calculator(calculator=calculator, center=center)

    def __getstate__(self):
        state = super().__getstate__()
        state['sectors'] = {name: sub.__getstate__() for name, sub in self._sectors.items()}
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._sectors = {name: CalculatorEmulator.from_state(sub)
                         for name, sub in state['sectors'].items()}

    def __repr__(self):
        inner = ', '.join(f'{name}={sub!r}' for name, sub in self._sectors.items())
        return f'{type(self).__name__}({inner})'
