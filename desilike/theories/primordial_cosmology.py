"""
Primordial cosmology calculator backed by :mod:`cosmoprimo`.

Classes
-------
CosmoprimoCosmology
    Wraps a cosmoprimo Boltzmann solver.  A single :class:`~desilike.base.Calculator`
    that flips its ``_is_external`` flag per instance based on the engine: JAX-native
    engines (``'eisenstein_hu'``, …) run as pure JAX (fully jit/vmap/grad-able), while
    external solvers run via pure_callback + finite-difference derivatives.
    In both cases ``self.cosmo`` holds the current :class:`cosmoprimo.Cosmology` object.
"""

import numpy as np
import jax.numpy as jnp

from ..base import Calculator
from ..parameter import Parameter


# Engines that produce JAX-traceable outputs through cosmoprimo.Cosmology.clone.
_JAX_ENGINES = frozenset({'eisenstein_hu'})

# Parameter name conversion: desilike name → cosmoprimo clone kwarg.
_CONVERSIONS = {'logA': 'ln10^10A_s'}


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
    defaults = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200,
                    logA=3.044, n_s=0.9649, m_ncdm=0.06, w0_fld=-1., wa_fld=0.)
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
    return fiducial.clone(base='input', **kw)


class CosmoprimoCosmology(Calculator):
    r"""
    Cosmology calculator backed by :mod:`cosmoprimo`.

    A single :class:`~desilike.base.Calculator` whose ``_is_external`` flag is set per
    instance from *engine*: JAX-native engines (``'eisenstein_hu'``, …) run as pure JAX
    (``grad``/``jit``/``vmap``), external Boltzmann codes (``'camb'``, ``'class'``, …) run
    via ``pure_callback`` + finite-difference derivatives.

    In both cases the populated :class:`cosmoprimo.Cosmology` object is available as
    ``self.cosmo`` after each pipeline call and downstream theories can call
    ``get_fourier()``, ``comoving_angular_distance()``, etc.

    Uses ``cosmoprimo.Cosmology.clone(base='input', ...)`` so that engine state is
    cached from the fiducial cosmology and only the free parameters are updated.

    Parameters
    ----------
    engine : str, default='camb'
        Boltzmann solver engine.  JAX-native engines: ``'eisenstein_hu'``.
        External engines: ``'camb'``, ``'class'``, etc.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default=None
        Fiducial cosmology used both to seed parameter *default values* and as the
        base for ``clone``.  When ``None`` a default ``cosmoprimo.Cosmology(engine=engine)``
        is used.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    def __init__(self, engine='camb', fiducial=None):
        # Per-instance flag: JAX-traceable engines run as pure JAX, others as external.
        self._is_external = str(engine) not in _JAX_ENGINES
        defaults = _make_cosmo_parameters(fiducial)
        self.h = Parameter('h', value=defaults['h'],
                           prior=dict(limits=[0.3, 1.0]),
                           ref=dict(dist='norm', loc=defaults['h'], scale=0.05),
                           latex='h')
        self.omega_b = Parameter('omega_b', value=defaults['omega_b'],
                                 prior=dict(limits=[0.01, 0.04]),
                                 ref=dict(dist='norm', loc=defaults['omega_b'], scale=0.001),
                                 latex=r'\omega_b')
        self.omega_cdm = Parameter('omega_cdm', value=defaults['omega_cdm'],
                                   prior=dict(limits=[0.05, 0.3]),
                                   ref=dict(dist='norm', loc=defaults['omega_cdm'], scale=0.005),
                                   latex=r'\omega_\mathrm{cdm}')
        self.logA = Parameter('logA', value=defaults['logA'],
                              prior=dict(limits=[2., 4.]),
                              ref=dict(dist='norm', loc=defaults['logA'], scale=0.1),
                              latex=r'\ln(10^{10}A_s)')
        self.n_s = Parameter('n_s', value=defaults['n_s'],
                             prior=dict(limits=[0.7, 1.3]),
                             ref=dict(dist='norm', loc=defaults['n_s'], scale=0.01),
                             latex='n_s')
        self.m_ncdm = Parameter('m_ncdm', value=defaults['m_ncdm'],
                                fixed=True, latex=r'\sum m_\nu')
        self.w0_fld = Parameter('w0_fld', value=defaults['w0_fld'],
                                fixed=True, latex='w_0')
        self.wa_fld = Parameter('wa_fld', value=defaults['wa_fld'],
                                fixed=True, latex='w_a')

    def __post_init__(self, engine='camb', fiducial=None):
        self._engine = str(engine)
        # Build (or resolve) the fiducial once, forcing ``engine`` so that subsequent
        # per-call ``.clone(base='input', ...)`` use the requested engine (not the
        # fiducial's default, e.g. CLASS for the named 'DESI'/'Planck2018' fiducials).
        import cosmoprimo
        if fiducial is None:
            self._fiducial = cosmoprimo.Cosmology(engine=self._engine)
        else:
            self._fiducial = _get_fiducial(fiducial).clone(engine=self._engine)

    def _current_params(self, as_float=False):
        """Return dict of current parameter values in desilike-name form.

        When ``as_float=True`` all values are cast to Python float (for external engines).
        Otherwise JAX tracers are preserved (for JAX-native engines under jit/grad).
        """
        names = ('h', 'omega_b', 'omega_cdm', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld')
        if as_float:
            return {name: np.asarray(getattr(self, name).value).reshape(-1)[0].item() for name in names}
        return {name: getattr(self, name).value for name in names}

    def tree_flatten(self):
        # Expose the parameter vector that defines the cosmology as the single array
        # output.  This is what actually changes between calls; it also guarantees the
        # ExternalCalculator's pure_callback has a non-empty output (an empty output is
        # elided by XLA, so __call__ would never run).  Downstream calculators still read
        # the live ``self.cosmo`` object set as a side effect of __call__.
        names = ('h', 'omega_b', 'omega_cdm', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld')
        marker = jnp.concatenate([jnp.ravel(jnp.asarray(getattr(self, name).value)) for name in names])
        return [marker], {'engine': self._engine}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj._engine = aux['engine']
        obj._param_vector = children[0]
        return obj

    def __call__(self):
        # JAX engines: keep tracers (clone is differentiable). External engines: plain floats.
        self.cosmo = _build_cosmo(self._fiducial, self._current_params(as_float=self._is_external))
        # Return None: cosmo is a Python object exposed via self.cosmo, read directly by
        # downstream calculators; returning it would break the JAX pipeline output (and, for
        # the external path, call_kind='none' avoids the dtype-object crash).
