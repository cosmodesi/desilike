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
from ..parameter import Parameter, VariableCollection


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
    params : VariableCollection, optional
        Cosmological parameters (names in ``{h, theta_MC_100, omega_cdm, omega_b, logA,
        n_s, tau_reio, m_ncdm, N_eff, w0_fld, wa_fld, Omega_k}``).  When ``None`` they are
        built via :meth:`propose_params`.  The chosen collection is stored as :attr:`params`.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default=None
        Fiducial cosmology used both to seed parameter *default values* and as the
        base for ``clone``.  When ``None`` a default ``cosmoprimo.Cosmology(engine=engine)``
        is used.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls, *args, engine='camb', fiducial=None, **kwargs):
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

    def __getitem__(self, name):
        """Return the current value of cosmological parameter *name*."""
        return self.cosmo[name]

    def tree_flatten(self):
        # Expose the parameter vector that defines the cosmology as the single array
        # output.  This is what actually changes between calls; it also guarantees the
        # pure_callback has a non-empty output (an empty output is
        # elided by XLA, so __call__ would never run).  Downstream calculators still read
        # the live ``self.cosmo`` object set as a side effect of __call__.
        marker = jnp.concatenate([jnp.ravel(jnp.asarray(param.value)) for param in self.params.values()])
        return [marker], {'engine': self._engine}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj._engine = aux['engine']
        obj._param_vector = children[0]
        return obj

    def __call__(self):
        # JAX engines: keep tracers (clone is differentiable). External engines: plain floats.
        params = {basename: np.asarray(param.value).reshape(-1)[0].item() if self._is_external else param.value for basename, param in self.params.items()}
        self.cosmo = _build_cosmo(self._fiducial,  params)
        # Return None: cosmo is a Python object exposed via self.cosmo, read directly by
        # downstream calculators; returning it would break the JAX pipeline output (and, for
        # the external path, call_kind='none' avoids the dtype-object crash).
