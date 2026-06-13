.. _developer-calculator:

######################
Writing a Calculator
######################

This page explains how to write a :class:`~desilike.base.Calculator` — the building
block of every **desilike** pipeline — and walks through two worked examples:
a minimal toy pipeline and a real galaxy-clustering theory model.


Overview
========

Every computation step in **desilike** is a *calculator*: a Python object whose
``__init__`` wires the dependency graph and whose ``__call__`` performs the
computation.  Calculators are linked by passing them as constructor arguments to
each other; :func:`~desilike.base.compile` then traces the graph, assigns an
evaluation order, and returns a :class:`~desilike.base.CompiledGraph` that is
fully compatible with JAX transforms (``jit``, ``grad``, ``vmap``).

Every calculator inherits from :class:`~desilike.base.Calculator`.  Two execution
modes are available, controlled by the class attribute ``_is_external``:

* ``_is_external = False`` (default) — **JAX-native mode**.  ``__call__`` must use
  JAX ops; automatic differentiation works natively.
* ``_is_external = True`` — **non-JAX (external) mode**.  ``__call__`` may use
  arbitrary Python/NumPy.  JAX wraps the call via ``pure_callback`` and computes
  gradients by finite differences (step and accuracy order configurable per
  parameter via ``fd_eps`` / ``fd_acc``).

Both modes follow the same three-method lifecycle.


The lifecycle
=============

Every calculator subclass implements up to four methods; the last two are part
of the JAX pytree protocol and are always required.

``__init__(self, *args, **kwargs)``
-----------------------------------

**Create and wire all nodes here.**

*Nodes* are :class:`~desilike.parameter.Parameter` instances and other calculators
passed as dependencies.  They must be stored as **public** (no leading underscore)
attributes so that :func:`~desilike.base.build_graph` can find them by scanning
``self.__dict__``.  Nodes nested inside lists, tuples, or dicts are also
discovered automatically.

You may call ``dep.update(...)`` on a calculator dependency here — ``update`` is
the mechanism for a parent to configure its children before the graph is compiled.
After :func:`~desilike.base.compile` is called, the graph is frozen; use
:meth:`~desilike.base.Calculator.clone` or reconstruct instead.

Non-node scalar/array data that is **not** a ``Parameter`` (e.g. a wavenumber
grid, a fixed scale factor) may also be stored in ``__init__`` on plain
(underscore-prefixed or not) attributes, but it will not be tracked as a
dependency.

``__post_init__(self, *args, **kwargs)``
-----------------------------------------

**Non-node setup only.**

Called by :func:`~desilike.base.compile` in dependency order (a dependency's
``__post_init__`` runs before its dependents').  Use this for heavy NumPy
configuration, helper objects, FFTLog plans, etc.  The same ``*args, **kwargs``
that were passed to ``__init__`` are forwarded.

**Must not** create new ``Parameter`` instances or ``Calculator`` dependencies;
those must exist after ``__init__`` so the graph can be built.

``__call__(self)``
-------------------

**Pure computation.**

Read parameter values via the :class:`~desilike.parameter.Parameter` objects
stored in ``__init__`` — they act as plain numbers/arrays inside JAX expressions
(``__jax_array__`` is implemented) so ``self.A * k ** self.ns`` works directly.
Read upstream outputs from dependency attributes set by their own ``__call__``.

Store all outputs as attributes (e.g. ``self.poles``) so downstream calculators
can read them.  Return the primary output array, a tuple of arrays, ``None``
(when outputs live only in attributes), or ``self``.

JAX pytree protocol
===================

Every calculator must register itself as a JAX pytree.  This is done automatically
when :func:`~desilike.base.Calculator.__init_subclass__` is called (i.e. at class
definition time), using the two methods below.

``tree_flatten(self) → (children, aux)``
-----------------------------------------

``children`` is a **list of JAX arrays** — all output attributes that downstream
calculators or JAX transforms need to differentiate through.  ``aux`` is any
static, non-JAX data that ``tree_unflatten`` needs to reconstruct a shell of the
object (e.g. ``k`` wavenumbers, ``ells``).

``tree_unflatten(cls, aux, children) → instance``
--------------------------------------------------

A classmethod that reconstructs a *shell* of the object: set the output
attributes from ``children`` (and any static data from ``aux``), but do **not**
restore constructor arguments or dependencies.  The shell is used by JAX to pass
results between compiled stages.

.. code-block:: python

    def tree_flatten(self):
        return [self.poles], None           # children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.poles = children[0]
        return obj


Parameters
==========

:class:`~desilike.parameter.Parameter` is the named, prior-carrying leaf of every
pipeline.  Key constructor arguments:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Argument
     - Meaning
   * - ``name``
     - Parameter name, optionally namespace-prefixed (``'galaxy.b1'``).
   * - ``value``
     - Default / starting value.
   * - ``prior``
     - Prior distribution as a dict: ``{'dist': 'norm', 'loc': 0., 'scale': 1.}``
       or ``{'limits': [0., 4.]}`` for a uniform prior.
   * - ``ref``
     - Reference (proposal) distribution for initial samples. Defaults to prior.
   * - ``latex``
     - LaTeX string (without ``$``).
   * - ``fixed``
     - ``True`` to fix the parameter (excluded from sampling/profiling).
   * - ``derived``
     - ``False`` (default), ``True`` (computed by the pipeline),
       ``'marg'`` / ``'best'`` (analytic marginalization / profile likelihood),
       or a string expression ``'{omega_m} * {h}**2'``.
   * - ``fd_eps``
     - Finite-difference step for external (``_is_external=True``) gradients.
   * - ``fd_acc``
     - Stencil accuracy order (2, 4, …) for external gradients.

Inside ``__call__`` a :class:`~desilike.parameter.Parameter` acts as a number or
array through ``__jax_array__``; ``self.b1 ** 2`` and ``self.b1.value ** 2`` are
equivalent.


Minimal example
===============

The following reproduces the test pipeline from ``desilike/tests/test_base.py``:
a cosmology (external, uses NumPy), a power spectrum (JAX), and a Gaussian
chi-squared likelihood (JAX).

.. code-block:: python

    import numpy as np
    import jax
    import jax.numpy as jnp

    from desilike.base import Calculator, compile
    from desilike.parameter import Parameter

    K = np.linspace(0.01, 0.3, 30)


    # ── step 1: a non-JAX cosmology ──────────────────────────────────────────────

    class Cosmology(Calculator):
        """Growth factor D = omega_m^0.55 / (1+z)."""
        _is_external = True   # enables pure_callback + FD gradients

        def __init__(self, omega_m, z):
            # Store Parameter nodes as public attributes.
            self.omega_m = omega_m
            self.z = z

        def __call__(self):
            # Arbitrary Python/NumPy — JAX wraps it via pure_callback.
            self.growth_factor = np.array(self.omega_m ** 0.55 / (1.0 + self.z))
            return self

        def tree_flatten(self):
            # Expose the output arrays that downstream calculators need.
            return [self.growth_factor], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.growth_factor = children[0]
            return obj


    # ── step 2: a JAX-native power spectrum ─────────────────────────────────────

    class PowerSpectrum(Calculator):
        """P(k) = A * k^ns * D^2."""

        def __init__(self, cosmo, A, ns):
            # 'cosmo' is a Calculator dep; A and ns are Parameter nodes.
            self.cosmo = cosmo
            self.A = A
            self.ns = ns

        def __call__(self):
            # Parameter objects act as arrays inside JAX expressions.
            self.pk = self.A * jnp.array(K) ** self.ns * self.cosmo.growth_factor ** 2
            return self.pk

        def tree_flatten(self):
            return [self.pk], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.pk = children[0]
            return obj


    # ── step 3: a Gaussian chi-squared likelihood ────────────────────────────────

    class GaussianChi2(Calculator):

        def __init__(self, spectrum, data, sigma=0.1):
            self.spectrum = spectrum          # Calculator dep (public attribute)
            self._data = jnp.asarray(data)   # static data (not a node)
            self._sigma = sigma

        def __call__(self):
            self.loglikelihood = -0.5 * jnp.sum(
                ((self.spectrum.pk - self._data) / self._sigma) ** 2)
            return self.loglikelihood

        def tree_flatten(self):
            return [self.loglikelihood], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.loglikelihood = children[0]
            return obj


    # ── assemble and compile ─────────────────────────────────────────────────────

    omega_m = Parameter('omega_m', value=0.3, prior={'dist': 'norm', 'loc': 0.3, 'scale': 0.01})
    z       = Parameter('z',       value=0.5, fixed=True)
    A       = Parameter('A',       value=1.0, prior={'limits': [0.5, 2.0]})
    ns      = Parameter('ns',      value=0.96, prior={'limits': [0.8, 1.2]})

    data = np.random.default_rng(0).normal(1.0, 0.1, len(K))
    cosmo      = Cosmology(omega_m=omega_m, z=z)
    spectrum   = PowerSpectrum(cosmo=cosmo, A=A, ns=ns)
    likelihood = GaussianChi2(spectrum=spectrum, data=data)

    pipe = compile(likelihood)

    # Evaluate eagerly
    params = {p.name: p.value for p in pipe.params}
    logL = float(pipe(params))

    # Gradient via JAX (finite-difference through Cosmology, exact through the rest)
    grad = jax.grad(pipe)(params)

    # JIT-compiled call
    logL_jit = float(jax.jit(pipe)(params))

    # Batch over omega_m
    batch = {**params, 'omega_m': jnp.linspace(0.25, 0.35, 8)}
    logL_batch = jax.vmap(pipe)({'omega_m': batch['omega_m'],
                                  'A': jnp.full(8, params['A']),
                                  'ns': jnp.full(8, params['ns'])})

Post-compile, outputs are available on the calculator objects:

.. code-block:: python

    pipe(params)
    print(spectrum.pk)          # jax array, shape (30,)
    print(cosmo.growth_factor)  # numpy scalar


Galaxy-clustering theory: ``KaiserTracerSpectrum2Poles``
=========================================================

The following is a simplified version of
:class:`~desilike.theories.galaxy_clustering.KaiserTracerSpectrum2Poles`
from :mod:`desilike.theories.galaxy_clustering.full_shape`.
It illustrates the patterns used by real theory calculators:

* a separate PT sub-calculator (``KaiserPTSpectrum2Poles``) wired as a dep
* ``update()`` to configure the dep from the tracer constructor
* ``__post_init__`` for a non-node scalar
* ``tree_flatten`` / ``tree_unflatten`` for the multipole output

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp

    from desilike.base import Calculator, compile
    from desilike.parameter import Parameter
    from desilike.theories.galaxy_clustering import (
        KaiserPTSpectrum2Poles, ShapeFitSpectrum2Template)


    class MyTracerSpectrum2Poles(Calculator):
        r"""
        Kaiser tracer power spectrum multipoles (simplified example).

        :math:`P_\ell(k) = (b_1^2 P_{dd} + 2 b_1 P_{d\theta} + P_{\theta\theta})_\ell + s_{n,0} \cdot 10^4 \,\delta_{\ell 0}`

        Parameters
        ----------
        k : array, default=None
            Output wavenumbers [h/Mpc].
        pt : KaiserPTSpectrum2Poles, default=None
            Matter PT module. Created with default settings if None.
        ells : tuple of int, default=(0, 2, 4)
            Multipole orders.
        template : template calculator, default=None
            Power spectrum template; forwarded to the PT module when given.
        shotnoise : float, default=1e4
            Shot-noise scale [h/Mpc]^3. ``sn0`` is in units of this.
        """

        def __init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, shotnoise=1e4):
            # ── Parameters (nodes) ────────────────────────────────────────────
            # Store as public attributes so build_graph picks them up.
            self.b1  = Parameter('b1',  value=1.5, prior={'limits': [0., 4.]},
                                 ref={'limits': [1., 2.]}, latex='b_1')
            self.sn0 = Parameter('sn0', value=0.,  prior={'dist': 'norm', 'loc': 0., 'scale': 1000.},
                                 ref={'dist': 'norm', 'loc': 0., 'scale': 0.1}, latex='s_{n,0}')

            # ── Static config (not nodes) ─────────────────────────────────────
            if k is None:
                k = np.linspace(0.01, 0.2, 101)
            self.k    = np.asarray(k, dtype='f8')
            self.ells = tuple(ells)

            # ── Calculator dep ────────────────────────────────────────────────
            if pt is None:
                pt = KaiserPTSpectrum2Poles()
            self.pt = pt
            # update() configures the dep while we are still inside __init__.
            self.pt.update(k=self.k, ells=self.ells)
            if template is not None:
                self.pt.update(template=template)

        def __post_init__(self, k=None, pt=None, ells=(0, 2, 4), template=None, shotnoise=1e4):
            # Non-node, non-JAX config goes here.
            # __post_init__ runs at compile() after all deps' __post_init__ have run,
            # so self.pt is fully initialised and readable.
            self._shotnoise = float(shotnoise)

        def __call__(self):
            # Shot-noise: only in the monopole (ell == 0).
            sn_mask = jnp.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None]
            sn = sn_mask * self.sn0.value * self._shotnoise   # or: self.sn0 * self._shotnoise

            # Read PT outputs set by KaiserPTSpectrum2Poles.__call__.
            pk_dd = self.pt.table['pk_dd']   # shape (n_ell, n_k)
            pk_dt = self.pt.table['pk_dt']
            pk_tt = self.pt.table['pk_tt']

            # Linear bias model.
            self.poles = self.b1 ** 2 * pk_dd + 2. * self.b1 * pk_dt + pk_tt + sn
            return self.poles

        def tree_flatten(self):
            # children: JAX arrays that downstream nodes or JAX transforms need.
            # aux: static data needed by tree_unflatten to rebuild the shell.
            return [self.poles], None

        @classmethod
        def tree_unflatten(cls, aux, children):
            obj = object.__new__(cls)
            obj.poles = children[0]
            return obj


Usage:

.. code-block:: python

    import numpy as np
    from desilike.base import compile
    from desilike.theories.galaxy_clustering import ShapeFitSpectrum2Template

    k    = np.linspace(0.01, 0.2, 101)
    ells = (0, 2)

    template = ShapeFitSpectrum2Template(z=0.8)
    theory   = MyTracerSpectrum2Poles(k=k, ells=ells, template=template)
    pipe     = compile(theory)

    params = {p.name: p.value for p in pipe.params}
    poles  = pipe(params)   # shape (n_ell, n_k)
    print(theory.poles)     # same array, available after the call


Checklist
=========

When writing a new calculator, verify that:

* Every :class:`~desilike.parameter.Parameter` and every calculator dependency is
  stored as a **public attribute** in ``__init__`` (not ``__post_init__``).
* ``__init__`` calls ``dep.update(...)`` for any configuration passed to a child.
* ``__post_init__`` does **not** create new nodes or call ``update()``.
* ``tree_flatten`` returns **all** arrays that downstream nodes or JAX transforms
  must differentiate through.
* ``tree_unflatten`` does **not** call ``__init__``; it sets output attributes
  directly on a bare instance (``object.__new__(cls)``).
* Static arrays (wavenumber grids, masks) that never change go in ``aux`` (the
  second return value of ``tree_flatten``); JAX-differentiable outputs go in
  ``children`` (the first).
