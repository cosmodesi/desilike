"""Density-split galaxy clustering theory models."""

import numpy as np
from statistics import NormalDist

from desilike import jax as desilike_jax
from desilike.jax import numpy as jnp
from desilike.jax import jit

from .base import ProjectToMultipoles
from .full_shape import BaseTracerPTPowerSpectrumMultipoles, FOLPSv2PowerSpectrumMultipoles


_QUANTILES = (1, 2, 3, 4, 5)
_MODEL_TREE = 'tree'
_MODEL_1LOOP = '1-loop'
_MODEL = _MODEL_1LOOP
_MODELS = (_MODEL_TREE, _MODEL_1LOOP)
_SMOOTHING_KERNELS = ('gaussian', 'tophat')
_SMOOTHING_APMODES = ('observed', 'physical')
_COMPOSITE_INDEPENDENT_QUANTILES = (1, 2, 4, 5)
_COMPOSITE_COEFFICIENT_PREFIXES = ('c1', 'c2', 'c3')
_COMPOSITE_DERIVATIVE_PREFIXES = ('e0', 'e2', 'e4')
_COMPOSITE_PARAMETER_PREFIXES = _COMPOSITE_COEFFICIENT_PREFIXES + _COMPOSITE_DERIVATIVE_PREFIXES
_STOCHASTIC_PARAMETER_PREFIXES = ('s0qg', 's2qg', 's2muqg')
_STOCHASTIC_TREE_PREFIXES = ('s0qg',)
_ANISOTROPIC_STOCHASTIC_PREFIXES = ('s0muqg',)
_P2_MOMENT_NAMES = ('one', 'F2', 'G2', 'S2', 'Rb', 'Rf')
_P2_MOMENT_INDEX = {name: index for index, name in enumerate(_P2_MOMENT_NAMES)}


def _normalize_quantiles(quantiles):
    if np.ndim(quantiles) == 0:
        quantiles = (quantiles,)
    quantiles = tuple(int(quantile) for quantile in quantiles)
    invalid = [quantile for quantile in quantiles if quantile not in _QUANTILES]
    if invalid:
        raise ValueError('quantiles must be drawn from {}; found {}'.format(_QUANTILES, invalid))
    if len(set(quantiles)) != len(quantiles):
        raise ValueError('quantiles must be unique')
    return quantiles


def _composite_parameter_name(prefix, quantile):
    return '{}q{:d}'.format(prefix, quantile)


def _composite_parameter_names(prefixes=_COMPOSITE_PARAMETER_PREFIXES, quantiles=_COMPOSITE_INDEPENDENT_QUANTILES):
    return [_composite_parameter_name(prefix, quantile) for prefix in prefixes for quantile in quantiles]


def _composite_sum_rule_parameter(params, prefix, quantile):
    if quantile == 3:
        return -sum(params[_composite_parameter_name(prefix, q)] for q in _COMPOSITE_INDEPENDENT_QUANTILES)
    return params[_composite_parameter_name(prefix, quantile)]


def _stochastic_parameter_name(prefix, quantile):
    return '{}{:d}'.format(prefix, quantile)


def _stochastic_parameter_names(prefixes=_STOCHASTIC_PARAMETER_PREFIXES, quantiles=_COMPOSITE_INDEPENDENT_QUANTILES):
    return [_stochastic_parameter_name(prefix, quantile) for prefix in prefixes for quantile in quantiles]


def _stochastic_sum_rule_parameter(params, prefix, quantile):
    if quantile == 3:
        return -sum(params[_stochastic_parameter_name(prefix, q)] for q in _COMPOSITE_INDEPENDENT_QUANTILES)
    return params[_stochastic_parameter_name(prefix, quantile)]


def _gaussian_quantile_coefficients(nquantiles=5):
    r"""
    Return normal-ordered Gaussian selection coefficients for equal-probability bins.

    The coefficients follow Eq. (35)-style Hermite moments for an indicator
    field divided by its bin probability, with the expansion convention
    c1 delta + c2 delta**2 / 2 + c3 delta**3 / 6.
    """
    normal = NormalDist()
    edges = [-np.inf] + [normal.inv_cdf(i / nquantiles) for i in range(1, nquantiles)] + [np.inf]
    probability = 1. / nquantiles

    def phi(x):
        if not np.isfinite(x):
            return 0.
        return np.exp(-0.5 * x**2) / np.sqrt(2. * np.pi)

    coefficients = {}
    for iq, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        phil, phiu = phi(lower), phi(upper)
        h2_lower = 0. if not np.isfinite(lower) else lower * phil
        h2_upper = 0. if not np.isfinite(upper) else upper * phiu
        h3_lower = 0. if not np.isfinite(lower) else (lower**2 - 1.) * phil
        h3_upper = 0. if not np.isfinite(upper) else (upper**2 - 1.) * phiu
        coefficients[iq] = dict(c1=(phil - phiu) / probability,
                                c2=(h2_lower - h2_upper) / probability,
                                c3=(h3_lower - h3_upper) / probability)
    return coefficients


def _smoothing_window(k, radius, kernel='gaussian'):
    x = k * radius
    if kernel == 'gaussian':
        return jnp.exp(-0.5 * x**2)
    if kernel == 'tophat':
        xsafe = jnp.where(x == 0., 1., x)
        window = 3. * (jnp.sin(xsafe) - xsafe * jnp.cos(xsafe)) / xsafe**3
        return jnp.where(x == 0., 1., window)
    raise ValueError('smoothing_kernel must be one of {}; found {}'.format(_SMOOTHING_KERNELS, kernel))


def _observed_k_like(k, target):
    k = jnp.asarray(k)
    return k[(Ellipsis,) + (None,) * (target.ndim - k.ndim)]


def _smoothing_k(kobs, kap, apmode='observed'):
    if apmode == 'observed':
        return _observed_k_like(kobs, kap)
    if apmode == 'physical':
        return kap
    raise ValueError('smoothing_apmode must be one of {}; found {}'.format(_SMOOTHING_APMODES, apmode))


def _log_interp_extrap(x, xp, fp):
    x = jnp.asarray(x)
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    xmin = jnp.maximum(jnp.min(jnp.where(xp > 0., xp, jnp.inf)), 1e-300)
    logx = jnp.log(jnp.maximum(x, xmin))
    logxp = jnp.log(jnp.maximum(xp, 1e-300))
    fp = jnp.nan_to_num(fp, nan=1e-300, posinf=1e300, neginf=1e-300)
    logfp = jnp.log(jnp.maximum(fp, 1e-300))

    index = jnp.searchsorted(logxp, logx, side='right') - 1
    index = jnp.clip(index, 0, logxp.size - 2)
    x0, x1 = logxp[index], logxp[index + 1]
    y0, y1 = logfp[index], logfp[index + 1]
    y = y0 + (y1 - y0) * (logx - x0) / (x1 - x0)
    y = jnp.nan_to_num(y, nan=-100., posinf=100., neginf=-100.)
    return jnp.exp(jnp.clip(y, -100., 100.))


def _composite_loop_quadrature(nq=80, nx=32, nphi=16, qmin=1e-4, qmax=5.):
    tq, wtq = np.polynomial.legendre.leggauss(int(nq))
    xmin, xmax = np.log(float(qmin)), np.log(float(qmax))
    t = 0.5 * (xmax - xmin) * tq + 0.5 * (xmax + xmin)
    q = np.exp(t)
    wq = 0.5 * (xmax - xmin) * wtq * q**3

    x, wx = np.polynomial.legendre.leggauss(int(nx))
    phi = (np.arange(int(nphi), dtype='f8') + 0.5) * (2. * np.pi / int(nphi))
    wphi = np.full(int(nphi), 2. * np.pi / int(nphi), dtype='f8')
    return q, wq, x, wx, phi, wphi


def composite_p2_moments(k, mu, kt, pklin, growth_rate, smoothing_radius=10., smoothing_kernel='gaussian',
                         nq=80, nx=32, nphi=16, qmin=1e-4, qmax=5.):
    r"""
    Return JAX moments for the composite ``c2 P2,g`` loop.

    The dense output has shape ``(6, 3, *k.shape)``. The first axis follows
    ``_P2_MOMENT_NAMES``; the second contracts with ``b1**2``, ``b1``, and
    ``1`` after the two Z1 factors are expanded.
    """
    k = jnp.asarray(k)
    mu = jnp.asarray(mu)
    k, mu = jnp.broadcast_arrays(k, mu)
    kt = jnp.asarray(kt)
    pklin = jnp.asarray(pklin)
    f = jnp.asarray(growth_rate)
    radius = jnp.asarray(smoothing_radius)

    q, wq, x, wx, phi, wphi = (
        jnp.asarray(array) for array in _composite_loop_quadrature(nq=nq, nx=nx, nphi=nphi, qmin=qmin, qmax=qmax)
    )
    pk_q = _log_interp_extrap(q, kt, pklin)
    w_q = _smoothing_window(q, radius, kernel=smoothing_kernel)

    q = q[:, None, None]
    wq = wq[:, None, None]
    pk_q = pk_q[:, None, None]
    w_q = w_q[:, None, None]
    x = x[None, :, None]
    wx = wx[None, :, None]
    phi = phi[None, None, :]
    wphi = wphi[None, None, :]

    sqrt_1mx2 = jnp.sqrt(jnp.maximum(0., 1. - x**2))
    cosphi = jnp.cos(phi)
    measure = wq * wx * wphi / (2. * jnp.pi)**3
    q_weight = measure * w_q * pk_q
    eps = 1e-12

    def one_point(kmu):
        kk, mm = kmu
        sin_mu = jnp.sqrt(jnp.maximum(0., 1. - mm**2))
        mu1 = x * mm - sqrt_1mx2 * cosphi * sin_mu
        p2 = kk**2 + q**2 - 2. * kk * q * x
        p = jnp.sqrt(jnp.maximum(p2, eps**2))
        mu2 = (kk * mm - q * mu1) / p
        x12 = (kk * x - q) / p

        pk_p = _log_interp_extrap(p, kt, pklin)
        w_p = _smoothing_window(p, radius, kernel=smoothing_kernel)
        base = q_weight * w_p * pk_p

        q_over_p = q / p
        p_over_q = p / q
        F2 = 5. / 7. + 0.5 * x12 * (q_over_p + p_over_q) + 2. / 7. * x12**2
        G2 = 3. / 7. + 0.5 * x12 * (q_over_p + p_over_q) + 4. / 7. * x12**2
        S2 = x12**2 - 1. / 3.

        mu1sq, mu2sq = mu1**2, mu2**2
        full_shape = jnp.ones_like(mu2sq)
        z_moments = jnp.stack([
            full_shape,
            f * (mu1sq + mu2sq),
            f**2 * mu1sq * mu2sq,
        ], axis=0)
        rb_kernel = 0.5 * f * mm * kk * (mu1 / q + mu2 / p)
        rf_kernel = 0.5 * f**2 * mm * kk * (mu1 * mu2sq / q + mu2 * mu1sq / p)
        kernels = jnp.stack([
            full_shape,
            F2 * full_shape,
            f * mm**2 * G2 * full_shape,
            S2 * full_shape,
            rb_kernel,
            rf_kernel,
        ], axis=0)
        return jnp.sum(base[None, None, ...] * kernels[:, None, ...] * z_moments[None, ...], axis=(-3, -2, -1))

    flat = jnp.stack([jnp.ravel(k), jnp.ravel(mu)], axis=-1)
    moments = desilike_jax.map(one_point, flat)
    moments = jnp.moveaxis(moments, 0, -1)
    return jnp.reshape(moments, (len(_P2_MOMENT_NAMES), 3) + k.shape)


def contract_p2_moments(moments, b1, b2, bs):
    """Contract bias-independent moments into ``P2,g(k, mu)``."""
    moments = jnp.asarray(moments)

    def zcontract(name):
        moment = moments[_P2_MOMENT_INDEX[name]]
        return b1**2 * moment[0] + b1 * moment[1] + moment[2]

    return (b1 * zcontract('F2') + 0.5 * b2 * zcontract('one')
            + bs * zcontract('S2') + zcontract('G2')
            + b1 * zcontract('Rb') + zcontract('Rf'))


class DensitySplitTracerPowerSpectrumMultipoles(BaseTracerPTPowerSpectrumMultipoles):
    r"""
    FOLPS-backed density-split quantile-galaxy cross-power multipoles.

    ``model='tree'`` is the strict Kaiser composite model. ``model='1-loop'``
    uses the current composite one-loop implementation: c1 propagation of the
    deterministic FOLPS galaxy spectrum, derivative counterterms, explicit
    c2 composite loop, and query-galaxy stochastic terms. The optional
    ``qg_anisotropic_stochastic`` flag adds an empirical
    ``shotnoise * s0muqg_a * mu^2`` stochastic term. Quantile 3 coefficients
    are derived by the five-bin partition rule. ``k`` is expressed in
    ``h / Mpc`` and ``smoothing_radius`` in ``Mpc / h``.
    """

    config_fn = 'density_split.yaml'
    _pt_cls = FOLPSv2PowerSpectrumMultipoles
    _default_options = dict(prior_basis='physical_aap', tracer=None, fsat=None, sigv=None, shotnoise=1e4,
                            model=_MODEL, folps_model='FOLPSD', bias_scheme='folps', IR_resummation=True,
                            damping='lor', b3_coev=True, backend='jax', sigma8_fid=None, h_fid=None,
                            smoothing_radius=10., smoothing_kernel='gaussian', smoothing_apmode='observed',
                            qg_anisotropic_stochastic=False,
                            composite_c1_prior_scale=2., composite_c2_prior_scale=5.,
                            composite_c3_prior_scale=10., composite_derivative_prior_scale=50.,
                            composite_loop_nq=80, composite_loop_nx=32, composite_loop_nphi=16,
                            composite_loop_qmin=1e-4, composite_loop_qmax=5.)

    def initialize(self, k=None, ells=(0, 2, 4), quantiles=_QUANTILES, pt=None, template=None, z=None, mu=20,
                   smoothing_radius=10., smoothing_kernel='gaussian', smoothing_apmode='observed', model=_MODEL,
                   prior_basis='physical_aap', tracers=None, composite_c1_prior_scale=2.,
                   qg_anisotropic_stochastic=False,
                   composite_c2_prior_scale=5., composite_c3_prior_scale=10.,
                   composite_derivative_prior_scale=50., composite_loop_nq=80, composite_loop_nx=32,
                   composite_loop_nphi=16, composite_loop_qmin=1e-4, composite_loop_qmax=5., **kwargs):
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.array(k, dtype='f8')
        self.ells = tuple(ells)
        self.quantiles = _normalize_quantiles(quantiles)
        self.model = str(model).lower()
        if self.model not in _MODELS:
            raise ValueError('model must be one of {}; found {}'.format(_MODELS, model))
        self.prior_basis = str(prior_basis)
        self.smoothing_radius = float(smoothing_radius)
        if self.smoothing_radius < 0.:
            raise ValueError('smoothing_radius must be non-negative')
        self.smoothing_kernel = str(smoothing_kernel).lower()
        if self.smoothing_kernel not in _SMOOTHING_KERNELS:
            raise ValueError('smoothing_kernel must be one of {}; found {}'.format(_SMOOTHING_KERNELS, smoothing_kernel))
        self.smoothing_apmode = str(smoothing_apmode).lower()
        if self.smoothing_apmode not in _SMOOTHING_APMODES:
            raise ValueError('smoothing_apmode must be one of {}; found {}'.format(_SMOOTHING_APMODES, smoothing_apmode))
        self.qg_anisotropic_stochastic = bool(qg_anisotropic_stochastic)

        composite_prior_scales = {
            'composite_c1_prior_scale': composite_c1_prior_scale,
            'composite_c2_prior_scale': composite_c2_prior_scale,
            'composite_c3_prior_scale': composite_c3_prior_scale,
            'composite_derivative_prior_scale': composite_derivative_prior_scale,
        }
        for name, value in composite_prior_scales.items():
            if float(value) <= 0.:
                raise ValueError('{} must be positive'.format(name))
        for name, value in {'composite_loop_nq': composite_loop_nq, 'composite_loop_nx': composite_loop_nx,
                            'composite_loop_nphi': composite_loop_nphi}.items():
            if int(value) <= 0:
                raise ValueError('{} must be positive'.format(name))
        if float(composite_loop_qmin) <= 0. or float(composite_loop_qmax) <= float(composite_loop_qmin):
            raise ValueError('composite_loop_qmin and composite_loop_qmax must satisfy 0 < qmin < qmax')

        kwargs = dict(kwargs)
        kwargs.update(model=self.model, prior_basis=self.prior_basis, smoothing_radius=self.smoothing_radius,
                      smoothing_kernel=self.smoothing_kernel, smoothing_apmode=self.smoothing_apmode,
                      qg_anisotropic_stochastic=self.qg_anisotropic_stochastic,
                      composite_c1_prior_scale=float(composite_c1_prior_scale),
                      composite_c2_prior_scale=float(composite_c2_prior_scale),
                      composite_c3_prior_scale=float(composite_c3_prior_scale),
                      composite_derivative_prior_scale=float(composite_derivative_prior_scale),
                      composite_loop_nq=int(composite_loop_nq), composite_loop_nx=int(composite_loop_nx),
                      composite_loop_nphi=int(composite_loop_nphi), composite_loop_qmin=float(composite_loop_qmin),
                      composite_loop_qmax=float(composite_loop_qmax), mu=mu)
        self._set_options(k=self.k, ells=self.ells, tracers=tracers, **kwargs)
        if self.model == _MODEL_1LOOP and (self.options['backend'] != 'jax' or desilike_jax.jax is None):
            raise ValueError("density-split model '1-loop' requires backend='jax'")
        pt_kwargs = dict(kwargs)
        pt_kwargs['model'] = self.options['folps_model']
        pt_kwargs['A_full'] = False
        pt_kwargs['ells'] = self.ells
        self._set_pt(pt=pt, template=template, z=z, **pt_kwargs)
        self._set_from_pt()
        self.to_poles = ProjectToMultipoles(mu=mu, ells=self.ells)
        self.mu = self.to_poles.mu

        keep = (self._folps_parameter_names()
                + self._model_composite_parameter_names()
                + self._model_stochastic_parameter_names())
        self.init.params = self.init.params.select(basename=keep)
        self._configure_composite_parameters()
        self._configure_stochastic_parameters()

    def _configure_composite_parameters(self):
        coefficients = _gaussian_quantile_coefficients(nquantiles=5)
        coefficient_priors = {
            'c1': dict(scale=self.options['composite_c1_prior_scale'], ref_scale=0.25, fixed=False),
            'c2': dict(scale=self.options['composite_c2_prior_scale'], ref_scale=0.5, fixed=self.model == _MODEL_TREE),
            'c3': dict(scale=self.options['composite_c3_prior_scale'], ref_scale=1., fixed=True),
        }
        for prefix, setup in coefficient_priors.items():
            for quantile in _COMPOSITE_INDEPENDENT_QUANTILES:
                name = _composite_parameter_name(prefix, quantile)
                loc = coefficients[quantile][prefix]
                for param in self.init.params.select(basename=name):
                    param.update(value=loc,
                                 prior=dict(dist='norm', loc=loc, scale=setup['scale']),
                                 ref=dict(dist='norm', loc=loc, scale=setup['ref_scale']),
                                 fixed=setup['fixed'])

        for prefix in _COMPOSITE_DERIVATIVE_PREFIXES:
            for quantile in _COMPOSITE_INDEPENDENT_QUANTILES:
                for param in self.init.params.select(basename=_composite_parameter_name(prefix, quantile)):
                    param.update(value=0.,
                                 prior=dict(dist='norm', loc=0., scale=self.options['composite_derivative_prior_scale']),
                                 ref=dict(dist='norm', loc=0., scale=5.),
                                 fixed=self.model == _MODEL_TREE)

    def _configure_stochastic_parameters(self):
        stochastic_priors = {
            's0qg': dict(scale=2., ref_scale=1.),
            's0muqg': dict(scale=2., ref_scale=1.),
            's2qg': dict(scale=50., ref_scale=5.),
            's2muqg': dict(scale=50., ref_scale=5.),
        }
        for prefix, setup in stochastic_priors.items():
            for quantile in _COMPOSITE_INDEPENDENT_QUANTILES:
                for param in self.init.params.select(basename=_stochastic_parameter_name(prefix, quantile)):
                    param.update(value=0.,
                                 prior=dict(dist='norm', loc=0., scale=setup['scale']),
                                 ref=dict(dist='norm', loc=0., scale=setup['ref_scale']),
                                 fixed=False)

    def _folps_parameter_names(self):
        if self.prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
            if self.model == _MODEL_TREE:
                return ['b1p']
            return ['b1p', 'b2p', 'bsp', 'b3p', 'alpha0p', 'alpha2p', 'alpha4p', 'ctp', 'X_FoG_pp']
        if self.prior_basis == 'standard':
            if self.model == _MODEL_TREE:
                return ['b1']
            return ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'ct', 'X_FoG_p']
        raise ValueError("Unknown prior_basis='{}'.".format(self.prior_basis))

    def _model_composite_parameter_names(self):
        if self.model == _MODEL_TREE:
            return _composite_parameter_names(prefixes=('c1',))
        return _composite_parameter_names()

    def _model_stochastic_parameter_names(self):
        prefixes = list(_STOCHASTIC_TREE_PREFIXES if self.model == _MODEL_TREE else _STOCHASTIC_PARAMETER_PREFIXES)
        if self.qg_anisotropic_stochastic:
            prefixes[1:1] = list(_ANISOTROPIC_STOCHASTIC_PREFIXES)
        return _stochastic_parameter_names(prefixes=tuple(prefixes))

    def _standard_folps_pars(self, params):
        b1 = params['b1']
        pars = [params.get(name, 0.) for name in ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'ct']]
        return b1, pars + [0., 0., params.get('X_FoG_p', 0.)]

    def _physical_aap_folps_pars(self, params):
        sigma8 = self.pt.sigma8
        f = self.pt.fsigma8 / sigma8
        qpar, qper = self.pt.qpar, self.pt.qper
        A_AP = 1. / (qper**2 * qpar)
        sqrt_A_AP = A_AP**0.5
        b1L = params['b1p'] / sigma8 / sqrt_A_AP - 1.
        b2L = params.get('b2p', 0.) / sigma8**2 / sqrt_A_AP
        bK2 = params.get('bsp', 0.) / sigma8**2 / sqrt_A_AP
        btd = params.get('b3p', 0.) / A_AP / sigma8**4
        b1E = 1. + b1L
        b2E = b2L
        if self.options['b3_coev']:
            btd = 23. / 42. * (b1E - 1.)
        bsE = 2. * bK2
        b3E = 64. / 105. * (-5. / 4. * bsE - btd)
        ctildeE = params.get('ctp', 0.)
        a0t, a2t, a4t = (params.get(name, 0.) / A_AP / sigma8**2 for name in ['alpha0p', 'alpha2p', 'alpha4p'])
        alpha0 = b1E**2 * a0t
        alpha2 = b1E * f * (a0t + a2t)
        alpha4 = f**2 * a2t + b1E * f * a4t
        pars = [b1E, b2E, bsE, b3E, alpha0, alpha2, alpha4, ctildeE]
        return b1E, pars + [0., 0., params.get('X_FoG_pp', 0.)]

    def _folps_pars(self, params):
        if self.prior_basis == 'standard':
            return self._standard_folps_pars(params)
        if self.prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
            return self._physical_aap_folps_pars(params)
        raise ValueError("Unknown prior_basis='{}'.".format(self.prior_basis))

    def _folps_pkmu(self, pars, shotnoise=None):
        import folps as folpsv2
        table = (self.pt.kt, *self.pt.pt.table, *self.pt.pt.scalars)
        table_now = (self.pt.kt, *self.pt.pt.table_now, *self.pt.pt.scalars_now)
        if shotnoise is None:
            shotnoise = 1. / self.nbar
        pars = list(pars[:-1]) + [shotnoise, pars[-1]]
        ncols = len(table)
        import folps.folps as _folps_module
        _folps_module.A_full_status = getattr(self.pt.pt, 'A_full', False)
        _folps_module.use_TNS_model_status = getattr(self.pt.pt, 'remove_DeltaP', False)
        if getattr(self, '_get_folps_pkmu', None) is None:

            def _get_folps_pkmu(pars, bias_scheme, damping, *table):
                folps_rsdmps_class = folpsv2.RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
                pars = folps_rsdmps_class.set_bias_scheme(pars=pars, bias_scheme=bias_scheme)
                return folps_rsdmps_class.get_rsd_pkmu(self.pt.pt.kap, self.pt.pt.muap, pars, table[:ncols], table[ncols:],
                                                       IR_resummation=True, damping=damping)

            self._get_folps_pkmu = jit(static_argnums=(1, 2))(_get_folps_pkmu) if self.options['backend'] == 'jax' else _get_folps_pkmu
        array = jnp.array(pars) if self.options['backend'] == 'jax' else np.array(pars)
        return self._get_folps_pkmu(array, self.options['bias_scheme'], self.options['damping'], *table, *table_now)

    def _linear_matter_pk(self):
        import folps as folpsv2
        table = (self.pt.kt, *self.pt.pt.table, *self.pt.pt.scalars)
        if getattr(self, '_get_linear_matter_pk', None) is None:

            def _get_linear_matter_pk(kap, *table):
                folps_rsdmps_class = folpsv2.RSDMultipolesPowerSpectrumCalculator(model='FOLPSD')
                return folps_rsdmps_class.interp_table(kap, table, False)[0]

            self._get_linear_matter_pk = jit(_get_linear_matter_pk) if self.options['backend'] == 'jax' else _get_linear_matter_pk
        return self._get_linear_matter_pk(self.pt.pt.kap, *table)

    def _composite_p2_moments(self):
        if getattr(self, '_get_composite_p2_moments', None) is None:
            self._get_composite_p2_moments = jit(static_argnames=('smoothing_kernel', 'nq', 'nx', 'nphi', 'qmin', 'qmax'))(composite_p2_moments)
        return self._get_composite_p2_moments(
            self.pt.pt.kap, self.pt.pt.muap, self.pt.kt, self.pt.pt.table[0],
            self.pt.fsigma8 / self.pt.sigma8, self.smoothing_radius,
            smoothing_kernel=self.smoothing_kernel,
            nq=self.options['composite_loop_nq'],
            nx=self.options['composite_loop_nx'],
            nphi=self.options['composite_loop_nphi'],
            qmin=self.options['composite_loop_qmin'],
            qmax=self.options['composite_loop_qmax'])

    def calculate(self, **params):
        self._set_from_pt()
        self.nbar = 1e-4
        jac, kap, muap = self.pt.pt.jac, self.pt.pt.kap, self.pt.pt.muap
        f = self.pt.fsigma8 / self.pt.sigma8
        mu2 = muap**2
        ksmooth = _smoothing_k(self.k, kap, apmode=self.smoothing_apmode)
        window = _smoothing_window(ksmooth, self.smoothing_radius, kernel=self.smoothing_kernel)
        b1, pars = self._folps_pars(params)
        pgg_lin = (b1 + f * mu2)**2 * self._linear_matter_pk()

        if self.model == _MODEL_TREE:
            pgg_base = pgg_lin
            p2g = None
        else:
            pgg_base = self._folps_pkmu(pars, shotnoise=0.)
            p2g = contract_p2_moments(self._composite_p2_moments(), pars[0], pars[1], pars[2])

        power = []
        shotnoise = self.options['shotnoise']
        for quantile in self.quantiles:
            c1q = _composite_sum_rule_parameter(params, 'c1', quantile)
            pkmu = c1q * window * pgg_base
            stochastic_pkmu = _stochastic_sum_rule_parameter(params, 's0qg', quantile)
            if self.qg_anisotropic_stochastic:
                stochastic_pkmu = stochastic_pkmu + _stochastic_sum_rule_parameter(params, 's0muqg', quantile) * mu2
            if self.model == _MODEL_1LOOP:
                c2q = _composite_sum_rule_parameter(params, 'c2', quantile)
                e0q = _composite_sum_rule_parameter(params, 'e0', quantile)
                e2q = _composite_sum_rule_parameter(params, 'e2', quantile)
                e4q = _composite_sum_rule_parameter(params, 'e4', quantile)
                s2qg = _stochastic_sum_rule_parameter(params, 's2qg', quantile)
                s2muqg = _stochastic_sum_rule_parameter(params, 's2muqg', quantile)
                pkmu = pkmu + c2q * p2g
                pkmu = pkmu - 2. * kap**2 * window * (e0q + e2q * mu2 + e4q * mu2**2) * pgg_lin
                stochastic_pkmu = stochastic_pkmu + kap**2 * (s2qg + s2muqg * mu2)
            pkmu = pkmu + shotnoise * stochastic_pkmu
            power.append(self.to_poles(jac * pkmu))
        self.power = jnp.stack(power, axis=0)

    def get(self):
        return self.power

    def __getstate__(self):
        state = self.to_poles.__getstate__()
        for name in ['k', 'z', 'ells', 'quantiles', 'smoothing_radius', 'smoothing_kernel',
                     'smoothing_apmode', 'model', 'power']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state
