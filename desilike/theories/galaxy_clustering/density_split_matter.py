"""Kaiser matter and polynomial density-selection spectra, evaluated with JAX.

The quadratic operator is O2 = (delta_R**2 - <delta_R**2>) / 2, where
delta_R is the Gaussian-smoothed redshift-space matter field. Its mean is
subtracted, but neither its linear response nor its constant power is removed.
O3 = (delta_R**3 - <delta_R**3>) / 6 follows the same convention.
This is a selection extension, not a loop-complete matter/RSD model.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ...base import Calculator
from ...parameter import Parameter, VariableCollection
from ..primordial_cosmology import CosmoprimoCosmology


_QUANTILES = (1, 2, 3, 4, 5)
_INDEPENDENT_QUANTILES = (1, 2, 4, 5)


def _matter_z2(a, b, cosine, mua, mub, f):
    """Symmetric second-order matter RSD kernel (standard EdS F2 and G2)."""
    f2 = 5. / 7. + 0.5 * cosine * (a / b + b / a) + 2. / 7. * cosine**2
    g2 = 3. / 7. + 0.5 * cosine * (a / b + b / a) + 4. / 7. * cosine**2
    los_sum = a * mua + b * mub
    sum_squared = jnp.maximum(a*a + b*b + 2. * a*b*cosine, 1.e-30)
    mapping = 0.5 * f * los_sum * (
        mua / a * (1. + f * mub**2) + mub / b * (1. + f * mua**2))
    return f2 + f * los_sum**2 / sum_squared * g2 + mapping


def _loop_quadrature(nq, nx, nphi, qmin, qmax):
    """Nodes and d^3q/(2 pi)^3 weights, using log(q) and cos(k,q)."""
    for name, count in [('nq', nq), ('nx', nx), ('nphi', nphi)]:
        if int(count) != count or count < 2:
            raise ValueError(f'{name} must be an integer >= 2')
    if not np.isfinite([qmin, qmax]).all() or not 0. < qmin < qmax:
        raise ValueError('require finite 0 < qmin < qmax')
    nq, nx, nphi = int(nq), int(nx), int(nphi)
    t, wt = np.polynomial.legendre.leggauss(nq)
    span = np.log(qmax / qmin) / 2.
    q = np.exp(np.log(qmin) + span * (t + 1.))[:, None, None]
    x, wx = np.polynomial.legendre.leggauss(nx)
    x = x[None, :, None]
    phi = (np.arange(nphi) + 0.5) * (2. * np.pi / nphi)
    weights = span * wt[:, None, None] * q**3 * wx[None, :, None]
    weights = weights / ((2. * np.pi)**2 * nphi)
    return tuple(jnp.asarray(v) for v in (q, x, np.cos(phi)[None, None, :], weights))


def _quadratic_spectra(k, mu, pk_grid, pklin, f, radius, quadrature):
    """Return (P_O2,m, P_O2,O2) on broadcast (k, mu) coordinates.

    All three cyclic tree-bispectrum terms are summed before integration.
    The 1/2 in O2 cancels the bispectrum's factor 2; Wick's two contractions
    leave a factor 1/2 in its auto-spectrum. No shot noise is included.
    """
    q, x, cosphi, weights = quadrature
    log_grid, log_power = jnp.log(pk_grid), jnp.log(pklin)

    def power(kk):
        return jnp.exp(jnp.interp(jnp.log(jnp.maximum(kk, pk_grid[0])), log_grid, log_power))

    pq = power(q)
    wq = jnp.exp(-0.5 * (q * radius)**2)

    def one_mode(kmu):
        kk, mm = kmu
        p = jnp.sqrt(jnp.maximum(kk**2 + q**2 - 2. * kk*q*x, 1.e-30))
        muq = x * mm + jnp.sqrt(jnp.maximum((1. - x*x) * (1. - mm*mm), 0.)) * cosphi
        mup = (kk * mm - q * muq) / p
        qp = (kk*x - q) / p
        p_minus_k = (q*x - kk) / p
        zq, zp, zk = 1. + f * muq**2, 1. + f * mup**2, 1. + f * mm**2
        pp, pk = power(p), power(kk)
        wp = jnp.exp(-0.5 * (p * radius)**2)
        bispectrum_half = (
            zq * zp * _matter_z2(q, p, qp, muq, mup, f) * pq * pp
            + zp * zk * _matter_z2(p, kk, p_minus_k, mup, -mm, f) * pp * pk
            + zk * zq * _matter_z2(kk, q, -x, -mm, muq, f) * pk * pq)
        p2m = jnp.sum(weights * wq * wp * bispectrum_half)
        p22 = 0.5 * jnp.sum(weights * (wq * wp * zq * zp)**2 * pq * pp)
        return jnp.stack([p2m, p22])

    k, mu = jnp.broadcast_arrays(k, mu)
    modes = jnp.stack([k.ravel(), mu.ravel()], axis=-1)
    result = jax.lax.map(one_mode, modes)
    return jnp.moveaxis(result, -1, 0).reshape((2,) + k.shape)


def _norm2(a):
    return jnp.maximum(jnp.sum(a * a, axis=-1), 1.e-30)


def _fg2(a, b):
    aa, bb, ab = _norm2(a), _norm2(b), jnp.sum(a * b, axis=-1)
    common = .5 * ab * (1. / aa + 1. / bb)
    return 5./7. + common + 2./7.*ab**2/(aa*bb), 3./7. + common + 4./7.*ab**2/(aa*bb)


def _fg3(a, b, c):
    """Symmetrized EdS recursion, including both 1+2 and 2+1 partitions."""
    f3, g3 = 0., 0.
    for u, v, w in ((a, b, c), (b, a, c), (c, a, b)):
        vw = v + w
        uv = jnp.sum(u * vw, axis=-1)
        alpha = 1. + uv / _norm2(u)
        reverse = 1. + uv / _norm2(vw)
        beta = _norm2(u + vw) * uv / (2. * _norm2(u) * _norm2(vw))
        f2, g2 = _fg2(v, w)
        f3 += 7.*alpha*f2 + (7.*reverse + 4.*beta)*g2
        g3 += 3.*alpha*f2 + (3.*reverse + 12.*beta)*g2
    return f3 / 54., g3 / 54.


def _z1_vector(a, f):
    return 1. + f * a[..., 2]**2 / _norm2(a)


def _z2_vector(a, b, f):
    f2, g2 = _fg2(a, b)
    total = a + b
    ua, ub = a[..., 2] / _norm2(a), b[..., 2] / _norm2(b)
    return (f2 + f*total[..., 2]**2/_norm2(total)*g2
            + .5*f*total[..., 2]*(ua + ub) + .5*f**2*total[..., 2]**2*ua*ub)


def _matter_z3(a, b, c, f):
    """Matter Z3 from exp(-ik_z u_z)(1+delta); LOS is z, no bias terms."""
    total = a + b + c
    kz = total[..., 2]
    f3, g3 = _fg3(a, b, c)
    result = f3 + f*kz**2/_norm2(total)*g3
    for u, v, w in ((a, b, c), (b, a, c), (c, a, b)):
        f2, g2 = _fg2(v, w)
        u1 = u[..., 2] / _norm2(u)
        u2 = (v + w)[..., 2] / _norm2(v + w) * g2
        result += (f*kz*(u1*f2 + u2) + f**2*kz**2*u1*u2) / 3.
        result += f**2*kz**2/6. * v[..., 2]*w[..., 2]/(_norm2(v)*_norm2(w))
    return result + f**3*kz**3/6. * a[..., 2]*b[..., 2]*c[..., 2]/(_norm2(a)*_norm2(b)*_norm2(c))


def _linear_power(k, grid, power):
    value = jnp.exp(jnp.interp(jnp.log(jnp.maximum(k, grid[0])), jnp.log(grid), jnp.log(power)))
    return jnp.where(k > 0., value, 0.)


def _matter_trispectrum(vectors, grid, power, f):
    """Connected tree T3111 + T2211 for four closing external momenta.

    A nonlinear leg at k_i receives -k_j from a linear external leg j.
    Enumerating those contractions directly avoids sign ambiguities in T2211.
    """
    from itertools import combinations

    p = [_linear_power(jnp.sqrt(_norm2(v)), grid, power) for v in vectors]
    z = [_z1_vector(v, f) for v in vectors]
    result = 0.
    for d in range(4):
        a, b, c = [i for i in range(4) if i != d]
        result += 6.*z[a]*z[b]*z[c]*p[a]*p[b]*p[c]*_matter_z3(vectors[a], vectors[b], vectors[c], f)
    for a, b in combinations(range(4), 2):
        for c in (i for i in range(4) if i not in (a, b)):
            internal = vectors[a] + vectors[c]
            result += (4.*z[a]*z[b]*p[a]*p[b]
                       * _linear_power(jnp.sqrt(jnp.sum(internal*internal, axis=-1)), grid, power)
                       * _z2_vector(-vectors[a], internal, f)
                       * _z2_vector(-vectors[b], -internal, f))
    return result


def _sobol_normals(log2_nodes, seed, chunk_size):
    """Fixed Gaussian importance nodes; generation is outside traced evaluation."""
    from scipy.special import ndtri
    from scipy.stats import qmc

    for name, value, minimum in [('log2_nodes', log2_nodes, 4), ('chunk_size', chunk_size, 1), ('seed', seed, 0)]:
        if int(value) != value or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    count = 2**int(log2_nodes)
    if count % chunk_size:
        raise ValueError('chunk_size must divide the Sobol node count')
    uniform = qmc.Sobol(6, scramble=True, seed=int(seed)).random_base2(int(log2_nodes))
    return ndtri(uniform).reshape(-1, int(chunk_size), 2, 3)


def _trispectrum_integral(k, mu, grid, power, f, radius, normals):
    """Integral W(q) W(p) W(k-q-p) T / 6 with Gaussian importance sampling."""
    k, mu = jnp.broadcast_arrays(k, mu)

    def one_mode(kmu):
        kk, mm = kmu
        external = jnp.array([0., 0., kk])
        st = jnp.sqrt(jnp.maximum(1. - mm*mm, 0.))

        def rotate(v):
            return jnp.stack([mm*v[..., 0] - st*v[..., 2], v[..., 1],
                              st*v[..., 0] + mm*v[..., 2]], axis=-1)

        def accumulate(value, sample):
            q = external/3. + jnp.sqrt(2./3.)/radius * sample[:, 0]
            p = external/3. + (-sample[:, 0]/jnp.sqrt(6.) + sample[:, 1]/jnp.sqrt(2.))/radius
            r = external - q - p
            # Reflection cancels odd transverse powers exactly. The resulting
            # angular dependence is an even polynomial terminating at mu^12.
            total = 0.
            for reflection in (jnp.array([1., 1., 1.]), jnp.array([-1., 1., 1.])):
                vectors = [rotate(v*reflection) for v in (q, p, r)] + [rotate(-external)]
                total += jnp.sum(_matter_trispectrum(vectors, grid, power, f)) / 2.
            return value + total, None

        result, _ = jax.lax.scan(accumulate, jnp.array(0., dtype=power.dtype), normals)
        normalization = jnp.exp(-(kk*radius)**2/6.) / (3.**1.5*(2.*jnp.pi)**3*radius**6)
        return result / (normals.shape[0]*normals.shape[1]) * normalization / 6.

    return jax.lax.map(one_mode, jnp.stack([k.ravel(), mu.ravel()], axis=-1)).reshape(k.shape)


def _legendre(mu, ells):
    values = [jnp.ones_like(mu), mu]
    for ell in range(2, max(ells) + 1):
        values.append(((2*ell-1)*mu*values[-1] - (ell-1)*values[-2])/ell)
    return jnp.stack([values[ell] for ell in ells])


def _interpolate_poles(k, mu, grid, poles, ells, method='linear'):
    """Signed interpolation in log k, with no extrapolation.

    The cubic rule uses four adjacent nodes (one-sided at the boundaries).
    It reproduces cubic polynomials and avoids a boundary-curvature assumption.
    Linear interpolation remains available for independent convergence checks.
    """
    x, nodes = jnp.log(k), jnp.log(grid)
    if method == 'cubic' and len(grid) >= 4:
        start = jnp.clip(jnp.searchsorted(nodes, x) - 2, 0, len(grid)-4)
        indices = start[..., None] + jnp.arange(4)
        selected = nodes[indices]
        weights = []
        for i in range(4):
            weight = jnp.ones_like(x)
            for j in range(4):
                if i != j:
                    weight *= (x-selected[..., j])/(selected[..., i]-selected[..., j])
            weights.append(weight)
        values = jnp.sum(poles[:, indices]*jnp.stack(weights,axis=-1)[None],axis=-1)
        values = jnp.where((x >= nodes[0]) & (x <= nodes[-1]), values, jnp.nan)
    else:
        values = jax.vmap(lambda pole: jnp.interp(x, nodes, pole, left=jnp.nan, right=jnp.nan))(poles)
    return jnp.sum(values * _legendre(mu, ells), axis=0)


def _cubic_convolutions(k, mu, grid, power, f, radius, quadrature, inner_k, inner_poles, interpolation='linear'):
    """P11 * (P12, P22) using signed intrinsic multipoles of the inner leg."""
    q, x, cp, weights = quadrature
    pq = _linear_power(q, grid, power) * jnp.exp(-(q*radius)**2)
    k, mu = jnp.broadcast_arrays(k, mu)

    def one_mode(kmu):
        kk, mm = kmu
        p = jnp.sqrt(jnp.maximum(kk**2 + q**2 - 2.*kk*q*x, 1.e-30))
        muq = x*mm + jnp.sqrt(jnp.maximum((1.-x*x)*(1.-mm*mm), 0.))*cp
        mup = (kk*mm - q*muq)/p
        p11 = pq * (1. + f*muq**2)**2
        return jnp.stack([jnp.sum(weights*p11*_interpolate_poles(p, mup, inner_k, poles, (0,2,4,6,8), method=interpolation))
                          for poles in inner_poles])
    result = jax.lax.map(one_mode, jnp.stack([k.ravel(), mu.ravel()], axis=-1))
    return jnp.moveaxis(result, -1, 0).reshape((2,) + k.shape)



class DensitySplitPowerSpectrumKernels(Calculator):
    """Cosmology-only quadratic or cubic selection basis.

    ``operator_order=2`` returns five spectra; order 3 returns nine.
    ``kernels`` has shape (nbasis, nells, nk), in ``kernel_names`` order. AP uses
    distances relative to ``fiducial`` and a radius fixed in true coordinates.
    Numerical controls are constructor options; they carry no fit parameters.
    Five positive-mu Gauss nodes integrate the intrinsic ell<=8 expansion
    exactly; cubic selection uses seven nodes through ell=12. AP uses 64.
    ``cubic_options`` controls fixed Sobol log2_nodes/seed/chunk_size and the
    nk/ninner interpolation grids. AP coordinates outside the padded true-k
    grid produce NaN rather than extrapolated predictions. Defaults are
    starting resolutions; covariance-level convergence must be checked.
    """

    kernel_names = ('p1m', 'p2m', 'p11', 'p12', 'p22')
    cubic_kernel_names = ('p1m', 'p2m', 'p3m', 'p11', 'p12', 'p13', 'p22', 'p23', 'p33')
    default_cubic_options = dict(log2_nodes=21, seed=42, chunk_size=4096, nk=48, ninner=256, interpolation='cubic')
    default_quadrature = dict(nq=120, nx=32, nphi=16, qmin=1.e-4, qmax=5., nmu=5, nklin=2048)

    def __init__(self, k, z=0., ells=None, smoothing_radius=10.,
                 rsd=True, ap=False, cosmo=None, engine='class', fiducial='DESI',
                 nq=120, nx=32, nphi=16, qmin=1.e-4, qmax=5., nmu=None, nklin=2048,
                 operator_order=2, cubic_options=None):
        if operator_order not in (2, 3):
            raise ValueError('operator_order must be 2 or 3')
        self.operator_order = int(operator_order)
        self.kernel_names = self.cubic_kernel_names if operator_order == 3 else type(self).kernel_names
        self.cubic_options = dict(self.default_cubic_options)
        if cubic_options is not None:
            if set(cubic_options) - self.cubic_options.keys():
                raise ValueError('unknown cubic numerical option')
            self.cubic_options.update(cubic_options)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)
        self.ells = tuple(range(0, 13 if operator_order == 3 else 9, 2)) if ells is None else tuple(ells)
        self.smoothing_radius = float(smoothing_radius)
        self.rsd, self.ap = bool(rsd), bool(ap)
        if nmu is None:
            nmu = 64 if self.ap else (7 if self.operator_order == 3 else self.default_quadrature['nmu'])
        self.cosmo = cosmo if cosmo is not None else CosmoprimoCosmology(engine=engine, fiducial=fiducial)
        self.quadrature = dict(nq=nq, nx=nx, nphi=nphi, qmin=qmin, qmax=qmax, nmu=nmu, nklin=nklin)
        self._fiducial = fiducial

    def __post_init__(self, *args, **kwargs):
        from scipy.special import eval_legendre

        _validate_coordinates(self.k, self.z, self.ells, self.rsd, self.smoothing_radius, quadratic=True)
        if self.ap and (not self.rsd or self.z == 0.):
            raise ValueError('AP requires redshift space and z > 0')
        options = self.quadrature
        self._quadrature = _loop_quadrature(**{name: options[name] for name in ('nq', 'nx', 'nphi', 'qmin', 'qmax')})
        nmu, nklin = options['nmu'], options['nklin']
        intrinsic_degree = 12 if self.operator_order == 3 else 8
        minimum_nmu = max(intrinsic_degree//2 + 1, (max(self.ells) + intrinsic_degree + 4)//4)
        if int(nmu) != nmu or nmu < minimum_nmu:
            raise ValueError('nmu is too small for the requested multipoles')
        if int(nklin) != nklin or nklin < 16:
            raise ValueError('nklin must be an integer >= 16')
        nmu, nklin = int(nmu), int(nklin)
        mu, weights = np.polynomial.legendre.leggauss(2 * nmu)
        self._mu = mu[nmu:]
        self._projection = np.array([(2 * ell + 1) * weights[nmu:] * eval_legendre(ell, self._mu) for ell in self.ells])
        # Separate the exact linear grid from the integration grid so c2=0
        # preserves the baseline Kaiser calculation (including its AP interpolation).
        self._linear_k = (np.geomspace(max(1.e-5, self.k.min() / 2.), self.k.max() * 2., 512)
                          if self.ap else self.k)
        self._loop_k = np.geomspace(min(1.e-6, options['qmin'] / 10., self.k.min() / 10.),
                                   options['qmax'] + 4. * self.k.max(), nklin)
        requirements = {
            'fourier.pk': [{'of': 'delta_cb', 'z': self.z, 'k': grid} for grid in (self._linear_k, self._loop_k)],
            'fourier.sigma8_z': [{'of': field, 'z': self.z} for field in ('delta_cb', 'theta_cb')],
        }
        if self.ap:
            from ..primordial_cosmology import _get_fiducial
            reference = _get_fiducial(self._fiducial).clone(engine='class')
            self._DH_fid = float(299792.458 / (100. * reference.efunc(self.z)))
            self._DM_fid = float(reference.comoving_angular_distance(self.z))
            requirements.update({name: [{'z': self.z}] for name in (
                'background.efunc', 'background.comoving_transverse_distance')})
        if self.operator_order == 3:
            self._prepare_cubic()
        self.cosmo.add_requirements(requirements)

    def _prepare_cubic(self):
        from scipy.special import eval_legendre

        options = self.cubic_options
        if options['interpolation'] not in ('linear', 'cubic'):
            raise ValueError('interpolation must be linear or cubic')
        for name in ('nk', 'ninner'):
            if int(options[name]) != options[name] or options[name] < 2:
                raise ValueError(f'{name} must be an integer >= 2')
        if self.quadrature['qmax'] * self.smoothing_radius < 8.:
            raise ValueError('cubic variance requires qmax * smoothing_radius >= 8')
        self._normals = _sobol_normals(options['log2_nodes'], options['seed'], options['chunk_size'])
        lower, upper = self.k[0], self.k[-1]
        if self.ap:
            lower, upper = lower / 2., upper * 2.
        self._cubic_k = (np.geomspace(lower, upper, int(options['nk'])) if upper > lower else np.array([lower]))
        self._inner_k = np.geomspace(min(1.e-8, lower/100.), self.quadrature['qmax'] + upper,
                                     int(options['ninner']))
        # The Gaussian importance support must fit in the provider grid. No
        # clipping of ultraviolet samples to the provider's last power value.
        largest_node = np.max(np.linalg.norm(self._normals, axis=-1))
        if 3.*largest_node/self.smoothing_radius + 2.*upper > self._loop_k[-1]:
            raise ValueError('increase qmax to cover the cubic importance samples')
        mu, weights = np.polynomial.legendre.leggauss(14)
        self._cubic_mu = mu[7:]
        self._cubic_projection = np.array([(2*ell+1)*weights[7:]*eval_legendre(ell, self._cubic_mu)
                                          for ell in range(0, 13, 2)])

    def _cubic_basis(self, k, mu, power, f, p1m, p11, p12, window):
        radius = self.smoothing_radius
        q, _, _, weights = self._quadrature
        variance = jnp.sum(weights * _linear_power(q, self._loop_k, power)
                           * jnp.exp(-(q*radius)**2)) * self.quadrature['nphi']
        variance *= 1. + 2.*f/3. + f**2/5.
        ik, im = jnp.asarray(self._inner_k)[:, None], jnp.asarray(self._cubic_mu)[None, :]
        inner = _quadratic_spectra(ik, im, self._loop_k, power, f, radius, self._quadrature)
        inner = inner.at[0].multiply(jnp.exp(-.5*(ik*radius)**2))
        inner_poles = jnp.einsum('lm,bkm->blk', jnp.asarray(self._cubic_projection[:5]), inner)
        ck, cm = jnp.asarray(self._cubic_k)[:, None], jnp.asarray(self._cubic_mu)[None, :]
        convolutions = _cubic_convolutions(ck, cm, self._loop_k, power, f, radius,
                                           self._quadrature, self._inner_k, inner_poles, self.cubic_options['interpolation'])
        connected = _trispectrum_integral(ck, cm, self._loop_k, power, f, radius, jnp.asarray(self._normals))
        intrinsic = jnp.concatenate([connected[None], convolutions], axis=0)
        poles = jnp.einsum('lm,bkm->blk', jnp.asarray(self._cubic_projection), intrinsic)
        additions = [_interpolate_poles(k, mu, self._cubic_k, pole, tuple(range(0, 13, 2)), method=self.cubic_options['interpolation']) for pole in poles]
        p3m = variance/2.*p1m + additions[0]
        return p3m, window*p3m, variance/2.*p12 + additions[1], variance**2/4.*p11 + additions[2]/3.

    def __call__(self):
        fourier = self.cosmo.get_fourier()
        linear = fourier.pk(of='delta_cb', z=self.z, k=self._linear_k)
        loop_power = fourier.pk(of='delta_cb', z=self.z, k=self._loop_k)
        f = (fourier.sigma8_z(of='theta_cb', z=self.z) / fourier.sigma8_z(of='delta_cb', z=self.z)
             if self.rsd else 0.)
        k, mu = jnp.asarray(self.k)[:, None], jnp.asarray(self._mu)[None, :]
        jac = 1.
        if self.ap:
            background = self.cosmo.get_background()
            qpar = (299792.458 / (100. * background.efunc(z=self.z))) / self._DH_fid
            qper = background.comoving_transverse_distance(z=self.z) / self._DM_fid
            factor = jnp.sqrt(1. + mu**2 * ((qper / qpar)**2 - 1.))
            k, mu = k / qper * factor, mu * qper / qpar / factor
            jac = 1. / (qpar * qper**2)
            linear = jnp.exp(jnp.interp(jnp.log(k), jnp.log(self._linear_k), jnp.log(linear)))
        else:
            linear = linear[:, None]
        matter = linear * (1. + f * mu**2)**2
        window = jnp.exp(-0.5 * (k * self.smoothing_radius)**2)
        p2m, p22 = _quadratic_spectra(k, mu, self._loop_k, loop_power, f, self.smoothing_radius, self._quadrature)
        basis = jnp.stack(jnp.broadcast_arrays(window * matter, p2m, window**2 * matter, window * p2m, p22))
        if self.operator_order == 3:
            p1m, p2m, p11, p12, p22 = basis
            p3m, p13, p23, p33 = self._cubic_basis(k, mu, loop_power, f, p1m, p11, p12, window)
            basis = jnp.stack([p1m, p2m, p3m, p11, p12, p13, p22, p23, p33])
        self.kernels = jnp.einsum('lm,bkm->blk', jnp.asarray(self._projection), jac * basis)
        return self.kernels

    def tree_flatten(self):
        return [self.kernels], {name: getattr(self, name) for name in (
            'k', 'z', 'ells', 'rsd', 'ap', 'smoothing_radius', 'quadrature',
            'operator_order', 'kernel_names', 'cubic_options')}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.kernels = children[0]
        for name, value in aux.items():
            setattr(obj, name, value)
        if not hasattr(obj, 'operator_order'):
            obj.operator_order = 2
            obj.kernel_names = cls.kernel_names
            obj.cubic_options = dict(cls.default_cubic_options)
        return obj


def _validate_coordinates(k, z, ells, rsd, radius, *, quadratic=False):
    if (k.ndim != 1 or not k.size or not np.isfinite(k).all()
            or np.any(k <= 0.) or np.any(np.diff(k) <= 0.)):
        raise ValueError('k must be a non-empty, finite, positive, strictly increasing one-dimensional array')
    if not np.isfinite(z) or z < 0.:
        raise ValueError('z must be finite and non-negative')
    if not ells or len(set(ells)) != len(ells):
        raise ValueError('ells must be non-empty and unique')
    if any(int(ell) != ell or ell < 0 or ell % 2 for ell in ells):
        raise ValueError('ells must be non-negative even integers')
    if not quadratic and any(ell not in (0, 2, 4) for ell in ells):
        raise ValueError('ells must be drawn from (0, 2, 4)')
    if not rsd and ells != (0,):
        raise ValueError('real-space density-split matter power supports only ell=0')
    if not np.isfinite(radius) or radius < 0. or (quadratic and radius == 0.):
        raise ValueError('smoothing_radius must be finite and positive for nonlinear selection, non-negative otherwise')


def _normalize_quantiles(quantiles):
    quantiles = tuple(int(quantile) for quantile in quantiles)
    if not quantiles or len(set(quantiles)) != len(quantiles):
        raise ValueError('quantiles must be non-empty and unique')
    if any(quantile not in _QUANTILES for quantile in quantiles):
        raise ValueError(f'quantiles must be drawn from {_QUANTILES}')
    return quantiles


def _gaussian_quantile_coefficients(nquantiles=5):
    """Return mean standardized-Gaussian density in each equal-probability bin."""
    from statistics import NormalDist

    normal = NormalDist()
    edges = [-np.inf] + [normal.inv_cdf(index / nquantiles)
                         for index in range(1, nquantiles)] + [np.inf]
    norm = np.sqrt(2. * np.pi)
    coefficients = {}
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        phi_low = 0. if not np.isfinite(low) else np.exp(-0.5 * low**2) / norm
        phi_high = 0. if not np.isfinite(high) else np.exp(-0.5 * high**2) / norm
        coefficients[index] = {'c1': nquantiles * (phi_low - phi_high)}
    return coefficients


class DensitySplitMatterPowerSpectrumMultipoles(Calculator):
    r"""Linear, quadratic or cubic density-split spectra with exact responses.

    The cosmology provider may be :class:`CosmoprimoCosmology`, an
    :class:`~desilike.theories.primordial_cosmology.ACECosmology` using MAPSE,
    or any compatible provider.  Alternatively, ``matter`` may supply precomputed
    Kaiser multipoles.  The output has shape ``(nquantiles, nells, nk)`` and is
    available as both ``power`` and ``poles``. Nonlinear selection uses a
    :class:`DensitySplitPowerSpectrumKernels` dependency (exact or emulated).
    It derives Q3 responses from Q1, Q2, Q4 and Q5. ``params`` overrides
    response defaults; integration options are forwarded only when constructing
    a selection kernel. ``quantile_fractions`` supplies five positive fractions
    for the weighted Q3 response identity (equal fifths by default). The linear ``matter=`` interface remains unchanged.
    """

    def __init__(self, k=None, z=0., ells=(0, 2, 4), quantiles=_QUANTILES,
                 smoothing_radius=10., rsd=True, cosmo=None, matter=None,
                 engine='class', fiducial='DESI', params=None,
                 model='linear', kernels=None, ap=False, quantile_fractions=None, **kernel_options):
        if model not in ('linear', 'quadratic', 'cubic', 'response'):
            raise ValueError("model must be linear, quadratic, cubic or response")
        if model == 'linear' and (kernels is not None or kernel_options):
            raise ValueError('kernels and integration options require quadratic or cubic selection')
        if model in ('quadratic', 'cubic', 'response') and matter is not None:
            raise ValueError('use kernels, not matter, for nonlinear selection')
        if kernels is not None and cosmo is not None:
            raise ValueError('provide either kernels or cosmo, not both')
        self.model = model
        self.operator_order = {'linear': 1, 'quadratic': 2, 'cubic': 3, 'response': 2}[model]
        self.quantile_fractions = np.asarray([.2]*5 if quantile_fractions is None else quantile_fractions, dtype='f8')
        if (self.quantile_fractions.shape != (5,) or not np.isfinite(self.quantile_fractions).all()
                or np.any(self.quantile_fractions <= 0.) or not np.isclose(self.quantile_fractions.sum(), 1.)):
            raise ValueError('quantile_fractions must be five positive fractions summing to one')
        self.quantiles = _normalize_quantiles(quantiles)
        self.response_quantiles = (tuple(q for q in _INDEPENDENT_QUANTILES if q in self.quantiles or 3 in self.quantiles)
                                   if model in ('quadratic', 'cubic', 'response') else self.quantiles)
        coefficients = _gaussian_quantile_coefficients()
        defaults = VariableCollection([
            Parameter(f'c1q{quantile}', value=coefficients[quantile]['c1'],
                      prior=dict(dist='norm', loc=coefficients[quantile]['c1'], scale=2.),
                      ref=dict(dist='norm', loc=coefficients[quantile]['c1'], scale=0.25),
                      fixed=False, latex=rf'c_{{1,{quantile}}}')
            for quantile in self.response_quantiles
        ])
        if model in ('quadratic', 'cubic', 'response'):
            defaults = defaults + VariableCollection([
                Parameter(f'c2q{q}', value=0., prior={},
                          ref=dict(dist='norm', loc=0., scale=5.), fixed=False,
                          latex=rf'c_{{2,{q}}}') for q in self.response_quantiles])
        if model == 'cubic':
            defaults = defaults + VariableCollection([
                Parameter(f'c3q{q}', value=0., prior={},
                          ref=dict(dist='norm', loc=0., scale=5.), fixed=False,
                          latex=rf'c_{{3,{q}}}') for q in self.response_quantiles])
        if model == 'response':
            if ap:
                raise ValueError('response model currently requires AP unity')
            defaults = defaults + VariableCollection([
                Parameter(name, value=0., prior=dict(dist='norm', loc=0., scale=1.),
                          ref=dict(dist='norm', loc=0., scale=.1), fixed=False)
                for name in DensitySplitResponsePowerSpectrumBasis.parameter_names])
        if params is not None:
            defaults = defaults + VariableCollection(params)
        for param in defaults:
            setattr(self, param.basename, param)
        self.response_params = [defaults[f'c1q{q}'] for q in self.response_quantiles]
        self.quadratic_params = ([defaults[f'c2q{q}'] for q in self.response_quantiles]
                                 if model in ('quadratic', 'cubic', 'response') else [])

        self.cubic_params = ([defaults[f'c3q{q}'] for q in self.response_quantiles]
                             if model == 'cubic' else [])

        if matter is not None and cosmo is not None:
            raise ValueError('provide either matter or cosmo, not both')
        self.matter = matter
        self.cosmo = None

        if k is None:
            k = getattr(kernels if kernels is not None else matter, 'k', None)
        if k is None:
            k = np.linspace(0.01, 0.2, 101)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)
        self.ells = tuple(int(ell) for ell in ells)
        self.smoothing_radius = float(smoothing_radius)
        self.rsd = bool(rsd)
        self.ap = bool(ap)
        self.kernels = kernels
        if model in ('quadratic', 'cubic', 'response'):
            if kernels is None:
                kernel_type = DensitySplitResponsePowerSpectrumBasis if model == 'response' else DensitySplitPowerSpectrumKernels
                self.kernels = kernel_type(
                    k=self.k, z=self.z, ells=self.ells, smoothing_radius=self.smoothing_radius,
                    rsd=self.rsd, ap=self.ap, cosmo=cosmo, engine=engine,
                    fiducial=fiducial, operator_order=self.operator_order, **kernel_options)
            else:
                self.ap = bool(kernels.ap)
        else:
            self.cosmo = (CosmoprimoCosmology(engine=engine, fiducial=fiducial)
                          if matter is None and cosmo is None else cosmo)

    def __post_init__(self, *args, **kwargs):
        _validate_coordinates(self.k, self.z, self.ells, self.rsd, self.smoothing_radius,
                              quadratic=self.model in ('quadratic', 'cubic', 'response'))
        if self.model == 'response' and (self.ap or not isinstance(self.kernels, DensitySplitResponsePowerSpectrumBasis)):
            raise ValueError('response model requires response coefficient kernels and AP unity')
        if self.kernels is not None:
            if (not np.array_equal(self.k, self.kernels.k) or self.ells != tuple(self.kernels.ells)
                    or self.z != self.kernels.z or self.rsd != self.kernels.rsd
                    or self.smoothing_radius != self.kernels.smoothing_radius
                    or getattr(self.kernels, 'operator_order', 2) != self.operator_order):
                raise ValueError('selection kernel order, grid, multipoles, redshift, RSD and smoothing must match')
        elif self.matter is not None:
            if (not np.array_equal(self.k, self.matter.k)
                    or self.ells != tuple(self.matter.ells)
                    or self.z != float(self.matter.z)
                    or self.rsd != bool(self.matter.rsd)):
                raise ValueError('matter kernel grid, multipoles, redshift and RSD must match')
        else:
            self.cosmo.add_requirements({
                'fourier.pk': [{'of': 'delta_cb', 'z': self.z, 'k': self.k}],
                'fourier.sigma8_z': [
                    {'of': 'delta_cb', 'z': self.z},
                    {'of': 'theta_cb', 'z': self.z},
                ],
            })

    def __call__(self):
        if self.model in ('quadratic', 'cubic', 'response'):
            c1, c2 = self._responses(self.response_params), self._responses(self.quadratic_params)
            basis = self._selection_basis()
            self.power = jnp.stack([c1[q] * basis[0] + c2[q] * basis[1] for q in self.quantiles])
            if self.model == 'cubic':
                c3 = self._responses(self.cubic_params)
                self.power += jnp.stack([c3[q] * basis[2] for q in self.quantiles])
            self.poles = self.power
            return self.power
        matter_power = self._matter_power()
        window = jnp.exp(-0.5 * (jnp.asarray(self.k) * self.smoothing_radius)**2)
        responses = jnp.stack([param.value for param in self.response_params])
        self.power = responses[:, None, None] * matter_power[None, :, :] * window[None, None, :]
        self.poles = self.power
        return self.power

    def _selection_basis(self):
        if self.model == 'response':
            return self.kernels.evaluate(jnp.stack([getattr(self, name).value
                for name in self.kernels.parameter_names]))[1:]
        return self.kernels.kernels

    def _matter_power(self):
        if self.matter is not None:
            matter_power = jnp.asarray(self.matter.power)
            shape = (len(self.ells), self.k.size)
            if matter_power.shape not in (shape, (shape[0] * shape[1],)):
                raise ValueError('matter power must have shape (nells, nk) or be flattened in ell-major order')
            matter_power = matter_power.reshape(shape)
        else:
            fourier = self.cosmo.get_fourier()
            linear_power = fourier.pk(of='delta_cb', z=self.z, k=self.k)
            if self.rsd:
                sigma8 = fourier.sigma8_z(of='delta_cb', z=self.z)
                fsigma8 = fourier.sigma8_z(of='theta_cb', z=self.z)
                f = fsigma8 / sigma8
                factors = {0: 1. + 2. * f / 3. + f**2 / 5.,
                           2: 4. * f / 3. + 4. * f**2 / 7.,
                           4: 8. * f**2 / 35.}
                matter_power = jnp.stack([factors[ell] * linear_power for ell in self.ells])
            else:
                matter_power = linear_power[None, :]
        return matter_power

    def _responses(self, params):
        responses = {q: param.value for q, param in zip(self.response_quantiles, params)}
        if self.model in ('quadratic', 'cubic', 'response') and 3 in self.quantiles:
            responses[3] = -sum(self.quantile_fractions[q-1]*value for q, value in responses.items())/self.quantile_fractions[2]
        return responses

    def tree_flatten(self):
        return [self.power], {'k': self.k, 'z': self.z, 'ells': self.ells,
                              'quantiles': self.quantiles, 'rsd': self.rsd,
                              'smoothing_radius': self.smoothing_radius,
                              'model': self.model, 'ap': self.ap, 'operator_order': self.operator_order,
                              'quantile_fractions': self.quantile_fractions}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.power = obj.poles = children[0]
        for name, value in aux.items():
            setattr(obj, name, value)
        return obj


class DensitySplitPairPowerSpectrumMultipoles(DensitySplitMatterPowerSpectrumMultipoles):
    """Quantile-pair spectra with shared responses and an optional constant.

    S0 is in observed-coordinate (Mpc/h)^3, added after deterministic AP
    projection, before the measurement window. It is independent of the
    estimator's already-subtracted Poisson term. All 15 pairs may be predicted;
    the caller must select an independent data vector for inference.
    """

    def __init__(self, k=None, z=0., ells=(0, 2, 4), pairs=((1, 1),),
                 smoothing_radius=10., rsd=True, matter=None, smoothing_in_matter=False,
                 monopole=True, model='linear', kernels=None, cosmo=None,
                 engine='class', fiducial='DESI', params=None, ap=False, **kernel_options):
        self.pairs = tuple(tuple(sorted(pair)) for pair in pairs)
        if not self.pairs or len(set(self.pairs)) != len(self.pairs):
            raise ValueError('pairs must be non-empty and unique')
        if any(len(pair) != 2 or any(q not in _QUANTILES for q in pair) for pair in self.pairs):
            raise ValueError('pairs must contain two quantiles between 1 and 5')
        quantiles = tuple(dict.fromkeys(q for pair in self.pairs for q in pair))
        self.pair_smoothing_radius = float(smoothing_radius)
        self.smoothing_in_matter = bool(smoothing_in_matter)
        self.monopole = bool(monopole)
        super().__init__(k=k, z=z, ells=ells, quantiles=quantiles,
                         smoothing_radius=smoothing_radius, rsd=rsd, matter=matter,
                         model=model, kernels=kernels, cosmo=cosmo, engine=engine,
                         fiducial=fiducial, params=params, ap=ap, **kernel_options)
        if model == 'linear':
            self.ap = bool(getattr(matter, 'ap', False))
        self.monopole_params = []
        overrides = VariableCollection(params or [])
        for first, second in self.pairs:
            name = f's0q{first}q{second}'
            param = (overrides[name] if name in overrides else
                     Parameter(name, value=0., prior={}, ref=dict(dist='norm', loc=0., scale=1.e4),
                               fixed=not self.monopole or 0 not in self.ells, latex=rf'S_{{0,{first}{second}}}'))
            setattr(self, name, param)
            self.monopole_params.append(param)

        self.derivative_params = []
        if model == 'response':
            for q,r in self.pairs:
                delattr(self, f's0q{q}q{r}')
            independent = tuple(q for q in _INDEPENDENT_QUANTILES if q in self.quantiles or 3 in self.quantiles)
            needed = {tuple(sorted((i,j))) for q,r in self.pairs
                      for i in (_INDEPENDENT_QUANTILES if q==3 else (q,))
                      for j in (_INDEPENDENT_QUANTILES if r==3 else (r,))}
            self.residual_pairs = tuple((q,r) for i,q in enumerate(independent)
                                        for r in independent[i:] if (q,r) in needed)
            self.residual_templates = getattr(self.kernels, 'residual_templates',
                                              response_residual_templates(self.k,self.ells))
            self.residual_params = []
            for order in ('s0', 's2', 's2parallel'):
                for q,r in self.residual_pairs:
                    name = f'{order}q{q}q{r}'
                    param = (overrides[name] if name in overrides else
                             Parameter(name, value=0., prior={}, ref=dict(dist='norm', loc=0., scale=1.e4),
                                       fixed=not self.monopole or not np.any(np.asarray(self.residual_templates)[('s0','s2','s2parallel').index(order)])))
                    setattr(self, name, param)
                    self.residual_params.append(param)
            self.monopole_params = self.residual_params[:len(self.residual_pairs)]
            self.derivative_params = self.residual_params[len(self.residual_pairs):]

    def _response_residual(self):
        # Delta_q3 = -sum_i f_i Delta_qi / f_3 applies to each residual matrix.
        def weight(q,i):
            return (-self.quantile_fractions[i-1]/self.quantile_fractions[2] if q==3 else float(q==i))
        transform = jnp.asarray([[weight(q,i)*weight(r,j)+(weight(q,j)*weight(r,i) if i!=j else 0.)
                                 for i,j in self.residual_pairs] for q,r in self.pairs])
        amplitudes = jnp.stack([p.value for p in self.residual_params]).reshape(3,-1) @ transform.T
        return jnp.einsum('tq,tlk->qlk', amplitudes, self.residual_templates)

    def __call__(self):
        c1 = self._responses(self.response_params)
        if self.model in ('quadratic', 'cubic', 'response'):
            c2 = self._responses(self.quadratic_params)
            basis = self._selection_basis()
            p11, p12, p22 = [basis[i] for i in ((3,4,6) if self.model == 'cubic' else (2,3,4))]
            self.power = jnp.stack([
                c1[q] * c1[r] * p11 + (c1[q] * c2[r] + c2[q] * c1[r]) * p12 + c2[q] * c2[r] * p22
                for q, r in self.pairs])
            if self.model == 'cubic':
                c3 = self._responses(self.cubic_params)
                self.power += jnp.stack([
                    (c1[q]*c3[r] + c3[q]*c1[r])*basis[5]
                    + (c2[q]*c3[r] + c3[q]*c2[r])*basis[7] + c3[q]*c3[r]*basis[8]
                    for q, r in self.pairs])
        else:
            matter = self._matter_power()
            window = jnp.exp(-0.5 * (jnp.asarray(self.k) * self.pair_smoothing_radius)**2)
            if not self.smoothing_in_matter:
                matter = matter * window[None, :]**2
            self.power = jnp.stack([c1[q] * c1[r] * matter for q, r in self.pairs])
        if self.model == 'response':
            self.power += self._response_residual()
        elif self.monopole:
            amplitudes = jnp.stack([param.value for param in self.monopole_params])
            mask = jnp.asarray([ell == 0 for ell in self.ells])
            self.power = self.power + amplitudes[:, None, None] * mask[None, :, None]
        self.poles = self.power
        return self.power

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        aux.update(pairs=self.pairs, monopole=self.monopole,
                   pair_smoothing_radius=self.pair_smoothing_radius,
                   smoothing_in_matter=self.smoothing_in_matter)
        return children, aux


class MatterPowerTheory(Calculator):
    """Linear matter power in the refactored JAX graph format."""

    def __init__(self, k=None, z=0.0, ells=(0,), rsd=False, ap=False,
                 engine="class", smoothing_radius=0., cosmo=None, fiducial="DESI", ap_nmu=64):
        if ap and not rsd:
            raise ValueError("AP requires redshift space")
        self.k = np.asarray(k, dtype="f8")
        self.z = float(z)
        self.ells = tuple(ells)
        self.rsd = bool(rsd)
        self.ap = bool(ap)
        self.engine = str(engine)
        self.smoothing_radius = float(smoothing_radius)
        self.cosmo = cosmo if cosmo is not None else CosmoprimoCosmology(engine=engine, fiducial=fiducial)
        self._fiducial, self.ap_nmu = fiducial, int(ap_nmu)

    def __post_init__(self, **kwargs):
        if self.k.ndim != 1 or not self.k.size:
            raise ValueError("k must be a non-empty one-dimensional array")
        self._pk_k = (np.geomspace(max(1.e-5, self.k.min() / 2.), self.k.max() * 2., 512)
                      if self.ap else self.k)
        requirements = {
            "fourier.pk": [{"of": "delta_cb", "z": self.z, "k": self._pk_k}],
            "fourier.sigma8_z": [
                {"of": "delta_cb", "z": self.z},
                {"of": "theta_cb", "z": self.z},
            ],
        }
        if self.ap:
            requirements.update({
                "background.efunc": [{"z": self.z}],
                "background.comoving_transverse_distance": [{"z": self.z}],
            })
            from ..primordial_cosmology import _get_fiducial
            from scipy import special
            reference = _get_fiducial(self._fiducial).clone(engine="class")
            self._DH_fid = float(299792.458 / (100. * reference.efunc(self.z)))
            self._DM_fid = float(reference.comoving_angular_distance(self.z))
            mu, weights = np.polynomial.legendre.leggauss(2 * self.ap_nmu)
            self._mu = mu[self.ap_nmu:]
            weights = weights[self.ap_nmu:]
            self._legendre_weights = np.asarray([
                2. * weights * (2 * ell + 1) / 2. * special.eval_legendre(ell, self._mu)
                for ell in self.ells
            ])
        self.cosmo.add_requirements(requirements)

    def __call__(self):
        fourier = self.cosmo.get_fourier()
        pk = fourier.pk(of="delta_cb", z=self.z, k=self._pk_k)
        f = (fourier.sigma8_z(of="theta_cb", z=self.z)
             / fourier.sigma8_z(of="delta_cb", z=self.z))
        if self.ap:
            background = self.cosmo.get_background()
            qpar = (299792.458 / (100. * background.efunc(z=self.z))) / self._DH_fid
            qper = background.comoving_transverse_distance(z=self.z) / self._DM_fid
            qap = qpar / qper
            mu, k = jnp.asarray(self._mu)[None, :], jnp.asarray(self.k)[:, None]
            factor = jnp.sqrt(1. + mu**2 * (1. / qap**2 - 1.))
            kap, muap = k / qper * factor, mu / qap / factor
            anisotropic = jnp.exp(jnp.interp(jnp.log(kap), jnp.log(self._pk_k), jnp.log(pk)))
            anisotropic /= qpar * qper**2
            if self.smoothing_radius:
                anisotropic *= jnp.exp(-0.5 * (kap * self.smoothing_radius)**2)
            anisotropic *= (1. + f * muap**2)**2
            self.power = jnp.asarray(self._legendre_weights) @ anisotropic.T
        else:
            if self.smoothing_radius:
                pk *= jnp.exp(-0.5 * (jnp.asarray(self.k) * self.smoothing_radius)**2)
            if self.rsd:
                factors = {0: 1. + 2. * f / 3. + f**2 / 5.,
                           2: 4. * f / 3. + 4. * f**2 / 7.,
                           4: 8. * f**2 / 35.}
                self.power = jnp.stack([factors[ell] * pk for ell in self.ells])
            else:
                self.power = pk[None, :]
        self.poles = self.power
        return self.power

    def tree_flatten(self):
        return [self.power], dict(k=self.k, z=self.z, ells=self.ells, rsd=self.rsd,
                                  ap=self.ap, smoothing_radius=self.smoothing_radius)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.power = obj.poles = children[0]
        for name, value in aux.items():
            setattr(obj, name, value)
        return obj


def _pqm_ap_smoothing(smoothing_radius):
    radius = float(smoothing_radius)
    if not np.isfinite(radius) or radius < 0.:
        raise ValueError("smoothing_radius must be finite and non-negative")
    return dict(model_version=1, kernel="gaussian", radius=radius, coordinates="true")


class SmoothedAPMatterPowerTheory(MatterPowerTheory):
    """AP matter kernel with Gaussian smoothing in true coordinates."""

    def __init__(self, k=None, z=0., ells=(0, 2, 4), rsd=True, ap=True,
                 smoothing_radius=10., engine="class", cosmo=None, fiducial="DESI", ap_nmu=64):
        if not ap:
            raise ValueError("the smoothed AP kernel requires ap=True")
        super().__init__(k=k, z=z, ells=ells, rsd=rsd, ap=ap, engine=engine,
                         smoothing_radius=_pqm_ap_smoothing(smoothing_radius)["radius"],
                         cosmo=cosmo, fiducial=fiducial, ap_nmu=ap_nmu)


class APDensitySplitMatterPowerSpectrumMultipoles(DensitySplitMatterPowerSpectrumMultipoles):
    """Apply exact quantile responses to an already smoothed AP kernel."""

    def __init__(self, k=None, z=0., ells=(0, 2, 4), quantiles=(1, 2, 3, 4, 5),
                 smoothing_radius=10., rsd=True, matter=None):
        if matter is None:
            raise ValueError("provide a smoothed AP matter kernel")
        self.ap_smoothing_radius = float(smoothing_radius)
        super().__init__(k=k, z=z, ells=ells, quantiles=quantiles,
                         smoothing_radius=0., rsd=rsd, matter=matter)
        if not self.rsd or not getattr(matter, "ap", False):
            raise ValueError("the pQm AP wrapper requires an RSD AP kernel")
        if self.ap_smoothing_radius != float(getattr(matter, "smoothing_radius", 0.)):
            raise ValueError("matter kernel smoothing radius must match pQm")
        self.ap = True

    def __call__(self):
        power = jnp.asarray(self.matter.power)
        shape = (len(self.ells), self.k.size)
        if power.shape not in (shape, (shape[0] * shape[1],)):
            raise ValueError("matter power must have shape (nells, nk) or be ell-major flattened")
        responses = jnp.stack([param.value for param in self.response_params])
        self.power = responses[:, None, None] * power.reshape(shape)[None, :, :]
        self.poles = self.power
        return self.power



def _kaiser_multipoles(
    linear_power: np.ndarray,
    growth_rate: float,
    ells: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Return linear matter Kaiser multipoles flattened in ell-major order."""
    f = growth_rate
    factors = {
        0: 1.0 + 2.0 * f / 3.0 + f**2 / 5.0,
        2: 4.0 * f / 3.0 + 4.0 * f**2 / 7.0,
        4: 8.0 * f**2 / 35.0,
    }
    power = np.asarray(linear_power)
    return np.concatenate([factors[ell] * power for ell in ells])


def _collapsed_matter_z3(k, q, f):
    """Exact symmetrized Z3(k,q,-q), with the zero-pair partition removed.

    F2(q,-q)=G2(q,-q)=0. Their products with the collapsed alpha/beta
    factors vanish in the paired limit. Removing this partition analytically
    avoids assigning a regulator-dependent value to a zero-vector denominator.
    The remaining k+/-q denominators are nonzero on the integration nodes.
    """
    kk = jnp.sum(k*k, axis=-1)
    qq = jnp.sum(q*q, axis=-1)
    kz, qz = k[..., 2], q[..., 2]
    cross = jnp.cross(k, q)
    collinear = jnp.sum(cross*cross, axis=-1) <= 1.e-28*kk*qq
    f3, g3, mapping = 0., 0., 0.
    for u, w in ((q, -q), (-q, q)):
        vw = k + w
        uu = qq
        vv = jnp.where(collinear, 1., jnp.sum(vw*vw, axis=-1))
        uv = jnp.sum(u*vw, axis=-1)
        alpha, reverse = 1.+uv/uu, 1.+uv/vv
        beta = kk*uv/(2.*uu*vv)
        f2, g2 = _fg2(k, w)
        f3 += 7.*alpha*f2 + (7.*reverse+4.*beta)*g2
        g3 += 3.*alpha*f2 + (3.*reverse+12.*beta)*g2
        u1, u2 = u[..., 2]/uu, vw[..., 2]/vv*g2
        mapping += (f*kz*(u1*f2+u2)+f*f*kz*kz*u1*u2)/3.
    generic = (f3/54.+f*kz*kz/kk*g3/54.+mapping
               - f*f*kz*kz*qz*qz/(6.*qq*qq)*(1.+f*kz*kz/kk))
    # In the collinear limit the symmetrized EdS kernels reduce exactly to
    # their one-dimensional displacement form, including q=+/-k.
    return jnp.where(collinear, -(1.+f*kz*kz/kk)**3*kk/(6.*qq), generic)


def _matter_loop_nodes(nq, nx, nphi):
    if nq < 4 or nq % 2:
        raise ValueError('one-loop nq must be even and >= 4')
    t, wt = np.polynomial.legendre.leggauss(nq//2)
    x, wx = np.polynomial.legendre.leggauss(nx)
    phi = (np.arange(nphi)+.5)*(2.*np.pi/nphi)
    return tuple(jnp.asarray(a) for a in (t,wt,x,wx,phi))


def _matter_one_loop(k, mu, grid, power, f, nodes, qmin, qmax, radius=None):
    """Matter P22 and P13; fold P22 onto q<=|k-q| to align soft limits.

    The P22 regulator is symmetric in its two internal legs. The polar range
    is mapped analytically rather than applying a discontinuous theta mask.
    Radial integration is split at k/2, where that range changes. Both pieces
    use the same infrared cutoff and are summed before angular integration.
    """
    t,wt,x,wx,phi = nodes
    def one_mode(kmu):
        kk,mm = kmu
        split = jnp.clip(kk/2.,qmin,qmax)
        lo = jnp.log(jnp.array([qmin,split]))[:,None]
        hi = jnp.log(jnp.array([split,qmax]))[:,None]
        q = jnp.exp(lo+(hi-lo)*(t[None,:]+1.)/2.).reshape(-1,1,1)
        radial = (((hi-lo)/2.)*wt[None,:]).reshape(-1,1,1)*q**3
        vector = jnp.array([kk*jnp.sqrt(1.-mm*mm),0.,kk*mm])
        def qvector(cosine):
            sine = jnp.sqrt(jnp.maximum(1.-cosine*cosine,0.))
            # Polar axis is k; azimuthal origin lies in the k-LOS plane.
            xx = q*(cosine*jnp.sqrt(1.-mm*mm)-sine*jnp.cos(phi)*mm)
            yy = q*sine*jnp.sin(phi)
            zz = q*(cosine*mm+sine*jnp.cos(phi)*jnp.sqrt(1.-mm*mm))
            return jnp.stack(jnp.broadcast_arrays(xx,yy,zz),axis=-1)
        lower = jnp.maximum(-1.,(kk*kk+q*q-qmax*qmax)/(2.*kk*q))
        upper = jnp.minimum(1.,kk/(2.*q))
        half = jnp.maximum(upper-lower,0.)/2.
        cosine = lower+half*(x[None,:,None]+1.)
        q22 = qvector(cosine)
        pvec = vector-q22
        pp = _linear_power(jnp.sqrt(jnp.sum(pvec*pvec,axis=-1)),grid,power)
        pq = _linear_power(q,grid,power)
        term22 = 4.*half*_z2_vector(q22,pvec,f)**2*pq*pp
        q13 = qvector(x[None,:,None])
        term13 = 6.*(1.+f*mm*mm)*_linear_power(kk,grid,power)*pq*_collapsed_matter_z3(vector,q13,f)
        weights = radial*wx[None,:,None]/((2.*jnp.pi)**2*len(phi))
        total = jnp.sum(weights*(term22+term13))
        p22 = jnp.sum(weights*term22)
        # Return a separately recorded P13 and the directly summed total.
        if radius is None:
            return jnp.stack([p22,jnp.sum(weights*term13),total])
        # Common positive quadrature for the second-order field Gram matrix.
        pp2 = jnp.sum(pvec*pvec,axis=-1)
        kernel2 = (.5*jnp.exp(-.5*radius**2*(q*q+pp2))
                   *(1.+f*q22[...,2]**2/(q*q))*(1.+f*pvec[...,2]**2/pp2))
        mixed = jnp.sum(weights*4.*half*_z2_vector(q22,pvec,f)*kernel2*pq*pp)
        auto = jnp.sum(weights*4.*half*kernel2**2*pq*pp)
        return jnp.stack([p22,jnp.sum(weights*term13),total,mixed,auto])
    k,mu = jnp.broadcast_arrays(k,mu)
    values = jax.lax.map(one_mode,jnp.stack([k.ravel(),mu.ravel()],axis=-1))
    return jnp.moveaxis(values,-1,0).reshape(((3 if radius is None else 5),)+k.shape)


class DensitySplitOneLoopPowerSpectrumKernels(DensitySplitPowerSpectrumKernels):
    """Raw quadratic basis through PL^2, with separately exposed EFT templates.

    Calling returns five bare one-loop spectra in the existing quadratic order.
    ``matter_pieces`` contains Kaiser, matter P22 and matter P13 multipoles.
    ``counterterms`` has axes (a0/a2/a4, W**0/W**1/W**2, ell, k), for
    -2 (k/pivot)^2 Z1 PL (a0+a2 mu^2+a4 mu^4). Coefficients are dimensionless.
    ``with_counterterms`` applies these amplitudes exactly after evaluation.
    This unresummed baseline adds no composite EFT or stochastic operators.
    """
    default_quadrature = dict(nq=120,nx=32,nphi=16,qmin=1.e-5,qmax=10.,nmu=5,nklin=4096)

    def __init__(self, *args, pivot=.1, **kwargs):
        if kwargs.get('operator_order',2) != 2:
            raise ValueError('one-loop basis supports only operator_order=2')
        for name,value in self.default_quadrature.items():
            if name != 'nmu':
                kwargs.setdefault(name,value)
        super().__init__(*args,**kwargs)
        self.pivot = float(pivot)
        if not np.isfinite(self.pivot) or self.pivot <= 0.:
            raise ValueError('pivot must be positive and finite')

    def __post_init__(self,*args,**kwargs):
        super().__post_init__(*args,**kwargs)
        self._matter_nodes = _matter_loop_nodes(*[int(self.quadrature[n]) for n in ('nq','nx','nphi')])

    def __call__(self):
        # Preserve the existing composite calculation and AP treatment exactly.
        super().__call__()
        fourier = self.cosmo.get_fourier()
        power = fourier.pk(of='delta_cb',z=self.z,k=self._loop_k)
        linear = fourier.pk(of='delta_cb',z=self.z,k=self._linear_k)
        f = (fourier.sigma8_z(of='theta_cb',z=self.z)/fourier.sigma8_z(of='delta_cb',z=self.z)
             if self.rsd else 0.)
        k,mu = jnp.asarray(self.k)[:,None],jnp.asarray(self._mu)[None,:]
        jac = 1.
        if self.ap:
            background = self.cosmo.get_background()
            qpar = (299792.458/(100.*background.efunc(z=self.z)))/self._DH_fid
            qper = background.comoving_transverse_distance(z=self.z)/self._DM_fid
            factor = jnp.sqrt(1.+mu*mu*((qper/qpar)**2-1.))
            k,mu = k/qper*factor,mu*qper/qpar/factor
            jac = 1./(qpar*qper*qper)
            linear = jnp.exp(jnp.interp(jnp.log(k),jnp.log(self._linear_k),jnp.log(linear)))
        else:
            linear = linear[:,None]
        loops = _matter_one_loop(k,mu,self._loop_k,power,f,self._matter_nodes,
                                 self.quadrature['qmin'],self.quadrature['qmax'])
        w = jnp.exp(-.5*(k*self.smoothing_radius)**2)
        projection = jnp.asarray(self._projection)
        def project(values):
            return jnp.einsum('lm,...km->...lk',projection,jac*values)
        self.kernels = self.kernels.at[0].add(project(w*loops[2]))
        self.kernels = self.kernels.at[2].add(project(w*w*loops[2]))
        self.matter_pieces = project(jnp.stack(jnp.broadcast_arrays(linear*(1.+f*mu*mu)**2,loops[0],loops[1])))
        self.counterterms = project(jnp.stack([jnp.stack(jnp.broadcast_arrays(
            *[-2.*(k/self.pivot)**2*(1.+f*mu*mu)*linear*mu**ell*w**n for n in range(3)]))
            for ell in (0,2,4)]))
        return self.kernels

    def with_counterterms(self, amplitudes):
        """Return the five EFT spectra without reevaluating cosmology or loops."""
        correction = jnp.einsum('a,awlk->wlk',jnp.asarray(amplitudes),self.counterterms)
        return self.kernels.at[0].add(correction[1]).at[2].add(correction[2])

    def tree_flatten(self):
        _,aux = super().tree_flatten()
        aux['pivot'] = self.pivot
        return [self.kernels,self.matter_pieces,self.counterterms],aux

    @classmethod
    def tree_unflatten(cls,aux,children):
        obj = object.__new__(cls)
        obj.kernels,obj.matter_pieces,obj.counterterms = children
        for name,value in aux.items():
            setattr(obj,name,value)
        return obj



def _quadratic_propagator(k, mu, grid, power, f, radius, quadrature):
    """One-leg O2 response: P2m = Gamma2 Z1 PL + second-order cross power.

    These are the two external-PL cyclic bispectrum terms, divided by Z1 PL
    analytically, before integration. No division of projected spectra occurs.
    """
    q,x,cp,weights = quadrature
    pq = _linear_power(q,grid,power)
    def one_mode(kmu):
        kk,mm = kmu
        p = jnp.sqrt(jnp.maximum(kk**2+q**2-2*kk*q*x,1.e-30))
        muq = x*mm+jnp.sqrt((1.-x*x)*(1.-mm*mm))*cp
        mup = (kk*mm-q*muq)/p
        value = ((1.+f*mup*mup)*_matter_z2(p,kk,(q*x-kk)/p,mup,-mm,f)*_linear_power(p,grid,power)
                 +(1.+f*muq*muq)*_matter_z2(kk,q,-x,-mm,muq,f)*pq)
        return jnp.sum(weights*jnp.exp(-.5*radius**2*(q*q+p*p))*value)
    k,mu = jnp.broadcast_arrays(k,mu)
    return jax.lax.map(one_mode,jnp.stack([k.ravel(),mu.ravel()],axis=-1)).reshape(k.shape)


class DensitySplitResponsePowerSpectrumKernels(DensitySplitOneLoopPowerSpectrumKernels):
    """Experimental matched one-leg/two-leg completion of the quadratic basis.

    The common regularization is exp(-d), with d = k^2 sigma_displacement^2
    [1+f(2+f)mu^2]/2. Gamma_m = exp(-d)[Z1+Gamma_m_loop+d Z1+EFT],
    Gamma_2 = exp(-d)[Gamma_2_loop+W(beta0+beta2 f mu^2)]. Second-order
    mode coupling is multiplied by exp(-2d). Expanding at beta=0 through PL^2
    recovers the original one-loop spectra, including the EFT linear term.
    This selects higher-order contractions; it is not full two-loop theory.
    """
    def __init__(self,*args,intrinsic_nmu=5,**kwargs):
        kwargs.setdefault('nmu',16)
        super().__init__(*args,**kwargs)
        if int(intrinsic_nmu) != intrinsic_nmu or (intrinsic_nmu != 0 and intrinsic_nmu < 5):
            raise ValueError('intrinsic_nmu must be 0 (direct) or an integer >=5')
        self.intrinsic_nmu = int(intrinsic_nmu)

    def __call__(self):
        fourier = self.cosmo.get_fourier()
        power = fourier.pk(of='delta_cb',z=self.z,k=self._loop_k)
        linear = fourier.pk(of='delta_cb',z=self.z,k=self._linear_k)
        f = (fourier.sigma8_z(of='theta_cb',z=self.z)/fourier.sigma8_z(of='delta_cb',z=self.z)
             if self.rsd else 0.)
        k,mu = jnp.asarray(self.k)[:,None],jnp.asarray(self._mu)[None,:]
        if not self.ap and self.intrinsic_nmu:
            nodes,angular_weights = np.polynomial.legendre.leggauss(2*self.intrinsic_nmu)
            nodes,angular_weights = nodes[self.intrinsic_nmu:],angular_weights[self.intrinsic_nmu:]
            mu = jnp.asarray(nodes)[None,:]
        jac = 1.
        if self.ap:
            bg = self.cosmo.get_background()
            qpar = (299792.458/(100.*bg.efunc(z=self.z)))/self._DH_fid
            qper = bg.comoving_transverse_distance(z=self.z)/self._DM_fid
            factor = jnp.sqrt(1.+mu*mu*((qper/qpar)**2-1.))
            k,mu = k/qper*factor,mu*qper/qpar/factor
            jac = 1./(qpar*qper*qper)
            linear = jnp.exp(jnp.interp(jnp.log(k),jnp.log(self._linear_k),jnp.log(linear)))
        else:
            linear = linear[:,None]
        loops = _matter_one_loop(k,mu,self._loop_k,power,f,self._matter_nodes,
            self.quadrature['qmin'],self.quadrature['qmax'],radius=self.smoothing_radius)
        response = _quadratic_propagator(k,mu,self._loop_k,power,f,self.smoothing_radius,self._quadrature)
        q,_,_,weights = self._quadrature
        variance = (jnp.sum(weights*_linear_power(q,self._loop_k,power)/(q*q))
                    *self.quadrature['nphi']/3.)
        z1 = 1.+f*mu*mu
        d = .5*k*k*variance*(1.+f*(2.+f)*mu*mu)
        w = jnp.exp(-.5*(k*self.smoothing_radius)**2)
        self.components = jnp.stack(jnp.broadcast_arrays(linear,z1,loops[1]/(2.*z1*linear),
            response,loops[0],loops[3],loops[4],d,w,(k/self.pivot)**2,mu*mu,f*mu*mu))
        if not self.ap and self.intrinsic_nmu:
            # The undamped components terminate at ell=8. Reconstruct their
            # finite polynomial before applying damping on the finer mu grid.
            # Gamma_m is divided by Z1 before projection, so its degree is <=6.
            ells = tuple(range(0,9,2))
            angular = _legendre(jnp.asarray(nodes),ells)
            projection = angular*jnp.asarray(angular_weights)[None,:]*jnp.asarray([2*l+1 for l in ells])[:,None]
            poles = jnp.einsum('lm,bkm->blk',projection,self.components)
            self.components = jnp.einsum('blk,lm->bkm',poles,_legendre(jnp.asarray(self._mu),ells))
        self.projection = jac*jnp.asarray(self._projection)
        self.kernels = self.assemble()[1:]
        return self.kernels

    def assemble(self, matter=(0.,0.,0.), response=(0.,0.), damping=1., truncated=False):
        """Return (Pmm,P1m,P2m,P11,P12,P22), forming products before projection.

        EFT and beta coefficients remain exact, outside cosmology emulation.
        Damping variations are model-choice checks, not numerical tolerances.
        The truncated control ignores beta (which is defined only for completion).
        """
        pl,z1,gm,g2,mm,m2,two,d,w,k2,mu2,fmu2 = self.components
        a,b = jnp.asarray(matter),jnp.asarray(response)
        ct = -k2*(a[0]+a[1]*mu2+a[2]*mu2**2)
        if truncated:
            pmm = (z1*z1+2*z1*(gm+ct))*pl+mm
            p2m,p22 = g2*z1*pl+m2,two
        else:
            d = damping*d
            gm = z1+gm+d*z1+ct
            g2 = g2+w*(b[0]+b[1]*fmu2)
            factor = jnp.exp(-2*d)
            pmm = factor*(gm*gm*pl+mm)
            p2m = factor*(g2*gm*pl+m2)
            p22 = factor*(g2*g2*pl+two)
        values = jnp.stack([pmm,w*pmm,p2m,w*w*pmm,w*p2m,p22])
        return jnp.einsum('lm,bkm->blk',self.projection,values)

    def with_counterterms(self, amplitudes):
        return self.assemble(matter=amplitudes)[1:]

    def tree_flatten(self):
        _,aux = DensitySplitPowerSpectrumKernels.tree_flatten(self)
        aux['pivot'] = self.pivot
        aux['intrinsic_nmu'] = self.intrinsic_nmu
        return [self.kernels,self.components,self.projection],aux

    @classmethod
    def tree_unflatten(cls,aux,children):
        obj = object.__new__(cls)
        obj.kernels,obj.components,obj.projection = children
        for name,value in aux.items():
            setattr(obj,name,value)
        return obj



def response_residual_templates(k, ells, pivot=.1):
    """Observed-coordinate S0, S2, S2parallel multipoles, before the window."""
    k2 = (jnp.asarray(k)/pivot)**2
    mono = jnp.asarray([ell==0 for ell in ells])[:,None]
    mu2 = jnp.asarray([1./3 if ell==0 else 2./3 if ell==2 else 0. for ell in ells])[:,None]
    return jnp.stack([mono*jnp.ones_like(k2), mono*k2, mu2*k2])


class DensitySplitResponsePowerSpectrumBasis(DensitySplitResponsePowerSpectrumKernels):
    """Cosmology-only coefficients of the exact five-nuisance polynomial.

    Shape is (21,6,nells,nk), with spectra Pmm,P1m,P2m,P11,P12,P22.
    Feature order is 1, five linear parameters, then upper-triangular products.
    Gaussian nuisance priors belong to consuming calculators, never this basis.
    """
    parameter_names = ('a0','a2','a4','beta0','beta2')
    feature_pairs = tuple((i,j) for i in range(5) for j in range(i,5))
    spectrum_names = ('pmm','p1m','p2m','p11','p12','p22')
    default_response_quadrature = dict(nq=240,nx=64,nphi=24,nklin=8192,nmu=24,qmin=1.e-5,qmax=10.)

    def __init__(self,*args,**kwargs):
        if kwargs.get('ap',False):
            raise ValueError('response coefficient basis currently requires AP unity')
        for name,value in self.default_response_quadrature.items():
            kwargs.setdefault(name,value)
        super().__init__(*args,**kwargs)

    @staticmethod
    def features(x):
        x = jnp.asarray(x)
        return jnp.concatenate([jnp.ones(1), x, jnp.stack([x[i]*x[j]
            for i,j in DensitySplitResponsePowerSpectrumBasis.feature_pairs])])

    def __call__(self):
        super().__call__()
        def evaluate(x):
            return self.assemble(x[:3],x[3:])
        origin = evaluate(jnp.zeros(5))
        positive = jax.vmap(evaluate)(jnp.eye(5))
        negative = jax.vmap(evaluate)(-jnp.eye(5))
        linear = (positive-negative)/2
        products = []
        for i,j in self.feature_pairs:
            products.append((positive[i]+negative[i])/2-origin if i==j else
                            evaluate(jnp.eye(5)[i]+jnp.eye(5)[j])-positive[i]-positive[j]+origin)
        self.coefficients = jnp.concatenate([origin[None],linear,jnp.stack(products)])
        return self.coefficients

    def evaluate(self,parameters):
        return jnp.einsum('c,c...->...',self.features(parameters),self.coefficients)

    def tree_flatten(self):
        aux = {name: getattr(self,name) for name in (
            'k','z','ells','rsd','ap','smoothing_radius','quadrature',
            'operator_order','kernel_names','cubic_options','pivot','intrinsic_nmu')}
        if hasattr(self,'residual_templates'):
            aux['residual_templates'] = np.asarray(self.residual_templates)
        return [self.coefficients],aux

    @classmethod
    def tree_unflatten(cls,aux,children):
        obj = object.__new__(cls)
        obj.coefficients = children[0]
        for name,value in aux.items():
            setattr(obj,name,value)
        return obj



class WindowedDensitySplitResponseBasis(DensitySplitResponsePowerSpectrumBasis):
    """Apply one common periodic-box window before nuisance assembly/emulation.

    The window is supplied by the fitting project. Its columns are ell-major
    true-lattice samples; rows are measured multipoles. No smoothing is added.
    """
    def __init__(self, basis, matrix, k, ells=(0,2,4)):
        self.basis = basis
        self.matrix = np.asarray(matrix)
        self.k, self.ells = np.asarray(k), tuple(ells)
        for name in ('z','rsd','ap','smoothing_radius','quadrature','operator_order',
                     'kernel_names','cubic_options','pivot','intrinsic_nmu'):
            setattr(self,name,getattr(basis,name))
        if self.matrix.shape != (len(self.k)*len(self.ells),len(basis.k)*len(basis.ells)):
            raise ValueError('window and response basis shapes differ')
        residual = np.asarray(response_residual_templates(basis.k,basis.ells)).reshape(3,-1)
        self.residual_templates = (residual@self.matrix.T).reshape(3,len(self.ells),len(self.k))

    def __post_init__(self,*args,**kwargs):
        pass

    def __call__(self):
        coefficients = self.basis.coefficients.reshape(21,6,-1)
        self.coefficients = jnp.einsum('ij,cbj->cbi',self.matrix,coefficients).reshape(21,6,len(self.ells),len(self.k))
        return self.coefficients


class ResponseMatterPowerSpectrumMultipoles(Calculator):
    """Matter-only control with the same completed spectrum and three free EFT coefficients."""
    def __init__(self,kernels,params=None):
        self.kernels = kernels
        self.k,self.ells,self.z = kernels.k,kernels.ells,kernels.z
        defaults = VariableCollection([Parameter(name,value=0.,prior=dict(dist='norm',loc=0.,scale=1.),
            ref=dict(dist='norm',loc=0.,scale=.1),fixed=False) for name in ('a0','a2','a4')])
        if params is not None:
            defaults = defaults + VariableCollection(params)
        for p in defaults:
            setattr(self,p.basename,p)

    def __call__(self):
        self.power = self.kernels.evaluate(jnp.array([self.a0.value,self.a2.value,self.a4.value,0.,0.]))[0]
        return self.power

    def tree_flatten(self):
        return [self.power],dict(k=self.k,ells=self.ells,z=self.z)

    @classmethod
    def tree_unflatten(cls,aux,children):
        obj=object.__new__(cls);obj.power=children[0]
        for name,value in aux.items():setattr(obj,name,value)
        return obj
