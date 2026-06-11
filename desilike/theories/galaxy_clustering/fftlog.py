"""
JAX-native FFTLog implementation for P(k) → ξ(r) Hankel transforms.

Adapted from jax-power/jaxpower/fftlog.py (https://github.com/cosmodesi/jax-power).
Only the subset needed for :class:`PowerToCorrelation` is included.
"""

import numpy as np
from scipy.special import loggamma as numpy_loggamma
import jax
from jax import numpy as jnp


def _loggamma(x):
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(numpy_loggamma, result_shape, x)


def pad(array, pad_width, axis=-1, extrap=0):
    """Pad *array* along *axis* with extrapolation mode *extrap*.

    Parameters
    ----------
    pad_width : int or (int, int)
        Number of points to add on left and right.
    extrap : float or 'log' or 'edge' or tuple thereof
        Padding value, or one of the named modes.
    """
    array = jnp.asarray(array)
    try:
        pad_width_left, pad_width_right = pad_width
    except (TypeError, ValueError):
        pad_width_left = pad_width_right = pad_width
    try:
        extrap_left, extrap_right = extrap
    except (TypeError, ValueError):
        extrap_left = extrap_right = extrap

    axis = axis % array.ndim
    to_axis = [1] * array.ndim
    to_axis[axis] = -1

    def index(i):
        return jnp.full(1, i, dtype='i4')

    if extrap_left == 'edge':
        end = jnp.take(array, index(0), axis=axis)
        pad_left = jnp.repeat(end, pad_width_left, axis=axis)
    elif extrap_left == 'log':
        end = jnp.take(array, index(0), axis=axis)
        ratio = jnp.take(array, index(1), axis=axis) / end
        exp = jnp.arange(-pad_width_left, 0).reshape(to_axis)
        pad_left = end * ratio ** exp
    else:
        pad_left = jnp.full(array.shape[:axis] + (pad_width_left,) + array.shape[axis + 1:], extrap_left)

    if extrap_right == 'edge':
        end = jnp.take(array, index(-1), axis=axis)
        pad_right = jnp.repeat(end, pad_width_right, axis=axis)
    elif extrap_right == 'log':
        end = jnp.take(array, index(-1), axis=axis)
        ratio = jnp.take(array, index(-2), axis=axis) / end
        exp = jnp.arange(1, pad_width_right + 1).reshape(to_axis)
        pad_right = end / ratio ** exp
    else:
        pad_right = jnp.full(array.shape[:axis] + (pad_width_right,) + array.shape[axis + 1:], extrap_right)

    return jnp.concatenate([pad_left, array, pad_right], axis=axis)


class JAXFFTEngine:
    def __init__(self, size, nparallel=1, **kwargs):
        self.size = size
        self.nparallel = nparallel

    def forward(self, fun):
        return jnp.fft.rfft(fun, axis=-1)

    def backward(self, fun):
        return jnp.fft.irfft(fun.conj(), n=self.size, axis=-1)


def get_fft_engine(engine, size, nparallel=1, **kwargs):
    if isinstance(engine, str):
        if engine.lower() == 'jax':
            return JAXFFTEngine(size=size, nparallel=nparallel, **kwargs)
        raise ValueError(f'Unknown FFT engine {engine!r}')
    return engine


class BaseKernel:
    def __call__(self, z):
        return self.eval(z)

    def __eq__(self, other):
        return other.__class__ == self.__class__


class SphericalBesselJKernel(BaseKernel):
    """Mellin transform of the spherical Bessel kernel j_nu."""

    def __init__(self, nu):
        self.nu = nu

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.nu == self.nu

    def eval(self, z):
        return jnp.exp(jnp.log(2) * (z - 1.5)
                       + _loggamma(0.5 * (self.nu + z))
                       - _loggamma(0.5 * (3 + self.nu - z)))


@jax.tree_util.register_pytree_node_class
class FFTlog:
    r"""
    FFTLog algorithm for integrals of the form

        G(y) = ∫ x dx F(x) K(xy)

    Implementation adapted from jax-power/jaxpower/fftlog.py.
    """

    def __init__(self, x, kernel, q=0, minfolds=2, lowring=True, xy=1, engine='jax', **engine_kwargs):
        self.inparallel = isinstance(kernel, (tuple, list))
        if not self.inparallel:
            kernel = [kernel]
        kernel = list(kernel)
        if np.ndim(q) == 0:
            q = [q] * len(kernel)
        q = list(q)
        self._x = jnp.asarray(x)
        if not self.inparallel:
            self._x = self._x[None, :]
        elif self._x.ndim == 1:
            self._x = jnp.tile(self._x[None, :], (len(kernel), 1))
        if np.ndim(xy) == 0:
            xy = [xy] * len(kernel)
        xy = list(xy)
        self._setup(kernel, q, minfolds=minfolds, lowring=lowring, xy=xy)
        self.set_fft_engine(engine, **engine_kwargs)

    def set_fft_engine(self, engine='jax', **engine_kwargs):
        self._engine = get_fft_engine(engine, size=self.padded_size, nparallel=self.nparallel, **engine_kwargs)

    @property
    def x(self):
        return self._x if self.inparallel else self._x[0]

    @property
    def y(self):
        return self._y if self.inparallel else self._y[0]

    @property
    def nparallel(self):
        return self._x.shape[0]

    @property
    def size(self):
        return self._x.shape[-1]

    def _setup(self, kernels, qs, minfolds=2, lowring=True, xy=1.):
        self.delta = jnp.log(self._x[:, -1] / self._x[:, 0]) / (self.size - 1)
        self.padded_size = self.size
        if minfolds:
            nfolds = (self.size * minfolds - 1).bit_length()
            self.padded_size = 2**nfolds
        npad = self.padded_size - self.size
        self.padded_size_in_left = npad // 2
        self.padded_size_in_right = npad - npad // 2
        self.padded_size_out_left = npad - npad // 2
        self.padded_size_out_right = npad // 2

        if lowring:
            self.lnxy = jnp.array(
                [delta / jnp.pi * jnp.angle(kernel(q + 1j * np.pi / delta))
                 for kernel, delta, q in zip(kernels, self.delta, qs)],
                dtype=self.x.dtype)
        else:
            self.lnxy = jnp.log(jnp.array(xy))

        self._y = jnp.exp(self.lnxy - self.delta)[:, None] / self._x[:, ::-1]

        m = np.arange(0, self.padded_size // 2 + 1)
        self._padded_u = []
        self._padded_prefactor = []
        self._padded_postfactor = []
        self._padded_x = pad(self._x, (self.padded_size_in_left, self.padded_size_in_right), axis=-1, extrap='log')
        self._padded_y = pad(self._y, (self.padded_size_out_left, self.padded_size_out_right), axis=-1, extrap='log')
        prev_kernel, prev_q, prev_delta, prev_u = None, None, None, None
        for kernel, padded_x, padded_y, lnxy, delta, q in zip(kernels, self._padded_x, self._padded_y, self.lnxy, self.delta, qs):
            self._padded_prefactor.append(padded_x**(-q))
            self._padded_postfactor.append(padded_y**(-q))
            if kernel is prev_kernel and q == prev_q and delta == prev_delta:
                u = prev_u
            else:
                u = prev_u = kernel(q + 2j * np.pi / self.padded_size / delta * m)
            self._padded_u.append(u * jnp.exp(-2j * jnp.pi * lnxy / self.padded_size / delta * m))
            prev_kernel, prev_q, prev_delta = kernel, q, delta
        self._padded_u = jnp.array(self._padded_u)
        self._padded_prefactor = jnp.array(self._padded_prefactor)
        self._padded_postfactor = jnp.array(self._padded_postfactor)

    def tree_flatten(self):
        children = (self._x, self._y, self._padded_x, self._padded_y,
                    self._padded_u, self._padded_prefactor, self._padded_postfactor)
        aux_data = {name: getattr(self, name)
                    for name in ['inparallel', 'padded_size',
                                 'padded_size_in_left', 'padded_size_in_right',
                                 'padded_size_out_left', 'padded_size_out_right', '_engine']}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        (new._x, new._y, new._padded_x, new._padded_y,
         new._padded_u, new._padded_prefactor, new._padded_postfactor) = children
        return new

    def __call__(self, fun, extrap=0, keep_padding=False):
        fun = jnp.asarray(fun)
        padded_fun = pad(fun, (self.padded_size_in_left, self.padded_size_in_right), axis=-1, extrap=extrap)
        fftloged = (self._engine.backward(
            self._engine.forward(padded_fun * self._padded_prefactor) * self._padded_u)
            * self._padded_postfactor)
        if not keep_padding:
            y = self._y
            fftloged = fftloged[..., self.padded_size_out_left:self.padded_size_out_left + self.size]
        else:
            y = self._padded_y
        if not self.inparallel:
            y = y[0]
            fftloged = jnp.reshape(fftloged, fun.shape if not keep_padding else fun.shape[:-1] + (self.padded_size,))
        return y, fftloged


@jax.tree_util.register_pytree_node_class
class PowerToCorrelation:
    r"""JAX-native P(k) → ξ(r) transform via FFTLog.

    .. math::
        \xi_{\ell}(s) = \frac{(-i)^{\ell}}{2\pi^{2}} \int dk\, k^{2} P_{\ell}(k) j_{\ell}(ks)

    Parameters
    ----------
    k : array_like
        Log-spaced input wavenumbers (shared across all multipoles).
    ell : int or list of int, default=0
        Multipole order(s). A list triggers parallel transforms.
    q : float, default=0
        Power-law tilt to regularise the integration.
    lowring : bool, default=True
        Use the low-ringing output coordinate condition.
    kwargs : dict
        Forwarded to :class:`FFTlog`.
    """

    def __init__(self, k, ell=0, q=0, lowring=True, **kwargs):
        if np.ndim(ell) == 0:
            kernel = SphericalBesselJKernel(ell)
        else:
            kernel = [SphericalBesselJKernel(ell_) for ell_ in ell]
        fftlog = FFTlog(k, kernel, q=1.5 + q, lowring=lowring, **kwargs)
        fftlog._padded_prefactor = fftlog._padded_prefactor * fftlog._padded_x**3 / (2 * np.pi)**1.5
        ell_arr = np.atleast_1d(ell)
        # (-i)^ell phase — imaginary part of odd poles is passed in, so phase simplifies
        phase = (-1)**(ell_arr // 2)
        fftlog._padded_postfactor = fftlog._padded_postfactor * phase[:, None]
        self._fftlog = fftlog

    def __call__(self, fun, **kwargs):
        return self._fftlog(fun, **kwargs)

    @property
    def y(self):
        return self._fftlog.y

    def tree_flatten(self):
        return (self._fftlog,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        (new._fftlog,) = children
        return new
