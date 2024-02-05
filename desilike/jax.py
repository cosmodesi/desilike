import functools
import logging

logging.getLogger('jax._src.lib.xla_bridge').addFilter(logging.Filter('No GPU/TPU found, falling back to CPU.'))


# jax array types
array_types = ()

try:
    # raise ImportError
    import jax, jaxlib
    from jax.config import config; config.update('jax_enable_x64', True)
    from jax import numpy, scipy
    array_types = []
    for line in ['jaxlib.xla_extension.DeviceArrayBase', 'type(numpy.array(0))', 'jax.core.Tracer']:
        try:
            array_types.append(eval(line))
        except AttributeError:
            pass
    array_types = tuple(array_types)
except ImportError:
    jax = None
    import numpy
    import scipy


def jit(*args, **kwargs):
    """Return :mod:`jax` just-in-time compiler."""

    def get_wrapper(func):
        if jax is None:
            return func
        return jax.jit(func, **kwargs)

    if kwargs or not args:
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))

    return get_wrapper(args[0])


def use_jax(array):
    """Whether to use jax.numpy depending on whether array is jax's object."""
    return isinstance(array, tuple(array_types))


def dist_name(dist):
    """
    Return distribution name, which should work with either scipy (where dist is a :class:`rv_continuous` instance)
    or jax implementation (where dist is a module).
    """
    name = getattr(dist, 'name', None)
    if name is None: name = dist.__name__.split('.')[-1]
    return name


def fallback(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        args, kwargs = func(self, *args, **kwargs)
        ofunc = getattr(self.odist, func.__name__)
        #print(any(use_jax(arg) for arg in args), func)
        if not any(use_jax(arg) for arg in args):
            return ofunc(*args, **kwargs)
        return getattr(self.dist, func.__name__, ofunc)(*args, **kwargs)

    return wrapper


class rv_frozen(object):
    """
    ``jax`` currently does not implement scipy's frozen random variate.
    Here is an ersatz.
    """
    def __init__(self, dist, *args, **kwds):
        self.dist = dist
        from scipy import stats
        self.odist = getattr(stats, dist_name(dist))
        self.args = args
        self.kwds = kwds

    @fallback
    def pdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def logpdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def cdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def logcdf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def ppf(self, q):
        return (q,) + self.args, self.kwds

    @fallback
    def isf(self, q):
        return (q,) + self.args, self.kwds

    @fallback
    def rvs(self, size=None, random_state=None):
        return self.args, {**self.kwds, 'size': size, 'random_state': random_state}

    @fallback
    def sf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def logsf(self, x):
        return (x,) + self.args, self.kwds

    @fallback
    def stats(self, moments='mv'):
        return self.args, {**self.kwds, 'moments': moments}

    @fallback
    def median(self):
        return self.args, self.kwds

    @fallback
    def mean(self):
        return self.args, self.kwds

    @fallback
    def var(self):
        return self.args, self.kwds

    @fallback
    def std(self):
        return self.args, self.kwds

    @fallback
    def moment(self, order=None, **kwds):
        return (order,) + self.args, {**self.kwds, **kwds}

    @fallback
    def entropy(self):
        return self.args, self.kwds

    @fallback
    def interval(self, confidence=None, **kwds):
        return (confidence,) + self.args, {**self.kwds, **kwds}

    @fallback
    def support(self):
        return self.args, self.kwds



class InterpolatedUnivariateSpline(object):

    """Taken from https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/816069f0e69d75ec83689406623839b53fbf43fc/jax_cosmo/scipy/interpolate.py#L44."""

    def __init__(self, x, y, k=3, endpoints="not-a-knot", coefficients=None):
        """JAX implementation of kth-order spline interpolation.

        This class aims to reproduce scipy's InterpolatedUnivariateSpline
        functionality using JAX. Not all of the original class's features
        have been implemented yet, notably
        - `w`    : no weights are used in the spline fitting.
        - `bbox` : we assume the boundary to always be [x[0], x[-1]].
        - `ext`  : extrapolation is always active, i.e., `ext` = 0.
        - `k`    : orders `k` > 3 are not available.
        - `check_finite` : no such check is performed.

        (The relevant lines from the original docstring have been included
        in the following.)

        Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
        Spline function passes through all provided points. Equivalent to
        `UnivariateSpline` with s = 0.

        Parameters
        ----------
        x : (N,) array_like
            Input dimension of data points -- must be strictly increasing
        y : (N,) array_like
            input dimension of data points
        k : int, optional
            Degree of the smoothing spline.  Must be 1 <= `k` <= 3.
        endpoints : str, optional, one of {'natural', 'not-a-knot'}
            Endpoint condition for cubic splines, i.e., `k` = 3.
            'natural' endpoints enforce a vanishing second derivative
            of the spline at the two endpoints, while 'not-a-knot'
            ensures that the third derivatives are equal for the two
            left-most `x` of the domain, as well as for the two
            right-most `x`. The original scipy implementation uses
            'not-a-knot'.
        coefficients: list, optional
            Precomputed parameters for spline interpolation. Shouldn't be set
            manually.

        See Also
        --------
        UnivariateSpline : Superclass -- allows knots to be selected by a
            smoothing condition
        LSQUnivariateSpline : spline for which knots are user-selected
        splrep : An older, non object-oriented wrapping of FITPACK
        splev, sproot, splint, spalde
        BivariateSpline : A similar class for two-dimensional spline interpolation

        Notes
        -----
        The number of data points must be larger than the spline degree `k`.

        The general form of the spline can be written as
          f[i](x) = a[i] + b[i](x - x[i]) + c[i](x - x[i])^2 + d[i](x - x[i])^3,
          i = 0, ..., n-1,
        where d = 0 for `k` = 2, and c = d = 0 for `k` = 1.

        The unknown coefficients (a, b, c, d) define a symmetric, diagonal
        linear system of equations, Az = s, where z = b for `k` = 1 and `k` = 2,
        and z = c for `k` = 3. In each case, the coefficients defining each
        spline piece can be expressed in terms of only z[i], z[i+1],
        y[i], and y[i+1]. The coefficients are solved for using
        `numpy.linalg.solve` when `k` = 2 and `k` = 3.

        """
        # Verify inputs
        k = int(k)
        assert k in (1, 2, 3), "Order k must be in {1, 2, 3}."
        x = numpy.atleast_1d(x)
        y = numpy.atleast_1d(y)
        assert len(x) == len(y), "Input arrays must be the same length."
        assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."
        n_data = len(x)

        # Difference vectors
        h = numpy.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
        p = numpy.diff(y)  # y[i+1] - y[i]

        if coefficients is None:
            # Build the linear system of equations depending on k
            # (No matrix necessary for k=1)
            if k == 1:
                assert n_data > 1, "Not enough input points for linear spline."
                coefficients = p / h

            if k == 2:
                assert n_data > 2, "Not enough input points for quadratic spline."
                assert endpoints == "not-a-knot"  # I have only validated this
                # And actually I think it's probably the best choice of border condition

                # The knots are actually in between data points
                knots = (x[1:] + x[:-1]) / 2.0
                # We add 2 artificial knots before and after
                knots = numpy.concatenate(
                    [
                        numpy.array([x[0] - (x[1] - x[0]) / 2.0]),
                        knots,
                        numpy.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                    ]
                )
                n = len(knots)
                # Compute interval lenghts for these new knots
                h = numpy.diff(knots)
                # postition of data point inside the interval
                dt = x - knots[:-1]

                # Now we build the system natrix
                A = numpy.diag(
                    numpy.concatenate(
                        [
                            numpy.ones(1),
                            (
                                2 * dt[1:]
                                - dt[1:] ** 2 / h[1:]
                                - dt[:-1] ** 2 / h[:-1]
                                + h[:-1]
                            ),
                            numpy.ones(1),
                        ]
                    )
                )

                A += numpy.diag(
                    numpy.concatenate([-numpy.array([1 + h[0] / h[1]]), dt[1:] ** 2 / h[1:]]),
                    k=1,
                )
                A += numpy.diag(
                    numpy.concatenate([numpy.atleast_1d(h[0] / h[1]), numpy.zeros(n - 3)]), k=2
                )

                A += numpy.diag(
                    numpy.concatenate(
                        [
                            h[:-1] - 2 * dt[:-1] + dt[:-1] ** 2 / h[:-1],
                            -numpy.array([1 + h[-1] / h[-2]]),
                        ]
                    ),
                    k=-1,
                )
                A += numpy.diag(
                    numpy.concatenate([numpy.zeros(n - 3), numpy.atleast_1d(h[-1] / h[-2])]),
                    k=-2,
                )

                # And now we build the RHS vector
                s = numpy.concatenate([numpy.zeros(1), 2 * p, numpy.zeros(1)])

                # Compute spline coefficients by solving the system
                coefficients = numpy.linalg.solve(A, s)

            if k == 3:
                assert n_data > 3, "Not enough input points for cubic spline."
                if endpoints not in ("natural", "not-a-knot"):
                    print("Warning : endpoints not recognized. Using natural.")
                    endpoints = "natural"

                # Special values for the first and last equations
                zero = numpy.array([0.0])
                one = numpy.array([1.0])
                A00 = one if endpoints == "natural" else numpy.array([h[1]])
                A01 = zero if endpoints == "natural" else numpy.array([-(h[0] + h[1])])
                A02 = zero if endpoints == "natural" else numpy.array([h[0]])
                ANN = one if endpoints == "natural" else numpy.array([h[-2]])
                AN1 = (
                    -one if endpoints == "natural" else numpy.array([-(h[-2] + h[-1])])
                )  # A[N, N-1]
                AN2 = zero if endpoints == "natural" else numpy.array([h[-1]])  # A[N, N-2]

                # Construct the tri-diagonal matrix A
                A = numpy.diag(numpy.concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
                upper_diag1 = numpy.diag(numpy.concatenate((A01, h[1:])), k=1)
                upper_diag2 = numpy.diag(numpy.concatenate((A02, numpy.zeros(n_data - 3))), k=2)
                lower_diag1 = numpy.diag(numpy.concatenate((h[:-1], AN1)), k=-1)
                lower_diag2 = numpy.diag(numpy.concatenate((numpy.zeros(n_data - 3), AN2)), k=-2)
                A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

                # Construct RHS vector s
                center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
                s = numpy.concatenate((zero, center, zero))
                # Compute spline coefficients by solving the system
                coefficients = numpy.linalg.solve(A, s)

        # Saving spline parameters for evaluation later
        self.k = k
        self._x = x
        self._y = y
        self._coefficients = coefficients

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self._x, self._y, self._coefficients)
        aux_data = {"endpoints": self._endpoints, "k": self.k}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y, coefficients = children
        return cls(x, y, coefficients=coefficients, **aux_data)

    def __call__(self, x):
        """Evaluation of the spline.

        Notes
        -----
        Values are extrapolated if x is outside of the original domain
        of knots. If x is less than the left-most knot, the spline piece
        f[0] is used for the evaluation; similarly for x beyond the
        right-most point.

        """
        if self.k == 1:
            t, a, b = self._compute_coeffs(x)
            result = a + b * t

        if self.k == 2:
            t, a, b, c = self._compute_coeffs(x)
            result = a + b * t + c * t**2

        if self.k == 3:
            t, a, b, c, d = self._compute_coeffs(x)
            result = a + b * t + c * t**2 + d * t**3

        return result

    def _compute_coeffs(self, xs):
        """Compute the spline coefficients for a given x."""
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = numpy.concatenate(
                [
                    numpy.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    numpy.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = numpy.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = numpy.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]
        h = numpy.diff(knots)[ind]

        if self.k == 1:
            a = y[ind]
            result = (t, a, coefficients[ind])

        if self.k == 2:
            dt = (x - knots[:-1])[ind]
            b = coefficients[ind]
            b1 = coefficients[ind + 1]
            a = y[ind] - b * dt - (b1 - b) * dt**2 / (2 * h)
            c = (b1 - b) / (2 * h)
            result = (t, a, b, c)

        if self.k == 3:
            c = coefficients[ind]
            c1 = coefficients[ind + 1]
            a = y[ind]
            a1 = y[ind + 1]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            result = (t, a, b, c, d)

        return result

    def derivative(self, x, n=1):
        """Analytic nth derivative of the spline.

        The spline has derivatives up to its order k.

        """
        assert n in range(self.k + 1), "Invalid n."

        if n == 0:
            result = self.__call__(x)
        else:
            # Linear
            if self.k == 1:
                t, a, b = self._compute_coeffs(x)
                result = b

            # Quadratic
            if self.k == 2:
                t, a, b, c = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t
                if n == 2:
                    result = 2 * c

            # Cubic
            if self.k == 3:
                t, a, b, c, d = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t + 3 * d * t**2
                if n == 2:
                    result = 2 * c + 6 * d * t
                if n == 3:
                    result = 6 * d

        return result

    def antiderivative(self, xs):
        """
        Computes the antiderivative of first order of this spline
        """
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = numpy.concatenate(
                [
                    numpy.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    numpy.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = numpy.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = numpy.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]

        if self.k == 1:
            a = y[:-1]
            b = coefficients
            h = numpy.diff(knots)
            cst = numpy.concatenate([numpy.zeros(1), numpy.cumsum(a * h + b * h**2 / 2)])
            return cst[ind] + a[ind] * t + b[ind] * t**2 / 2

        if self.k == 2:
            h = numpy.diff(knots)
            dt = x - knots[:-1]
            b = coefficients[:-1]
            b1 = coefficients[1:]
            a = y - b * dt - (b1 - b) * dt**2 / (2 * h)
            c = (b1 - b) / (2 * h)
            cst = numpy.concatenate(
                [numpy.zeros(1), numpy.cumsum(a * h + b * h**2 / 2 + c * h**3 / 3)]
            )
            return cst[ind] + a[ind] * t + b[ind] * t**2 / 2 + c[ind] * t**3 / 3

        if self.k == 3:
            h = numpy.diff(knots)
            c = coefficients[:-1]
            c1 = coefficients[1:]
            a = y[:-1]
            a1 = y[1:]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            cst = numpy.concatenate(
                [
                    numpy.zeros(1),
                    numpy.cumsum(a * h + b * h**2 / 2 + c * h**3 / 3 + d * h**4 / 4),
                ]
            )
            return (
                cst[ind]
                + a[ind] * t
                + b[ind] * t**2 / 2
                + c[ind] * t**3 / 3
                + d[ind] * t**4 / 4
            )

    def integral(self, a, b):
        """
        Compute a definite integral over a piecewise polynomial.
        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1
        xs = numpy.array([a, b])
        return sign * numpy.diff(self.antiderivative(xs))