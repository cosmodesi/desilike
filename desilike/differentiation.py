import itertools

import numpy as np

from desilike import PipelineError
from .base import _args_or_kwargs
from .parameter import Parameter, ParameterCollection, ParameterArray, Samples, Deriv, ParameterPriorError
from .utils import BaseClass, expand_dict, is_sequence
from .jax import jax
from .jax import numpy as jnp
from . import mpi


def deriv_ncoeffs(order, acc=2):
    """Return number of coefficients given input derivative order and accuracy."""
    return 2 * ((order + 1) // 2) - 1 + acc


def coefficients(order, acc, coords, idx):
    """
    Calculate the finite difference coefficients for given derivative order and accuracy order.
    Assume that the underlying grid is non-uniform.

    Adapted from https://github.com/maroba/findiff/blob/master/findiff/coefs.py

    Parameters
    ----------
    order : int
        The derivative order (positive integer).

    acc : int
        The accuracy order (even positive integer).

    coords : np.ndarray
        The coordinates of the axis for the partial derivative.

    idx : int
        Index of the grid position where to calculate the coefficients.

    Returns
    -------
    coeffs, offsets
    """
    import math

    if acc % 2 or acc <= 0:
        raise ValueError('Accuracy order acc must be positive EVEN integer')

    if order < 0:
        raise ValueError('Derive degree must be positive integer')

    order, acc = int(order), int(acc)

    ncoeffs = deriv_ncoeffs(order, acc=acc)
    nside = ncoeffs // 2
    ncoeffs += (order % 2 == 0)

    def _build_rhs(offsets, order):
        """The right hand side of the equation system matrix"""
        b = [0 for _ in offsets]
        b[order] = math.factorial(order)
        return np.array(b, dtype='float')

    def _build_matrix_non_uniform(p, q, coords, k):
        """Constructs the equation matrix for the finite difference coefficients of non-uniform grids at location k"""
        A = [[1] * (p + q + 1)]
        for i in range(1, p + q + 1):
            line = [(coords[k + j] - coords[k])**i for j in range(-p, q + 1)]
            A.append(line)
        return np.array(A, dtype='float')

    if idx < nside:
        matrix = _build_matrix_non_uniform(0, ncoeffs - 1, coords, idx)

        offsets = list(range(ncoeffs))
        rhs = _build_rhs(offsets, order)

        return np.linalg.solve(matrix, rhs), np.array(offsets)

    if idx >= len(coords) - nside:
        matrix = _build_matrix_non_uniform(ncoeffs - 1, 0, coords, idx)

        offsets = list(range(-ncoeffs + 1, 1))
        rhs = _build_rhs(offsets, order)

        return np.linalg.solve(matrix, rhs), np.array(offsets)

    matrix = _build_matrix_non_uniform(nside, nside, coords, idx)

    offsets = list(range(-nside, nside + 1))
    rhs = _build_rhs(offsets, order)

    return np.linalg.solve(matrix, rhs), np.array([p for p in range(-nside, nside + 1)])


def deriv_nd(X, Y, orders, center=None, atol=0.):
    """
    Compute n-dimensional derivative.

    Parameters
    ----------
    X : array
        Array of shape (nsamples, ndim), with ndim the number of variables.

    Y : array
        Array of shape (nsamples, ysize), with ysize the size of the vector to derive.

    orders : list
        List of tuples (derivation axis between 0 and ndim - 1, derivative order, derivative accuracy).

    center : array, default=None
        The center around which to take derivatives, of size ndim.
        If ``None``, defaults to the median of input ``X``.

    atol : list, float
        Absolute tolerance to find the center.

    Returns
    -------
    deriv : array
        Derivative of Y, of size ysize.
    """
    uorders = []
    for axis, order, acc in orders:
        if not order: continue
        uorders.append((axis, order, acc))
    orders = uorders
    if center is None:
        center = [np.median(np.unique(xx)) for xx in X.T]
    if np.ndim(atol) == 0:
        atol = [atol] * X.shape[1]
    atol = list(atol)
    if not len(orders):
        toret = Y[np.all([np.isclose(xx, cc, rtol=0., atol=at) for xx, cc, at in zip(X.T, center, atol)], axis=0)]
        if not toret.size:
            raise ValueError('Global center point not found')
        return toret[0]
    axis, order, acc = orders[-1]
    ncoeffs = deriv_ncoeffs(order, acc=acc)
    coord = np.unique(X[..., axis])
    if coord.size < ncoeffs:
        raise ValueError('Grid is not large enough ({:d} < {:d}) to estimate {:d}-th order derivative'.format(coord.size, ncoeffs, order))
    cidx = np.flatnonzero(np.isclose(coord, center[axis], rtol=0., atol=atol[axis]))
    if not cidx.size:
        raise ValueError('Global center point not found')
    cidx = cidx[0]
    toret = 0.
    for coeff, offset in zip(*coefficients(order, acc, coord, cidx)):
        mask = X[..., axis] == coord[cidx + offset]
        ncenter = center.copy()
        ncenter[axis] = coord[cidx + offset]
        # We could fill in atol[axis] = 0., but it should be useless?
        y = deriv_nd(X[mask], Y[mask], orders[:-1], center=ncenter, atol=atol)
        toret += y * coeff
    return toret


def deriv_grid(grids, current_order=0):
    """
    Return grid of points where to compute function to estimate its derivatives.

    Parameters
    ----------
    grids : list
        List of tuples (1D grid coordinates, array of (minimum) derivative orders corresponding to 1D grid, derivative accuracy).

    Returns
    -------
    grid : list
        List of coordinates.
    """
    grid, orders, maxorder = grids[-1]
    toret = []
    for order in np.unique(orders)[::-1]:
        if order == 0 or order + current_order <= maxorder:
            mask = orders == order
            if len(grids) > 1:
                mgrid = deriv_grid(grids[:-1], current_order=order + current_order)
            else:
                mgrid = [[]]
            toret += [mg + [gg] for mg in mgrid for gg in grid[mask]]
    return toret


class Differentiation(BaseClass):

    """Estimate derivatives of ``calculator`` quantities, with auto- or finite-differentiation."""

    def __init__(self, calculator, getter=None, order=1, method=None, accuracy=2, delta_scale=1., mpicomm=None):
        """
        Initialize differentiation.

        Parameters
        ----------
        calculator : BaseCalculator
            Input calculator.

        getter : callable, default=None
            Function (without input arguments) that returns a quantity,
            or a list or dictionary mapping names to quantities from ``calculator`` to be differentiated.
            If ``None``, defaults to derived parameters.

        order : int, dict, default=1
            A dictionary mapping parameter name (including wildcard) to maximum derivative order.
            If a single value is provided, applies to all varied parameters.

        method : str, dict, default=None
            A dictionary mapping parameter name (including wildcard) to method to use to estimate derivatives,
            either 'auto' for automatic differentiation, or 'finite' for finite differentiation.
            If ``None``, 'auto' will be used if possible, else 'finite'.
            If a single value is provided, applies to all varied parameters.

        accuracy : int, dict, default=2
            A dictionary mapping parameter name (including wildcard) to derivative accuracy (number of points used to estimate it).
            If a single value is provided, applies to all varied parameters.
            Not used if ``method = 'auto'``  for this parameter.

        delta_scale : float, default=1.
            Parameter grid ranges for the estimation of finite derivatives are inferred from parameters' :attr:`Parameter.delta`.
            These values are then scaled by ``delta_scale`` (< 1. means smaller ranges).

        mpicomm : mpi.COMM_WORLD, default=None
            MPI communicator. If ``None``, defaults to ``calculator``'s :attr:`BaseCalculator.mpicomm`.
        """
        if mpicomm is None:
            mpicomm = calculator.mpicomm
        self.mpicomm = mpicomm
        self.calculator = calculator
        self.calculator()  # dry run
        self.pipeline = self.calculator.runtime_info.pipeline
        self.varied_params = self.calculator.varied_params
        # In case of likelihood marginalization self.calculator.runtime_info.pipeline._varied_params is changed
        # Make sure these parameters are included in all_params with + self.varied_params
        self.all_params = self.calculator.all_params.select(derived=False) + self.varied_params
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        if mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))

        for name, item in zip(['order', 'method', 'accuracy'], [order, method, accuracy]):
            setattr(self, name, expand_dict(item, self.varied_params.names()))

        for param, value in self.order.items():
            if value is None: value = 0
            self.order[param] = int(value)

        self.getter = getter

        if getter is None:
            calculators, fixed, varied = self.pipeline._classify_derived(self.pipeline.calculators)
            varied_by_calculator = []
            for cc, vv in zip(calculators, varied):
                base_names = cc.runtime_info.base_names
                tmp = ParameterCollection()
                for v in vv:
                    if v in base_names:
                        p = self.pipeline.params[base_names[v]]
                        if p.derived: tmp.set(p.copy())
                varied_by_calculator.append(tmp)

            if not any(varied_by_calculator):
                raise ValueError('No varied parameter is derived, so nothing to differentiate')

            def getter():
                toret = {}
                for calculator, varied in zip(calculators, varied_by_calculator):
                    state = calculator.__getstate__()
                    for param in varied:
                        name = param.basename
                        if name in state: value = state[name]
                        else: value = getattr(calculator, name)
                        toret[param] = value = jnp.array(value)
                        param._shape = value.shape  # a bit hacky, but no need to update parameters for this...
                return toret

            self.getter = getter

        for param, method in self.method.items():
            if self.order[param] == 0:
                method = self.method[param] = 'auto'
                continue
            if method in [None, 'auto']:
                try:
                    self._calculate({param: [self.pipeline.input_values[param]]}, autoderivs=[(), (param,)])  # This takes time because the model is evaluated for each parameter
                except Exception as exc:
                    if method is None:
                        method = 'finite'
                    else:
                        raise ValueError('Cannot use auto-differentiation (with jax) for parameter {}'.format(param)) from exc
                else:
                    method = 'auto'
            if self.method[param] is None and mpicomm.rank == 0:
                self.log_info('Using {}-differentiation for parameter {}.'.format(method, param))
            self.method[param] = method
            if method == 'finite':
                value = self.accuracy[param]
                if value is None:
                    raise ValueError('accuracy not specified for parameter {}'.format(param))
                value = int(value)
                if value < 1:
                    raise ValueError('accuracy is {} < 1 for parameter {}'.format(value, param))
                if value % 2:
                    raise ValueError('accuracy is {} for parameter {}, but it must be a positive EVEN integer'.format(value, param))
                self.accuracy[param] = value

        self._grid_center, grids = {}, []
        for param in self.varied_params:
            center = param.delta[0]
            if self.method[param.name] == 'finite' and self.order[param.name]:
                size = deriv_ncoeffs(self.order[param.name], acc=self.accuracy[param.name])
                delta, limits = param.delta[1:], param.prior.limits
                if not (limits[0] <= center <= limits[-1]):
                    raise ValueError('for {} center {} is not within prior limits {}'.format(param.name, center, limits))
                delta = tuple(delta_scale * dd for dd in delta)
                if any(dd <= 0 for dd in delta):
                    raise ValueError('for {} delta {} is not > 0'.format(param.name, delta))
                hsize = size // 2
                grid_min = limits[1] - hsize * (delta[0] + delta[1])  # if we start from upper limit
                grid_min = max(limits[0], min(center - delta[0] * hsize, grid_min))
                grid = [grid_min + np.arange(hsize + 1) * delta[0]]  # below center
                center = grid[0][-1]
                grid.append(center + np.arange(1, hsize + 1) * delta[1])  # above center
                grid = np.concatenate(grid)
                if grid[-1] > limits[1]:
                    raise ValueError('for {}, cannot fit {:d} steps in prior limits {} with delta = {}; increase prior limits or decrease delta'.format(param.name, size, limits, delta))
                cindex = hsize
                order = np.zeros(len(grid), dtype='i')
                for ord in range(self.order[param.name], 0, -1):
                    s = deriv_ncoeffs(ord, acc=self.accuracy[param.name])
                    order[cindex - s // 2:cindex + s // 2 + 1] = ord
                order[cindex] = 0
                grid = (grid, order, self.order[param.name])
                if mpicomm.rank == 0:
                    self.log_info('{} grid is {}.'.format(param, grid[0]))
            else:
                grid = (np.array([center]), np.array([0]), 0)
            self._grid_center[param.name] = center
            grids.append(grid)

        self._grid_samples = self._grid_cidx = None
        if mpicomm.rank == 0:
            samples = np.array(deriv_grid(grids)).T
            self._grid_samples = Samples(samples, params=self.varied_params)
            self._grid_cidx = True
            for array, grid in zip(self._grid_samples, grids):
                grid = grid[0]
                center = grid[len(grid) // 2]
                atol = 0.
                self._grid_cidx &= np.isclose(array, center, rtol=0., atol=atol)
            self._grid_cidx = tuple(np.flatnonzero(self._grid_cidx))
            assert len(self._grid_cidx) == 1
            self.log_info('Differentiation will evaluate {:d} points.'.format(len(self._grid_samples)))
        self._grid_cidx = mpicomm.bcast(self._grid_cidx, root=0)
        autoparams, autoorder, self.autoderivs = [], [], []
        for param, method in self.method.items():
            autoparams.append(param)
            autoorder.append(self.order[param] if method == 'auto' else 0)
        self.autoderivs.append(())
        for maxorder in range(1, max([0] + autoorder) + 1):
            self.autoderivs.append([autoparams[i] for i, o in enumerate(autoorder) if o >= maxorder])
        #self.mpicomm = mpicomm

    @property
    def mpicomm(self):
        return self._mpicomm

    @mpicomm.setter
    def mpicomm(self, mpicomm):
        mpicomm_bak = getattr(self, '_mpicomm', None)
        if mpicomm_bak is not None and mpicomm is not mpicomm_bak:
            # Broadcast self._grid_samples to the new rank = 0 processes
            ranks = mpicomm_bak.allgather(mpicomm_bak.rank if mpicomm.rank == 0 else None)
            ranks = [rank for rank in ranks if rank is not None]
            for mpiroot in ranks:
                grid_samples = Samples.bcast(self._grid_samples, mpiroot=mpiroot, mpicomm=mpicomm_bak)
                if grid_samples is not None: self._grid_samples = grid_samples
        self._mpicomm = mpicomm

    def _calculate(self, params, autoderivs=None):
        if autoderivs is None:
            autoderivs = self.autoderivs

        mpicomm = self.pipeline.mpicomm

        names = self.mpicomm.bcast(list(params) if self.mpicomm.rank == 0 else None, root=0)
        values = []
        for name in names:
            value = np.atleast_1d(params[name]) if self.mpicomm.rank == 0 else None
            values.append(value)
            csize = self.mpicomm.bcast(value.size if self.mpicomm.rank == 0 else None)
        global getter_inst, getter_size
        getter_inst, getter_size = None, None

        def __calculate(*values):
            global getter_inst, getter_size
            assert len(names) == len(values)
            self.pipeline.calculate(dict(zip(names, values)))
            toret = self.getter()
            getter_inst = toret
            if hasattr(toret, 'values'):
                toret = list(toret.values())
            getter_size = int(is_sequence(toret))
            if getter_size:
                getter_size = len(toret)
            else:
                toret = [toret]
            toret = list(toret)
            if not toret:
                raise ValueError('getter returns nothing to differentiate')
            return toret

        getter_samples, errors = [], []
        max_chunk_size = getattr(self, '_mpi_max_chunk_size', 100)
        nchunks = (csize // max_chunk_size) + 1

        for ichunk in range(nchunks):  # divide in chunks to save memory for MPI comm
            self.pipeline.mpicomm = mpi.COMM_SELF
            chunk_params = {}
            for name, value in zip(names, values):
                chunk_params[name] = mpi.scatter(value[csize * ichunk // nchunks:csize * (ichunk + 1) // nchunks] if self.mpicomm.rank == 0 else None, mpicomm=self.mpicomm, mpiroot=0)
                chunk_size = len(chunk_params[name])

            for ivalue in range(chunk_size):
                chunk_values = [chunk_params[name][ivalue] for name in params]
                toret = []
                try:
                    try:
                        jac = __calculate
                        for iautoderiv, autoderiv in enumerate(autoderivs[1:]):
                            if jax is None:
                                raise ValueError('jax is required to compute the Jacobian')
                            argnums = [names.index(p) for p in autoderiv]
                            funcname = 'jacfwd' # if iautoderiv else 'jacrev'
                            jac = getattr(jax, funcname)(jac, argnums=argnums, has_aux=False, holomorphic=False)
                            #jac = jax.jacfwd(jac, argnums=argnums, has_aux=False, holomorphic=False)
                            toret.append(jac(*chunk_values))
                            #jax.vjp(tmp, has_aux=False)[1](jnp.ones(len(autoderiv)))
                    except Exception as exc:
                        raise exc
                    finally:
                        getter_samples.append([__calculate(*chunk_values)] + toret)
                except Exception as exc:
                    errors.append(exc)

            errors = self.mpicomm.allreduce(errors)
            self.pipeline.mpicomm = mpicomm
            if errors:
                raise PipelineError('got these errors: {}'.format(errors))
            tmp_samples = self.mpicomm.reduce(getter_samples, root=0)
            if self.mpicomm.rank == 0:
                getter_samples += tmp_samples

        for getter_size, getter_inst in (self.mpicomm.gather((getter_size, getter_inst), root=0) or []):
            if getter_size is not None: break

        if self.mpicomm.rank == 0:
            toret = [[[None for isample in range(csize)] for iautoderiv in range(len(autoderivs))] for igetter in range(max(getter_size, 1))]
            for isample in range(csize):
                items = getter_samples[isample]
                for ideriv, derivs in enumerate(items):
                    for iitem, item in enumerate(derivs):
                        toret[iitem][ideriv][isample] = item
        return toret, getter_inst, getter_size

    def run(self, *args, **kwargs):
        params = _args_or_kwargs(args, kwargs)
        # Getter, or calculator, dict[param1, param2]
        self.center = {}
        # print(self.pipeline.input_values)
        for param in self.all_params:
            self.center[param.name] = params.get(param.name, self.pipeline.input_values[param.name])
        if self.mpicomm.rank == 0:
            samples = self._grid_samples.copy()
            for param in self.all_params:
                if param.name in self._grid_center:
                    offset = self.center[param.name] - self._grid_center[param.name]
                    samples[param] = self._grid_samples[param] + offset
                else:
                    samples[param] = np.full(samples.shape, self.center[param.name])
        nsamples = self.mpicomm.bcast(samples.size if self.mpicomm.rank == 0 else None, root=0)
        getter_samples, getter_inst, getter_size = self._calculate(samples.to_dict(params=self.all_params) if self.mpicomm.rank == 0 else {})
        if self.mpicomm.rank == 0:
            finiteparams, finiteorder, finiteaccuracy = [], [], []
            for param in self._grid_samples.names():
                if self.method[param] == 'finite':
                    finiteparams.append(param)
                    finiteorder.append(self.order[param])
                    finiteaccuracy.append(self.accuracy[param])
            getter_samples = [[np.array(s) for s in getter_sample] for getter_sample in getter_samples]
            #self.getter_samples = getter_samples
            degrees, derivatives = [], [[] for i in range(max(getter_size, 1))]
            cidx = self._grid_cidx
            if finiteparams:
                X = np.concatenate([samples[param].reshape(nsamples, 1) for param in finiteparams], axis=-1)
                ndim = X.shape[1]
                center = X[cidx]

            autodegrees, autoindices = [Deriv()], [()]
            for autoorder, autoderiv in enumerate(self.autoderivs):
                nautodegrees, nautoindices = [], []
                for autodegree, autoindex in zip(autodegrees, autoindices):
                    for iautoparam, autoparam in enumerate(autoderiv or (None,)):
                        if autoorder > 0:
                            nautodegree = autodegree + Deriv([autoparam])
                            nautoindex = autoindex + (iautoparam,)
                        else:
                            nautodegree = autodegree
                            nautoindex = autoindex
                        if nautodegree in degrees:
                            continue
                        nautodegrees.append(nautodegree)
                        nautoindices.append(nautoindex)
                        degrees.append(nautodegree)
                        Y = [getter_sample[autoorder][(slice(None),) + nautoindex + (Ellipsis,)] for getter_sample in getter_samples]
                        if autodegree:  # with jax nan derivatives are zero derivatives...
                            for y in Y: y[np.isnan(y)] = 0.
                        for iy, y in enumerate(Y): derivatives[iy].append(y[cidx])
                        # Now finite differentiation
                        yshapes = [y.shape[samples.ndim:] for y in Y]
                        Y = [y.reshape(nsamples, -1) for y in Y]
                        for order in range(1, max(finiteorder + [0]) + 1):
                            for indices in itertools.product(range(ndim), repeat=order):
                                orders = np.bincount(indices, minlength=ndim).astype('i4')
                                if sum(orders) + autoorder > min(order for o, order in zip(orders, finiteorder) if o):
                                    continue
                                degree = nautodegree + Deriv(dict(zip(finiteparams, orders)))
                                if degree in degrees:
                                    continue
                                orders = [(iparam, order, accuracy) for iparam, (order, accuracy) in enumerate(zip(orders, finiteaccuracy)) if order > 0]
                                dx = [deriv_nd(X, y, orders, center=center, atol=0.) for y in Y]
                                if any(np.isnan(ddx).any() for ddx in dx):
                                    raise ValueError('some derivatives are NaN')
                                degrees.append(degree)
                                for iy, (ddx, yshape) in enumerate(zip(dx, yshapes)): derivatives[iy].append(ddx.reshape(yshape))
                autodegrees = nautodegrees
                autoindices = nautoindices
            toret = derivatives = [ParameterArray(derivative, derivs=degrees, param=Parameter('param_{:d}'.format(ideriv), shape=derivative[0].shape)) for ideriv, derivative in enumerate(derivatives)]
            if isinstance(getter_inst, dict):
                toret = Samples()
                for param in self.varied_params:
                    toret[param] = ParameterArray(self.center[param.name], param=param)
                for param, derivative in zip(getter_inst, derivatives):
                    derivative.param = Parameter(param)
                    toret[param] = derivative
                toret.attrs['center'] = self.center
            elif not getter_size:
                toret = toret[0]
        self.samples = toret

    def __call__(self, *args, **kwargs):
        """
        Return derivatives for input parameter values.
        If ``getter`` returns a list (resp. dict), a list (resp. :class:`Samples`) of derivatives."""
        self.run(*args, **kwargs)
        return self.samples
