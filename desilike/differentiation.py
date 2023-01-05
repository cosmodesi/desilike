import itertools

import numpy as np

from .parameter import Parameter, ParameterCollection, ParameterArray, Samples, Deriv
from .utils import BaseClass, expand_dict, is_sequence
from .jax import jax
from .jax import numpy as jnp


def deriv_ncoeffs(order, acc=2):
    return 2 * ((order + 1) // 2) - 1 + acc


def coefficients(order, acc, coords, idx):
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.
    Assumes that the underlying grid is non-uniform.

    Taken from https://github.com/maroba/findiff/blob/master/findiff/coefs.py

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


def deriv_nd(X, Y, orders, center=None):
    uorders = []
    for axis, order, acc in orders:
        if not order: continue
        uorders.append((axis, order, acc))
    orders = uorders
    if center is None:
        center = [np.median(np.unique(xx)) for xx in X.T]
    if not len(orders):
        toret = Y[np.all([xx == cc for xx, cc in zip(X.T, center)], axis=0)]
        if not toret.size:
            raise ValueError('Global center point not found')
        return toret[0]
    axis, order, acc = orders[-1]
    ncoeffs = deriv_ncoeffs(order, acc=acc)
    coord = np.unique(X[..., axis])
    if coord.size < ncoeffs:
        raise ValueError('Grid is not large enough ({:d} < {:d}) to estimate {:d}-th order derivative'.format(coord.size, ncoeffs, order))
    cidx = np.flatnonzero(coord == center[axis])
    if not cidx.size:
        raise ValueError('Global center point not found')
    cidx = cidx[0]
    toret = 0.
    for coeff, offset in zip(*coefficients(order, acc, coord, cidx)):
        mask = X[:, axis] == coord[cidx + offset]
        y = deriv_nd(X[mask], Y[mask], orders[:-1])
        toret += y * coeff
    return toret


class Differentiation(BaseClass):

    def __init__(self, calculator, getter=None, method=None, order=1, ref_scale=1e-1, accuracy=2, mpicomm=None):
        if mpicomm is None:
            mpicomm = calculator.mpicomm
        self.mpicomm = mpicomm
        self.calculator = calculator
        self.calculator()  # dry run
        self.pipeline = self.calculator.runtime_info.pipeline
        self.varied_params = self.calculator.varied_params
        if not self.varied_params:
            raise ValueError('No parameters to be varied!')
        if self.mpicomm.rank == 0:
            self.log_info('Varied parameters: {}.'.format(self.varied_params.names()))

        for name, item in zip(['method', 'order', 'accuracy'], [method, order, accuracy]):
            setattr(self, name, expand_dict(item, self.varied_params.names()))

        for param, value in self.order.items():
            if value is None: value = 0
            self.order[param] = int(value)

        self.getter = getter

        if getter is None:
            calculators, fixed, varied = self.pipeline._classify_derived(self.pipeline.calculators)
            varied_by_calculator = []
            for cc, vv in zip(calculators, varied):
                bp = cc.runtime_info.base_params
                varied_by_calculator.append(ParameterCollection([bp[k].copy() for k in vv if k in bp and bp[k].derived]))

            def getter():
                toret = {}
                for calculator, varied in zip(calculators, varied_by_calculator):
                    state = calculator.__getstate__()
                    for param in varied:
                        name = param.basename
                        if name in state: value = state[name]
                        else: value = getattr(calculator, name)
                        param._shape = value.shape  # a bit hacky, but no need to update parameters for this...
                        toret[param] = jnp.array(value)
                return toret

            self.getter = getter

        for param, method in self.method.items():
            if self.order[param] == 0:
                method = self.method[param] = 'auto'
                continue
            if method in [None, 'auto']:
                try:
                    self.autoderivs = [(), (param,)]
                    self._calculate()
                except Exception as exc:
                    if method is None:
                        method = 'finite'
                    else:
                        raise ValueError('Cannot use auto-differentiation (with jax) for parameter {}'.format(param)) from exc
                else:
                    method = 'auto'
            if self.method[param] is None and self.mpicomm.rank == 0:
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

        size, sphere = {}, {}
        for param in self.varied_params.names():
            if self.method[param] == 'finite':
                size[param] = deriv_ncoeffs(self.order[param], acc=self.accuracy[param])
                sphere[param] = self.order[param]
            else:
                size[param] = 1
                sphere[param] = 0

        from desilike.samplers import GridSampler
        sampler = GridSampler(self.calculator, size=size, ref_scale=ref_scale, sphere=sphere, mpicomm=self.mpicomm)
        # sphere is not None, so samples will *end* with the central point, making sure param_values correspond to the user last run(**params) call
        self._grid_center = {param.name: param.value for param in sampler.varied_params}
        self._grid_samples = sampler.samples.deepcopy() if self.mpicomm.rank == 0 else None

        autoparams, autoorder, self.autoderivs = [], [], []
        for param, method in self.method.items():
            autoparams.append(param)
            autoorder.append(self.order[param] if method == 'auto' else 0)
        self.autoderivs.append(())
        for maxorder in range(1, max([0] + autoorder) + 1):
            self.autoderivs.append([autoparams[i] for i, o in enumerate(autoorder) if o >= maxorder])

    def _calculate(self, **params):

        def getter():
            toret = self.getter()
            self.getter_inst = toret
            if hasattr(toret, 'values'):
                toret = list(toret.values())
            self.getter_size = int(is_sequence(toret))
            if self.getter_size:
                self.getter_size = len(toret)
            else:
                toret = [toret]
            return list(toret)

        params = {**self.pipeline.param_values, **params}
        params, values = list(params.keys()), list(params.values())

        def __calculate(*values):
            self.pipeline.param_values.update(dict(zip(params, values)))
            values = self.pipeline.params.eval(**self.pipeline.param_values)
            for calculator in self.pipeline.calculators:  # start by first calculator, end by the last one
                runtime_info = calculator.runtime_info
                runtime_info.set_param_values(values, full=True, force=True)
                result = runtime_info.calculate()
            return getter()

        toret = []
        try:
            jac = __calculate
            jacs = []
            for iautoderiv, autoderiv in enumerate(self.autoderivs[1:]):
                if jax is None:
                    raise ValueError('jax is required to compute the Jacobian')
                argnums = [params.index(p) for p in autoderiv]
                jac = getattr(jax, 'jacfwd' if iautoderiv else 'jacrev')(jac, argnums=argnums, has_aux=False, holomorphic=False)
                toret.append(jac(*values))
        except Exception as exc:
            raise exc
        finally:
            self._getter_sample = [__calculate(*values)] + toret
        return toret

    def _more_derived(self, ipoint):
        self._getter_samples[ipoint] = self._getter_sample

    def run(self, **params):
        # Getter, or calculator, dict[param1, param2]
        self.center = {}
        for param in self.varied_params:
            self.center[param.name] = params.get(param.name, self.pipeline.param_values[param.name])
        if self.mpicomm.rank == 0:
            samples = self._grid_samples.copy()
            for param in self.varied_params:
                offset = self.center[param.name] - self._grid_center[param.name]
                samples[param] = self._grid_samples[param] + offset
        nsamples = self.mpicomm.bcast(samples.size if self.mpicomm.rank == 0 else None, root=0)
        self._getter_samples = {}
        calculate_bak, more_derived_bak = self.pipeline.calculate, self.pipeline.more_derived
        self.pipeline.calculate, self.pipeline.more_derived = self._calculate, self._more_derived
        self.pipeline.mpicalculate(**(samples.to_dict(params=self.varied_params) if self.mpicomm.rank == 0 else {}))
        self.pipeline.calculate, self.pipeline.more_derived = calculate_bak, more_derived_bak
        states = self.mpicomm.gather(self._getter_samples, root=0)

        toret = None
        if self.mpicomm.rank == 0:
            finiteparams, finiteorder, finiteaccuracy = [], [], []
            for param, method in self.method.items():
                if method == 'finite':
                    finiteparams.append(param)
                    finiteorder.append(self.order[param])
                    finiteaccuracy.append(self.accuracy[param])
            self._getter_samples = [[[None for i in range(nsamples)] for i in range(len(self.autoderivs))] for i in range(max(self.getter_size, 1))]
            for state in states:
                for istate, items in state.items():
                    for ideriv, derivs in enumerate(items):
                        for iitem, item in enumerate(derivs):
                            self._getter_samples[iitem][ideriv][istate] = item
            del states
            self._getter_samples = [[np.array(s) for s in getter_samples] for getter_samples in self._getter_samples]
            degrees, derivatives = [], [[] for i in range(max(self.getter_size, 1))]
            if finiteparams:
                X = np.concatenate([samples[param].reshape(nsamples, 1) for param in finiteparams], axis=-1)
                ndim = X.shape[1]
                #center = np.array([np.median(np.unique(xx)) for xx in X.T])
                center = [self.center[param] for param in finiteparams]
                cidx = np.flatnonzero(np.all([xx == cc for xx, cc in zip(X.T, center)], axis=0))
                if not cidx.size:
                    raise ValueError('Global center point not found')
                cidx = tuple(cidx)
            else:
                cidx = (0,)
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
                        Y = [getter_samples[autoorder][(slice(None),) + nautoindex + (Ellipsis,)] for getter_samples in self._getter_samples]
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
                                dx = [deriv_nd(X, y, orders, center=center) for y in Y]
                                if any(np.isnan(ddx).any() for ddx in dx):
                                    raise ValueError('Some derivatives are NaN')
                                degrees.append(degree)
                                for iy, (ddx, yshape) in enumerate(zip(dx, yshapes)): derivatives[iy].append(ddx.reshape(yshape))
                autodegrees = nautodegrees
                autoindices = nautoindices
            toret = derivatives = [ParameterArray(derivative, derivs=degrees, param=Parameter('param_{:d}'.format(ideriv), shape=derivative[0].shape)) for ideriv, derivative in enumerate(derivatives)]
            if isinstance(self.getter_inst, dict):
                toret = Samples()
                for param in self.varied_params:
                    toret[param] = ParameterArray(self.center[param.name], param=param)
                for param, derivative in zip(self.getter_inst.keys(), derivatives):
                    derivative.param = Parameter(param)
                    toret[param] = derivative
            elif not self.getter_size:
                toret = toret[0]
            self.samples = toret

    def __call__(self, **params):
        self.run(**params)
        return self.samples
