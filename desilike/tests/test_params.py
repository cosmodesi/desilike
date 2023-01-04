import numpy as np

from desilike.parameter import Parameter


class ParameterArray(np.ndarray):

    def __new__(cls, value, param, copy=False, dtype=None, **kwargs):
        """
        Initalize :class:`array`.

        Parameters
        ----------
        value : array
            Local array value.

        copy : bool, default=False
            Whether to copy input array.

        dtype : dtype, default=None
            If provided, enforce this dtype.
        """
        value = np.array(value, copy=copy, dtype=dtype, **kwargs)
        obj = value.view(cls)
        obj.param = param
        return obj

    def __array_finalize__(self, obj):
        self.param = getattr(obj, 'param', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, ParameterArray):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, ParameterArray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], ParameterArray):
                inputs[0].param = self.param
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(ParameterArray)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        for result in results:
            if isinstance(result, ParameterArray):
                result.param = self.param

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.param, self)

    def __reduce__(self):
        # See https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        # Get the parent's __reduce__ tuple
        pickled_state = super(ParameterArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.param.__getstate__(),)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.param = Parameter.from_state(state[-1])  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(ParameterArray, self).__setstate__(state[:-1])

    def __getstate__(self):
        return {'value': self.view(np.ndarray), 'param': self.param.__getstate__()}

    @classmethod
    def from_state(cls, state):
        return cls(state['value'], Parameter.from_state(state['param']))


class ParameterArray(np.ndarray):

    def __new__(cls, value, param=None, derivs=None, copy=False, dtype=None, **kwargs):
        """
        Initalize :class:`array`.

        Parameters
        ----------
        value : array
            Local array value.

        copy : bool, default=False
            Whether to copy input array.

        dtype : dtype, default=None
            If provided, enforce this dtype.
        """
        value = np.array(value, copy=copy, dtype=dtype, **kwargs)
        obj = value.view(cls)
        obj.param = None if param is None else Parameter(param)
        obj.derivs = None if derivs is None else [frozenset(str(p) for p in deriv) for deriv in derivs]
        return obj

    def __array_finalize__(self, obj):
        self.param = getattr(obj, 'param', None)
        self.derivs = getattr(obj, 'derivs', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, ParameterArray):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, ParameterArray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], ParameterArray):
                inputs[0].param = self.param
                inputs[0].derivs = self.derivs
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(ParameterArray)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        for result in results:
            if isinstance(result, ParameterArray):
                result.param = self.param
                result.derivs = self.derivs

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.param, self.derivs, self)

    def __reduce__(self):
        # See https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
        # Get the parent's __reduce__ tuple
        pickled_state = super(ParameterArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.param.__getstate__(), self.derivs)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.param = Parameter.from_state(state[-2])  # Set the info attribute
        self.derivs = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(ParameterArray, self).__setstate__(state[:-2])

    def __getstate__(self):
        return {'value': self.view(np.ndarray), 'param': None if self.param is None else self.param.__getstate__(), 'derivs': self.derivs}

    def _index(self, deriv):
        if self.derivs is not None:
            deriv_types = (Parameter, str)
            tuple_deriv = (deriv,) if not isinstance(deriv, tuple) else deriv
            if all(isinstance(param, deriv_types) for param in tuple_deriv):
                deriv = self.derivs.index(frozenset(str(param) for param in tuple_deriv))
        return deriv

    def __getitem__(self, deriv):
        return super(ParameterArray, self).__getitem__(self._index(deriv))

    def __setitem__(self, deriv, item):
        return super(ParameterArray, self).__setitem__(self._index(deriv), item)

    @classmethod
    def from_state(cls, state):
        return cls(state['value'], None if state.get('param', None) is None else Parameter.from_state(state['param']), state.get('derivs', None))


if __name__ == '__main__':

    param = Parameter('a')
    array = ParameterArray(np.ones(4), param=param)

    array = ParameterArray(np.ones((1, 4)), param=param, derivs=[(param,)])
    print((array + array)[param])
    array[param] += 1.
    print((array + array)[param])

    array = ParameterArray(np.ones((1, 4)))
    array *= 2
