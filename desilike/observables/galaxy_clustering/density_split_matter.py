"""Window projection for density-split spectra with quantile or pair axes."""
import numpy as np
import jax.numpy as jnp
from ...base import Calculator


class WindowedDensitySplitPowerSpectrumMultipolesObservable(Calculator):
    """Adapter from quantile theory to an lsstypes window."""

    def __init__(self, data=None, theory=None, window=None, name="pqm"):
        self.name = str(name)
        self.data = data
        self.theory = theory
        self.window = window
        self.flatdata = np.asarray(data.value(), dtype="f8").ravel()
        value = np.asarray(window.value(), dtype="f8")
        if value.shape[0] != self.flatdata.size:
            raise ValueError(f"{self.name} data and window observable sizes do not match")

    def __call__(self):
        matrix = jnp.asarray(self.window.value())
        flatpower = jnp.ravel(self.theory.power)
        if matrix.shape[1] != flatpower.size:
            raise ValueError(f"{self.name} theory and window theory sizes do not match")
        self.flattheory = matrix @ flatpower
        return self.flattheory

    def tree_flatten(self):
        return [self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.flattheory = children[0]
        return obj
