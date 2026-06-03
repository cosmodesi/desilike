"""
pklin_mg_emulator_jax.py
------------------------
JAX (traceable / jit- & vmap-able) version of ``MgPkEmulator``
(:mod:`pklin_mg_emulator`).  Same weights, same SimpleChains MLP arithmetic, but
implemented with ``jax.numpy`` so the emulator can be traced by ``jax.jit`` /
``jax.vmap`` -- clearing "Wall 1" (the numpy ``np.array`` on tracers in
``MgPkEmulator._input_vector``).

The forward pass and (de)normalisation are identical to the numpy emulator, so
predictions match bit-for-bit (validated in
``tests/test_mg_emulator_jax.py``).

Differences from the numpy class, required for traceability:
  - ``predict_*`` take the 14 inputs and return JAX arrays (no ``.tolist()`` /
    ``np.maximum`` concretisation);
  - layer weights are pre-unpacked into ``jnp`` arrays at construction;
  - ``predict_scalars`` returns the 5-vector (order: ``SCALAR_NAMES``) rather than
    a dict.
"""

import json
import numpy as np
import jax
import jax.numpy as jnp


def _load_emulator_files(path):
    weights = np.load(f"{path}/weights.npy")
    inminmax = np.load(f"{path}/inminmax.npy")
    outminmax = np.load(f"{path}/outminmax.npy")
    with open(f"{path}/nn_setup.json") as f:
        nn_dict = json.load(f)
    return weights, inminmax, outminmax, nn_dict


def _unpack_layers(weights, nn_dict):
    """Pre-unpack flat SimpleChains weights into a list of (W, b) jnp arrays."""
    n_input = nn_dict["n_input_features"]
    n_output = nn_dict["n_output_features"]
    hidden = [v["n_neurons"] for v in nn_dict["layers"].values()]
    sizes = [n_input] + hidden + [n_output]
    params, offset = [], 0
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        W = weights[offset: offset + n_in * n_out].reshape(n_out, n_in, order='F')
        offset += n_in * n_out
        b = weights[offset: offset + n_out]
        offset += n_out
        params.append((jnp.asarray(W, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64)))
    return params


def _forward(params, x):
    out = x
    n = len(params)
    for i, (W, b) in enumerate(params):
        out = W @ out + b
        if i < n - 1:
            out = jnp.tanh(out)
    return out


class JaxMgPkEmulator:
    """JAX, jit/vmap-able emulator for MG plin, pnw and scalars."""

    SCALAR_NAMES = ["sigma8_z", "sigma8_0", "da_z", "chi_z", "e_z"]

    def __init__(self, path_plin=None, path_pnw=None, path_scalars=None):
        if path_scalars is None:
            raise ValueError("path_scalars must be provided.")
        if path_plin is None and path_pnw is None:
            raise ValueError("At least one of path_plin or path_pnw must be provided.")
        self.path_plin, self.path_pnw, self.path_scalars = path_plin, path_pnw, path_scalars

        w, inm, outm, nn = _load_emulator_files(path_scalars)
        self._sc = (_unpack_layers(w, nn), jnp.asarray(inm), jnp.asarray(outm))

        if path_plin is not None:
            w, inm, outm, nn = _load_emulator_files(path_plin)
            self._plin = (_unpack_layers(w, nn), jnp.asarray(inm), jnp.asarray(outm))
            self.k_plin = jnp.asarray(np.load(f"{path_plin}/k.npy"))
        if path_pnw is not None:
            w, inm, outm, nn = _load_emulator_files(path_pnw)
            self._pnw = (_unpack_layers(w, nn), jnp.asarray(inm), jnp.asarray(outm))
            self.k_pnw = jnp.asarray(np.load(f"{path_pnw}/k.npy"))

    @staticmethod
    def _input_vector(z, ln10As, ns, H0, ombh2, omch2,
                      mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4):
        return jnp.stack([z, ln10As, ns, H0, ombh2, omch2,
                          mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4]).astype(jnp.float64)

    @staticmethod
    def _run(emu, x):
        params, inminmax, outminmax = emu
        x_norm = (x - inminmax[:, 0]) / (inminmax[:, 1] - inminmax[:, 0])
        out_norm = _forward(params, x_norm)
        return out_norm * (outminmax[:, 1] - outminmax[:, 0]) + outminmax[:, 0]

    def predict_scalars(self, z, ln10As, ns, H0, ombh2, omch2,
                        mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4):
        x = self._input_vector(z, ln10As, ns, H0, ombh2, omch2,
                               mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        return self._run(self._sc, x)   # jnp array, order = SCALAR_NAMES

    def predict_plin(self, z, ln10As, ns, H0, ombh2, omch2,
                     mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4, sigma8_z=None):
        x = self._input_vector(z, ln10As, ns, H0, ombh2, omch2,
                               mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        if sigma8_z is None:
            sigma8_z = self._run(self._sc, x)[0]
        pk = jnp.maximum(self._run(self._plin, x) * sigma8_z ** 2, 0.0)
        return self.k_plin, pk

    def predict_pnw(self, z, ln10As, ns, H0, ombh2, omch2,
                    mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4, sigma8_z=None):
        x = self._input_vector(z, ln10As, ns, H0, ombh2, omch2,
                               mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        if sigma8_z is None:
            sigma8_z = self._run(self._sc, x)[0]
        pk = jnp.maximum(self._run(self._pnw, x) * sigma8_z ** 2, 0.0)
        return self.k_pnw, pk

    def predict_all(self, z, ln10As, ns, H0, ombh2, omch2,
                    mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4):
        """Return (k, plin, pnw, scalars) -- scalars is the 5-vector (SCALAR_NAMES)."""
        x = self._input_vector(z, ln10As, ns, H0, ombh2, omch2,
                               mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        scalars = self._run(self._sc, x)
        sigma8_z = scalars[0]
        plin = jnp.maximum(self._run(self._plin, x) * sigma8_z ** 2, 0.0)
        pnw = jnp.maximum(self._run(self._pnw, x) * sigma8_z ** 2, 0.0)
        return self.k_plin, plin, pnw, scalars
