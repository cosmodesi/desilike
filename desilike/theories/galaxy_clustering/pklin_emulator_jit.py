"""
pklin_emulator_jit.py
---------------------
JAX-based emulator for plin and pnw predictions using trained SimpleChains weights
(GR, mnu/w0/wa CDM).  Copied from desilike-Arnaud-bk so the MG no-wiggle
approximation ``pnw_MG = pnw_GR * plin_MG / plin_GR`` can reuse the *same*
original GR emulators.

The only change from the reference is that ``jaxmapse`` is imported lazily (inside
``_compute_growth_factor``): the GR-ratio no-wiggle path always supplies ``D``
explicitly (and the ``As * D^2`` normalisation cancels in the wiggle ratio
``pnw_GR / plin_GR`` anyway), so it never needs jaxmapse.

Usage:
    from .pklin_emulator_jit import PkEmulator

    emu = PkEmulator(path_plin="/path/plin", path_pnw="/path/pnw")
    k, plin, pnw = emu.predict_both(z=0.5, ln10As=3.0, ns=0.96, H0=67.0,
                                    ombh2=0.022, omch2=0.12, Mnu=0.06,
                                    w0=-1.0, wa=0.0, D=1.0)
"""

import numpy as np
import json
import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# File loading (numpy, runs once at init)
# ---------------------------------------------------------------------------

def _load_emulator_files(path):
    """Load weights and metadata from an emulator directory."""
    weights   = np.load(f"{path}/weights.npy")
    inminmax  = np.load(f"{path}/inminmax.npy")   # shape: (9, 2)
    outminmax = np.load(f"{path}/outminmax.npy")  # shape: (nk, 2)
    k         = np.load(f"{path}/k.npy")
    with open(f"{path}/nn_setup.json") as f:
        nn_dict = json.load(f)
    return weights, inminmax, outminmax, k, nn_dict


def _parse_architecture(nn_dict):
    """Extract layer sizes from nn_dict."""
    n_input      = nn_dict["n_input_features"]
    n_output     = nn_dict["n_output_features"]
    hidden_sizes = [v["n_neurons"] for v in nn_dict["layers"].values()]
    return [n_input] + hidden_sizes + [n_output]


def _unpack_weights(weights, layer_sizes):
    """
    Unpack flat weight vector into list of (W, b) JAX arrays.
    SimpleChains layout: weights (column-major/Fortran order) then biases per layer.
    """
    params = []
    offset = 0
    for i in range(len(layer_sizes) - 1):
        n_in  = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        W = jnp.array(weights[offset: offset + n_in * n_out].reshape(n_out, n_in, order='F'))
        offset += n_in * n_out
        b = jnp.array(weights[offset: offset + n_out])
        offset += n_out
        params.append((W, b))
    return params


# ---------------------------------------------------------------------------
# JAX forward pass — JIT-compatible via static n_layers
# ---------------------------------------------------------------------------

def _forward_pass_jax(x, params, n_layers):
    """JAX forward pass through the MLP: tanh on all hidden layers, linear output."""
    out = x
    for i in range(n_layers):
        W, b = params[i]
        out  = W @ out + b
        if i < n_layers - 1:
            out = jnp.tanh(out)
    return out


# ---------------------------------------------------------------------------
# Growth factor (only used when D is not supplied explicitly)
# ---------------------------------------------------------------------------

def _compute_growth_factor(z, ombh2, omch2, Mnu, h, w0, wa):
    """Compute D(z) using jaxmapse, matching Effort.jl's D_z."""
    import jaxmapse  # lazy: the GR-ratio path always passes D explicitly
    cosmo = jaxmapse.w0waCDMCosmology(
        ln10As=3.0,   # does not affect D(z)
        ns=0.96,      # does not affect D(z)
        h=h,
        omega_b=ombh2,
        omega_c=omch2,
        m_nu=Mnu,
        w0=w0,
        wa=wa,
    )
    return float(cosmo.D_z(z))


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PkEmulator:
    """
    JAX-based emulator for plin and pnw matter power spectra (GR).
    Supports JIT compilation and user-supplied growth factor D.

    Parameters
    ----------
    path_plin : str, optional
        Path to plin emulator directory containing:
        weights.npy, inminmax.npy, outminmax.npy, k.npy, nn_setup.json
    path_pnw : str, optional
        Path to pnw emulator directory (same structure).

    Notes
    -----
    - predict_plin / predict_pnw / predict_both accept an optional D argument.
      If D is not provided, it is computed internally via jaxmapse.
    - get_jit_predict_plin / get_jit_predict_pnw return JIT-compiled functions
      for fast repeated evaluation. D must be provided explicitly for JIT calls.
    """

    def __init__(self, path_plin=None, path_pnw=None):
        if path_plin is None and path_pnw is None:
            raise ValueError("At least one of path_plin or path_pnw must be provided.")

        self.path_plin = path_plin
        self.path_pnw  = path_pnw

        if path_plin is not None:
            self._plin = self._load(path_plin)

        if path_pnw is not None:
            self._pnw = self._load(path_pnw)

    def _load(self, path):
        """Load and preprocess emulator files into JAX arrays."""
        weights, inminmax, outminmax, k, nn_dict = _load_emulator_files(path)
        layer_sizes = _parse_architecture(nn_dict)
        params      = _unpack_weights(weights, layer_sizes)
        return {
            "k":          jnp.array(k),
            "inminmax":   jnp.array(inminmax),
            "outminmax":  jnp.array(outminmax),
            "params":     params,
            "n_layers":   len(layer_sizes) - 1,
        }

    def _get_D(self, z, ombh2, omch2, Mnu, H0, w0, wa, D=None):
        """Return D(z) — either user-supplied or computed via jaxmapse."""
        if D is not None:
            return float(D)
        return _compute_growth_factor(z, ombh2, omch2, Mnu, H0 / 100, w0, wa)

    def _predict(self, emu, z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D=None):
        """Core prediction — normalise, forward pass, denormalise, rescale."""
        x_raw     = jnp.array([z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa])
        inminmax  = emu["inminmax"]
        outminmax = emu["outminmax"]
        params    = emu["params"]
        n_layers  = emu["n_layers"]

        # Normalise input to [0, 1]
        x_norm = (x_raw - inminmax[:, 0]) / (inminmax[:, 1] - inminmax[:, 0])

        # Forward pass
        pk_norm = _forward_pass_jax(x_norm, params, n_layers)

        # Denormalise output
        pk_rescaled = pk_norm * (outminmax[:, 1] - outminmax[:, 0]) + outminmax[:, 0]

        # Undo As * D(z)^2 rescaling
        As     = jnp.exp(jnp.array(ln10As)) * 1e-10
        D_val  = self._get_D(z, ombh2, omch2, Mnu, H0, w0, wa, D)
        factor = As * D_val**2
        pk     = jnp.maximum(pk_rescaled * factor, 0.0)

        return np.array(emu["k"]), np.array(pk)

    def predict_plin(self, z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D=None):
        """Predict P_lin(k).  Returns (k [h/Mpc], plin [(Mpc/h)^3])."""
        if self.path_plin is None:
            raise ValueError("path_plin was not provided at initialisation.")
        return self._predict(self._plin, z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D)

    def predict_pnw(self, z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D=None):
        """Predict P_nw(k).  Returns (k [h/Mpc], pnw [(Mpc/h)^3])."""
        if self.path_pnw is None:
            raise ValueError("path_pnw was not provided at initialisation.")
        return self._predict(self._pnw, z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D)

    def predict_both(self, z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D=None):
        """Predict both plin and pnw. D is computed once and reused.

        Returns (k, plin, pnw)."""
        if self.path_plin is None or self.path_pnw is None:
            raise ValueError("Both path_plin and path_pnw must be provided for predict_both.")
        D_val = self._get_D(z, ombh2, omch2, Mnu, H0, w0, wa, D)
        k,    plin = self.predict_plin(z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D=D_val)
        _,    pnw  = self.predict_pnw( z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D=D_val)
        return k, plin, pnw

    def get_jit_predict_plin(self):
        """Return a JIT-compiled prediction function for plin (D must be provided)."""
        return self._make_jit_fn(self._plin)

    def get_jit_predict_pnw(self):
        """Return a JIT-compiled prediction function for pnw (D must be provided)."""
        return self._make_jit_fn(self._pnw)

    def _make_jit_fn(self, emu):
        """Build and return a JIT-compiled predict function for a given emulator."""
        inminmax  = emu["inminmax"]
        outminmax = emu["outminmax"]
        params    = emu["params"]
        n_layers  = emu["n_layers"]
        k_np      = jnp.array(emu["k"])   # return as numpy always

        @jax.jit
        def _jit_core(x_raw, As, D):
            x_norm      = (x_raw - inminmax[:, 0]) / (inminmax[:, 1] - inminmax[:, 0])
            pk_norm     = _forward_pass_jax(x_norm, params, n_layers)
            pk_rescaled = pk_norm * (outminmax[:, 1] - outminmax[:, 0]) + outminmax[:, 0]
            pk          = jnp.maximum(pk_rescaled * As * D**2, 0.0)
            return pk

        def predict_jit(z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa, D):
            x_raw = jnp.array([z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa])
            As    = jnp.exp(jnp.array(ln10As)) * 1e-10
            pk    = _jit_core(x_raw, As, jnp.array(D))
            return k_np, jnp.array(pk)

        return predict_jit
