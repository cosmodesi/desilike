"""
pklin_mg_emulator.py
--------------------
Emulator for ISiTGR MG plin, pnw, and scalar predictions (binned mu/Sigma).

Input parameters (14):
    z, ln10As, ns, H0, ombh2, omch2,
    mu1, mu2, mu3, mu4,            # gravitational slip bins
    Sigma1, Sigma2, Sigma3, Sigma4 # light deflection bins

Fixed during training: mnu=0.06 eV, w0=-1, wa=0 (LCDM background)

Scalar outputs: sigma8_z, sigma8_0, da_z [Mpc], chi_z [Mpc], e_z [H(z)/H0]

The plin/pnw emulators were trained on spectra rescaled by sigma8_z^2, so
sigma8_z from the scalar emulator is required to recover the physical spectrum.

Usage:
    from pklin_mg_emulator import MgPkEmulator

    emu = MgPkEmulator(
        path_plin="/path/to/plin",
        path_pnw="/path/to/pnw",
        path_scalars="/path/to/scalars",
    )

    scalars = emu.predict_scalars(z=0.5, ln10As=3.04, ns=0.965, H0=67.4,
                                  ombh2=0.0224, omch2=0.12,
                                  mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
                                  Sigma1=1.0, Sigma2=1.0, Sigma3=1.0, Sigma4=1.0)
    # returns dict: sigma8_z, sigma8_0, da_z, chi_z, e_z

    k, plin = emu.predict_plin(z=0.5, ln10As=3.04, ns=0.965, H0=67.4,
                                ombh2=0.0224, omch2=0.12,
                                mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
                                Sigma1=1.0, Sigma2=1.0, Sigma3=1.0, Sigma4=1.0)

    k, pnw = emu.predict_pnw(...)

    # Efficient: compute scalars once, pass sigma8_z to both spectra
    k, plin, pnw, scalars = emu.predict_all(...)
"""

import numpy as np
import json


def _load_emulator_files(path):
    weights   = np.load(f"{path}/weights.npy")
    inminmax  = np.load(f"{path}/inminmax.npy")
    outminmax = np.load(f"{path}/outminmax.npy")
    with open(f"{path}/nn_setup.json") as f:
        nn_dict = json.load(f)
    return weights, inminmax, outminmax, nn_dict


def _forward_pass(x, weights, nn_dict):
    """MLP forward pass using SimpleChains-format weights (column-major, tanh hidden, linear out)."""
    n_input      = nn_dict["n_input_features"]
    n_output     = nn_dict["n_output_features"]
    hidden_sizes = [v["n_neurons"] for v in nn_dict["layers"].values()]
    layer_sizes  = [n_input] + hidden_sizes + [n_output]

    offset = 0
    out    = x.astype(np.float64)

    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        W       = weights[offset: offset + n_in * n_out].reshape(n_out, n_in, order='F')
        offset += n_in * n_out
        b       = weights[offset: offset + n_out]
        offset += n_out
        out     = W @ out + b
        if i < len(layer_sizes) - 2:
            out = np.tanh(out)

    return out


class MgPkEmulator:
    """
    Emulator for MG plin, pnw, and scalars (ISiTGR binned mu/Sigma parameterisation).

    Parameters
    ----------
    path_plin : str, optional
    path_pnw  : str, optional
    path_scalars : str
        Required — sigma8_z from the scalar emulator rescales plin and pnw.
    """

    SCALAR_NAMES = ["sigma8_z", "sigma8_0", "da_z", "chi_z", "e_z"]

    def __init__(self, path_plin=None, path_pnw=None, path_scalars=None):
        if path_scalars is None:
            raise ValueError("path_scalars must be provided (needed for sigma8_z rescaling).")
        if path_plin is None and path_pnw is None:
            raise ValueError("At least one of path_plin or path_pnw must be provided.")

        self.path_plin    = path_plin
        self.path_pnw     = path_pnw
        self.path_scalars = path_scalars

        (self._w_sc, self._inminmax_sc,
         self._outminmax_sc, self._nn_sc) = _load_emulator_files(path_scalars)

        if path_plin is not None:
            (self._w_plin, self._inminmax_plin,
             self._outminmax_plin, self._nn_plin) = _load_emulator_files(path_plin)
            self.k_plin = np.load(f"{path_plin}/k.npy")

        if path_pnw is not None:
            (self._w_pnw, self._inminmax_pnw,
             self._outminmax_pnw, self._nn_pnw) = _load_emulator_files(path_pnw)
            self.k_pnw = np.load(f"{path_pnw}/k.npy")

    @staticmethod
    def _input_vector(z, ln10As, ns, H0, ombh2, omch2,
                      mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4):
        return np.array([z, ln10As, ns, H0, ombh2, omch2,
                         mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4])

    def _run(self, weights, inminmax, outminmax, nn_dict, x):
        x_norm   = (x - inminmax[:, 0]) / (inminmax[:, 1] - inminmax[:, 0])
        out_norm = _forward_pass(x_norm, weights, nn_dict)
        return out_norm * (outminmax[:, 1] - outminmax[:, 0]) + outminmax[:, 0]

    def predict_scalars(self, z, ln10As, ns, H0, ombh2, omch2,
                        mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4):
        """
        Predict scalar cosmological quantities.

        Returns
        -------
        dict with keys: sigma8_z, sigma8_0, da_z [Mpc], chi_z [Mpc], e_z [H(z)/H0]
        """
        x = self._input_vector(z, ln10As, ns, H0, ombh2, omch2,
                               mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        scalars = self._run(self._w_sc, self._inminmax_sc, self._outminmax_sc, self._nn_sc, x)
        return dict(zip(self.SCALAR_NAMES, scalars.tolist()))

    def _predict_pk(self, weights, inminmax, outminmax, k, nn_dict, x, sigma8_z):
        pk_rescaled = self._run(weights, inminmax, outminmax, nn_dict, x)
        return k, np.maximum(pk_rescaled * sigma8_z**2, 0.0)

    def predict_plin(self, z, ln10As, ns, H0, ombh2, omch2,
                     mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4,
                     sigma8_z=None):
        """
        Predict P_lin(k).

        sigma8_z : float, optional
            If provided, skips an internal scalar emulator call. Pass from
            predict_all() to compute sigma8_z only once.

        Returns (k [h/Mpc], plin [(Mpc/h)^3])
        """
        if self.path_plin is None:
            raise ValueError("path_plin was not provided.")
        x = self._input_vector(z, ln10As, ns, H0, ombh2, omch2,
                               mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        if sigma8_z is None:
            sigma8_z = self.predict_scalars(z, ln10As, ns, H0, ombh2, omch2,
                                            mu1, mu2, mu3, mu4,
                                            Sigma1, Sigma2, Sigma3, Sigma4)["sigma8_z"]
        return self._predict_pk(self._w_plin, self._inminmax_plin, self._outminmax_plin,
                                self.k_plin, self._nn_plin, x, sigma8_z)

    def predict_pnw(self, z, ln10As, ns, H0, ombh2, omch2,
                    mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4,
                    sigma8_z=None):
        """
        Predict P_nw(k).

        Returns (k [h/Mpc], pnw [(Mpc/h)^3])
        """
        if self.path_pnw is None:
            raise ValueError("path_pnw was not provided.")
        x = self._input_vector(z, ln10As, ns, H0, ombh2, omch2,
                               mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        if sigma8_z is None:
            sigma8_z = self.predict_scalars(z, ln10As, ns, H0, ombh2, omch2,
                                            mu1, mu2, mu3, mu4,
                                            Sigma1, Sigma2, Sigma3, Sigma4)["sigma8_z"]
        return self._predict_pk(self._w_pnw, self._inminmax_pnw, self._outminmax_pnw,
                                self.k_pnw, self._nn_pnw, x, sigma8_z)

    def predict_all(self, z, ln10As, ns, H0, ombh2, omch2,
                    mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4):
        """
        Predict plin, pnw, and scalars in one call (scalar emulator runs once).

        Returns (k, plin, pnw, scalars_dict)
        """
        if self.path_plin is None or self.path_pnw is None:
            raise ValueError("Both path_plin and path_pnw must be provided for predict_all.")
        scalars  = self.predict_scalars(z, ln10As, ns, H0, ombh2, omch2,
                                        mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4)
        sigma8_z = scalars["sigma8_z"]
        k, plin  = self.predict_plin(z, ln10As, ns, H0, ombh2, omch2,
                                     mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4,
                                     sigma8_z=sigma8_z)
        _, pnw   = self.predict_pnw(z, ln10As, ns, H0, ombh2, omch2,
                                    mu1, mu2, mu3, mu4, Sigma1, Sigma2, Sigma3, Sigma4,
                                    sigma8_z=sigma8_z)
        return k, plin, pnw, scalars
