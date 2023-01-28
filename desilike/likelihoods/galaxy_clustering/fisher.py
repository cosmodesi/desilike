import numpy as np
from scipy import special

from desilike.jax import numpy as jnp
from desilike.base import EnsembleCalculator
from desilike.likelihoods.base import BaseGaussianLikelihood
from desilike import utils


class SNWeightedPowerSpectrumLikelihood(BaseGaussianLikelihood):
    r"""
    Likelihood for Fisher forecasts, integrating anisotropic signal-to-noise
    over cosine angle to the line-of-sight :math:`\mu` and wavenumber :math:`k`.

    Parameters
    ----------
    data : dict, default=None
        Parameters to be passed to ``theories`` to generate fiducial measurement.
    
    theories : list, BaseCalculator
        List of theories.

    footprints : list, BaseFootprint
        List of (or single) footprints for input ``theories``.
    
    klim : dict, default=None
        Wavenumber cut, e.g. ``(0.01, 0.2)``.
    
    mu : int, default=50
        Number of :math:`\mu`-bins to use (in :math:`[0, 1]`).
    """
    def initialize(self, data=None, theories=None, footprints=None, klim=None, mu=50):
        if not utils.is_sequence(theories):
            theories = [theories]
        self.theories = theories
        if not utils.is_sequence(footprints):
            footprints = [footprints] * len(self.theories)
        self.footprints = footprints
        if klim is not None:
            k = np.linspace(*klim, num=100)
            for theory in self.theories: theory.init.update(k=k)
        self.theories = EnsembleCalculator(calculators=theories)
        if np.ndim(mu) == 0:
            self.mu = np.linspace(0., 1., mu)
        else:
            self.mu = np.asarray(mu)
        muw = utils.weights_trapz(self.mu)
        prefactor = 4 * np.pi / (2 * (2 * np.pi)**3) * muw
        self.flatdata, self.precision = [], []
        self.theories(**(data or {}))
        for theory, footprint in zip(self.theories, self.footprints):
            pkmu = self._get_pkmu(theory)
            precision = prefactor * footprint.volume * (theory.k**2 * utils.weights_trapz(k))[:, None] * (pkmu + footprint.shotnoise)**(-2)
            self.flatdata.append(pkmu.ravel())
            self.precision.append(precision.ravel())
        self.flatdata, self.precision = np.concatenate(self.flatdata), np.concatenate(self.precision)
        self.runtime_info.requires = self.theories
        super(SNWeightedPowerSpectrumLikelihood, self).initialize(data=self.flatdata, precision=self.precision)

    def _get_pkmu(self, theory):
        pkell = theory.power
        pkmu = 0.
        for ill, ell in enumerate(theory.ells): pkmu += pkell[ill][:, None] * special.legendre(ell)(self.mu)
        return pkmu

    @property
    def flattheory(self):
        return jnp.concatenate([self._get_pkmu(theory).ravel() for theory in self.theories], axis=0)
