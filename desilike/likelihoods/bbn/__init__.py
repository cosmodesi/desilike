"""Big Bang Nucleosynthesis (BBN) likelihoods."""


import jax.numpy as jnp
from desilike.base import GaussianLikelihood


class BaseBBNLikelihood(GaussianLikelihood):
    r"""Cosmological likelihood from Big Bang Nucleosynthesis (BBN).

    The class provides access to cosmological results from BBN studies. The
    likelihoods are Gaussian likelihoods on cosmological parameters such as
    :math:`\omega_\mathrm{b}` and :math:`N_\mathrm{eff}`.
    """
    def __init__(self, mean, covariance, quantities=('omega_b',), cosmo=None):
        """Initialize the model.

        Parameters
        ----------
        mean : array_like
            Mean of cosmological parameters.
        covariance : array_like
            Covariance of cosmological parameters.
        quantities : array_like
            Name of cosmological parameters.
        cosmo : BasePrimordialCosmology, default=None
            Cosmology calculator. If None, defaults to ``Cosmoprimo()``.

        """
        if cosmo is None:
            from desilike.theories.primordial_cosmology import CosmoprimoCosmology
            cosmo = CosmoprimoCosmology(fiducial='DESI')
        self.cosmo = cosmo
        self._quantities = list(quantities)
        self.cosmo.add_requirements({f'params.{quantity}': None for quantity in self._quantities})
        self.flatdata = jnp.asarray(mean)
        self.precision = jnp.linalg.inv(jnp.asarray(covariance))

    def __call__(self):
        self.flattheory = jnp.array([self.cosmo[quantity] for quantity in self._quantities])
        return super().__call__()


class Schoneberg2024BBNLikelihood(BaseBBNLikelihood):
    r"""BBN measurement from Schoneberg (2024).

    Reference
    ---------
    https://arxiv.org/abs/2401.15054

    """
    def __init__(self, cosmo=None):
        mean = [0.02196, 2.904]
        covariance = [[4.03112260e-07, 7.30390042e-05],
                      [7.30390042e-05, 4.52831584e-02]]
        quantities = ['omega_b', 'N_eff']
        super().__init__(mean=mean, covariance=covariance, quantities=quantities, cosmo=cosmo)