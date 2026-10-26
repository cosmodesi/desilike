"""Big Bang Nucleosynthesis (BBN) likelihoods."""


import jax.numpy as jnp
from desilike.base import GaussianLikelihood
from desilike.parameter import Variable


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
        self.flatdata = Variable(f'{type(self).__name__}.flatdata', value=jnp.asarray(mean))
        self.precision = jnp.linalg.inv(jnp.asarray(covariance))

    def __post_init__(self, *args, **kwargs):
        self.cosmo.add_requirements({f'params.{quantity}': None for quantity in self._quantities})

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

#: Observed primordial abundances, as ``(quantity, mean, standard deviation)``, unscaled.
#: ``quantity`` names a :class:`~desilike.theories.bbn.LINXAbundances` output.
BBN_MEASUREMENTS = {
    # Aver, Olive & Skillman (2015), https://arxiv.org/abs/1503.08146 -- used in the LINX scripts
    'Aver2015': ('Y_p', 0.2449, 0.0040),
    # Aver et al. (2021), https://arxiv.org/abs/2010.04180
    'Aver2021': ('Y_p', 0.2453, 0.0034),
    # Aver et al. (2026), LBT Y_p Project IV, https://arxiv.org/abs/2601.22238 -- as used in arXiv:2609.12065
    'Aver2026': ('Y_p', 0.2458, 0.0013),
    # Cooke, Pettini & Steidel (2018), https://arxiv.org/abs/1710.11129
    'Cooke2018': ('D_H', 2.527e-5, 0.030e-5),
}


class LINXBBNLikelihood(GaussianLikelihood):
    r"""BBN likelihood on observed primordial abundances, with theory predictions from LINX.

    Unlike :class:`Schoneberg2024BBNLikelihood`, a compressed Gaussian on
    :math:`(\omega_\mathrm{b}, N_\mathrm{eff})`, this compares abundances computed at each point
    by :class:`~desilike.theories.bbn.LINXAbundances` with independent Gaussian measurements,
    so it holds for any parameters the theory varies (neutron lifetime, nuclear rates, ...).

    Reference
    ---------
    https://arxiv.org/abs/2408.14538

    Parameters
    ----------
    theory : LINXAbundances, default=None
        Abundance calculator. If ``None``, defaults to ``LINXAbundances(cosmo=cosmo)``.
    cosmo : PrimordialCosmology, default=None
        Cosmology calculator, used only when ``theory`` is ``None``.
    measurements : list, default=('Aver2021', 'Cooke2018')
        Keys of :data:`BBN_MEASUREMENTS` ('Aver2015', 'Aver2021', 'Aver2026', 'Cooke2018'), or ``(quantity, mean, std)`` tuples.
    """
    def __init__(self, theory=None, cosmo=None, measurements=('Aver2021', 'Cooke2018')):
        if theory is None:
            from desilike.theories.bbn import LINXAbundances
            theory = LINXAbundances(cosmo=cosmo)
        self.theory = theory
        measurements = [BBN_MEASUREMENTS[measurement] if isinstance(measurement, str) else tuple(measurement)
                        for measurement in measurements]
        self._quantities = [quantity for quantity, mean, std in measurements]
        self.flatdata = Variable(f'{type(self).__name__}.flatdata', value=jnp.array([mean for quantity, mean, std in measurements]))
        self.precision = jnp.diag(jnp.array([std for quantity, mean, std in measurements])**(-2))

    def __call__(self):
        self.flattheory = jnp.array([getattr(self.theory, quantity) for quantity in self._quantities])
        return super().__call__()
