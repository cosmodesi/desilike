import numpy as np

from desilike.base import BaseCalculator
from desilike.theories.primordial_cosmology import external_cosmo, Cosmoprimo


class P1DPowerSpectrumExtractor(BaseCalculator):
    r"""
    Extract P1D shape parameters :math:`\Delta_{\star}^{2}` and :math:`n_{\star}` from the linear power spectrum (in velocity units).

    Parameters
    ----------
    z : float, default=3.
        Pivot redshift :math:`z_{\star}`.

    qstar : float, default=0.009
        Pivot wavenumber :math:`q_{\star}` in velocity units (km/s).

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.


    Reference
    ---------
    https://arxiv.org/pdf/2209.09895.pdf
    """
    config_fn = 'power_template.yaml'

    def initialize(self, z=3., qstar=0.009, cosmo=None):
        self.z = float(z)
        self.qstar = float(qstar)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        if external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'pk_interpolator': {'z': self.z, 'of': [('delta_cb', 'delta_cb')]}},
                                   'background': {'efunc': {'z': self.z}}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo()
            self.cosmo.params = self.params.copy()
        self.params.clear()

    def calculate(self):
        fo = self.cosmo.get_fourier()
        self.pk_dd_interpolator = fo.pk_interpolator(of='delta_cb').to_1d(z=self.z)
        q = np.geomspace(0.5 * self.qstar, 2. * self.qstar, 100)
        m = 100 * self.cosmo.efunc(self.z) / (1. + self.z)
        pq = m**3 * self.pk_dd_interpolator(q * m)
        # Eq. C7 of https://arxiv.org/pdf/2209.09895.pdf
        # lowest power first
        coeffs = np.polynomial.polynomial.Polynomial.fit(np.log(q / self.qstar), np.log(pq), deg=2, full=False).convert().coef
        self.delta2star = np.exp(coeffs[0]) * self.qstar**3 / (2. * np.pi**2)
        self.nstar = coeffs[1]
        self.alphastar = 2. * coeffs[2]
