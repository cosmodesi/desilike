"""Union3 type Ia supernovae likelihoods."""

import os

import jax.numpy as jnp

from desilike.parameter import Parameter, VariableCollection
from .base import BaseSNLikelihood


class _BaseUnion3SNLikelihood(BaseSNLikelihood):
    """Shared setup for the two Union3 data releases (see :class:`Union3SNLikelihood`
    and :class:`Union3p1SNLikelihood`).

    Each release ships as a single FITS file (read with ``fitsio``) packing the whole
    Hubble diagram into one ``(n+1, n+1)`` matrix: the first row (excluding the corner)
    is redshift, the first column is the distance modulus, and the remaining ``(n, n)``
    block is the *precision* (inverse covariance) matrix directly (no inversion needed).

    The dependence on ``h`` is absorbed into the nuisance offset ``dM``: since
    cosmoprimo's ``luminosity_distance`` is in Mpc/h, ``H0 * d_L^{physical} = 100 * h
    * (d_L / h) = 100 * d_L``, so no explicit division by ``h`` is needed here.
    """
    installer_section = 'Union3SNLikelihood'
    _zname = 'zcmb'

    @classmethod
    def propose_params(cls):
        return VariableCollection([Parameter('dM', value=-9.2, prior=dict(limits=[-20., 20.]), latex=r'\Delta \mathcal{M}_B')])

    def read_light_curve_params(self, fn):
        import fitsio
        data = fitsio.read(fn)
        return {'zcmb': data[0, 1:], 'mb': data[1:, 0]}

    def read_covariance(self, fn):
        # NB: this is the precision matrix directly, not the covariance (see __init__).
        import fitsio
        data = fitsio.read(fn)
        return data[1:, 1:]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatdata = jnp.asarray(self.light_curve_params['mb'])
        self.precision = jnp.asarray(self.covariance)
        self.cosmo.add_requirements({'background.luminosity_distance': [{'z': self.light_curve_params['zcmb']}]})

    def __call__(self):
        z = self.light_curve_params['zcmb']
        dL = self.cosmo.get_background().luminosity_distance(z=z)
        self.flattheory = 5 * jnp.log10(100 * dL) + 25 + self.dM.value
        return super().__call__()

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download

        data_fn = os.path.join(data_dir, cls.data_file)

        if installer.reinstall or not exists_path(data_fn):
            github = 'https://raw.githubusercontent.com/rubind/union3_release/master/'
            download(github + cls.data_file, data_fn)

        installer.write({cls.installer_section: {'data_dir': data_dir}})


class Union3SNLikelihood(_BaseUnion3SNLikelihood):
    """
    Likelihood for the Union3 & UNITY1.5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/pdf/2311.12098.pdf
    """
    data_file = covariance_file = 'mu_mat_union3_cosmo=2_mu.fits'


class Union3p1SNLikelihood(_BaseUnion3SNLikelihood):
    """
    Likelihood for the Union3.1 & UNITY1.8 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/pdf/2311.12098.pdf
    """
    data_file = covariance_file = 'mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits'
