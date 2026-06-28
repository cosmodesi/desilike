"""DES-Y5 type Ia supernovae likelihoods."""

import os

import numpy as np
import jax.numpy as jnp

from desilike.parameter import Parameter, VariableCollection
from .base import BaseSNLikelihood


class _BaseDESY5SNLikelihood(BaseSNLikelihood):
    """Shared setup for the two DES-Y5 data releases (see :class:`DESY5v1SNLikelihood`
    and :class:`DESY5DovekieSNLikelihood`): distance-modulus theory and ``install()``.

    Reference
    ---------
    https://arxiv.org/abs/2401.02929
    """
    installer_section = 'DESY5SNLikelihood'
    _zname = 'zHD'

    @classmethod
    def propose_params(cls):
        return VariableCollection([Parameter('Mb', value=0., prior=dict(limits=[-5., 5.]), latex='M_b')])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flatdata = self.light_curve_params['MU'] - 5 * np.log10((1 + self.light_curve_params['zHEL']) / (1 + self.light_curve_params['zHD']))
        self.flatdata = jnp.asarray(flatdata)

    def __post_init__(self, *args, **kwargs):
        self.cosmo.add_requirements({'background.luminosity_distance': [{'z': self.light_curve_params['zHD']}]})

    def __call__(self):
        z = self.light_curve_params['zHD']
        dL = self.cosmo.get_background().luminosity_distance(z=z)
        self.flattheory = 5 * jnp.log10(dL / self.cosmo['h']) + 25 + self.Mb.value
        return super().__call__()

    @classmethod
    def install(cls, installer):
        try:
            data_dir = installer[cls.installer_section]['data_dir']
        except KeyError:
            data_dir = installer.data_dir(cls.installer_section)

        from desilike.install import exists_path, download, extract

        data_fn = os.path.join(data_dir, cls.data_file)
        cov_fn = os.path.join(data_dir, cls.covariance_file)

        if installer.reinstall or not exists_path(data_fn):
            # Only .txt files are served gzipped upstream; .csv/.npz files are not.
            for fn in [data_fn, cov_fn]:
                fngz = fn.replace('.txt', '.txt.gz')
                download(os.path.join(cls.github_dir, os.path.basename(fngz)), fngz)
                if fngz.endswith('.gz'):
                    extract(fngz, fn, remove=True)

        installer.write({cls.installer_section: {'data_dir': data_dir}})


class DESY5v1SNLikelihood(_BaseDESY5SNLikelihood):
    """
    Likelihood for the DES-Y5 type Ia supernovae sample, original v1.0 data release
    matching the DES-SN5YR Y5 cosmology paper.

    Covariance is a plain-text stat+sys systematics matrix; the per-SN statistical
    variance (``MUERR_FINAL``) is added on the diagonal.

    Reference
    ---------
    https://arxiv.org/abs/2401.02929
    """
    data_file = 'DES-SN5YR_HD.csv'
    covariance_file = 'STAT+SYS.txt'
    github_dir = 'https://raw.githubusercontent.com/des-science/DES-SN5YR/v1.0/4_DISTANCES_COVMAT/'

    def read_light_curve_params(self, fn):
        return super().read_light_curve_params(fn, header='', sep=',', skip='#')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.covariance = self.covariance + np.diag(self.light_curve_params['MUERR_FINAL']) ** 2
        self.precision = jnp.linalg.inv(jnp.asarray(self.covariance))


class DESY5DovekieSNLikelihood(_BaseDESY5SNLikelihood):
    """
    Likelihood for the DES-Y5 type Ia supernovae sample, "Dovekie" recalibration
    (``main`` branch of the data repository).

    Light-curve parameters use a ``VARNAMES:``/``SN:``-prefixed text format, and the
    covariance file directly stores the upper triangle of the *precision* (inverse
    covariance) matrix as a ``.npz`` archive (unpacking logic per
    ``5_COSMOLOGY/Dovekie_cosmosis_likelihood.py`` in the data repository), already
    including all statistical and systematic contributions (no extra diagonal term
    is added).

    Reference
    ---------
    https://arxiv.org/abs/2401.02929
    """
    data_file = 'DES-Dovekie_HD.csv'
    covariance_file = 'STAT+SYS.npz'
    github_dir = 'https://raw.githubusercontent.com/des-science/DES-SN5YR/main/4_DISTANCES_COVMAT/'

    def read_light_curve_params(self, fn):
        """Parse the 'VARNAMES:'/'SN:'-prefixed, whitespace-separated Dovekie light-curve file."""
        names, values = None, None
        with open(fn, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('VARNAMES:'):
                    names = line[len('VARNAMES:'):].split()
                    values = {name: [] for name in names}
                    continue
                if line.startswith('SN:'):
                    row = line[len('SN:'):].split()
                    for name, value in zip(names, row):
                        try: value = float(value)
                        except ValueError: pass  # str, e.g. CID
                        values[name].append(value)
        return {name: np.array(value) for name, value in values.items()}

    def read_covariance(self, fn):
        """Unpack the Dovekie .npz precision matrix (upper-triangle-packed, symmetrized)."""
        data = np.load(fn)
        n = int(data['nsn'][0])
        precision = np.zeros((n, n))
        precision[np.triu_indices(n)] = data['cov']
        i_lower = np.tril_indices(n, -1)
        precision[i_lower] = precision.T[i_lower]
        return precision

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.covariance already holds the full stat+sys precision matrix directly.
        self.precision = jnp.asarray(self.covariance)
