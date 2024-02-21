import os
from collections import UserDict

import numpy as np

from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseLikelihood, BaseGaussianLikelihood


class BaseSNLikelihood(BaseGaussianLikelihood):
    """
    Base likelihood for supernovae.

    Parameters
    ----------
    config_fn : str, Path
        Configuration file.

    data_dir : str, Path, default=None
        Data directory. Defaults to path saved in desilike's configuration,
        as provided by :class:`Installer` if likelihood has been installed.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.
    """
    def initialize(self, config_fn, data_dir=None, cosmo=None):
        if data_dir is None:
            from desilike.install import Installer
            data_dir = Installer()[self.installer_section]['data_dir']
        self.config = self.read_config(os.path.join(data_dir, config_fn))
        self.light_curve_params = self.read_light_curve_params(os.path.join(data_dir, self.config['data_file']))
        self.covariance = self.read_covariance(os.path.join(data_dir, self.config['mag_covmat_file']))
        self.cosmo = cosmo
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'background': {'luminosity_distance': {'z': np.linspace(0., 10., 1000)}}}
        elif self.cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            self.cosmo = Cosmoprimo()
        BaseLikelihood.initialize(self)

    def read_config(self, fn):

        class Parser(UserDict):

            def __init__(self, fn):
                data = {}
                with open(fn, 'r') as file:
                    for line in file.readlines():
                        kv = [v.strip() for v in line.split('=')]
                        if len(kv) == 1:
                            data[kv] = ''
                        elif len(kv) == 2:
                            data[kv[0]] = kv[1]
                        else:
                            raise ValueError('Could not read {} of {}'.format(line, fn))
                self.data = data

            def int(self, key):
                return int(self[key])

            def float(self, key):
                return float(self[key])

            def bool(self, key):
                toret = self[key].lower()
                if toret in ['true', 't']:
                    return True
                if toret in ['false', 'f']:
                    return False
                raise ValueError('Cannot interpret {} as bool'.format(key))

        return Parser(fn)

    def read_covariance(self, fn):
        if self.mpicomm.rank == 0:
            self.log_info('Loading covariance from {}'.format(fn))
        with open(fn, 'r') as file:
            size = int(file.readline())
        return np.loadtxt(fn, skiprows=1).reshape(size, size)

    def read_light_curve_params(self, fn, header='#', sep=' ', skip=None):
        if self.mpicomm.rank == 0:
            self.log_info('Loading light-curve from {}'.format(fn))
        with open(fn, 'r') as file:
            start = True
            for iline, line in enumerate(file.readlines()):
                if skip is not None:
                    if isinstance(skip, str):
                        if line.strip().startswith(skip):
                            continue
                    elif iline <= skip:
                        continue
                if start:
                    names = [name.strip() for name in line[len(header):].split(sep)]
                    values = {name: [] for name in names}
                    start = False
                    continue
                line = [el for el in line.split(sep) if el]
                for name, value in zip(names, line):
                    try: value = float(value)
                    except ValueError: pass  # str
                    values[name].append(value)
        return {name: np.array(value) for name, value in values.items()}
