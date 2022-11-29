import numpy as np
from scipy import constants

from .base import TrapzTheoryPowerSpectrumMultipoles
from .power_template import FixedPowerSpectrumTemplate


class PNGTracerPowerSpectrumMultipoles(TrapzTheoryPowerSpectrumMultipoles):

    config_fn = 'primordial_non_gaussianity.yaml'

    def initialize(self, *args, template=None, method='prim', mode='b-p', **kwargs):
        super(PNGTracerPowerSpectrumMultipoles, self).initialize(*args, **kwargs)
        self.kin = np.insert(self.k, 0, 1e-4)
        if template is None:
            template = FixedPowerSpectrumTemplate(k=self.kin)
        self.template = template
        self.template.update(k=self.kin)
        self.method = str(method)
        self.mode = str(mode)
        keep_params = ['b1', 'sigmas', 'sn0']
        if self.mode == 'bphi':
            keep_params += ['fnl_loc', 'bphi']
        elif self.mode == 'b-p':
            keep_params += ['fnl_loc', 'p']
        elif self.mode == 'bfnl':
            keep_params += ['bfnl_loc']
        else:
            raise ValueError('Unknown mode {}; it must be one of ["bphi", "b-p", "bfnl_loc"]'.format(self.mode))
        self.params = self.params.select(basename=keep_params)

    def calculate(self, b1=2., sigmas=0., sn0=0., **params):
        pk_dd = self.template.pk_dd
        cosmo = self.template.cosmo
        pk_prim = cosmo.get_primordial(mode='scalar').pk_interpolator()(self.kin)  # power_prim is ~ k^(n_s - 1)
        if self.method == 'prim':
            pphi_prim = 9 / 25 * 2 * np.pi**2 / self.kin**3 * pk_prim / cosmo.h**3
            alpha = 1. / (pk_dd / pphi_prim)**0.5
        else:
            # Normalization in the matter dominated era
            # https://arxiv.org/pdf/1904.08859.pdf eq. 2.3
            tk = (pk_dd / pk_prim / self.kin / (pk_dd[0] / pk_prim[0] / self.kin[0]))**0.5
            znorm = 10.
            normalized_growth_factor = cosmo.growth_factor(self.template.z) / cosmo.growth_factor(znorm) / (1 + znorm)
            alpha = 3. * cosmo.Omega0_m * 100**2 / (2. * (constants.c / 1e3)**2 * self.kin**2 * tk * normalized_growth_factor)
        # Remove first k, used to normalize tk
        pk_dd, alpha = pk_dd[1:], alpha[1:]
        if self.mode == 'bphi':
            fnl_loc = params['fnl_loc']
            bphi = params['bphi']
            bfnl_loc = bphi * fnl_loc
        elif self.mode == 'b-p':
            fnl_loc = params['fnl_loc']
            p = params.get('p', 1.)
            bfnl_loc = 2. * 1.686 * (b1 - p) * fnl_loc
        else:
            bfnl_loc = params['bfnl_loc']
        # bfnl_loc is typically 2 * delta_c * (b1 - p)
        bias = b1 + bfnl_loc * alpha
        fog = 1. / (1. + sigmas**2 * self.k[:, None]**2 * self.mu**2 / 2.)**2.
        pkmu = fog * (bias[:, None] + self.template.f * self.mu**2)**2 * pk_dd[:, None] + sn0
        self.power = self.to_poles(pkmu)
