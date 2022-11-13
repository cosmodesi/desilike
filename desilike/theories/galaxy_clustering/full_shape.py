import numpy as np

from .base import TrapzTheoryPowerSpectrumMultipoles
from .base import BaseTheoryPowerSpectrumMultipoles, BaseTheoryCorrelationFunctionMultipoles, BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles
from .power_template import FullPowerSpectrumTemplate  # to add calculator in the registry


class BasePTPowerSpectrumMultipoles(BaseTheoryPowerSpectrumMultipoles):

    config_fn = 'full_shape.yaml'

    def __init__(self, *args, template=None, **kwargs):
        super(BasePTPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.kin = np.geomspace(min(1e-3, self.k[0] / 2), max(1., self.k[0] * 2), 600)  # margin for AP effect
        if template is None:
            template = FullPowerSpectrumTemplate(k=self.kin)
        self.template = template
        self.template.k = self.kin


class BasePTCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionMultipoles):

    config_fn = 'full_shape.yaml'

    def __init__(self, s=None, ells=(0, 2, 4)):
        super(BasePTCorrelationFunctionMultipoles, self).__init__(s=s, ells=ells)
        self.kin = np.geomspace(min(1e-3, 1 / self.s[-1] / 2), max(2., 1 / self.s[0] * 2), 1000)  # margin for AP effect


class KaiserTracerPowerSpectrumMultipoles(BasePTPowerSpectrumMultipoles, TrapzTheoryPowerSpectrumMultipoles):

    def __init__(self, *args, mu=200, **kwargs):
        super(KaiserTracerPowerSpectrumMultipoles, self).__init__(*args, **kwargs)
        self.set_k_mu(k=self.k, mu=self.mu, ells=self.ells)

    def calculate(self, b1=1., sn0=0.):
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        f = self.template.f
        pkmu = (b1 + f * muap**2)**2 * np.interp(np.log10(kap), np.log10(self.kin), self.template.pk_tt / f**2) + sn0
        self.power = self.to_poles(pkmu)
        return self


class KaiserTracerCorrelationFunctionMultipoles(BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles):

    pass
