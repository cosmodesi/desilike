import numpy as np

from desilike.cosmo import is_external_cosmo
from desilike.likelihoods.base import BaseCalculator


def projection(size, order=None):
    from scipy import special
    if order is None: order = size // 2
    x = np.linspace(-1., 1., size)
    poly = np.array([special.chebyt(n)(x) for n in range(order + 1)])
    proj = np.linalg.solve(poly.dot(poly.T), poly)
    return proj, poly


class ClTheory(BaseCalculator):
    r"""
    Theory CMB :math:`C_{\ell}^{xy}`.

    Parameters
    ----------
    cls : dict, default=None
        Dictionary mapping types :math:`xy \in \{tt, ee, bb, te, pp, tp, ep\}` of :math:`C_{\ell}^{xy}` to max :math:`\ell`.

    lensing : bool, default=None
        If ``True``, add lensing to theory :math:`C_{\ell}^{xy}`.
        If ``None``, compute lensing if potential :math:`C_{\ell}^{xy}` (:math:`xy \in \{pp, tp, ep\}`) are required in ``cls``.

    non_linear : str, bool, default=None
        If ``True``, or string (e.g. 'mead'), add non-linear correction to theory :math:`C_{\ell}^{xy}`.
        If ``None``, compute non-linear correction with 'mead' if potential :math:`C_{\ell}^{xy}` (:math:`xy \in \{pp, tp, ep\}`),
        or B-modes :math:`C_{\ell}^{bb}` beyond :math:`\ell > 50` are required in ``cls``.

    unit : str, default=None
        Unit, either ``None`` (no unit), or 'muK' (micro-Kelvin).

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo()``.

    T0 : float, default=None
        If ``unit`` is 'muK', CMB temperature to assume. Defaults to :attr:`Cosmology.T_cmb`.

    """
    def initialize(self, cls=None, lensing=None, non_linear=None, unit=None, cosmo=None, T0=None):
        self.requested_cls = dict(cls or {})
        self.ell_max_lensed_cls, self.ell_max_lens_potential_cls = 0, 0
        for cl, ellmax in self.requested_cls.items():
            if cl in ['tt', 'ee', 'bb', 'te']: self.ell_max_lensed_cls = max(self.ell_max_lensed_cls, ellmax)
            elif cl in ['pp', 'tp', 'ep']: self.ell_max_lens_potential_cls = max(self.ell_max_lens_potential_cls, ellmax)
            elif cl in ['tb', 'eb']: pass  # zeros
            else: raise ValueError('Unknown Cl {}'.format(cl))
        if lensing is None:
            lensing = bool(self.ell_max_lens_potential_cls)
        ellmax = max(self.ell_max_lensed_cls, self.ell_max_lens_potential_cls)
        if non_linear is None:
            if bool(self.ell_max_lens_potential_cls) or max(ellmax if 'b' in cl.lower() else 0 for cl, ellmax in self.requested_cls.items()) > 50:
                non_linear = 'mead'
            else:
                non_linear = ''
        self.unit = unit
        allowed_units = [None, 'muK']
        if self.unit not in allowed_units:
            raise ValueError('Input unit must be one of {}, found {}'.format(allowed_units, self.unit))
        self.T0 = T0
        self.cosmo = cosmo
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'harmonic': {}}
            if self.ell_max_lensed_cls:
                self.cosmo_requires['harmonic']['lensed_cl'] = {'ellmax': self.ell_max_lensed_cls}
            if self.ell_max_lens_potential_cls:
                self.cosmo_requires['harmonic']['lens_potential_cl'] = {'ellmax': self.ell_max_lens_potential_cls}
        else:
            if self.cosmo is None:
                from desilike.theories.primordial_cosmology import Cosmoprimo
                self.cosmo = Cosmoprimo()
            requires = {'lensing': self.cosmo.init.get('lensing', False) or lensing,
                        'ellmax_cl': max(self.cosmo.init.get('ellmax_cl', 0), ellmax),
                        'non_linear': self.cosmo.init.get('non_linear', '') or non_linear}
            self.cosmo.init.update(**requires)

    def calculate(self):
        self.cls = {}
        T0 = self.T0 if self.T0 is not None else self.cosmo.T0_cmb
        hr = self.cosmo.get_harmonic()
        if self.ell_max_lensed_cls:
            lensed_cl = hr.lensed_cl(ellmax=self.ell_max_lensed_cls)
        if self.ell_max_lens_potential_cls:
            lens_potential_cl = hr.lens_potential_cl(ellmax=self.ell_max_lens_potential_cls)
        for cl, ellmax in self.requested_cls.items():
            if cl in ['tb', 'eb']:
                tmp = np.zeros(ellmax + 1, dtype='f8')
            if 'p' in cl:
                tmp = lens_potential_cl[cl][:ellmax + 1]
            else:
                tmp = lensed_cl[cl][:ellmax + 1]
            if self.unit == 'muK':
                npotential = cl.count('p')
                unit = (T0 * 1e6)**(2 - npotential)
                tmp = tmp * unit
            self.cls[cl] = tmp

    def get(self):
        return self.cls

    def __getstate__(self):
        state = {}
        for name in ['requested_cls', 'unit']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return {**state, **self.cls}

    def __setstate__(self, state):
        state = state.copy()
        self.unit = state.pop('unit')
        self.cls = state
