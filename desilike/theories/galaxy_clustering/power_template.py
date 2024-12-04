import re

import numpy as np
from cosmoprimo import PowerSpectrumBAOFilter, PowerSpectrumInterpolator1D

from desilike.jax import numpy as jnp
from desilike.base import BaseCalculator
from desilike.cosmo import is_external_cosmo
from desilike.parameter import ParameterCollection
from desilike.theories.primordial_cosmology import get_cosmo, Cosmoprimo, Cosmology, constants
from .base import APEffect


_kw_interp = dict(extrap_kmin=1e-7, extrap_kmax=1e2)


def _bcast_shape(array, shape, axis=0):
    # Return array with shape, and size matching along axis
    array = jnp.atleast_1d(array)
    tshape = [1 for s in shape]
    tshape[axis] = shape[axis]
    return array.reshape(tshape)


class BasePowerSpectrumExtractor(BaseCalculator):

    """Base class to extract shape parameters from linear power spectrum."""
    config_fn = 'power_template.yaml'

    def initialize(self, z=1., with_now=False, cosmo=None, fiducial='DESI'):
        self.z = np.asarray(z, dtype='f8')
        self.fiducial = get_cosmo(fiducial)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        params = self.params.select(derived=True)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'of': [('delta_cb', 'delta_cb')]}}, 'thermodynamics': {'rs_drag': None}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
            self.cosmo.init.params = [param for param in self.params if param not in params]
        self.init.params = params
        BasePowerSpectrumExtractor._set_base(self, fiducial=True)
        self.with_now = with_now
        if self.with_now:
            self.filter = PowerSpectrumBAOFilter(self.pk_dd_interpolator_fid, engine=self.with_now, cosmo=self.fiducial, cosmo_fid=self.fiducial)
            self.pknow_dd_interpolator_fid = self.filter.smooth_pk_interpolator()

    def _set_base(self, fiducial=False, with_now=False):
        cosmo = self.fiducial if fiducial else self.cosmo
        fo = cosmo.get_fourier()
        state = {}
        state['sigma8'] = fo.sigma8_z(self.z, of='delta_cb')
        state['fsigma8'] = fo.sigma8_z(self.z, of='theta_cb')
        state['f'] = state['fsigma8'] / state['sigma8']
        state['pk_dd_interpolator'] = fo.pk_interpolator(of='delta_cb', **_kw_interp).to_1d(z=self.z)
        state['pk_tt_interpolator'] = fo.pk_interpolator(of='theta_cb', **_kw_interp).to_1d(z=self.z)
        k0 = 1e-3
        state['f0'] = np.sqrt(state['pk_tt_interpolator'](k0) / state['pk_dd_interpolator'](k0))
        if hasattr(self, 'k'):  # ShapeFitPowerSpectrumExtractor has no k
            state['fk'] = np.sqrt(state['pk_tt_interpolator'](self.k) / state['pk_dd_interpolator'](self.k))
        if with_now:
            self.filter(state['pk_dd_interpolator'], cosmo=cosmo)
            state['pknow_dd_interpolator'] = self.filter.smooth_pk_interpolator()
        for name, value in state.items(): setattr(self, name + ('_fid' if fiducial else ''), value)


class BasePowerSpectrumTemplate(BasePowerSpectrumExtractor):

    """Base class for linear power spectrum template."""
    # See calculate(self) for the list of attributes that must be set by calculate

    config_fn = 'power_template.yaml'
    _interpolator_k = np.logspace(-5., 2., 1000)  # more than classy

    def initialize(self, k=None, z=1., with_now=False, apmode='qparqper', fiducial='DESI', only_now=False, eta=1. / 3., cosmo=None):
        self.z = np.asarray(z, dtype='f8')
        self.cosmo = self.fiducial = get_cosmo(fiducial)
        if k is None: k = np.logspace(-3., 1., 400)
        self.k = np.array(k, dtype='f8')
        self.cosmo_requires = {}
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, mode=apmode, eta=eta, cosmo=cosmo)
        ap_params = ParameterCollection()
        for param in list(self.init.params):
            if param.basename in ['qpar', 'qper', 'qap', 'qiso'] + (['DM', 'DH', 'DH_over_DM', 'DV'] if apmode == 'geometry' else []):
                ap_params.set(param)
                del self.init.params[param]
        self.apeffect.init.params = ap_params
        BasePowerSpectrumExtractor._set_base(self, fiducial=True)
        self.with_now = with_now
        if only_now and not self.with_now:
            if isinstance(only_now, bool):
                only_now = 'peakaverage'  # default
            self.with_now = only_now
        self.only_now = bool(only_now)
        self.pk_dd_fid = self.pk_dd_interpolator_fid(self.k)
        if self.with_now:
            self.filter = PowerSpectrumBAOFilter(self.pk_dd_interpolator_fid, engine=self.with_now, cosmo=self.fiducial, cosmo_fid=self.fiducial)
            self.pknow_dd_interpolator_fid = self.filter.smooth_pk_interpolator()
            self.pknow_dd_fid = self.pknow_dd_interpolator_fid(self.k)

    def calculate(self):
        # These are the quantities that should be set by a BasePowerSpectrumTemplate-inherited class.
        # See BasePowerSpectrumExtrator._set_base for how to get these quantities from self.cosmo
        # fk is sqrt(pk_tt / pk_dd)
        # f0 is the limit of fk for k -> 0
        for name in ['sigma8', 'fsigma8', 'f', 'f0', 'fk', 'pk_dd_interpolator', 'pk_dd']:
            setattr(self, name, getattr(self, name + '_fid'))
        if self.with_now:
            for name in ['pknow_dd_interpolator', 'pknow_dd']:
                setattr(self, name, getattr(self, name + '_fid'))
        if self.only_now:
            for name in ['dd_interpolator', 'dd']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name + '_fid'))

    @property
    def eta(self):
        return self.apeffect.eta

    @property
    def qpar(self):
        return self.apeffect.qpar

    @property
    def qper(self):
        return self.apeffect.qper

    def ap_k_mu(self, k, mu):
        return self.apeffect.ap_k_mu(k, mu)

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'fiducial', 'only_now', 'with_now', 'qpar', 'qper', 'eta']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for suffix in ['', '_fid']:
            #for name in ['sigma8', 'fsigma8', 'f', 'f0', 'pk_dd_interpolator', 'pk_dd'] + ['pknow_dd_interpolator', 'pknow_dd']:
            for name in ['sigma8', 'fsigma8', 'f', 'f0', 'fk', 'pk_dd', 'pknow_dd']:
                name = name + suffix
                if hasattr(self, name):
                    state[name] = getattr(self, name)
        for name in ['fiducial']:
            if name in state:
                state[name] = state[name].__getstate__()
        #for name in list(state):
        #    if 'interpolator' in name:
        #        value = state.pop(name)
        #        state[name + '_k'] = self._interpolator_k
        #        state[name + '_pk'] = value(self._interpolator_k)
        return state

    def __setstate__(self, state):
        state = dict(state)
        for name in ['fiducial']:
            if name in state:
                cosmo = state.pop(name)
                if not hasattr(self, name):
                    setattr(self, name, Cosmology.from_state(cosmo))
        #for name in list(state):
        #    if ('interpolator' in name) and name.endswith('_k'):
        #        k = state.pop(name)
        #        pk = state.pop(name[:-1] + 'pk')
        #        state[name[:-2]] = PowerSpectrumInterpolator1D(k, pk)
        if hasattr(self, 'apeffect'):
            tmpap = self.apeffect
        else:
            class TmpAPEffect(object): pass
            TmpAPEffect.ap_k_mu = APEffect.ap_k_mu
            state['apeffect'] = tmpap = TmpAPEffect()
        tmpap.qpar, tmpap.qper, tmpap.eta = state.pop('qpar'), state.pop('qper'), state.pop('eta')
        self.__dict__.update(state)


class FixedPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    """
    Fixed power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    with_now : str, default=False
        If provided, also compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    """
    def initialize(self, *args, **kwargs):
        super(FixedPowerSpectrumTemplate, self).initialize(*args, apmode='qparqper', **kwargs)
        self.apeffect.init.params = dict(qpar=dict(value=1.), qper=dict(value=1.))
        self.apeffect()  # qpar, qper = 1. as a default
        self.runtime_info.requires = []  # remove APEffect dependence


class DirectPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    """
    Direct power spectrum template, i.e. parameterized in terms of base cosmological parameters.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    with_now : str, default=False
        If provided, also compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    """
    def initialize(self, *args, cosmo=None, **kwargs):
        engine = kwargs.pop('engine', 'class')
        super(DirectPowerSpectrumTemplate, self).initialize(*args, apmode='geometry', **kwargs)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        # keep only derived parameters, others are transferred to Cosmoprimo
        params = self.init.params.select(derived=True)
        if is_external_cosmo(self.cosmo):
            # cosmo_requires only used for external bindings (cobaya, cosmosis, montepython): specifies the input theory requirements
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'k': self.k, 'of': [('delta_cb', 'delta_cb')]}}, 'thermodynamics': {'rs_drag': None}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial, engine=engine)
            # transfer the parameters of the template (Omega_m, logA, h, etc.) to Cosmoprimo
            self.cosmo.init.params = [param for param in self.params if param not in params]
        self.init.params = params
        # Alcock-Paczynski effect, that is known given the cosmo and fiducial
        self.apeffect.init.update(cosmo=self.cosmo)
        if is_external_cosmo(self.cosmo):
            # update cosmo_requires with background quantities
            self.cosmo_requires.update(self.apeffect.cosmo_requires)

    def calculate(self):
        # compute the power spectrum for the current cosmo
        BasePowerSpectrumExtractor._set_base(self, with_now=self.with_now)
        self.pk_dd = self.pk_dd_interpolator(self.k)
        if self.with_now:
            self.pknow_dd = self.pknow_dd_interpolator(self.k)
        if self.only_now:  # only used if we want to take wiggles out of our model (e.g. for BAO)
            for name in ['dd_interpolator', 'dd']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))


class BAOExtractor(BasePowerSpectrumExtractor):
    """
    Extract BAO parameters from base cosmological parameters.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    eta : float, default=1./3.
        Relation between 'qpar', 'qper' and 'qiso', 'qap' parameters:
        ``qiso = qpar ** eta * qper ** (1 - eta)``.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    """
    config_fn = 'power_template.yaml'
    conflicts = [('DM_over_rd', 'qper'), ('DH_over_rd', 'qper'), ('DM_over_DH', 'qap'), ('DV_over_rd', 'qiso')]

    @staticmethod
    def _params(params, rs_drag_varied=False):
        if rs_drag_varied:
            params['rs_drag'] = dict(value=100., prior=dict(limits=[10., 1000.]), ref=dict(dist='norm', loc=100., scale=10.), latex=r'r_\mathrm{d}')
        return params

    def initialize(self, z=1., eta=1. / 3., cosmo=None, fiducial='DESI', rs_drag_varied=False):
        self.z = np.asarray(z, dtype='f8')
        self.eta = float(eta)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        params = self.init.params.select(derived=True) + self.init.params.select(basename=['rs_drag'])
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
            self.cosmo_requires['thermodynamics'] = {'rs_drag': None}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
            self.cosmo.init.params = [param for param in self.params if param not in params]
        self.init.params = params
        if self.fiducial is not None:
            self._set_base(fiducial=True)

    def calculate(self, rs_drag=None):
        self._set_base(rs_drag=rs_drag)

    def _set_base(self, fiducial=False, rs_drag=None):
        cosmo = self.fiducial if fiducial else self.cosmo
        state = {}
        state['rd'] = cosmo.rs_drag if rs_drag is None else rs_drag
        state['DH'] = (constants.c / 1e3) / (100. * cosmo.efunc(self.z))
        state['DM'] = cosmo.comoving_angular_distance(self.z)
        state['DV'] = state['DH']**self.eta * state['DM']**(1. - self.eta) * self.z**(1. / 3.)
        state['DH_over_rd'] = state['DH'] / state['rd']
        state['DM_over_rd'] = state['DM'] / state['rd']
        state['DH_over_DM'] = state['DH'] / state['DM']
        state['DV_over_rd'] = state['DV'] / state['rd']
        for name, value in state.items(): setattr(self, name + ('_fid' if fiducial else ''), value)

    def get(self):
        if self.fiducial is not None:
            self.qpar = self.DH_over_rd / self.DH_over_rd_fid
            self.qper = self.DM_over_rd / self.DM_over_rd_fid
            self.qiso = self.DV_over_rd / self.DV_over_rd_fid
            self.qap = self.DH_over_DM / self.DH_over_DM_fid
        return self

#from desilike.jax import register_pytree_node_class

#@register_pytree_node_class
class BAOPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    """
    BAO power spectrum template.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    apmode : str, default='qparqper'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    """
    def initialize(self, *args, with_now='peakaverage', **kwargs):
        super(BAOPowerSpectrumTemplate, self).initialize(*args, with_now=with_now, **kwargs)
        # Set DM_over_rd, etc.
        BAOExtractor._set_base(self, fiducial=True)

    def calculate(self, df=1.):
        super(BAOPowerSpectrumTemplate, self).calculate()
        self.f = self.f_fid * df
        self.f0 = self.f0_fid * df
        self.fk = self.fk_fid * df

    def get(self):
        self.DH_over_rd = self.qpar * self.DH_over_rd_fid
        self.DM_over_rd = self.qper * self.DM_over_rd_fid
        self.DV_over_rd = self.qpar**self.eta * self.qper**(1. - self.eta) * self.DV_over_rd_fid
        self.DH_over_DM = self.qpar / self.qper * self.DH_over_DM_fid
        return self

    def __getstate__(self):
        state = super().__getstate__()
        for name in ['DH_over_rd_fid', 'DM_over_rd_fid', 'DH_over_DM_fid', 'DV_over_rd_fid']:
            state[name] = getattr(self, name)
        return state


class BAOPhaseShiftExtractor(BAOExtractor):
    """
    Extract BAO + phase shift parameters from base cosmological parameters.

    Reference
    ---------
    https://arxiv.org/pdf/1803.10741

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    eta : float, default=1./3.
        Relation between 'qpar', 'qper' and 'qiso', 'qap' parameters:
        ``qiso = qpar ** eta * qper ** (1 - eta)``.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    """
    conflicts = BAOExtractor.conflicts + [('baoshift',)]

    def _set_base(self, fiducial=False, **kwargs):
        cosmo = self.fiducial if fiducial else self.cosmo
        super(BAOPhaseShiftExtractor, self)._set_base(fiducial=fiducial, **kwargs)
        setattr(self, 'N_eff' + ('_fid' if fiducial else ''), cosmo.N_eff)

    def get(self):
        super(BAOPhaseShiftExtractor, self).get()
        if self.fiducial is not None:
            a_nu = 8.0 / 7.0 * ((11.0 / 4.0)**(4.0 / 3.0))
            self.baoshift = (self.N_eff * (self.N_eff_fid + a_nu)) / (self.N_eff_fid * (self.N_eff + a_nu))
        return self


def _interp(k, k1, pk1):
    from desilike.jax import numpy as jnp
    from desilike.jax import interp1d
    return interp1d(jnp.log10(k), jnp.log10(k1), pk1, method='cubic')


class BAOPhaseShiftPowerSpectrumTemplate(BAOPowerSpectrumTemplate):
    r"""
    BAO power spectrum template, including :math:`N_\mathrm{eff}`-induced phase shift, following Baumann et al 2018.

    parameterization of the BAO phase shift due to the effective number of neutrino species.
    From the Baumann et al 2018, best fit values for parameters in this function for a range of cosmologies are:

    phi_inf = 0.227
    kstar = 0.0324 h/Mpc
    epsilon = 0.872

    Reference
    ---------
    https://arxiv.org/pdf/1803.10741

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    apmode : str, default='qparqper'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance
    """
    def initialize(self, *args, phiinf=0.227, kstar=0.0324, epsilon=0.872, **kwargs):
        super(BAOPhaseShiftPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        self.phiinf = float(phiinf)
        self.kstar = float(kstar)
        self.epsilon = float(epsilon)

    def calculate(self, df=1., baoshift=1.):
        super(BAOPhaseShiftPowerSpectrumTemplate, self).calculate(df=df)
        kshift = self.phiinf / (1.0 + (self.kstar / self.k)**self.epsilon) / self.fiducial.rs_drag  # eq. 3.3 of https://arxiv.org/pdf/1803.10741
        k = np.geomspace(self.pk_dd_interpolator_fid.extrap_kmin, self.pk_dd_interpolator_fid.extrap_kmax, 2000)
        wiggles = _interp(jnp.clip(self.k + (baoshift - 1.) * kshift, k[0], k[-1]), k, self.pk_dd_interpolator_fid(k) - self.pknow_dd_interpolator_fid(k))
        self.pk_dd = self.pknow_dd_fid + wiggles
        if self.only_now:  # only used if we want to take wiggles out of our model (e.g. for BAO)
            for name in ['dd']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))


class StandardPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    r"""
    Extract standard RSD parameters :math:`(q_{\parallel}, q_{\perp}, df)`.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    r : float, default=8.
        Sphere radius to estimate the normalization of the linear power spectrum.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    """
    conflicts = BAOExtractor.conflicts + [('df', 'fsigmar')]

    def initialize(self, *args, eta=1. / 3., r=8., **kwargs):
        self.eta = float(eta)
        self.r = float(r)
        super(StandardPowerSpectrumExtractor, self).initialize(*args, **kwargs)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires['thermodynamics'] = {'rs_drag': None}
            self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
        self._set_base(fiducial=True)

    def calculate(self):
        self._set_base()

    def _set_base(self, fiducial=False):
        BAOExtractor._set_base(self, fiducial=fiducial)
        cosmo = self.fiducial if fiducial else self.cosmo
        DV = getattr(self, 'DV', self.DV_fid)
        r = self.r * DV / self.DV_fid
        fo = cosmo.get_fourier()
        state = {}
        state['sigmar'] = fo.sigma_rz(r, z=self.z, of='delta_cb')
        state['fsigmar'] = fo.sigma_rz(r, z=self.z, of='theta_cb')
        state['f'] = state['fsigmar'] / state['sigmar']
        for name, value in state.items(): setattr(self, name + ('_fid' if fiducial else ''), value)

    def get(self):
        BAOExtractor.get(self)
        self.df = self.fsigmar / self.fsigmar_fid
        return self


class StandardPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    r"""
    Standard power spectrum template, in terms of :math:`f` and Alcock-Paczynski parameters.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    r : float, default=8.
        Sphere radius to estimate the normalization of the linear power spectrum.

    apmode : str, default='qparqper'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    with_now : str, default=None
        If not ``None``, compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').
    """
    def initialize(self, *args, r=8., **kwargs):
        self.r = float(r)
        super(StandardPowerSpectrumTemplate, self).initialize(*args, **kwargs)
        StandardPowerSpectrumExtractor._set_base(self, fiducial=True)

    def calculate(self, df=1.):
        super(StandardPowerSpectrumTemplate, self).calculate()
        self.f = self.f_fid * df
        self.f0 = self.f0_fid * df
        self.fk = self.fk_fid * df

    def get(self):
        return self


class ShapeFitPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    """
    Extract ShapeFit parameters from linear power spectrum.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    kp : float, default=0.03
        Pivot point in ShapeFit parameterization.

    a : float, default=0.6
        :math:`a` parameter in ShapeFit parameterization.

    n_varied : bool, default=False
        Use second order ShapeFit parameter ``n``.
        This choice changes the definition of parameter ``m``.

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.


    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    https://arxiv.org/pdf/2212.04522.pdf
    """
    def initialize(self, *args, kp=0.03, a=0.6, eta=1. / 3., n_varied=False, dfextractor='Ap', r=8., with_now='peakaverage', **kwargs):
        self.kp, self.a = float(kp), float(a)
        self.n_varied = bool(n_varied)
        self.dfextractor = dfextractor
        allowed_dfextractor = ['Ap', 'fsigmar']
        if self.dfextractor not in allowed_dfextractor:
            raise ValueError('dfextractor must be one of {}, found {}'.format(allowed_dfextractor, self.dfextractor))
        self.r = float(r)
        super(ShapeFitPowerSpectrumExtractor, self).initialize(*args, with_now=with_now, **kwargs)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires['primordial'] = {'pk_interpolator': None}
            self.cosmo_requires['thermodynamics'] = {'rs_drag': None}
            self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
        self.eta = float(eta)
        self._set_base(fiducial=True)

    def calculate(self):
        self._set_base()

    def _set_base(self, fiducial=False):
        BasePowerSpectrumExtractor._set_base(self, fiducial=fiducial, with_now=True)
        BAOExtractor._set_base(self, fiducial=fiducial)
        cosmo = self.fiducial if fiducial else self.cosmo
        s = cosmo.rs_drag / self.fiducial.rs_drag
        kp = self.kp / s
        state = {}
        state['n'] = cosmo.n_s
        dk = 1e-2
        k = kp * np.array([1. - dk, 1. + dk])
        # No need to include 1/s^3 factors here, as we care about the slope
        if self.n_varied: pk_prim = cosmo.get_primordial().pk_interpolator()(k) * k
        else: pk_prim = np.ones_like(k)
        suffix = '_fid' if fiducial else ''
        pknow_dd_interpolator = getattr(self, 'pknow_dd_interpolator' + suffix)
        f = getattr(self, 'f' + suffix)
        pknow_dd = pknow_dd_interpolator(k)
        state['m'] = np.diff(np.log(pknow_dd / _bcast_shape(pk_prim, pknow_dd.shape, axis=0)), axis=0)[0] / np.diff(np.log(k))[0]
        # Eq. 3.11 of https://arxiv.org/abs/2106.07641
        state['Ap'] = 1. / s**3 * pknow_dd_interpolator(kp)
        state['f_sqrt_Ap'] = f * state['Ap']**0.5
        # Eq. 3.11 of https://arxiv.org/pdf/2212.04522.pdf
        dm = state['m'] - getattr(self, 'm_fid', state['m'])
        state['f_sigmar'] = f * pknow_dd_interpolator.sigma_r(self.r * s) * np.exp(dm / (2 * self.a) * np.tanh(self.a * self.fiducial.rs_drag / self.r))
        for name, value in state.items(): setattr(self, name + suffix, value)

    def get(self):
        BAOExtractor.get(self)
        self.dn = self.n - self.n_fid
        self.dm = self.m - self.m_fid
        if self.dfextractor == 'Ap':
            self.df = self.f_sqrt_Ap / self.f_sqrt_Ap_fid
        else:
            self.df = self.f_sigmar / self.f_sigmar_fid
        return self


class ShapeFitPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    r"""
    ShapeFit power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    kp : float, default=0.03
        Pivot point in ShapeFit parameterization.

    a : float, default=0.6
        :math:`a` parameter in ShapeFit parameterization.

    apmode : str, default='qparqper'
        Alcock-Paczynski parameterization:

        - 'qiso': single istropic parameter 'qiso'
        - 'qap': single, Alcock-Paczynski parameter 'qap'
        - 'qisoqap': two parameters 'qiso', 'qap'
        - 'qparqper': two parameters 'qpar' (scaling along the line-of-sight), 'qper' (scaling perpendicular to the line-of-sight)

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    https://arxiv.org/pdf/2212.04522.pdf
    """
    def initialize(self, *args, kp=0.03, a=0.6, r=8., with_now='peakaverage', **kwargs):
        self.a = float(a)
        self.kp = float(kp)
        self.n_varied = self.init.params['dn'].varied
        self.r = float(r)
        super(ShapeFitPowerSpectrumTemplate, self).initialize(*args, with_now=with_now, **kwargs)
        ShapeFitPowerSpectrumExtractor._set_base(self, fiducial=True)

    def calculate(self, df=1., dm=0., dn=0.):
        super(ShapeFitPowerSpectrumTemplate, self).calculate()
        factor = _bcast_shape(jnp.exp(dm / self.a * jnp.tanh(self.a * jnp.log(self.k / self.kp)) + dn * jnp.log(self.k / self.kp)), self.pk_dd_fid.shape, axis=0)
        #factor = np.exp(dm * np.log(self.k / self.kp))
        self.pk_dd = self.pk_dd_fid * factor
        if self.with_now:
            self.pknow_dd = self.pknow_dd_fid * factor
        if self.only_now:
            self.pk_dd = self.pknow_dd
        self.n = self.n_fid + dn
        self.m = self.m_fid + dm
        self.f = self.f_fid * df
        self.f0 = self.f0_fid * df
        self.fk = self.fk_fid * df
        self.f_sqrt_Ap = self.f * self.Ap_fid**0.5

    def get(self):
        return self


class BandVelocityPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    r"""
    Extract band power parameters.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    kp : array
        Pivot :math:`k` where to evaluate the velocity divergence power spectrum :math:`P_{\theta \theta}`.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    """
    _base_param_name = 'dptt'

    def initialize(self, *args, eta=1. / 3., kp=None, **kwargs):
        super(BandVelocityPowerSpectrumExtractor, self).initialize(*args, **kwargs)
        self.apeffect = APEffect(z=self.z, cosmo=self.cosmo, fiducial=self.fiducial, eta=eta, mode='geometry')
        self.kp = kp
        if kp is None: raise ValueError('Please provide kp')
        else: self.kp = np.asarray(kp, dtype='f8')
        self._set_base(fiducial=True)

    def calculate(self):
        self._set_base()

    def _set_base(self, fiducial=False):
        cosmo = self.fiducial if fiducial else self.cosmo
        qiso = 1. if fiducial else self.apeffect.qiso
        r = 8. * qiso
        fo = cosmo.get_fourier()
        state = {}
        state['sigmar'] = fo.sigma_rz(r, self.z, of='delta_cb')
        state['fsigmar'] = fo.sigma_rz(r, self.z, of='theta_cb')
        state['f'] = state['fsigmar'] / state['sigmar']
        state['pk_tt_interpolator'] = fo.pk_interpolator(of='theta_cb', **_kw_interp)
        state['pk_tt'] = state['pk_tt_interpolator'](self.kp / qiso) / qiso**3
        state['qap'] = 1. if fiducial else self.apeffect.qap
        for name, value in state.items(): setattr(self, name + ('_fid' if fiducial else ''), value)

    def get(self):
        dptt = self.pk_tt / self.pk_tt_fid
        setattr(self, self._base_param_name, dptt)
        for i, dptt in enumerate(dptt):
            setattr(self, '{}{:d}'.format(self._base_param_name, i), dptt)
        self.df = self.fsigmar / self.fsigmar_fid
        return self


def BandVelocityPowerSpectrumCalculator(calculator, extractor=None, **kwargs):

    Calculator = calculator.__class__
    if extractor is None:
        extractor = BandVelocityPowerSpectrumExtractor()
    extractor.init.update(kwargs)

    def initialize(self):
        extractor.runtime_info.initialize()
        calculator.runtime_info.initialize()
        self.runtime_info.requires = [extractor]

    def calculate(self, **params):
        extractor.calculate()
        extractor.get()
        calculator(**{param.name: getattr(extractor, param.basename) for param in calculator_params}, **params)
        self.__dict__.update({name: value for name, value in calculator.__dict__.items() if name not in ['info', 'runtime_info']})

    def __getstate__(self):
        return calculator.runtime_info.pipeline.derived.to_dict()

    calculator()
    for calculator in calculator.runtime_info.pipeline.calculators:
        kp = getattr(calculator, 'kp', None)
        if kp is not None:
            extractor.init.setdefault('kp', kp)
        z = getattr(calculator, 'z')
        if z is not None:
            extractor.init.setdefault('z', z)

    cosmo_requires = getattr(extractor, 'cosmo_requires', {})
    params, calculator_params = ParameterCollection(), ParameterCollection()
    for param in calculator.all_params:
        param = param.copy()
        if re.match(r'{}(-?\d+)'.format(extractor._base_param_name), param.basename) or param.basename in ['df', 'qap']:
            calculator_params.set(param)
        else:
            params.set(param)

    new_cls = type(Calculator)(Calculator.__name__, (Calculator,),
                               {'initialize': initialize, 'calculate': calculate, '_params': params, 'config_fn': None, 'cosmo_requires': cosmo_requires, '__getstate__': __getstate__})
    return new_cls()


class BandVelocityPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    r"""
    Band velocity power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    kp : array
        Pivot :math:`k` where to change the value of the velocity divergence power spectrum :math:`P_{\theta \theta}`.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    with_now : str, default=None
        If not ``None``, compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').
    """
    _base_param_name = BandVelocityPowerSpectrumExtractor._base_param_name

    def initialize(self, *args, kp=None, **kwargs):
        super(BandVelocityPowerSpectrumTemplate, self).initialize(*args, apmode='qap', **kwargs)
        re_param_template = re.compile(r'{}(-?\d+)'.format(self._base_param_name))
        params = self.init.params.select(basename=re_param_template)
        nkp = len(params)
        if kp is None:
            if not nkp:
                raise ValueError('No parameter {}* found'.format(self._base_param_name))
            step = (self.k[-1] - self.k[0]) / nkp
            self.kp = (self.k[0] + step / 2., self.k[-1] - step / 2.)
        else:
            self.kp = np.array(kp)
        if nkp:
            if len(self.kp) == 2:
                self.kp = np.linspace(*self.kp, num=nkp)
            self.kp = np.array(self.kp, dtype='f8')
            if self.kp.size != nkp:
                raise ValueError('{:d} (!= {:d} parameters {}*) points have been provided'.format(self.kp.size, nkp, self._base_param_name))
        basenames = []
        for ikp, kp in enumerate(self.kp):
            basename = '{}{:d}'.format(self._base_param_name, ikp)
            basenames.append(basename)
            if basename not in self.params:
                value = 1.
                self.init.params[basename] = dict(value=value, prior={'limits': [0, 3]}, ref={'dist': 'norm', 'loc': value, 'scale': 0.01}, delta=0.005)
            self.init.params[basename].update(latex=r'(P / P^{{\mathrm{{fid}}}})_{{\{0}\{0}}}(k={1:.3f})'.format('theta', kp))
        params = self.init.params.basenames(basename=re_param_template)
        if set(params) != set(basenames):
            raise ValueError('Found parameters {}, but expected {}'.format(params, basenames))
        if self.kp[0] < self.k[0]:
            raise ValueError('Theory k starts at {0:.2e} but first point is {1:.2e} < {0:.2e}'.format(self.k[0], self.kp[0]))
        if self.kp[-1] > self.k[-1]:
            raise ValueError('Theory k ends at {0:.2e} but last point is {1:.2e} > {0:.2e}'.format(self.k[-1], self.kp[-1]))
        ekp = np.concatenate([[self.k[0]], self.kp, [self.k[-1]]], axis=0)
        self.templates = []
        for ip, kp in enumerate(self.kp):
            diff = self.k - kp
            mask_neg = diff < 0
            diff[mask_neg] /= (ekp[ip] - kp)
            diff[~mask_neg] /= (ekp[ip + 2] - kp)
            self.templates.append(np.maximum(1. - diff, 0.))
        self.templates = np.array(self.templates)
        fo = self.fiducial.get_fourier()
        #self.sigma8_fid = fo.sigma8_z(self.z, of='delta_cb')
        #self.fsigma8_fid = fo.sigma8_z(self.z, of='theta_cb')
        #self.f_fid = self.fsigma8_fid / self.sigma8_fid
        self.pk_tt_interpolator_fid = fo.pk_interpolator(of='theta_cb', **_kw_interp).to_1d(z=self.z)
        self.pk_tt_fid = self.pk_tt_interpolator_fid(self.k)
        self.pk_dd_fid = self.pk_tt_fid / self.f_fid**2
        if self.with_now:
            self.filter = PowerSpectrumBAOFilter(self.pk_tt_interpolator_fid, engine=self.with_now, cosmo=self.cosmo, cosmo_fid=self.fiducial)
            self.pknow_tt_fid = self.filter.smooth_pk_interpolator()(self.k)
            self.pknow_dd_fid = self.pknow_tt_fid / self.f_fid**2
        self.sigma8 = fo.sigma8_z(self.z, of='delta_cb')
        self.fsigma8 = fo.sigma8_z(self.z, of='theta_cb')

    def calculate(self, df=1., **params):
        self.f = self.f_fid * df
        self.f0 = self.f0_fid * df
        self.fk = self.fk_fid * df
        dptt = jnp.array([params['{}{:d}'.format(self._base_param_name, ii)] - 1. for ii in range(len(self.templates))])
        factor = _bcast_shape(1. + jnp.dot(dptt, self.templates), self.pk_tt_fid.shape, axis=0)
        self.pk_tt = self.pk_tt_fid * factor
        self.pk_dd = self.pk_tt / self.f**2
        if self.with_now:
            self.pknow_tt = self.pknow_tt_fid * factor
            self.pknow_dd = self.pknow_tt / self.f**2
        if self.only_now:
            for name in ['dd', 'tt']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))

    def get(self):
        return self


def _kernel_tophat_lowx(x2):
    r"""
    Maclaurin expansion of :math:`W(x) = 3 (\sin(x)-x\cos(x))/x^{3}` to :math:`\mathcal{O}(x^{10})`.
    Necessary numerically because at low x W(x) relies on the fine cancellation of two terms.

    Note
    ----
    Taken from https://github.com/LSSTDESC/CCL/blob/66397c7b53e785ae6ee38a688a741bb88d50706b/src/ccl_power.c
    """
    return 1. + x2 * (-1.0 / 10.0 + x2 * (1.0 / 280.0 + x2 * (-1.0 / 15120.0 + x2 * (1.0 / 1330560.0 + x2 * (-1.0 / 172972800.0)))))


def _kernel_tophat_highx(x):
    r"""Tophat function math:`W(x) = 3 (\sin(x)-x\cos(x))/x^{3}`."""
    return 3. * (np.sin(x) - x * np.cos(x)) / x**3


def kernel_tophat2(x):
    """Non-vectorized tophat function."""
    x = np.asarray(x, dtype='f8')
    toret = np.empty_like(x)
    mask = x < 0.1
    toret[mask] = _kernel_tophat_lowx(x[mask]**2)**2
    toret[~mask] = _kernel_tophat_highx(x[~mask])**2
    return toret


def _kernel_tophat_deriv_lowx(x):
    r"""
    Maclaurin expansion of the derivative of :math:`W(x) = 3 (\sin(x)-x\cos(x))/x^{3}` to :math:`\mathcal{O}(x^{9})`.
    Necessary numerically because at low x W(x) relies on the fine cancellation of two terms

    Note
    ----
    Taken from https://github.com/LSSTDESC/CCL/blob/66397c7b53e785ae6ee38a688a741bb88d50706b/src/ccl_power.c
    """
    x2 = x**2
    return x * (-2.0 / 10.0 + x2 * (4.0 / 280.0 + x2 * (-6.0 / 15120.0 + x2 * (8.0 / 1330560.0 + x2 * (-10.0 / 172972800.0)))))


def _kernel_tophat_deriv_highx(x):
    r"""Derivative of tophat function math:`W(x) = 3 (\sin(x)-x\cos(x))/x^{3}`."""
    return 3. * np.sin(x) / x**2 - 9. * (np.sin(x) - x * np.cos(x)) / x**4


def kernel_tophat2_deriv(x):
    """Derivative of tophat function."""
    x = np.asarray(x, dtype='f8')
    toret = np.empty_like(x)
    mask = x < 0.1
    toret[mask] = 2. * _kernel_tophat_lowx(x[mask]**2) * _kernel_tophat_deriv_lowx(x[mask])
    toret[~mask] = 2 * _kernel_tophat_highx(x[~mask]) * _kernel_tophat_deriv_highx(x[~mask])
    return toret


def kernel_gauss2(x):
    """Gaussian kernel."""
    return np.exp(-x**2)


def kernel_gauss2_deriv(x):
    """Derivative of Gaussian kernel."""
    return - 2. * x * np.exp(-x**2)


def integrate_sigma_r2(r, pk, kmin=1e-6, kmax=1e2, nk=2048, kernel=kernel_tophat2, **kwargs):
    r"""
    Return the variance of perturbations smoothed by a kernel :math:`W` of radius :math:`r`, i.e.:

    .. math::

        \sigma_{r}^{2} = \frac{1}{2 \pi^{2}} \int dk k^{2} P(k) W^{2}(kr)

    Parameters
    ----------
    r : float
        Smoothing radius.

    pk : callable
        Power spectrum.

    kmin : float, default=1e-6
        Minimum wavenumber.

    kmax : float, default=1e2
        Maximum wavenumber.

    nk : int, default=2048
        ``nk`` points between ``kmin`` and ``kmax``.

    kernel : callable, default=kernel_tophat2
        Kernel :math:`W^{2}`; defaults to (square of) top-hat kernel.

    Returns
    -------
    sigmar2 : float
        Variance of perturbations.
    """
    logk = np.linspace(np.log10(kmin), np.log10(kmax), nk)
    k = np.exp(logk)
    integrand = pk(k, **kwargs)
    k = _bcast_shape(k, shape=integrand.shape, axis=0)
    integrand *= kernel(k * r) * k**3  # extra k factor because log integration
    return 1. / 2. / np.pi**2 * np.trapz(integrand, x=logk, axis=0)


class WiggleSplitPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    r"""
    Extract wiggle-split parameters :math:`(q_{\mathrm{ap}}, q_{\mathrm{BAO}}, df, dm)`.

    Parameters
    ----------
    r : float, default=8.
        Smoothing radius to estimate the normalization of the linear power spectrum.

    kernel : str, default='gauss'
        Kernel for normalization of the linear power spectrum: 'gauss' or 'tophat'.

    eta : float, default=1./3.
        Relation between 'qpar', 'qper' and 'qiso', 'qap' parameters:
        ``qiso = qpar ** eta * qper ** (1 - eta)``.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    """
    def initialize(self, *args, r=8., kernel='gauss', eta=1. / 3., **kwargs):
        self.r = float(r)
        self.eta = float(eta)
        self.set_kernel(kernel=kernel)
        super(WiggleSplitPowerSpectrumExtractor, self).initialize(*args, **kwargs)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires['thermodynamics'] = {'rs_drag': None}
            self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
        self._set_base(fiducial=True)

    def set_kernel(self, kernel):
        if kernel == 'gauss':
            self.kernel, self.kernel_deriv = kernel_gauss2, kernel_gauss2_deriv
        elif kernel == 'tophat':
            self.kernel, self.kernel_deriv = kernel_tophat2, kernel_tophat2_deriv
        else:
            raise ValueError('Unknown kernel {}; should be one of ["tophat", "gauss"]')

    def calculate(self):
        self._set_base()

    def _set_base(self, fiducial=False):
        BAOExtractor._set_base(self, fiducial=fiducial)
        cosmo = self.fiducial if fiducial else self.cosmo
        DV = getattr(self, 'DV', self.DV_fid)
        r = self.r * DV / self.DV_fid
        fo = cosmo.get_fourier()
        state = {}
        state['pk_tt_interpolator'] = fo.pk_interpolator(of='theta_cb', **_kw_interp)
        state['pk_dd_interpolator'] = fo.pk_interpolator(of='delta_cb', **_kw_interp)
        state['fsigmar'] = integrate_sigma_r2(r, state['pk_tt_interpolator'], kernel=self.kernel)**0.5
        state['sigmar'] = integrate_sigma_r2(r, state['pk_dd_interpolator'], kernel=self.kernel)**0.5
        state['f'] = state['fsigmar'] / state['sigmar']
        state['m'] = - integrate_sigma_r2(r, state['pk_tt_interpolator'], kernel=self.kernel_deriv) / state['fsigmar']**2 - 3.
        for name, value in state.items(): setattr(self, name + ('_fid' if fiducial else ''), value)

    def get(self):
        self.qbao = self.DV_over_rd / self.DV_over_rd_fid
        self.qap = self.DH_over_DM / self.DH_over_DM_fid
        self.dm = self.m - self.m_fid
        self.df = self.fsigmar / self.fsigmar_fid
        return self


class WiggleSplitPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    r"""
    ShapeFit power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    r : float, default=8.
        Smoothing radius to estimate the normalization of the linear power spectrum.

    kernel : str, default='gauss'
        Kernel for normalization of the linear power spectrum: 'gauss' or 'tophat'.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    with_now : str, default='peakaverage'
        Compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').
    """
    def initialize(self, *args, r=8., kernel='gauss', with_now='peakaverage', **kwargs):
        self.r = float(r)
        WiggleSplitPowerSpectrumExtractor.set_kernel(self, kernel=kernel)
        super(WiggleSplitPowerSpectrumTemplate, self).initialize(*args, apmode='qap', with_now=with_now, **kwargs)
        WiggleSplitPowerSpectrumExtractor._set_base(self, fiducial=True)
        self.filter = PowerSpectrumBAOFilter(self.pk_tt_interpolator_fid, engine=self.with_now, cosmo=self.cosmo, cosmo_fid=self.fiducial)
        self.pknow_tt_interpolator_fid = self.filter.smooth_pk_interpolator()

    def calculate(self, df=1., dm=0., qbao=1.):
        super(WiggleSplitPowerSpectrumTemplate, self).calculate()
        self.f = self.f_fid * df
        self.f0 = self.f0_fid * df
        self.fk = self.fk_fid * df
        kp = 0.05
        k = k = np.geomspace(self.pk_dd_interpolator_fid.extrap_kmin, self.pk_dd_interpolator_fid.extrap_kmax, 2000)
        k = k[(k > k[0] * 2.) & (k < k[-1] / 2.)]  # to avoid hitting boundaries with qbao
        factor = (k / kp)**dm
        pknow_tt = self.pknow_tt_interpolator_fid(k)
        wiggles = self.pk_tt_interpolator_fid(k / qbao) - self.pknow_tt_interpolator_fid(k / qbao)
        factor = _bcast_shape(factor, pknow_tt.shape, axis=0)
        pk_tt_interpolator = PowerSpectrumInterpolator1D(k, (pknow_tt + wiggles) * factor)
        norm = df**2 * self.fsigmar_fid**2 / integrate_sigma_r2(self.r, pk_tt_interpolator, kernel=self.kernel)
        self.pk_tt = pk_tt_interpolator(self.k) * norm
        self.pk_dd = self.pk_tt / self.f**2
        if self.with_now:
            pknow_tt_interpolator = PowerSpectrumInterpolator1D(k, pknow_tt * factor)
            norm = df * self.fsigmar_fid**2 / integrate_sigma_r2(self.r, pknow_tt_interpolator, kernel=self.kernel)
            self.pknow_tt = pknow_tt_interpolator(self.k) * norm
            self.pknow_dd = self.pknow_tt / self.f**2
        if self.only_now:
            for name in ['dd', 'tt']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))
        self.m = self.m_fid + dm

    def get(self):
        return self


def find_turn_over(pk_interpolator, **kwargs):
    # Find turn over, with parabolic interpolation
    k = pk_interpolator.k
    pk = pk_interpolator(k, **kwargs)
    imax = np.argmax(pk, axis=0).flat[0]
    logk = np.log10(k[imax - 1:imax + 2])
    logpk = np.log10(pk[imax - 1:imax + 2])
    # Parabola is a * (logk - logk0)^2 + b, we do not care about b
    c0, c1, c2 = logpk[0] / (logk[0] - logk[1]) / (logk[0] - logk[2]), logpk[1] / (logk[1] - logk[0]) / (logk[1] - logk[2]), logpk[2] / (logk[2] - logk[0]) / (logk[2] - logk[1])
    a = c0 + c1 + c2
    logk0 = (c0 * (logk[1] + logk[2]) + c1 * (logk[0] + logk[2]) + c2 * (logk[0] + logk[1])) / (2 * a)
    assert np.all(a <= 0.)
    assert np.all(logk[0] <= logk0) and np.all(logk0 <= logk[2])
    k0 = 10**logk0
    if kwargs: kwargs.setdefault('grid', False)
    return k0, pk_interpolator(k0, **kwargs)


class TurnOverPowerSpectrumExtractor(BasePowerSpectrumExtractor):
    """
    Extract turn over parameters from base cosmological parameters.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    eta : float, default=1./3.
        Relation between 'qpar', 'qper' and 'qiso', 'qap' parameters:
        ``qiso = qpar ** eta * qper ** (1 - eta)``.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance


    Reference
    ---------
    https://arxiv.org/pdf/2302.07484.pdf
    """
    config_fn = 'power_template.yaml'
    conflicts = [('DM_over_DH', 'qap'), ('DV_times_kTO', 'qto')]

    def initialize(self, *args, r=8., eta=1. / 3., **kwargs):
        self.eta = float(eta)
        super(TurnOverPowerSpectrumExtractor, self).initialize(*args, **kwargs)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires['background'] = {'efunc': {'z': self.z}, 'comoving_angular_distance': {'z': self.z}}
        if self.fiducial is not None:
            self._set_base(fiducial=True)

    def calculate(self):
        self._set_base()

    def _set_base(self, fiducial=False):
        cosmo = self.fiducial if fiducial else self.cosmo
        state = {}
        state['DH'] = (constants.c / 1e3) / (100. * cosmo.efunc(self.z))
        state['DM'] = cosmo.comoving_angular_distance(self.z)
        state['DV'] = state['DH']**self.eta * state['DM']**(1. - self.eta) * self.z**(1. / 3.)
        state['DH_over_DM'] = state['DH'] / state['DM']
        fo = cosmo.get_fourier()
        state['pk_dd_interpolator'] = fo.pk_interpolator(of='delta_cb', **_kw_interp).to_1d(z=self.z)
        state['kTO'], state['pkTO_dd'] = find_turn_over(state['pk_dd_interpolator'])
        state['DV_times_kTO'] = state['DV'] * state['kTO']
        for name, value in state.items(): setattr(self, name + ('_fid' if fiducial else ''), value)

    def get(self):
        if self.fiducial is not None:
            self.qap = self.DH_over_DM / self.DH_over_DM_fid
            self.qto = self.DV_times_kTO / self.DV_times_kTO_fid
        return self


class TurnOverPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    """
    TurnOver power spectrum template.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the growth rate. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance


    Reference
    ---------
    https://arxiv.org/pdf/2302.07484.pdf
    """
    def initialize(self, *args, **kwargs):
        super(TurnOverPowerSpectrumTemplate, self).initialize(*args, with_now=False, apmode='qap', **kwargs)
        TurnOverPowerSpectrumExtractor._set_base(self, fiducial=True)

    def calculate(self, df=1., m=0.6, n=0.9, qto=1., dpto=1.):
        super(TurnOverPowerSpectrumTemplate, self).calculate()
        kTO = self.kTO_fid * qto
        pkTO = self.pkTO_dd_fid * dpto
        x = _bcast_shape(jnp.log10(self.k), self.pk_dd.shape, axis=0) / jnp.log10(kTO) - 1
        self.pk_dd = jnp.empty_like(x)
        mask_m = x > 0
        self.pk_dd = jnp.where(mask_m, pkTO ** (1. - m * x ** 2), pkTO ** (1. - n * x ** 2))
        self.pknow_dd = self.pk_dd
        self.f = self.f_fid * df
        self.f0 = self.f0_fid * df
        self.fk = self.fk_fid * df
        self.DV_times_kTO = self.apeffect.qiso * self.DV_times_kTO_fid
        self.DH_over_DM = self.apeffect.qap * self.DH_over_DM_fid

    def get(self):
        return self


class DirectWiggleSplitPowerSpectrumTemplate(BasePowerSpectrumTemplate):
    """
    Same as :class:`DirectPowerSpectrumTemplate`, i.e. parameterized in terms of base cosmological parameters,
    but rescale the wiggly part of the power spectrum by ``qbao`` in order to marginalize over the sound horizon scale.
    The wiggle amplitude can also be modulated with the Gaussian damping ``sigmabao``.

    Parameters
    ----------
    k : array, default=None
        Theory wavenumbers where to evaluate the linear power spectrum.

    z : float, default=1.
        Effective redshift.

    with_now : str, default=False
        If provided, also compute smoothed, BAO-filtered, linear power spectrum with this engine (e.g. 'wallish2018', 'peakaverage').

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology, used to compute the linear power spectrum. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    Reference
    ----------
    https://arxiv.org/abs/2112.10749
    """
    def initialize(self, *args, cosmo=None, with_now='peakaverage', **kwargs):
        super(DirectWiggleSplitPowerSpectrumTemplate, self).initialize(*args, with_now=with_now, **kwargs)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        # keep only derived parameters and qbao, sigmabao, others are transferred to Cosmoprimo
        params = self.init.params.select(derived=True) + self.init.params.select(basename=['qbao', 'sigmabao'])
        if is_external_cosmo(self.cosmo):
            # cosmo_requires only used for external bindings (cobaya, cosmosis, montepython): specifies the input theory requirements
            self.cosmo_requires = {'fourier': {'sigma8_z': {'z': self.z, 'of': [('delta_cb', 'delta_cb'), ('theta_cb', 'theta_cb')]},
                                               'pk_interpolator': {'z': self.z, 'k': self.k, 'of': [('delta_cb', 'delta_cb')]}}, 'thermodynamics': {'rs_drag': None}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial)
            # transfer the parameters of the template (Omega_m, logA, h, etc.) to Cosmoprimo
            self.cosmo.init.params = [param for param in self.init.params if param not in params]
        self.init.params = params
        # Alcock-Paczynski effect, that is known given the cosmo and fiducial
        self.apeffect = APEffect(z=self.z, fiducial=self.fiducial, cosmo=self.cosmo, mode='geometry').runtime_info.initialize()
        if is_external_cosmo(self.cosmo):
            # update cosmo_requires with background quantities
            self.cosmo_requires.update(self.apeffect.cosmo_requires)

    def calculate(self, qbao=1., sigmabao=0.):
        # compute the power spectrum for the current cosmo
        BasePowerSpectrumExtractor._set_base(self, with_now=True)
        k = self.pk_dd_interpolator.k
        wiggles = _interp(self.k / qbao, k, self.pk_dd_interpolator(k) - self.pknow_dd_interpolator(k))
        wiggles *= _bcast_shape(jnp.exp(- (self.k * sigmabao)**2), wiggles.shape, axis=0)
        self.pknow_dd = self.pknow_dd_interpolator(self.k)
        self.pk_dd = self.pknow_dd + wiggles
        if self.only_now:  # only used if we want to take wiggles out of our model (e.g. for BAO)
            for name in ['dd']:
                setattr(self, 'pk_' + name, getattr(self, 'pknow_' + name))
