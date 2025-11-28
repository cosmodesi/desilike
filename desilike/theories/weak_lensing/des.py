import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os

from desilike.jax import numpy as jnp
from desilike.base import BaseCalculator
from desilike.cosmo import is_external_cosmo
from desilike.parameter import ParameterCollection
from desilike.theories.primordial_cosmology import get_cosmo, Cosmoprimo, Cosmology
from .base import TwoPTCalculator

_kw_interp = dict(extrap_kmin=1e-7, extrap_kmax=1e4)

data_path = '/Users/illyasviel/Desktop/Cosmology/DES/des-y3/des3'

class BasePowerSpectrumExtractor(BaseCalculator):

    """Base class to extract shape parameters from linear power spectrum."""
    config_fn = 'weak_lensing.yaml'

    def initialize(self, cosmo=None, fiducial='DESI'):
        self.z = np.linspace(0., 3., 300)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        params = self.params.select(derived=True)
        if is_external_cosmo(self.cosmo):
            self.cosmo_requires = {'fourier': {'pk_interpolator': {'z': self.z, 
                                                                   'of': [('delta_m', 'delta_m'), ('delta_m', 'phi_plus_psi'), ('phi_plus_psi', 'phi_plus_psi')],
                                                                   'non_linear': 'takahashi'}}, 
                                               'background': {'z': self.z, 'comoving_radial_distance': None, 'hubble_function': None}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(fiducial=self.fiducial, non_linear='halofit', engine='camb')
            self.cosmo.init.params = [param for param in self.params if param not in params]
        self.init.params = params
        BasePowerSpectrumExtractor._set_base(self, fiducial=True)

    def _set_base(self, fiducial=False):
        cosmo = Cosmology(non_linear='halofit', engine='camb') if fiducial else self.cosmo
        if not isinstance(cosmo, Cosmology): cosmo = cosmo.cosmo
        fo = cosmo.get_fourier()
        state = {}
        #state['pk_dd_interpolator'] = fo.pk_interpolator(of='delta_cb', **_kw_interp).to_1d(z=self.z)
        #state['pk_tt_interpolator'] = fo.pk_interpolator(of='theta_cb', **_kw_interp).to_1d(z=self.z)
        state['pk_mm_l_interpolator'] = fo.pk_interpolator(of='delta_m', **_kw_interp).to_1d(z=self.z)
        state['pk_mm_nl_interpolator'] = fo.pk_interpolator(of='delta_m', non_linear='halofit', **_kw_interp).to_1d(z=self.z)
        state['pk_ww_interpolator'] = fo.pk_interpolator(of='phi_plus_psi', **_kw_interp).to_1d(z=self.z)
        state['pk_mw_interpolator'] = fo.pk_interpolator(of=('delta_m', 'phi_plus_psi'), **_kw_interp).to_1d(z=self.z)
        for name, value in state.items(): setattr(self, name + ('_fid' if fiducial else ''), value)

class BasePowerSpectrumTemplate(BasePowerSpectrumExtractor):

    """Base class for linear power spectrum template."""
    # See calculate(self) for the list of attributes that must be set by calculate

    config_fn = 'weak_lensing.yaml'
    _interpolator_k = np.logspace(-5., 2., 1000)  # more than classy

    def initialize(self, k=None, fiducial='DESI', cosmo=None):
        self.z = np.linspace(0., 3., 300)
        self.fiducial = get_cosmo(fiducial)
        self.cosmo=Cosmoprimo(fiducial=self.fiducial, non_linear='takahashi', engine='camb')
        if k is None: k = np.logspace(-3., 1., 400)
        self.k = np.array(k, dtype='f8')
        self.cosmo_requires = {}
        BasePowerSpectrumExtractor._set_base(self, fiducial=True)

    def calculate(self):
        # These are the quantities that should be set by a BasePowerSpectrumTemplate-inherited class.
        # See BasePowerSpectrumExtrator._set_base for how to get these quantities from self.cosmo
        # fk is sqrt(pk_tt / pk_dd)
        # f0 is the limit of fk for k -> 0
        for name in ['f', 'f0', 'fk', 'pk_dd_interpolator', 'pk_dd']:
            setattr(self, name, getattr(self, name + '_fid'))
        for name in ['xip', 'xim', 'gammat', 'wtheta']:
            setattr(self, name, getattr(self, name))

    def __getstate__(self):
        state = {}
        for name in ['k', 'z', 'fiducial']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for suffix in ['', '_fid']:
            #for name in ['sigma8', 'fsigma8', 'f', 'f0', 'pk_dd_interpolator', 'pk_dd'] + ['pknow_dd_interpolator', 'pknow_dd']:
            for name in ['f', 'f0', 'fk', 'pk_dd', 'xip', 'xim', 'gammat', 'wtheta']:
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
        self.__dict__.update(state)

class DESWeakLensing3x2pt(BasePowerSpectrumTemplate):
    def initialize(self, *args, cosmo=None, fiducial='DESI', des_model='DES_3YR', ia_model='TATT', Limber=False, Weyl=False, fourier='binned_bessels',
                   nzbins=None, nwbins=None, bin_pairs=None, **kwargs):
        engine = kwargs.pop('engine', 'camb')
        super(DESWeakLensing3x2pt, self).initialize(*args, **kwargs)
        self.cosmo_requires = {}
        self.cosmo = cosmo
        self.fiducial = get_cosmo(fiducial)

        # DES data and settings initialization
        if 'DES' in des_model:
            year = des_model.split('_')[1]
            if year not in ['1YR', '3YR', '6YR']:
                raise ValueError(f'Invalid DES model: {des_model}. Supported models are: DES_1YR, DES_3YR, DES_6YR.')
        else:
            raise ValueError(f'Invalid DES model: {des_model}. Supported models are: DES_1YR, DES_3YR, DES_6YR.')
        self.des_model = des_model
        self.ia_model = ia_model
        self.Limber = Limber
        self.Weyl = Weyl    # True when running MG
        self.fourier = fourier
        self.nzbins = nzbins
        self.nwbins = nwbins
        self.bin_pairs = bin_pairs
        if year == '1YR':
            self.nzbins = 4
            self.nwbins = 5
            self.ia_model = 'NLA'  # only NLA is supported for DES Y1
            self.Limber = False  # Limber approximation not used in DES Y1
            self.fourier = 'binned_bessels'  # Fourier space integrals are done with binned Bessel functions in DES Y1
            self.bin_pairs = {'xip': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
                              'xim': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
                              'gammat': [(0, 0), (0, 1), (0, 2), (0, 3),
                                         (1, 0), (1, 1), (1, 2), (1, 3),
                                         (2, 0), (2, 1), (2, 2), (2, 3),
                                         (3, 0), (3, 1), (3, 2), (3, 3),
                                         (4, 0), (4, 1), (4, 2), (4, 3)],
                              'wtheta': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}
            
        elif year == '3YR':
            self.nzbins = 4
            self.nwbins = 6
            self.Limber = True  # Limber approximation used in DES Y3
            self.fourier = 'legendre'  # Fourier space integrals are done with Legendre polynomials in DES Y3
            self.bin_pairs = {'xip': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
                              'xim': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
                              'gammat': [(0, 0), (0, 1), (0, 2), (0, 3),
                                         (1, 0), (1, 1), (1, 2), (1, 3),
                                         (2, 0), (2, 1), (2, 2), (2, 3),
                                         (3, 0), (3, 1), (3, 2), (3, 3),
                                         (4, 0), (4, 1), (4, 2), (4, 3),
                                         (5, 0), (5, 1), (5, 2), (5, 3)],
                              'wtheta': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]}
        if self.ia_model not in ['NLA', 'TATT']:
            raise ValueError(f'Invalid IA model: {self.ia_model}. Supported models are: NLA, TATT.')
        if self.fourier not in ['binned_bessels', 'legendre']:
            raise ValueError(f'Invalid Fourier model: {self.fourier}. Supported models are: binned_bessels, legendre.')

        nz_source = np.loadtxt(os.path.join(data_path, 'DES_{}_final_nz_source.dat'.format(year)))
        self.zmid = nz_source[:, 1]
        self.zbin_sp = []
        for b in range(self.nzbins):
            self.zbin_sp += [InterpolatedUnivariateSpline(self.zmid, nz_source[:, b + 3])]
        nz_lens = np.loadtxt(os.path.join(data_path, 'DES_{}_final_nz_lens.dat'.format(year)))
        assert (np.array_equal(nz_lens[:, 1], self.zmid))
        self.zbin_w_sp = []
        for b in range(self.nwbins):
            self.zbin_w_sp += [InterpolatedUnivariateSpline(self.zmid, nz_lens[:, b + 3])]
        self.zmax = self.zmid[-1]
        self.zs = self.zmid[self.zmid <= self.zmax]
        self.z = self.zs

        # keep only derived parameters, others are transferred to Cosmoprimo
        params = self.init.params.select(derived=True)
        if is_external_cosmo(self.cosmo):
            # cosmo_requires only used for external bindings (cobaya, cosmosis, montepython): specifies the input theory requirements
            self.cosmo_requires = {'fourier': {'pk_interpolator': {'z': self.z, 
                                                                   'of': [('delta_m', 'delta_m'), ('delta_m', 'phi_plus_psi'), ('phi_plus_psi', 'phi_plus_psi')],
                                                                   'non_linear': (True, False)}}, 
                                               'background': {'z': self.z, 'comoving_radial_distance': None, 'efunc': None}}
        elif cosmo is None:
            self.cosmo = Cosmology(engine=engine, non_linear='takahashi')
            # transfer the parameters of the template (Omega_m, logA, h, etc.) to Cosmoprimo
            self.cosmo.params = [param for param in self.params if param not in params]
        self.init.params = params

    def calculate(self):
        BasePowerSpectrumExtractor._set_base(self)
        TwoPT = TwoPTCalculator(self.params, self.cosmo, self.pk_mm_nl_interpolator, self.pk_mm_l_interpolator, self.pk_ww_interpolator, self.pk_mw_interpolator,
                                   zbin_sp=self.zbin_sp, zbin_w_sp=self.zbin_w_sp, zs=self.zs,
                                   nzbins=self.nzbins, nwbins=self.nwbins, bin_pairs=self.bin_pairs,
                                   Limber=self.Limber, Weyl=self.Weyl, ia_model=self.ia_model, fourier=self.fourier)
        self.xip = TwoPT['xip']
        self.xim = TwoPT['xim']
        self.gammat = TwoPT['gammat']
        self.wtheta = TwoPT['wtheta']
        self.ell = TwoPT['ell']
        self.cl_xip = TwoPT['cl_xip']
        self.cl_xim = TwoPT['cl_xim']
        self.cl_gammat = TwoPT['cl_gammat']
        self.cl_wtheta = TwoPT['cl_wtheta']
        self.chis = TwoPT['chis']
        self.Hs = TwoPT['Hs']
        self.nz_lens = TwoPT['nz_lens']
        self.nz_source = TwoPT['nz_source']