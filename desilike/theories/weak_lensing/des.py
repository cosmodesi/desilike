import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os

from desilike.jax import numpy as jnp
from desilike.base import BaseCalculator
from desilike.cosmo import is_external_cosmo
from desilike.theories.primordial_cosmology import get_cosmo
from desilike.theories import Cosmoprimo
from .base import TwoPTCalculator

_kw_interp = dict(extrap_kmin=1e-7, extrap_kmax=1e4)

data_path = '/Users/illyasviel/Desktop/Cosmology/DES/des-y3/des-y3/des3'

class DESWeakLensing3x2pt(BaseCalculator):

    config_fn = 'weak_lensing.yaml'

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
        self.z = self.zmid[self.zmid <= self.zmax]

        # keep only derived parameters, others are transferred to Cosmoprimo
        params = self.init.params.select(derived=True)
        if is_external_cosmo(self.cosmo):
            # cosmo_requires only used for external bindings (cobaya, cosmosis, montepython): specifies the input theory requirements
            self.cosmo_requires = {'fourier': {'pk_interpolator': {'z': self.z, 
                                                                   'of': [('delta_m', 'delta_m'), ('delta_m', 'phi_plus_psi'), ('phi_plus_psi', 'phi_plus_psi')],
                                                                   'non_linear': (True, False)}}, 
                                               'background': {'z': self.z, 'comoving_radial_distance': None, 'efunc': None}}
        elif cosmo is None:
            self.cosmo = Cosmoprimo(engine=engine, non_linear='takahashi')
            # transfer the parameters of the template (Omega_m, logA, h, etc.) to Cosmoprimo
            self.cosmo.params = [param for param in self.params if param not in params]
            self.cosmo()
        self.init.params = params
        fo = self.cosmo.get_fourier()
        self.pk_mm_l_interpolator = fo.pk_interpolator(of='delta_m', **_kw_interp).to_1d(z=self.z)
        self.pk_mm_nl_interpolator = fo.pk_interpolator(of='delta_m', non_linear='halofit', **_kw_interp).to_1d(z=self.z)
        self.pk_ww_interpolator = fo.pk_interpolator(of='phi_plus_psi', **_kw_interp).to_1d(z=self.z)
        self.pk_mw_interpolator = fo.pk_interpolator(of=('delta_m', 'phi_plus_psi'), **_kw_interp).to_1d(z=self.z)

    def calculate(self):
        TwoPT = TwoPTCalculator(self.params, self.cosmo, self.pk_mm_nl_interpolator, self.pk_mm_l_interpolator, self.pk_ww_interpolator, self.pk_mw_interpolator,
                                   zbin_sp=self.zbin_sp, zbin_w_sp=self.zbin_w_sp, zs=self.z,
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