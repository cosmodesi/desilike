import numpy as np
from scipy import optimize

from cosmoprimo import Cosmology, CosmologyError, constants  # constants is imported by e.g. theories.galaxy_clustering.base

from desilike.base import BaseCalculator


class BasePrimordialCosmology(BaseCalculator):

    """Base primordial cosmology computation."""


conversions = {'logA': 'ln10^10A_s'}


def convert(params):
    return {conversions.get(name, name): value for name, value in params.items()}


def get_cosmo(cosmo):
    if cosmo is None:
        return cosmo
    import cosmoprimo
    if isinstance(cosmo, cosmoprimo.Cosmology):
        return cosmo
    if isinstance(cosmo, str):
        cosmo = (cosmo, {})
    if isinstance(cosmo, tuple):
        return getattr(cosmoprimo.fiducial, cosmo[0])(**cosmo[1])
    return cosmoprimo.Cosmology(**convert(cosmo))


def get_from_cosmo(cosmo, name):
    name = conversions.get(name, name)
    if name.lower().startswith('omega_'):
        name = name[:5] + '0' + name[5:]
    if name.startswith('omega'):
        return get_from_cosmo(cosmo, 'O' + name[1:]) * cosmo.h ** 2
    scale = None
    if name == 'm_ncdm':
        name = 'm_ncdm_tot'
    if name == 'theta_MC_100':
        name = 'theta_cosmomc'
        scale = 100.
    if name == 'k_pivot':
        return cosmo.k_pivot * cosmo.h
    if name == 'n_s':
        # Rafaela edit for axiclass: depending on what hyperparameters are set for EDE
        # calling ns as an attribute for the cosmology leads to segementation fault
        # Don't really know why...
        # that's why this quick fix:
        return cosmo['n_s']
    try:
        toret = getattr(cosmo, name)
    except AttributeError:
        try:
            toret = cosmo[name]  # parameter
        except (CosmologyError, KeyError) as exc:
            raise AttributeError from exc
    if not toret:
        return 0.
    if scale is not None:
        return scale * toret
    return toret


def _clone(self, params, base='input'):
    cparams = {}
    for name, value in params.items():
        cparams[conversions.get(name, name)] = value

    theta_MC_100 = cparams.pop('theta_MC_100', None)
    self.cosmo = self.fiducial.clone(base=base, **cparams)
    if theta_MC_100 is not None:
        if 'h' in cparams:
            raise ValueError('Cannot provide both theta_MC_100 and h')
        # With self.cosmo.get_thermodynamics().theta_cosmomc
        # Typically takes 18 iterations and ~0.8 s
        # The computation of the thermodynamics is the most time consuming
        # The 'theta_cosmomc' call takes ~0.1 s and is accurate within 3e-6 (rel.), ~1% of Planck errors
        self.cosmo = self.cosmo.solve('h', 'theta_MC_100', theta_MC_100, xtol=1e-6, rtol=1e-6)

    return self.cosmo


class Cosmoprimo(BasePrimordialCosmology):

    """Primordial cosmology calculation, based on :mod:`cosmoprimo`."""
    config_fn = 'primordial_cosmology.yaml'
    _likelihood_catch_errors = (CosmologyError,)

    @staticmethod
    def _params(params, engine=None):
        # engine-specific parameters
        if engine == 'axiclass':
            for name in ['w0_fld', 'wa_fld']: del params[name]  # do not seem supported by axiclass?
            params['fraction_axion_ac'] = dict(prior=dict(limits=[1e-4, 0.5]), ref=dict(dist='norm', loc=0.02, scale=0.1), proposal=0.01, delta=0.05, latex='f_{\rm ede}(a_c)')  # axiclass seems to complain when fraction_axion_ac is exactly 0
            params['log10_axion_ac'] = dict(value=-3.586941554, prior=dict(limits=[-4.5, -3.0]), ref=dict(dist='norm', loc=-3.5, scale=0.5), proposal=0.1, delta=0.1, latex='\log_{10}a_c')
        return params

    def initialize(self, fiducial=None, **kwargs):
        """
        Initialize :class:`Cosmoprimo`.

        Parameters
        ----------
        fiducial : str, tuple, dict, cosmoprimo.Cosmology
            Specifications for fiducial cosmology, which is used to fill in parameter values :attr:`Parameter.value` if provided.
            Either:

            - str: name of fiducial cosmology in :class:`cosmoprimo.fiucial`
            - tuple: (name of fiducial cosmology, dictionary of parameters to update)
            - dict: dictionary of parameters
            - :class:`cosmoprimo.Cosmology`: Cosmology instance

        **kwargs : dict
            Optionally, dictionary of parameters to update ``fiducial`` with.
        """
        # kwargs is engine, extra_params
        fiducial_input = bool(fiducial)
        if fiducial is None:
            self.fiducial = Cosmology()
        else:
            self.fiducial = get_cosmo(fiducial)
        engine = kwargs.get('engine', None)
        if engine == 'axiclass':
            extra_params = dict(scf_parameters=[2.72, 0.0], do_shooting=True, do_shooting_scf=True,
                                scf_potential='axion', n_axion=3, security_small_Omega_scf=0.001,
                                n_axion_security=2.09, use_big_theta_scf=True, scf_has_perturbations=True,
                                attractor_ic_scf=False, scf_tuning_index=0)
            kwargs['extra_params'] = {**kwargs.get('extra_params', {}), **extra_params}
            # Rafaela Gsponer edit:
            # axiclass has sometimes problems if hyperparameters are initialized
            # before cosmological parameters
            for param in self.params.select(basename='*axion*', input=True):
                kwargs.setdefault(param.name, param.value)
        self.fiducial = _clone(self, kwargs)
        if any(name in self.params.basenames(input=True) for name in ['h', 'H0']):
            for param in self.params.select(basename='theta_MC_100'):
                del self.params[param]
        if fiducial_input:
            for param in self.params:
                param.update(value=get_from_cosmo(self.fiducial, param.basename))
        basis = {param.name: param.value for param in self.params.select(input=True)}
        self.fiducial = _clone(self, basis)  # just to set the parameter basis
        self.cosmo_requires = {'fiducial': self.fiducial.__getstate__(), 'params': dict.fromkeys(self.params.basenames(input=True))}

    def calculate(self, **params):
        self.cosmo = _clone(self, params)

    def get(self):
        return self.cosmo

    def __getattr__(self, name):
        if 'cosmo' in self.__dict__:
            return get_from_cosmo(self.cosmo, name)
        raise AttributeError('{} has not attribute {}; try calling it first?'.format(self, name))

    def __getitem__(self, name):
        return self.cosmo.__getitem__(name)

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')