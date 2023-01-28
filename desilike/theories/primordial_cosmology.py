from cosmoprimo import Cosmology

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


def external_cosmo(cosmo):
    return isinstance(cosmo, str) and cosmo == 'external'


def get_from_cosmo(cosmo, name):
    name = conversions.get(name, name)
    if name.lower().startswith('omega_'):
        name = name[:5] + '0' + name[5:]
    if name.startswith('omega'):
        return get_from_cosmo(cosmo, 'O' + name[1:]) * cosmo.h ** 2
    if name == 'k_pivot':
        return cosmo.k_pivot * cosmo.h
    try:
        toret = getattr(cosmo, name)
    except AttributeError:
        toret = cosmo[name]
    if not toret:
        return 0.
    return toret


class Cosmoprimo(BasePrimordialCosmology):

    """Primordial cosmology calculation, based on :mod:`cosmoprimo`."""
    config_fn = 'primordial_cosmology.yaml'

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
            fiducial = Cosmology()
        else:
            fiducial = get_cosmo(fiducial)
        self.fiducial = fiducial.clone(**kwargs)
        if fiducial_input:
            for param in self.params:
                param.update(value=get_from_cosmo(self.fiducial, param.basename))
        self.cosmo_requires = {'fiducial': self.fiducial.__getstate__(), 'params': dict.fromkeys(self.params.basenames())}

    def calculate(self, **params):
        self.cosmo = self.fiducial.clone(**{name: float(value) for name, value in convert(params).items()})

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
