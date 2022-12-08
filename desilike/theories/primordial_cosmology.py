from cosmoprimo import Cosmology

from desilike.base import BaseCalculator


class BasePrimordialCosmology(BaseCalculator):

    pass


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
    return cosmoprimo.Cosmology(**cosmo)


def external_cosmo(cosmo):
    return isinstance(cosmo, str) and cosmo == 'external'


def get_from_cosmo(cosmo, name):
    if name.lower().startswith('omega_'):
        name = name[:5] + '0' + name[5:]
    if name.startswith('omega'):
        return get_from_cosmo(cosmo, 'O' + name[1:]) * cosmo.h ** 2
    if name == 'k_pivot':
        return cosmo.k_pivot * cosmo.h
    toret = getattr(cosmo, name)
    if not toret:
        return 0.
    return toret


class Cosmoprimo(BasePrimordialCosmology):

    config_fn = 'primordial_cosmology.yaml'

    def initialize(self, fiducial=None, **kwargs):
        # kwargs is engine, extra_params
        fiducial_input = bool(fiducial)
        if fiducial is not None:
            fiducial = get_cosmo(fiducial)
        else:
            fiducial = Cosmology()
        self.fiducial = fiducial.clone(**kwargs)
        if fiducial_input:
            for param in self.params:
                if not param.drop:
                    param.value = get_from_cosmo(self.fiducial, param.basename)
        self.cosmo_requires = {'fiducial': self.fiducial.__getstate__(), 'params': dict.fromkeys(self.params.basenames())}

    def calculate(self, **params):
        self.cosmo = self.fiducial.clone(**params)

    def get(self):
        return self.cosmo

    def __getattr__(self, name):
        if 'cosmo' in self.__dict__:
            return get_from_cosmo(self.cosmo, name)
        raise AttributeError('{} has not attribute {}'.format(self, name))

    @classmethod
    def install(cls, config):
        config.pip('git+https://github.com/cosmodesi/cosmoprimo')
