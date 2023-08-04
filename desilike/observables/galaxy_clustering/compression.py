from desilike.jax import numpy as jnp
from desilike.parameter import Parameter
from desilike.base import BaseCalculator
from desilike.samples import load_source
from desilike.theories.galaxy_clustering.power_template import BAOExtractor, StandardPowerSpectrumExtractor, ShapeFitPowerSpectrumExtractor, WiggleSplitPowerSpectrumExtractor, BandVelocityPowerSpectrumExtractor, TurnOverPowerSpectrumExtractor


def get_quantities(conflict_names):
    toret = []
    for conflicts in conflict_names:
        for conflict in conflicts:
            if conflict not in toret:
                toret.append(conflict)
    return toret


class BaseCompressionObservable(BaseCalculator):

    conflict_names = []
    meta_names = []

    def initialize(self, extractor=None, data=None, covariance=None, quantities=None, **kwargs):
        self.extractor = extractor
        self.extractor.init.update(**kwargs)
        if quantities is None:
            quantities = get_quantities(self.conflict_names)
        quantities = list(quantities)
        self.load_data(data=data, covariance=covariance, quantities=quantities, meta=[name for name in self.meta_names if name not in kwargs])

    def load_data(self, data=None, covariance=None, quantities=None, meta=None):
        if data is None:
            data = covariance
        if isinstance(data, dict):
            self.quantities = list(quantities)
            self.extractor(**data)
            self.calculate()
            self.flatdata = self.flattheory
        else:
            if meta:
                try:
                    source = load_source(data, params=meta, choice=True, return_type='dict')
                except (ValueError, AttributeError, KeyError):
                    pass
                else:
                    meta = {name: source.pop(name) for name in self.meta_names if name in source}
                    if self.mpicomm.rank == 0:
                        self.log_info('Found meta parameters {}.'.format(meta))
                    self.extractor.init.update(meta)
            source = load_source(data, params=quantities or None, choice=True, return_type='dict')
            quantities = [Parameter(quantity) for quantity in source.keys()]
            self.quantities = [quantity.basename for quantity in quantities]
            self.flatdata = [source[quantity.name] for quantity in quantities]
        if self.mpicomm.rank == 0:
            self.log_info('Found quantities {}.'.format(self.quantities))
        for conflicts in self.conflict_names:
            if all(quantity in self.quantities for quantity in conflicts):
                raise ValueError('Found conflicting quantities: {}'.format(conflicts))
        self.covariance = None
        if covariance is not None:
            self.covariance = load_source(covariance, params=quantities or None, cov=True, return_type='nparray')

    def calculate(self):
        self.flattheory = jnp.array([getattr(self.extractor, quantity) for quantity in self.quantities])

    def __getstate__(self):
        state = {}
        for name in ['flatdata', 'covariance', 'flattheory', 'quantities']:
            state[name] = getattr(self, name)
        return state


class BAOCompressionObservable(BaseCompressionObservable):
    """
    BAO observable: compare (compressed) BAO measurements
    (in terms of ratios of distances to the sound horizon scale at the drag epoch) to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        BAO parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for BAO parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['DM_over_rd', 'DH_over_rd', 'DV_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qiso', 'qap']``.

    z : float, default=None
        Effective redshift.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    **kwargs: dict
        Optional arguments for :class:`BAOExtractor`.
    """
    def initialize(self, *args, **kwargs):
        super(BAOCompressionObservable, self).initialize(*args, extractor=BAOExtractor(), **kwargs)


class StandardCompressionObservable(BaseCompressionObservable):
    """
    Standard RSD compression observable: compare compressed RSD measurements to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        Standard compressed parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for compressed parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['fsigmar', 'df', 'DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']``.

    z : float, default=None
        Effective redshift.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    **kwargs : dict
        Other optional arguments for :class:`StandardPowerSpectrumExtractor`, e.g., ``r``.
    """
    def initialize(self, *args, **kwargs):
        super(StandardCompressionObservable, self).initialize(*args, extractor=StandardPowerSpectrumExtractor(), **kwargs)


class ShapeFitCompressionObservable(BaseCompressionObservable):
    """
    ShapeFit observable: compare ShapeFit measurements to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        ShapeFit parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for ShapeFit parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['m', 'n', 'f_sqrt_Ap', 'dm', 'dn', 'df', 'DM_over_rd', 'DH_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qap', 'qiso']``.

    z : float, default=None
        Effective redshift.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    **kwargs : dict
        Other optional arguments for :class:`ShapeFitPowerSpectrumExtractor`, e.g., ``kp``, ``a``.


    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    """
    meta_names = ['kp']

    def initialize(self, *args, **kwargs):
        super(ShapeFitCompressionObservable, self).initialize(*args, extractor=ShapeFitPowerSpectrumExtractor(), **kwargs)
        self.extractor.init.update(n_varied='n' in self.quantities)


class WiggleSplitCompressionObservable(BaseCompressionObservable):
    """
    Wiggle-split observable: compare the amplitude and tilt of the velocity power spectrum, and BAO position to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        Wiggle-split parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for band power parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['df', 'dm', 'qap', 'qbao']``.

    z : float, default=None
        Effective redshift.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    **kwargs : dict
        Other optional arguments for :class:`WiggleSplitPowerSpectrumExtractor`.
    """
    meta_names = ['r']

    def initialize(self, *args, **kwargs):
        super(WiggleSplitCompressionObservable, self).initialize(*args, extractor=WiggleSplitPowerSpectrumExtractor(), **kwargs)


class BandVelocityCompressionObservable(BaseCompressionObservable):
    """
    Band velocity power observable: compare band power measurements to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        Band velocity power parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for band power parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['dptt*', 'df', 'qap']``.

    z : float, default=None
        Effective redshift.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    **kwargs : dict
        Other optional arguments for :class:`BandVelocityPowerSpectrumExtractor`.
    """
    meta_names = ['kp']

    def initialize(self, *args, **kwargs):
        super(BandVelocityCompressionObservable, self).initialize(*args, extractor=BandVelocityPowerSpectrumExtractor(), **kwargs)


class TurnOverCompressionObservable(BaseCompressionObservable):
    """
    Turn over observable: compare (compressed) turn over measurement
    (in terms of product of distance with the turn over scale :math:`k_{\mathrm{TO}}`) to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        BAO parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for BAO parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['DV_times_kTO', 'DM_over_DH', 'qto', 'qap']``.

    z : float, default=None
        Effective redshift.

    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:

        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`
        - tuple: (name of fiducial cosmology, dictionary of parameters to update)
        - dict: dictionary of parameters
        - :class:`cosmoprimo.Cosmology`: Cosmology instance

    **kwargs: dict
        Optional arguments for :class:`BAOExtractor`.
    """
    def initialize(self, *args, **kwargs):
        super(TurnOverCompressionObservable, self).initialize(*args, extractor=TurnOverPowerSpectrumExtractor(), **kwargs)