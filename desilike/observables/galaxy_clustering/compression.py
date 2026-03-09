import numpy as np
import lsstypes as types

from desilike.jax import numpy as jnp
from desilike.base import BaseCalculator
from desilike.theories.galaxy_clustering.power_template import BAOExtractor, BAOPhaseShiftExtractor, StandardPowerSpectrumExtractor, ShapeFitPowerSpectrumExtractor, WiggleSplitPowerSpectrumExtractor, BandVelocityPowerSpectrumExtractor, TurnOverPowerSpectrumExtractor


class BaseCompressionObservable(BaseCalculator):

    conflict_names = []

    def initialize(self, extractor: BaseCalculator=None, data: types.ObservableLike | np.ndarray=None,
                   covariance: None | types.CovarianceMatrix | np.ndarray=None,
                   parameters: list[str]=None, name: str=None, **kwargs):
        self.extractor = extractor
        self.extractor.init.update(**kwargs)
        self.name = str(name)
        self._format_compression_data(data=data, covariance=covariance, parameters=parameters)

    def _format_compression_data(self, data=None, covariance=None, parameters=None):
        """Set data, covariance."""
        custom_data = not isinstance(data, types.ObservableLike)
        custom_covariance = not isinstance(covariance, types.CovarianceMatrix)
        if custom_data:
            if parameters is None:
                raise ValueError('When input data is an array or None, provide parameters')
            else:
                parameters = list(parameters)
            if data is None:
                data = np.zeros(len(parameters), dtype='f8')
            data = [types.ObservableLeaf(value=np.atleast_1d(value)) for value in data]
            data = types.ObservableTree(data, parameters=parameters)
        self.data = data
        self.parameters = self.data.parameters
        self.flatdata = self.data.value()
        if self.mpicomm.rank == 0:
            self.log_info('Found parameters {}.'.format(self.parameters))
        for conflicts in self.conflict_names:
            if all(parameter in self.parameters for parameter in conflicts):
                raise ValueError('Found conflicting parameters: {}'.format(conflicts))
        self.covariance = None
        if covariance is not None:
            if custom_covariance:
                covariance = np.array(covariance)
                assert covariance.ndim == 2
                covariance = types.CovarianceMatrix(value=covariance, observable=data.clone(value=np.zeros_like(data.value())))
            elif not custom_data:  # match
                covariance = covariance.at.observable.match(data)
            assert covariance.shape[0] == data.size, 'covariance shape must match data size'
            self.covariance = covariance

    def calculate(self):
        """Set flattheory: obtained from :attr:`extractor`."""
        self.flattheory = jnp.array([getattr(self.extractor, parameter) for parameter in self.parameters])

    def get(self):
        return self.flattheory

    def __getstate__(self, varied=True, fixed=True):
        state = {}
        for name in (['data', 'flatdata', 'covariance', 'parameters', 'name'] if fixed else []) + (['flattheory'] if varied else []):
            state[name] = getattr(self, name)
        return state


class BAOCompressionObservable(BaseCompressionObservable):
    """
    BAO observable: compare (compressed) BAO measurements (in terms of ratios of distances
    to the sound horizon scale at the drag epoch) to theory predictions.

    Parameters
    ----------
    data : array, lsstypes.ObservableTree, default=None
        BAO parameters. Either:
        - flat array (of all parameters). Additionally provide list of parameters;
        - :class:`lsstypes.ObservableTree` (contains all necessary information);
        - ``None``. Set to 0.
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    parameters : list, tuple
        Parameters; choose from ``['DM_over_rd', 'DH_over_rd', 'DV_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qiso', 'qap']``.
    z : float, default=None
        Effective redshift.
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:
        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`;
        - :class:`cosmoprimo.Cosmology`: Cosmology instance.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs: dict
        Optional arguments for :class:`BAOExtractor`.
    """
    def initialize(self, *args, name='bao', **kwargs):
        super(BAOCompressionObservable, self).initialize(*args, extractor=BAOExtractor(), name=name, **kwargs)


class BAOPhaseShiftCompressionObservable(BaseCompressionObservable):
    """
    BAO observable with phase shift: compare (compressed) BAO measurements
    (in terms of ratios of distances to the sound horizon scale at the drag epoch, and shift) to theory predictions.

    Reference
    ---------
    https://arxiv.org/pdf/1803.10741

    Parameters
    ----------
    data : array, lsstypes.ObservableTree, default=None
        BAO parameters. Either:
        - flat array (of all parameters). Additionally provide list of parameters;
        - :class:`lsstypes.ObservableTree` (contains all necessary information);
        - ``None``. Set to 0.
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    parameters : list, tuple
        Parameters; choose from ``['DM_over_rd', 'DH_over_rd', 'DV_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qiso', 'qap', 'baoshift']``.
    z : float, default=None
        Effective redshift.
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:
        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`;
        - :class:`cosmoprimo.Cosmology`: Cosmology instance.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs: dict
        Optional arguments for :class:`BAOPhaseShiftExtractor`.
    """
    def initialize(self, *args, name='baoshift', **kwargs):
        super(BAOPhaseShiftCompressionObservable, self).initialize(*args, extractor=BAOPhaseShiftExtractor(), name=name, **kwargs)


class StandardCompressionObservable(BaseCompressionObservable):
    """
    Standard RSD compression observable: compare compressed RSD measurements to theory predictions.

    Parameters
    ----------
    data : array, lsstypes.ObservableTree, default=None
        BAO parameters. Either:
        - flat array (of all parameters). Additionally provide list of parameters;
        - :class:`lsstypes.ObservableTree` (contains all necessary information);
        - ``None``. Set to 0.
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    parameters : list, tuple
        Parameters; choose from ``['fsigmar', 'df', 'DM_over_rd', 'DH_over_rd', 'DV_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qiso', 'qap']``.
    z : float, default=None
        Effective redshift.
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:
        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`;
        - :class:`cosmoprimo.Cosmology`: Cosmology instance.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs: dict
        Optional arguments for :class:`StandardPowerSpectrumExtractor`, e.g., ``r`` (smoothing scale for ``fsigmar``).
    """
    def initialize(self, *args, name='standard', **kwargs):
        super(StandardCompressionObservable, self).initialize(*args, extractor=StandardPowerSpectrumExtractor(), name=name, **kwargs)


class ShapeFitCompressionObservable(BaseCompressionObservable):
    """
    ShapeFit observable: compare ShapeFit measurements to theory predictions.

    Reference
    ---------
    https://arxiv.org/abs/2106.07641

    Parameters
    ----------
    data : array, lsstypes.ObservableTree, default=None
        BAO parameters. Either:
        - flat array (of all parameters). Additionally provide list of parameters;
        - :class:`lsstypes.ObservableTree` (contains all necessary information);
        - ``None``. Set to 0.
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    parameters : list, tuple
        Parameters; choose from ``['m', 'n', 'f_sqrt_Ap', 'dm', 'dn', 'df', 'DM_over_rd', 'DH_over_rd', 'DV_over_rd', 'DM_over_DH', 'DV_over_rd', 'qpar', 'qper', 'qiso', 'qap']``.
    z : float, default=None
        Effective redshift.
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:
        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`;
        - :class:`cosmoprimo.Cosmology`: Cosmology instance.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs: dict
        Optional arguments for :class:`ShapeFitPowerSpectrumExtractor`, e.g., ``kp``, ``a``.
    """
    def initialize(self, *args, name='shapefit', **kwargs):
        super(ShapeFitCompressionObservable, self).initialize(*args, extractor=ShapeFitPowerSpectrumExtractor(), name=name, **kwargs)
        self.extractor.init.update(n_varied='n' in self.parameters)


class WiggleSplitCompressionObservable(BaseCompressionObservable):
    """
    Wiggle-split observable: compare the amplitude and tilt of the velocity power spectrum, and BAO position to theory predictions.

    Parameters
    ----------
    data : array, lsstypes.ObservableTree, default=None
        BAO parameters. Either:
        - flat array (of all parameters). Additionally provide list of parameters;
        - :class:`lsstypes.ObservableTree` (contains all necessary information);
        - ``None``. Set to 0.
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    parameters : list, tuple
        Parameters; choose from ``['df', 'dm', 'qap', 'qbao']``.
    z : float, default=None
        Effective redshift.
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:
        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`;
        - :class:`cosmoprimo.Cosmology`: Cosmology instance.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs: dict
        Optional arguments for :class:`WiggleSplitPowerSpectrumExtractor`.
    """
    def initialize(self, *args, name='wigglesplit', **kwargs):
        super(WiggleSplitCompressionObservable, self).initialize(*args, extractor=WiggleSplitPowerSpectrumExtractor(), name=name, **kwargs)


class BandVelocityCompressionObservable(BaseCompressionObservable):
    """
    Band velocity power observable: compare band power measurements to theory predictions.

    Parameters
    ----------
    data : array, lsstypes.ObservableTree, default=None
        BAO parameters. Either:
        - flat array (of all parameters). Additionally provide list of parameters;
        - :class:`lsstypes.ObservableTree` (contains all necessary information);
        - ``None``. Set to 0.
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    parameters : list, tuple
        Parameters; choose from ``['dptt', 'df', 'qap']``.
    z : float, default=None
        Effective redshift.
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:
        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`;
        - :class:`cosmoprimo.Cosmology`: Cosmology instance.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs : dict
        Optional arguments for :class:`BandVelocityPowerSpectrumExtractor`.
    """
    def initialize(self, *args, name='bandvelocity', **kwargs):
        super(BandVelocityCompressionObservable, self).initialize(*args, extractor=BandVelocityPowerSpectrumExtractor(), name=name, **kwargs)


class TurnOverCompressionObservable(BaseCompressionObservable):
    r"""
    Turn over observable: compare (compressed) turn over measurement
    (in terms of product of distance with the turn over scale :math:`k_{\mathrm{TO}}`) to theory predictions.

    Reference
    ---------
    https://arxiv.org/abs/2505.16153

    Parameters
    ----------
    data : array, lsstypes.ObservableTree, default=None
        BAO parameters. Either:
        - flat array (of all parameters). Additionally provide list of parameters;
        - :class:`lsstypes.ObservableTree` (contains all necessary information);
        - ``None``. Set to 0.
    covariance : array, lsstypes.CovarianceMatrix, default=None
        Covariance matrix. Either:
        - 2D array, of shape ``(len(data), len(data))``;
        - :class:`lsstypes.CovarianceMatrix` (contains all necessary information);
        - ``None``. Pass covariance to the likelihood.
    cosmo : BasePrimordialCosmology, default=None
        Cosmology calculator. Defaults to ``Cosmoprimo(fiducial=fiducial)``.
    parameters : list, tuple
        Parameters; choose from ``['DV_times_kTO', 'DM_over_DH', 'qto', 'qap']``.
    z : float, default=None
        Effective redshift.
    fiducial : str, tuple, dict, cosmoprimo.Cosmology, default='DESI'
        Specifications for fiducial cosmology. Either:
        - str: name of fiducial cosmology in :class:`cosmoprimo.fiducial`;
        - :class:`cosmoprimo.Cosmology`: Cosmology instance.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs: dict
        Optional arguments for :class:`TurnOverPowerSpectrumExtractor`.
    """
    def initialize(self, *args, name='turnover', **kwargs):
        super(TurnOverCompressionObservable, self).initialize(*args, extractor=TurnOverPowerSpectrumExtractor(), name=name, **kwargs)