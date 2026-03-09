from desilike.theories.lya.power_template import P1DPowerSpectrumExtractor
from desilike.observables.galaxy_clustering.compression import BAOCompressionObservable, BaseCompressionObservable


class P1DCompressionObservable(BaseCompressionObservable):
    """
    P1D compression observable: compare P1D compressed measurements to theory predictions.

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
        Parameters; choose from ``['delta2star', 'nstar', 'alphastar']``.
    z : float, default=None
        Effective redshift.
    name : str, optional
        Observable name. Used to match covariance matrix when creating likelihood of multiple observables.
        See :class:`ObservablesGaussianLikelihood`.
    **kwargs: dict
        Optional arguments for :class:`P1DPowerSpectrumExtractor`, e.g. ``z``, ``qstar``.
    """
    def initialize(self, *args, name='p1d', **kwargs):
        super(P1DCompressionObservable, self).initialize(*args, extractor=P1DPowerSpectrumExtractor(), name=name, **kwargs)
