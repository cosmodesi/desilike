from desilike.theories.lya.power_template import P1DPowerSpectrumExtractor
from desilike.observables.galaxy_clustering.compression import BAOCompressionObservable, BaseCompressionObservable


class P1DCompressionObservable(BaseCompressionObservable):
    """
    P1D compression observable: compare P1D compressed measurements to theory predictions.

    Parameters
    ----------
    data : str, Path, array, Profiles, Chain
        P1D compressed parameters. If array, provide corresponding ``quantities``.
        Else, chain, profiles or path to such objects.

    covariance : str, Path, 2D array, Profiles, Chain, ParameterCovariance
        Covariance for P1D compressed parameters. If 2D array, provide corresponding ``quantities``.
        Else, chain, profiles, covariance or path to such objects.

    quantities : list, tuple
        Quantities to take from ``data`` and ``covariance``:
        chose from ``['delta2star', 'nstar', 'alphastar']``.

    **kwargs : dict
        Optional arguments for :class:`P1DPowerSpectrumExtractor`, e.g. ``z``, ``qstar``.


    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    """
    def initialize(self, *args, **kwargs):
        super(P1DCompressionObservable, self).initialize(*args, extractor=P1DPowerSpectrumExtractor(), **kwargs)
