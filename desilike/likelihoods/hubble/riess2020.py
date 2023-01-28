from .base import H0Likelihood, MbLikelihood


class Riess2020H0Likelihood(H0Likelihood):
    r"""
    Local :math:`H_{0}` measurement from Riess et al. 2020.
    
    Reference
    ---------
    https://arxiv.org/abs/2012.08534
    """
    config_fn = 'riess2020.yaml'


class Riess2020MbLikelihood(MbLikelihood):
    """
    Magnitude measurement from Riess et al. 2020.
    
    Reference
    ---------
    https://arxiv.org/abs/2012.08534
    """
    config_fn = 'riess2020.yaml'
