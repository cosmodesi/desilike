"""Hubble parameter likelihoods from Riess et al. (2020)."""

from .base import H0Likelihood, MbLikelihood


class Riess2020H0Likelihood(H0Likelihood):
    r"""Local :math:`H_0` measurement from Riess et al. (2020).

    Reference
    ---------
    https://arxiv.org/abs/2012.08534

    """

    config_fn = "riess2020.yaml"
    name = "Riess2020H0"


class Riess2020MbLikelihood(MbLikelihood):
    """Magnitude measurement from Riess et al. (2020).

    Reference
    ---------
    https://arxiv.org/abs/2012.08534

    https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/H0/riess2020Mb.yaml

    """

    config_fn = "riess2020.yaml"
    name = "Riess2020Mb"
