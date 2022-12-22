from .base import H0Likelihood, MbLikelihood


class Riess2020H0Likelihood(H0Likelihood):

    r"""Local $H_{0}$ measurement from Riess 2020."""

    config_fn = 'riess2020.yaml'


class Riess2020MbLikelihood(MbLikelihood):

    """Magnitude measurement from Riess 2020."""

    config_fn = 'riess2020.yaml'
