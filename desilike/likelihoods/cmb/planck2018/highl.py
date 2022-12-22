from .base import BasePlanck2018ClikLikelihood


class TTHighlPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
    r"""High-$\ell$ temperature-only \textsc{plik} likelihood of Planck's 2018 data release."""
    config_fn = 'highl.yaml'


class TTTEEEHighlPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
    r"""High-$\ell$ temperature and polarization \textsc{plik} likelihood of Planck's 2018 data release."""
    config_fn = 'highl.yaml'
