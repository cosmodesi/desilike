from .base import BasePlanck2018ClikLikelihood


class TTLowlPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
    r"""Low-$\ell$ temperature-only \textsc{plik} likelihood of Planck's 2018 data release."""
    config_fn = 'lowl.yaml'


class EELowlPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
    r"""Low-$\ell$ temperature and polarization \textsc{plik} likelihood of Planck's 2018 data release."""
    config_fn = 'lowl.yaml'
