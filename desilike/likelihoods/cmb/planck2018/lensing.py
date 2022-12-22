from .base import BasePlanck2018ClikLikelihood


class LensingPlanck2018ClikLikelihood(BasePlanck2018ClikLikelihood):
    r"""Lensing likelihood of Planck's 2018 data release based on temperature+polarization map-based lensing reconstruction."""
    config_fn = 'lensing.yaml'
