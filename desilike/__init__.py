"""A Common Framework for Writing DESI Likelihoods."""

from .utils import setup_logging
from .base import BaseCalculator, PipelineError, vmap
from .parameter import Parameter, ParameterPrior, ParameterCollection, ParameterArray, Samples, ParameterCovariance, ParameterPrecision
from .differentiation import Differentiation
from .fisher import Fisher, LikelihoodFisher, FisherGaussianLikelihood


from .install import Installer
Installer().setenv()

__version__ = "0.0.1"
