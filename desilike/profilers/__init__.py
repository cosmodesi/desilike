"""Module providing common interfaces for likelihood/posterior profiling."""

from . import optimize
from .profiler import Profiler

__all__ = ['Profiler', 'optimize']
