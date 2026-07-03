"""
Observables for galaxy clustering.

Classes
-------
Spectrum2PolesObservable
    Power spectrum multipoles observable: applies window matrix to theory, compares to data.
Correlation2PolesObservable
    Correlation function multipoles observable: applies window matrix to theory, compares to data.
Spectrum3PolesObservable
    Bispectrum multipoles observable: applies window matrix to theory, compares to data.
BAOCompressionObservable
    Compare BAO distance measurements to :class:`~desilike.theories.galaxy_clustering.template.BAOTheory` predictions.
BAOPhaseShiftCompressionObservable
    Compare BAO + N_eff phase-shift measurements to :class:`~desilike.theories.galaxy_clustering.template.BAOPhaseShiftTheory` predictions.
TurnOverCompressionObservable
    Compare turn-over measurements to :class:`~desilike.theories.galaxy_clustering.template.TurnOverTheory` predictions.
"""

from .stats import Spectrum2PolesObservable, Correlation2PolesObservable, Spectrum3PolesObservable
from .compressed import (BAOCompressionObservable, BAOPhaseShiftCompressionObservable, TurnOverCompressionObservable)
