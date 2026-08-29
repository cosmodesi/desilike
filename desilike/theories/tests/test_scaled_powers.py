"""The full-shape dilation s-powers, keyed by name rather than by position.

Pins the mapping against the historical positional tuple, so the rewrite is provably equivalent.
The powers have no consumer today -- the `h` dilation routing is the one piece of the legacy
protocol not ported -- but they are measured numbers, and this keeps them honest for whoever
ports it.
"""

import pytest

from desilike.theories.galaxy_clustering.full_shape import FOLPSDEmulator as Scaled


# the tuple this replaced, in folps' documented order
HISTORICAL = (0, 0, 0, 0, 2, 2, 2, 0, 3, 5, 0, 0)


def test_name_keyed_powers_reproduce_the_positional_tuple():
    powers = tuple(Scaled._nuisance_scale_powers.get(name, 0)
                   for name in Scaled._nuisance_names)
    assert powers == HISTORICAL


def test_the_measured_channels_are_the_ones_rescaled():
    """alpha0/2/4 pick up s^2, the constant shot s^3, the k^2 shot s^5. ctilde and the FoG
    damping are exactly invariant and must not be rescaled -- the channels anticorrelate, so a
    partial rescaling is worse than none."""
    powers = Scaled._nuisance_scale_powers
    assert powers == {'alpha0': 2, 'alpha2': 2, 'alpha4': 2, 'alphashot0': 3, 'alphashot2': 5}
    for invariant in ('b1', 'b2', 'bs2', 'b3nl', 'ctilde', 'PshotP', 'X_FoG'):
        assert powers.get(invariant, 0) == 0


def test_every_named_power_is_a_real_folps_parameter():
    """A typo in the dict would silently rescale nothing, which is the failure the names are
    meant to prevent."""
    unknown = set(Scaled._nuisance_scale_powers) - set(Scaled._nuisance_names)
    assert not unknown, f'{unknown} are not folps nuisance parameters'
