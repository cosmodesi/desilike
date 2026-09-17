"""The full-shape dilation s-powers, keyed by name rather than by position.

Pins the mapping against the historical positional tuple, so the rewrite stays traceable, and
pins the ONE slot where the current powers deliberately depart from it (``X_FoG``, see below).
The powers are measured numbers and this keeps them honest.
"""

import pytest

from desilike.theories.galaxy_clustering.full_shape import FOLPSDEmulator as Scaled


# the tuple this replaced, in folps' documented order
HISTORICAL = (0, 0, 0, 0, 2, 2, 2, 0, 3, 5, 0, 0)

CURRENT = {'alpha0': 2, 'alpha2': 2, 'alpha4': 2, 'alphashot0': 3, 'alphashot2': 5, 'X_FoG': 1}


def test_the_positional_tuple_is_reproduced_except_for_X_FoG():
    """Every historical slot is kept but one: ``X_FoG`` was 0 and is now 1.

    It is not a rewrite artefact, it is a correction -- see
    :attr:`FOLPSDEmulator._nuisance_scale_powers`. Without it the emulated power spectrum
    departs from the exact one by 2.15 sigma_data rms at the top of the ACE ``h`` box.
    """
    powers = tuple(Scaled._nuisance_scale_powers.get(name, 0)
                   for name in Scaled._nuisance_names)
    differ = [name for name, new, old in zip(Scaled._nuisance_names, powers, HISTORICAL)
              if new != old]
    assert differ == ['X_FoG']
    assert Scaled._nuisance_scale_powers['X_FoG'] == 1


def test_the_measured_channels_are_the_ones_rescaled():
    """alpha0/2/4 pick up s^2, the constant shot s^3, the k^2 shot s^5, X_FoG s^1.

    ``ctilde`` is the only counterterm that is exactly invariant: it multiplies
    ``(k mu f0)^4 sigma2w^2``, and ``sigma2w`` is a table column that already carries the
    dilation. ``X_FoG`` multiplies ``k`` alone and so does not share that invariance.
    The rescaled channels anticorrelate, so a partial set is worse than none.
    """
    powers = Scaled._nuisance_scale_powers
    assert powers == CURRENT
    for invariant in ('b1', 'b2', 'bs2', 'b3nl', 'ctilde', 'PshotP'):
        assert powers.get(invariant, 0) == 0


def test_every_named_power_is_a_real_folps_parameter():
    """A typo in the dict would silently rescale nothing, which is the failure the names are
    meant to prevent."""
    unknown = set(Scaled._nuisance_scale_powers) - set(Scaled._nuisance_names)
    assert not unknown, f'{unknown} are not folps nuisance parameters'


def test_the_bispectrum_arm_preconditions_in_h_too():
    """``FOLPSD3PolesEmulator`` used to inherit ``precondition = ()``.

    That is what forced ``levels={'h': 4}`` on the whole pipeline: its ``pk_l`` row was fitted on
    a grid fixed in h/Mpc while the BAO wiggles slid through it, so it reached its floor at
    exactly 17 nodes (2.8e-03 at 5, 1.5e-04 at 9, 6.7e-06 at 17) while its no-wiggle twin was
    already there with 9. Measured on the bispectrum multipoles themselves, over
    h in [0.50041, 0.89957] at z = 0.8, median / max ``|dB/B|`` against the exact pipeline:
    1.2e-03 / 3.2e-02 at 5 nodes without it, 5.8e-06 / 4.3e-05 with.

    Unlike the power spectrum's, it is undone by resampling the rows FORWARD rather than by
    handing the assembly dilated ``q``'s -- ``folps.sigmas`` is called per evaluation on these
    rows and cuts at a fixed ``kT <= 0.4`` with a fixed ``k_BAO = 1/104``, so it is not
    homogeneous and would return the wrong damping in a reference frame. A consequence worth
    keeping: no bispectrum nuisance parameter needs an s-power, because the assembly never sees
    the reference frame.
    """
    from desilike.theories.galaxy_clustering.full_shape import FOLPSD3PolesEmulator

    assert FOLPSD3PolesEmulator.precondition == ('h',)

    # the two anchors the preconditioning reads have to travel as children of the pt, or
    # `transform` reads a KeyError at the first node
    probe = object.__new__(FOLPSD3PolesEmulator)
    probe.set_children_leafnames()
    for name in ('h_anchor', 'sigma_mpc_anchor'):
        assert name in probe.children_leafnames, probe.children_leafnames
