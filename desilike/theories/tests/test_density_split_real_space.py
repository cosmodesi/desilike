"""Focused checks for real-space density-split power-spectrum predictions."""

import numpy as np


def test_density_split_real_space_tree():
    from desilike.theories.galaxy_clustering import (
        DensitySplitTracerPowerSpectrumMultipoles,
        FixedPowerSpectrumTemplate,
    )

    k = np.array([0.02, 0.05])
    template = FixedPowerSpectrumTemplate(z=0.5, fiducial="DESI")
    tree = DensitySplitTracerPowerSpectrumMultipoles(
        k=k,
        ells=(0, 2, 4),
        quantiles=(5,),
        template=template,
        model="tree",
        prior_basis="standard",
        rsd=False,
    )
    c1 = 0.75
    power = np.asarray(tree(b1=1.0, c1q5=c1, s0qg5=0.0))
    linear = np.asarray(tree._linear_matter_pk())[:, 0]
    expected = c1 * np.exp(-0.5 * (k * tree.smoothing_radius) ** 2) * linear
    assert np.allclose(power[0, 0], expected)
    assert np.allclose(power[0, 1:], 0.0, atol=1e-6, rtol=0.0)
    assert tree.pt.fsigma8 > 0.0
    assert tree.pt.template.f > 0.0


def test_density_split_rsd_default_is_unchanged():
    from desilike.theories.galaxy_clustering import (
        DensitySplitTracerPowerSpectrumMultipoles,
        FixedPowerSpectrumTemplate,
    )

    k = np.array([0.02, 0.05])
    template = FixedPowerSpectrumTemplate(z=0.5, fiducial="DESI")
    explicit = DensitySplitTracerPowerSpectrumMultipoles(
        k=k,
        ells=(0, 2),
        quantiles=(1, 5),
        template=template,
        model="tree",
        rsd=True,
    )
    default = DensitySplitTracerPowerSpectrumMultipoles(
        k=k,
        ells=(0, 2),
        quantiles=(1, 5),
        template=template,
        model="tree",
    )
    assert np.allclose(explicit(), default())


def test_density_split_real_space_one_loop():
    from desilike import jax as desilike_jax
    from desilike.theories.galaxy_clustering import (
        DensitySplitTracerPowerSpectrumMultipoles,
        FixedPowerSpectrumTemplate,
    )

    if desilike_jax.jax is None:
        return
    theory = DensitySplitTracerPowerSpectrumMultipoles(
        k=np.array([0.02, 0.05]),
        ells=(0, 2, 4),
        quantiles=(1, 5),
        template=FixedPowerSpectrumTemplate(z=0.5, fiducial="DESI"),
        model="1-loop",
        prior_basis="standard",
        rsd=False,
        composite_loop_nq=4,
        composite_loop_nx=4,
        composite_loop_nphi=4,
    )
    power = np.asarray(theory())
    assert np.isfinite(power).all()
    assert np.allclose(power[:, 1:], 0.0, atol=1e-6, rtol=0.0)
    assert theory.pt.fsigma8 > 0.0
    assert theory.pt.template.f > 0.0
