import importlib
import pytest


def test_galaxy_clustering_imports():
    """
    Catch broken exports in desilike.theories.galaxy_clustering.__init__,
    such as trying to import names that are no longer defined in full_shape.py.
    """
    mod = importlib.import_module("desilike.theories.galaxy_clustering")

    assert hasattr(mod, "DirectPowerSpectrumTemplate")
    assert hasattr(mod, "fkptjaxTracerPowerSpectrumMultipoles")


def test_full_shape_imports():
    """
    Catch syntax errors or broken imports inside full_shape.py itself.
    """
    mod = importlib.import_module("desilike.theories.galaxy_clustering.full_shape")
    assert mod is not None


def test_cosmoprimo_wrapper_initializes():
    """
    Catch failures in desilike.theories.primordial_cosmology.Cosmoprimo
    before they appear later as PipelineError inside a full likelihood run.
    """
    from desilike.theories.primordial_cosmology import Cosmoprimo

    cosmo = Cosmoprimo()
    cosmo.runtime_info.initialize()

    assert cosmo.runtime_info.initialized


def test_direct_power_spectrum_template_constructs():
    """
    Basic construction test for the template object used in your pipeline.
    This catches missing dependencies and theory-stack regressions early.
    """
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
    from desilike.theories.primordial_cosmology import Cosmoprimo

    cosmo = Cosmoprimo()
    template = DirectPowerSpectrumTemplate(z=0.8, cosmo=cosmo)

    assert template is not None


@pytest.mark.parametrize("module_name", [
    "desilike.theories.galaxy_clustering",
    "desilike.theories.galaxy_clustering.full_shape",
    "desilike.theories.primordial_cosmology",
])
def test_core_modules_import(module_name):
    """
    Generic smoke test for critical desilike modules.
    """
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_optional_fkptjax_api_if_present():
    """
    If fkptjax is installed, verify the API expected by this branch.
    This is the exact class of failure you were seeing.
    """
    spec = importlib.util.find_spec("fkptjax")
    if spec is None:
        pytest.skip("fkptjax not installed")

    import fkptjax

    # Public API that should exist for the current public package
    assert importlib.util.find_spec("fkptjax.calculate_jax") is not None
    assert importlib.util.find_spec("fkptjax.util") is not None

    # Optional branch-specific API: only enforce if your branch expects it.
    expects_ode = False
    try:
        import desilike.theories.galaxy_clustering.full_shape as fs
        source = open(fs.__file__, "r").read()
        expects_ode = "from fkptjax.ode import" in source
    except Exception:
        pass

    if expects_ode:
        assert importlib.util.find_spec("fkptjax.ode") is not None, (
            "This desilike branch expects fkptjax.ode, but installed fkptjax "
            "does not provide it."
        )
