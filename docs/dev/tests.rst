.. _developer-tests:

##########
Unit Tests
##########

`desilike` uses `pytest` for automated unit testing. Every code change triggers tests on GitHub to ensure the library continues to work correctly. 

When contributing to `desilike`, it is critical that changes do not break existing tests. If you add new functionality, please include unit tests that verify it works as intended. Contributions that are untested or cause test failures will generally not be accepted.

Tests are organized per-module under ``desilike/{module}/tests/`` (e.g.
``desilike/samplers/tests/``, ``desilike/profilers/tests/``, etc.).
``pyproject.toml`` sets ``testpaths = ["desilike"]``, so running pytest from
the repository root discovers all tests automatically:

.. code-block:: bash

  pytest

To run only a specific module's tests:

.. code-block:: bash

  pytest desilike/samplers/tests/

