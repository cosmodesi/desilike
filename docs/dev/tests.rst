.. _developer-tests:

##########
Unit Tests
##########

`desilike` uses `pytest` for automated unit testing. Every code change triggers tests on GitHub to ensure the library continues to work correctly. 

When contributing to `desilike`, it is critical that changes do not break existing tests. If you add new functionality, please include unit tests that verify it works as intended. Contributions that are untested or cause test failures will generally not be accepted.

Tests are located in the :root:`/tests` directory. To run the tests locally, execute:

.. code-block:: bash

  pytest

