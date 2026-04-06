.. _developer-documentation:

#############
Documentation
#############

There are three components to `desilike`'s documentation: docstrings, Sphinx documentation, and Jupyter notebooks.

For docstrings, `desilike` follows the `NumPy docstring convention <https://numpydoc.readthedocs.io>`_. Please ensure that all code contributions are accompanied by complete and error-free docstrings. Tools like `ruff` or `darglint` can help check docstring completeness.

The Sphinx documentation for `desilike` is built using `Sphinx` and hosted on Read the Docs. Please follow the `Sphinx style guide <https://documentation-style-guide-sphinx.readthedocs.io>`_ when writing documentation. The documentation is stored in the :root:`docs` folder.

The documentation can be built by running the following commands in a terminal:

.. code-block:: bash

   cd /path/to/desilike/doc
   make html

Note that you may have to install dependencies listed in :root:`docs/requirements.txt` first. After building, the HTML pages can be accessed by opening ``_build/html/index.html`` with your web browser.

Finally, `desilike` also includes example notebooks stored in the :root:`nb` folder. Please consider contributing example notebooks when adding new functionality to `desilike`.

