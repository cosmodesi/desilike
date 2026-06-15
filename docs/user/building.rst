.. _user-building:

Building
========

Requirements
------------
Only strict requirements are:

  - numpy
  - scipy
  - pyyaml
  - mpi4py
  - jax
  - cosmoprimo (currently with pyclass to compute DESI fiducial cosmology)

Extra requirements are:

  * plotting: tabulate for nice tables; getdist, anesthetic to make nice contour plots

pip
---
To install **desilike**, simply run::

  python -m pip install git+https://github.com/cosmodesi/desilike

If you want to install extra requirements, run::

  python -m pip install git+https://github.com/cosmodesi/desilike#egg=desilike[plotting]

git
---

First::

  git clone https://github.com/cosmodesi/desilike.git

To install the code::

  pip install --user

Or in development mode (any change to Python code will take place immediately)::

  pip install --user -e .

You may want to avoid installing dependencies in your local $HOME (in particular if you load the cosmodesi environment)::

  pip install --no-deps --user -e .

Pipeline dependencies, samplers, profilers, emulators
-----------------------------------------------------
Requirements for samplers, profilers, emulators can be installed within Python,
with the provided installer.

Any calculator, profiler, sampler, etc. can be installed with :class:`~desilike.install.Installer`.
