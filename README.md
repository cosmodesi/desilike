# desilike

WARNING: this is ongoing work!

**desilike** is an attempt to provide a common framework for writing DESI likelihoods,
that can be imported in common cosmological inference codes (Cobaya, CosmoSIS, MontePython).

Stricly required, to import likelihoods in these inference codes, are:

  - theories
  - observables
  - likelihoods
  - bindings
  - (optionally: emulators)

Directories samples, samplers and profilers are provided for self-contained tests of provided likelihoods.
Example notebooks presenting most use cases are provided in directory nb/.

## TODO

.yaml or not .yaml file? e.g. parameter specifications could be given in docstrings, and automatically read out at runtime
(but this will make very long docstrings...)
Special treatment for sum of independent likelihoods?
Add curve_fit-type profiler.

## Documentation

Documentation WILL BE hosted on Read the Docs, [desilike docs](https://desilike.readthedocs.io/).

## Requirements

Only strict requirements are:

  - numpy
  - scipy
  - pyyaml
  - mpi4py
  - cosmoprimo (currently with pyclass to compute DESI fiducial cosmology)

Should be made optional in the future:
  - mpi4py
  - pyclass (by extending TabulatedDESI to power spectra)

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/cosmodesi/desilike
```
If you wish to use plotting routines (getdist, anesthetic), and tabulate for pretty tables:
```
python -m pip install git+https://github.com/cosmodesi/desilike#egg=desilike[plotting]
```
If you addtionally wish to be able to use analytic marginalization with jax:
```
python -m pip install git+https://github.com/cosmodesi/desilike#egg=desilike[plotting,jax]
```

### git

First:
```
git clone https://github.com/cosmodesi/desilike.git
```
To install the code:
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately):
```
python setup.py develop --user
```

## Other dependencies (theory codes, etc.)

Just define your calculator (most commonly your likelihood), then in a python script:
```
from desilike import Installer
Installer(user=True)(likelihood)
```
## License

**desilike** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/cosmodesi/desilike/blob/main/LICENSE).


## Acknowledgments

- Stephen Chen, Mark Maus and Martin White for wrappers for velocileptors: https://github.com/sfschen/velocileptors, https://github.com/martinjameswhite/CobayaLSS
- Cullan Howlett, Yan Xiang Lai for wrapper for pybird: https://github.com/pierrexyz/pybird, https://github.com/CullanHowlett/pybird
- Samuel Brieden, Hector Gil-Marin for ShapeFit
- Stephen Chen and Mark Maus for Taylor expansion emulator: https://github.com/sfschen/velocileptors_shapefit
- Stephen Chen and Joe DeRose for MLP emulator: https://github.com/sfschen/EmulateLSS
- Cobaya, CosmoSIS bindings inspired by firecrown: https://github.com/LSSTDESC/firecrown
- Inspiration from cobaya: https://github.com/CobayaSampler/cobaya
