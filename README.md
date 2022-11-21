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

Convenient built-in installer to install required dependencies (e.g. theory codes), depending on the likelihood.
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
  - mpi4py (should be optional in the future)
  - pyyaml
  - cosmoprimo

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/cosmodesi/desilike
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
