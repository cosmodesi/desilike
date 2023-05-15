# desilike

**desilike** is an attempt to provide a common framework for writing DESI likelihoods,
that can be imported in common cosmological inference codes (Cobaya, CosmoSIS, MontePython).

**desilike** has the following structure:

  - root directory: definition of parameters, base calculator classes, differentiation and Fisher routines, installation routines
  - theories: e.g. BAO, full-shape theory models
  - observables: e.g. power spectrum, correlation function
  - likelihoods: e.g. Gaussian likelihood of observables, a few external likelihoods (Pantheon, Planck)
  - bindings: automatic linkage with cobaya, cosmosis, montepython
  - emulators: emulate e.g. full-shape theory models, to speed up inference
  - samples: define chains, profiles data structures and plotting routines
  - samplers: many samplers for posterior sampling
  - profilers: profilers for posterior profiling

samples, samplers and profilers are provided for self-contained sampling / profiling of provided likelihoods.
Example notebooks presenting most use cases are provided in directory nb/.

## TODO

When autodiff is available, pass gradient to profilers / samplers whenever relavant.

## Documentation

Documentation in construction on Read the Docs, [desilike docs](https://desilike.readthedocs.io/).

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
- Pat McDonald, Eva Maria Mueller, Antony Lewis for thoughts
- Pat McDonald, Edmond Chaussidon, Uendert Andrade for early debugging
- Cobaya, CosmoSIS bindings inspired by firecrown: https://github.com/LSSTDESC/firecrown
- Inspiration from cobaya: https://github.com/CobayaSampler/cobaya