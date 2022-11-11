# desilike

WARNING: this is ongoing work!

**desilike** is an attempt to provide a common framework for writing DESI likelihoods,
that can be imported in common cosmological inference codes (Cobaya, CosmoSIS, MontePython).

Example notebooks presenting most use cases are provided in directory nb/.

## Documentation

Documentation is hosted on Read the Docs, [desilike docs](https://desilike.readthedocs.io/).

## Requirements

Only strict requirements are:

  - numpy
  - scipy

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/adematti/desilike
```

### git

First:
```
git clone https://github.com/adematti/desilike.git
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

**desilike** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/adematti/desilike/blob/main/LICENSE).
