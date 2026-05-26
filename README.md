# desilike

[![Unit Testing Status](https://img.shields.io/github/actions/workflow/status/cosmodesi/desilike/tests.yml?branch=main&label=tests)](https://github.com/cosmodesi/desilike/actions)
[![Documentation Status](https://img.shields.io/readthedocs/desilike)](https://desilike.readthedocs.io)
[![Code Coverage](https://img.shields.io/coverallsCoverage/github/cosmodesi/desilike)](https://coveralls.io/github/cosmodesi/desilike?branch=main)
[![License: MIT](https://img.shields.io/github/license/cosmodesi/desilike)](https://raw.githubusercontent.com/cosmodesi/desilike/main/LICENSE)

``desilike`` is the cosmological inference framework used by the Dark Energy Spectroscopic Instrument (DESI) collaboration. It enables writing likelihoods for the analysis of baryon acoustic oscillations (BAOs) and redshift-space distortions (RSDs) and to import them in commonly used cosmological inference codes (``Cobaya``, ``CosmoSIS``, ``MontePython``). Additionally, ``desilike`` provides support for Bayesian samplers and profile-likelihood analysis out of the box.

## Documentation

The documentation is hosted on Read the Docs: https://desilike.readthedocs.io/. Additionally, example notebooks presenting most use cases are provided in directory ``nb``.

## Installation

The latest development version of ``desilike`` can be installed with ``pip`` via:

```
python -m pip install git+https://github.com/cosmodesi/desilike
```

This will install all required dependencies. You can install optional dependencies such as ``jax`` or ``getdist`` as needed via ``pip``.

## Contributing

We welcome contributions from all members of the DESI collaboration. Details are describe in [CONTRIBUTING.md](https://github.com/cosmodesi/desilike/blob/main/CONTRIBUTING.md).

## License

``desilike`` is free software distributed under a 3-Clause BSD license. For details, see the [LICENSE](https://github.com/cosmodesi/desilike/blob/main/LICENSE).


## Acknowledgments

- Stephen Chen, Mark Maus, Martin White for velocileptors wrapper: https://github.com/sfschen/velocileptors, https://github.com/martinjameswhite/CobayaLSS
- Pierre Zhang, Cullan Howlett, Yan Xiang Lai for pybird wrapper: https://github.com/pierrexyz/pybird, https://github.com/CullanHowlett/pybird
- Hernan E. Noriega, Alejandro Aviles for folps wrapper: https://github.com/henoriega/FOLPS-nu
- Samuel Brieden, Hector Gil-Marin, Mark Maus for ShapeFit: https://arxiv.org/abs/2106.07641
- Stephen Chen, Mark Maus for Taylor expansion emulator: https://github.com/sfschen/velocileptors_shapefit
- Stephen Chen, Joe DeRose for MLP emulator: https://github.com/sfschen/EmulateLSS
- Pat McDonald, Eva Maria Mueller, Antony Lewis for helpful discussions
- Pat McDonald, Edmond Chaussidon, Uendert Andrade, Daniel Forero Sanchez, Batia Friedman-Shaw, Svyatoslav Trusov, Nathan Findlay, Enrique Paillas, Vincenzo Aronica for early debugging and feedback
- Ruiyang Zhao for systematics templates
- Benedict Bahr-Kalus for turnover scale analysis: https://arxiv.org/pdf/2302.07484.pdf
- Rodrigo Calderón for Pantheon+ with/out SH0ES and Union3 likelihoods
- Cobaya, CosmoSIS bindings inspired by firecrown: https://github.com/LSSTDESC/firecrown
- Inspiration from Cobaya: https://github.com/CobayaSampler/cobaya
