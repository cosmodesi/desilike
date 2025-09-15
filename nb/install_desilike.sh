#Note: First run the 'environment.yaml' file

#!/usr/bin/env bash
set -euo pipefail

#activate environment (assumes conda)
conda activate desilike_env

#upgrade pip and wheel
python -m pip install --upgrade pip setuptools wheel

#1) desilike (with extras for plotting and jax if desired)
python -m pip install git+https://github.com/cosmodesi/desilike#egg=desilike[plotting,jax]

#2) cosmoprimo with extras (class/camb/astropy/pyfftw if needed)
python -m pip install git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[class,camb,astropy,extras]

#3) pypower (official repository)
python -m pip install git+https://github.com/cosmodesi/pypower

#4) pyclass (compiles CLASS during installation)
python -m pip install git+https://github.com/adematti/pyclass

#5) useful packages that sometimes are not included in extras
python -m pip install getdist anesthetic tabulate

#6) fallback for zeus-mcmc (if conda install fails, pip will be used)
python -m pip install zeus-mcmc

#If everything worked you should see:
echo "Installation completed (if no errors above)."
