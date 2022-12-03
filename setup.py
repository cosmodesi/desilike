import os
import sys
from setuptools import setup


package_basename = 'desilike'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='cosmodesi',
      author_email='',
      description='Package for DESI likelihoods',
      license='BSD3',
      url='http://github.com/cosmodesi/desilike',
      install_requires=['numpy', 'scipy', 'pyyaml', 'cosmoprimo @ git+https://github.com/cosmodesi/cosmoprimo'],
      extras_require={'plotting': ['tabulate', 'getdist', 'anesthetic'], 'jax': ['jax[cpu]']},
      packages=[package_basename])
