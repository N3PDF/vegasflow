# Installation script for python
from setuptools import setup, find_packages
import os
import re

requirements = ['joblib', 'numpy']
if os.environ.get('READTHEDOCS') == 'True':
    requirements.append('tensorflow')

PACKAGE = 'vegasflow'

def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join('src', PACKAGE, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)

setup(name='vegasflow',
      version=get_version(),
      description='Monte Carlo integration with Tensorflow',
      author = 'S.Carrazza, J.Cruz-Martinez',
      author_email='stefano.carrazza@cern.ch, juan.cruz@mi.infn.it',
      url='https://github.com/N3PDF/VegasFlow',
      package_dir={'':'src'},
      packages=find_packages('src'),
      zip_safe=False,
      classifiers=[
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      install_requires= requirements,
      extras_require={
          'docs' : [
            'sphinx_rtd_theme',
            'recommonmark',
            'sphinxcontrib-bibtex',
            ],
          'examples' : [
            'cffi',
            ],
          'benchmark' : [
            'vegas', # Lepage's Vegas for benchmarking
            ],
          },
      python_requires='>=3.6,<3.8',
      long_description='See readthedocs webpage with the documentation'
)
