# Installation script for python
from setuptools import setup, find_packages

setup(name='vegasflow',
      version='0.1',
      description='Monte Carlo integration with Tensorflow',
      author = 'S.Carrazza, J.Cruz-Martinez',
      author_email='stefano.carrazza@cern.ch',
      url='https://github.com/N3PDF/VegasFlow',
      package_dir={'': 'vegasflow'},
      packages=find_packages('vegasflow'),
      zip_safe=False,
      classifiers=[
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      install_requires=[
          'numpy',
          'tensorflow',
          'cffi',
          'sphinx_rtd_theme',
          'recommonmark',
          'sphinxcontrib-bibtex',
          'vegas', # Lepage's Vegas for benchmarking
      ],
      setup_requires=[
          "cffi>1.0.0"
          ],
      python_requires='>=3.6'
)
