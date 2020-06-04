[![Tests](https://github.com/N3PDF/vegasflow/workflows/pytest/badge.svg)](https://github.com/N3PDF/vegasflow/actions?query=workflow%3A%22pytest%22)
[![Documentation Status](https://readthedocs.org/projects/vegasflow/badge/?version=latest)](https://vegasflow.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/226363558.svg)](https://zenodo.org/badge/latestdoi/226363558)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vegasflow/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)
[![AUR](https://img.shields.io/aur/version/oha)](https://aur.archlinux.org/packages/python-vegasflow/)]


# VegasFlow

VegasFlow is a Monte Carlo integration library written in Python and based on the [TensorFlow](https://www.tensorflow.org/) framework. It is developed with a focus on speed and efficiency, enabling researchers to perform very expensive calculation as quick and easy as possible.

Some of the key features of VegasFlow are:
- Integrates efficiently high dimensional functions on single (multi-threading) and multi CPU, single and multi GPU, many GPUs or clusters.

- Compatible with Python, C, C++ or Fortran.

- Implementation of different Monte Carlo algorithms.

## Documentation

[https://vegasflow.readthedocs.io/en/latest](https://vegasflow.readthedocs.io/en/latest)


## Installation

The package can be installed with pip:
```
python3 -m pip install vegasflow
```

as well as with `conda`, from the `conda-forge` channel:
```
conda install vegasflow -c conda-forge
```

If you prefer a manual installation just use:
```
python setup.py install
```
or if you are planning to extend or develop code just use:
```
python setup.py develop
```

## Examples

There are some examples in the `examples/` folder.

## Citation policy

If you use the package please cite the following paper and zenodo references:
- [https://doi.org/10.5281/zenodo.3691926](https://doi.org/10.5281/zenodo.3691926)
- [https://arxiv.org/abs/2002.12921](https://arxiv.org/abs/2002.12921)
