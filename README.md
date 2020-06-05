[![Tests](https://github.com/N3PDF/vegasflow/workflows/pytest/badge.svg)](https://github.com/N3PDF/vegasflow/actions?query=workflow%3A%22pytest%22)
[![Documentation Status](https://readthedocs.org/projects/vegasflow/badge/?version=latest)](https://vegasflow.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/226363558.svg)](https://zenodo.org/badge/latestdoi/226363558)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vegasflow/badges/installer/conda.svg)](https://anaconda.org/conda-forge/vegasflow)
[![AUR](https://img.shields.io/aur/version/python-vegasflow)](https://aur.archlinux.org/packages/python-vegasflow/)


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

as well as `conda`, from the `conda-forge` channel:
```
conda install vegasflow -c conda-forge
```

If you prefer a manual installation you can clone the repository and run:
```
git clone https://github.com/N3PDF/vegasflow.git
cd vegasflow
python setup.py install
```
or if you are planning to extend or develop the code just use:
```
python setup.py develop
```

## Examples

There are some examples in the `examples/` folder.

## Minimum Working Example
```
import tensorflow as tf
from vegasflow.vflow import vegas_wrapper

def integrand(x, **kwargs):
    """ Function:
       x_{1} * x_{2} ... * x_{n}
       x: array of dimension (events, n)
    """
    return tf.reduce_prod(x, axis=1)

dimensions = 8
iterations = 5
events_per_iteration = int(1e5)
vegas_wrapper(integrand, dimensions, iterations, events_per_iteration, compilable=True)
```

For more complicated examples please see the [documentation](https://vegasflow.readthedocs.io/en/latest)
or the [examples](https://github.com/N3PDF/vegasflow/tree/master/examples) folder.

Please feel free to [open an issue](https://github.com/N3PDF/vegasflow/issues/new) if you would like
some specific example or find any problems at all with the code or the documentation.

## Citation policy

If you use the package please cite the following paper and zenodo references:
- [https://doi.org/10.5281/zenodo.3691926](https://doi.org/10.5281/zenodo.3691926)
- [https://arxiv.org/abs/2002.12921](https://arxiv.org/abs/2002.12921)
