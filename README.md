[![DOI](https://zenodo.org/badge/226363558.svg)](https://zenodo.org/badge/latestdoi/226363558)
[![cpc](https://img.shields.io/badge/j.%20Computer%20Physics%20Communication-2020%2F107376-blue)](https://inspirehep.net/literature/1783000)

[![Tests](https://github.com/N3PDF/vegasflow/workflows/pytest/badge.svg)](https://github.com/N3PDF/vegasflow/actions?query=workflow%3A%22pytest%22)
[![Documentation Status](https://readthedocs.org/projects/vegasflow/badge/?version=latest)](https://vegasflow.readthedocs.io/en/latest/?badge=latest)
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
```bash
python3 -m pip install vegasflow
```

as well as `conda`, from the `conda-forge` channel:
```bash
conda install vegasflow -c conda-forge
```

If you prefer a manual installation you can clone the repository and run:
```bash
git clone https://github.com/N3PDF/vegasflow.git
cd vegasflow
python setup.py install
```
or if you are planning to extend or develop the code just use:
```bash
python setup.py develop
```

## Examples

There are some examples in the `examples/` folder.

## Minimum Working Example
```python
from vegasflow import vegas_wrapper
import tensorflow as tf

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

```latex
    @article{Carrazza:2020rdn,
        author = "Carrazza, Stefano and Cruz-Martinez, Juan M.",
        title = "{VegasFlow: accelerating Monte Carlo simulation across multiple hardware platforms}",
        eprint = "2002.12921",
        archivePrefix = "arXiv",
        primaryClass = "physics.comp-ph",
        reportNumber = "TIF-UNIMI-2020-8",
        doi = "10.1016/j.cpc.2020.107376",
        journal = "Comput. Phys. Commun.",
        volume = "254",
        pages = "107376",
        year = "2020"
    }


    @software{vegasflow_package,
        author       = {Juan Cruz-Martinez and
                        Stefano Carrazza},
        title        = {N3PDF/vegasflow: vegasflow v1.0},
        month        = feb,
        year         = 2020,
        publisher    = {Zenodo},
        version      = {v1.0},
        doi          = {10.5281/zenodo.3691926},
        url          = {https://doi.org/10.5281/zenodo.3691926}
    }
```
