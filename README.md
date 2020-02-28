[![Tests](https://github.com/N3PDF/vegasflow/workflows/pytest/badge.svg)](https://github.com/N3PDF/vegasflow/actions?query=workflow%3A%22pytest%22)
[![Documentation Status](https://readthedocs.org/projects/vegasflow/badge/?version=latest)](https://vegasflow.readthedocs.io/en/latest/?badge=latest)

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
pip install vegasflow
```
or if you are planning to extend or develop code just use:
```
python setup.py develop
```

## Examples

There are some examples in the `examples/` folder.

## Citation policy

If you use the theta package please cite the following  paper and zenodo references:
- Zenodo
- arXiv: