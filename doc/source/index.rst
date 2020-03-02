.. title::
    vegasflow's documentation!


========================
VegasFlow: accelerating Monte Carlo simulation across multiple hardware platforms
========================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3691926.svg
   :target: https://doi.org/10.5281/zenodo.3691926

.. contents::
   :local:
   :depth: 1

VegasFlow is a `Monte Carlo integration <https://en.wikipedia.org/wiki/Monte_Carlo_integration>`_ library written in Python and based on the `TensorFlow <https://www.tensorflow.org/>`_ framework.
It is developed with a focus on speed and efficiency, enabling researchers to perform very expensive calculation as quick and easy as possible.

Some of the key features of VegasFlow are:

    - Integrates efficiently high dimensional functions on single (multi-threading) and multi CPU, single and multi GPU, many GPUs or clusters.
    - Compatible with Python, C, C++ or Fortran.
    - Implementation of different Monte Carlo algorithms.


Motivation
==========

VegasFlow is developed within the Particle Physics group of the University of Milan.
Theoretical calculations in particle physics are incredibly time consuming operations, sometimes taking months in big clusters all around the world.

These expensive calculations are driven by the high dimensional phase space that need to be integrated but also by a lack of expertise in new techniques on high performance computation.
Indeed, while at the theoretical level these are some of the most complicated calculations performed by mankind; at the technical level most of these calculations are performed using very dated code and methodologies that are unable to make us of the available resources.

With VegasFlow we aim to fill this gap between theoretical calculations and technical performance by providing a framework which can automatically make the best of the machine in which it runs.
To that end VegasFlow is based on two technologies that together will enable a new age of research.

    - `TensorFlow <https://www.tensorflow.org/>`_: the framework developed by Google and made public in November of 2015 is a perfect combination between performance and usability. With a focus on Deep Learning, TensorFlow provides an algebra library able to easily run operations in many different devices: CPUs, GPUs, TPUs with little input by the developer. Write your code once.



How to cite ``vegaflow``?
=========================

When using ``vegasflow`` in your research, please cite the following publications:

https://arxiv.org/abs/2002.12921

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3691926.svg
   :target: https://doi.org/10.5281/zenodo.3691926

Bibtex:

.. code-block:: latex

    @article{Carrazza:2020rdn,
       author         = "Carrazza, Stefano and Cruz-Martinez, Juan M.",
       title          = "{VegasFlow: accelerating Monte Carlo simulation across
                         multiple hardware platforms}",
       year           = "2020",
       eprint         = "2002.12921",
       archivePrefix  = "arXiv",
       primaryClass   = "physics.comp-ph",
       reportNumber   = "TIF-UNIMI-2020-8",
       SLACcitation   = "%%CITATION = ARXIV:2002.12921;%%"
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

FAQ
===

Why the name ``vegasflow``?
---------------------------

It is a combination of the names `Vegas` and `Tensorflow`.




Indices and tables
==================

.. toctree::
    :maxdepth: 3
    :glob: 
    :caption: Contents:

    VegasFlow<self>
    how_to
    intalg
    apisrc/vegasflow
    

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
