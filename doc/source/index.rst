.. title::
    vegasflow's documentation!


=================================================================================
VegasFlow: accelerating Monte Carlo simulation across multiple hardware platforms
=================================================================================

.. image:: https://img.shields.io/badge/j.%20Computer%20Physics%20Communication-2020%2F107376-blue
   :target: https://doi.org/10.1016/j.cpc.2020.107376

.. image:: https://img.shields.io/badge/physics.comp--ph-arXiv%3A2002.12921-B31B1B
   :target: https://arxiv.org/abs/2002.12921

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3691926.svg
   :target: https://doi.org/10.5281/zenodo.3691926

.. contents::
   :local:
   :depth: 1

VegasFlow is a `Monte Carlo integration <https://en.wikipedia.org/wiki/Monte_Carlo_integration>`_ library written in Python and based on the `TensorFlow <https://www.tensorflow.org/>`_ framework.
It is developed with a focus on speed and efficiency, enabling researchers to perform very expensive calculation as quick and easy as possible.

Some of the key features of VegasFlow are:

   * Integrates efficiently high dimensional functions on single (multi-threading) and multi CPU, single and multi GPU, many GPUs or clusters.
   * Compatible with Python, C, C++ or Fortran.
   * Implementation of different Monte Carlo algorithms.

How to obtain the code
======================

Open Source
-----------
The ``vegasflow`` package is open source and available at https://github.com/N3PDF/vegasflow

Installation
------------
The package can be installed with pip:

.. code-block:: bash

  python3 -m pip install vegasflow

If you prefer a manual installation just use:

.. code-block:: bash

  git clone https://github.com/N3PDF/vegasflow
  cd vegasflow
  python3 setup.py install

or if you are planning to extend or develop code just use:

.. code-block:: bash

  python3 setup.py develop
  
It is also possible to install the package from repositories such as `conda-forge <https://anaconda.org/conda-forge/vegasflow>`_ or the `Arch User Repository <https://aur.archlinux.org/packages/python-vegasflow/>`_

.. code-block:: bash

  conda install vegasflow -c conda-forge
  yay -S python-vegasflow

Motivation
==========

VegasFlow is developed within the Particle Physics group of the University of Milan.
Theoretical calculations in particle physics are incredibly time consuming operations, sometimes taking months in big clusters all around the world.

These expensive calculations are driven by the high dimensional phase space that need to be integrated but also by a lack of expertise in new techniques on high performance computation.
Indeed, while at the theoretical level these are some of the most complicated calculations performed by mankind; at the technical level most of these calculations are performed using very dated code and methodologies that are unable to make us of the available resources.

With VegasFlow we aim to fill this gap between theoretical calculations and technical performance by providing a framework which can automatically make the best of the machine in which it runs.
To that end VegasFlow is based on two technologies that together will enable a new age of research.


How to cite ``vegaflow``?
=========================

When using ``vegasflow`` in your research, please cite the following publications:

.. image:: https://img.shields.io/badge/j.%20Computer%20Physics%20Communication-2020%2F107376-blue
   :target: https://doi.org/10.1016/j.cpc.2020.107376
   

.. image:: https://img.shields.io/badge/arXiv-physics.comp--ph%2F%20%20%20%202002.12921-%23B31B1B
   :target: https://arxiv.org/abs/2002.12921


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3691926.svg
   :target: https://doi.org/10.5281/zenodo.3691926

Bibtex:

.. code-block:: latex

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

FAQ
===

Why the name ``VegasFlow``?
---------------------------

It is a combination of the names `Vegas` and `Tensorflow`.

- **Vegas**: this integration algorithm, created originally by `G.P. Lepage <https://www.sciencedirect.com/science/article/pii/S001046559900209X>`_ sits at the core of many of the most advanced calculations in High Energy Physics, it powers `Madgraph <https://cp3.irmp.ucl.ac.be/projects/madgraph/>_`, `MCFM <https://mcfm.fnal.gov/>`_ or `Sherpa <https://sherpa.hepforge.org/doc/SHERPA-MC-2.2.8.html#VEGAS>`_ among others. Lepage's own implementation is available in `github <https://github.com/gplepage/vegas>`_.

- **TensorFlow**: the `tensorflow <https://www.tensorflow.org/>`_ is developed by Google and was made public in November of 2015. It is a perfect combination between performance and usability. With a focus on Deep Learning, TensorFlow provides an algebra library able to easily run operations in many different devices: CPUs, GPUs, TPUs with little input by the developer.

I have a problem I can't solve
------------------------------
Please, `open an issue <https://github.com/N3PDF/vegasflow/issues/new?assignees=&labels=bug&template=bug_report.md&title=>`_ in the github repository
or `check <https://github.com/N3PDF/vegasflow/issues>`_ whether someone has already asked the same question.
We will be happy to help.


Indices and tables
==================

.. toctree::
    :maxdepth: 3
    :glob: 
    :caption: Contents:

    VegasFlow<self>
    how_to
    intalg
    examples
    apisrc/vegasflow
    

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
