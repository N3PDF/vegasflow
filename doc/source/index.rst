.. title::
    vegasflow's documentation!

========================
VegasFlow: Gotta go fast
========================

VegasFlow is a MonteCarlo integration library written in Python and based on the `TensorFlow <https://www.tensorflow.org/>`_ framework.
It is developed with a focus on speed and efficiency, enabling researchers to perform very expensive calculation as quick as possible, as easy as possible.

Some of the key features of VegasFlow are:

    - Runs efficiently on CPU, GPU, many GPUs or clusters.
    - Compatible with Python, C, C++ or Fortran.
    - Implementation of several MonteCarlo algorithms.

Motivation
==========

VegasFlow is developed within the Particle Physics group of the University of Milan.
Theoretical calculations in particle physics are incredibly time consuming operations, sometimes taking months in big clusters all around the world.

These expensive calculations are driven by the high dimensional phase space that need to be integrated but also by a lack of expertise in new techniques on high performance computation.
Indeed, while at the theoretical level these are some of the most complicated calculations performed by mankind; at the technical level most of these calculations are performed using very dated code and methodologies that are unable to make us of the available resources.

With VegasFlow we aim to fill this gap between theoretical calculations and technical performance by providing a framework which can automatically make the best of the machine in which it runs.
To that end VegasFlow is based on two technologies that together will enable a new age of research.

    - `TensorFlow <https://www.tensorflow.org/>`_: the framework developed by Google and made public in November of 2015 is a perfect combination between performance and usability. With a focus on Deep Learning, TensorFlow provides an algebra library able to easily run operations in many different devices: CPUs, GPUs, TPUs with little input by the developer. Write your code once.
    - `Ray <https://ray.readthedocs.io/en/latest/>`_: also with a focus on Deep Learning, the Ray framework allows building distributed applications in a robust way, so that the burden of using several different machines is lifted away from the developer, that can instead focus in their own applications.


FAQ
===

Why the name ``vegasflow``?
---------------------------

It is a combination of the names `Vegas` and `Tensorflow`.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   how_to
   intalg

.. automodule:: vegasflow
    :members:
    :noindex:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
