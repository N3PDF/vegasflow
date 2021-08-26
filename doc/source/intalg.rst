.. _intalg-label:

======================
Integration algorithms
======================

This page lists the integration algorithms currently implemented.

.. contents::
   :local:
   :depth: 1

.. _vegas-label:

VegasFlow
=========

Overview
^^^^^^^^

This implementation of the Vegas algorithm closely follow the description of the importance sampling in the original `Vegas <https://doi.org/10.1016/0021-9991(78)90004-9>`_ paper.

An integration with the Vegas algorithm can be performed using the ``VegasFlow`` class.
Initializing the integrator requires to provide a number of dimensions with which initialize the grid and a target number of calls per iterations.

.. code-block:: python

    from vegasflow import VegasFlow
    dims = 4
    n_calls = int(1e6)
    vegas_instance = VegasFlow(dims, n_calls)

Once that is generated it is possible to register an integrand by calling the ``compile`` method.

.. code-block:: python

    def example_integrand(x, **kwargs):
        y = 0.0
        for d in range(dims):
            y += x[:,d]
        return y

    vegas_instance.compile(example_integrand)

Once this process has been performed we can start computing the result by simply calling the ``run_integration`` method to which we need to provided a number of iterations.
After each iteration the grid will be refined, producing more points (and hence reducing the error) in the regions where the integrand is larger.

.. code-block:: python

    n_iter = 3
    result = vegas_instance.run_integration(n_iter)

The output variable, in this example named ``result``, is a tuple variable where the first element is the result of the integration while the second element is the error of the integration.

Integration Wrapper
^^^^^^^^^^^^^^^^^^^

Although manually instantiating the integrator allows for a better fine-grained control
of the integration, it is also possible to use wrappers which automatically do most of the work
behind the scenes.

.. code-block:: python

   from vegasflow import vegas_wrapper
   n_iter = 5
   result = vegas_wrapper(example_integrand, dims, n_iter, n_calls)

Grid freezing
^^^^^^^^^^^^^

It is often useful to freeze the grid to compute the integration several times with a frozen grid, in order to do that we provide the ``freeze_grid`` method. Note that freezing the grid forces a recompilation of the integrand which means the first iteration after freezing can potentially be slow, after which it will become much faster as before as the part of the graph dedicated to the adjusting of the grid is dropped.

.. code-block:: python

    vegas_instance.freeze_grid()


Saving and loading a grid
^^^^^^^^^^^^^^^^^^^^^^^^^

On a related note, it is possible to save and load the grid from and to json files, in order to do that we can use the ``save_grid`` and ``load_grid`` methods at any point in the calculation.
Note, however, that loading a new grid will destroy the current grid.

.. code-block:: python

    json_file = "my_grid.json"
    vegas_instance.save_grid(json_file)
    vegas_instance.load_grid(json_file)

.. autoclass:: vegasflow.vflow.VegasFlow
    :noindex:
    :show-inheritance:
    :members: freeze_grid, unfreeze_grid, save_grid, load_grid


VegasFlowPlus
=============

Overview
^^^^^^^^
While ``VegasFlow`` is limited to the importance sampling algorithm,
``VegasFlowPlus`` includes the latest version of importance plus stratified sampling
from Lepage's `latest paper <https://arxiv.org/abs/2009.05112>`_.

The usage and interfaces exposed by ``VegasFlowPlus`` are equivalent to those
of ``VegasFlow``:


.. code-block:: python

    from vegasflow import VegasFlowPlus
    dims = 4
    n_calls = int(1e6)
    vegas_instance = VegasFlowPlus(dims, n_calls)

    def example_integrand(x, **kwargs):
        y = 0.0
        for d in range(dims):
            y += x[:,d]
        return y

    vegas_instance.compile(example_integrand)

    n_iter = 3
    result = vegas_instance.run_integration(n_iter)


As it can be seen, the only change has been to substitute the ``VegasFlow`` class
with ``VegasFlowPlus``.

.. note:: ``VegasFlowPlus`` does not support multi-device running, as it cannot break the integration in several pieces, an issue tracked at `#78 <https://github.com/N3PDF/vegasflow/issues/78>`_.

Integration Wrapper
^^^^^^^^^^^^^^^^^^^

A wrapper is also provided for simplicity:

.. code-block:: python

   from vegasflow import vegasflowplus_wrapper
   n_iter = 5
   result = vegasflowplus_wrapper(example_integrand, dims, n_iter, n_calls)

 
PlainFlow
=========

Overview
^^^^^^^^

We provide a very rudimentary Monte Carlo integrator which we name PlainFlow.
This provides a easy example on how to implement a new integration algorithm.

The usage pattern is similar to :ref:`vegas-label`.

.. code-block:: python

    from vegasflow import PlainFlow
    plain_instance = PlainFlow(dims, n_calls)
    plain_instance.compile(example_integrand)
    plain_instance.run_integration(n_iter)

Integration Wrapper
^^^^^^^^^^^^^^^^^^^

An integration wrapper is also provided as ``vegasflow.plain_wrapper``.

.. code-block:: python

   from vegasflow import plain_wrapper
   result = plain_wrapper(example_integrand, dims, n_iter, n_calls)


.. autoclass:: vegasflow.plain.PlainFlow
    :noindex:
    :show-inheritance:
