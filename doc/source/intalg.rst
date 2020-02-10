.. _intalg-label:

======================
Integration algorithms
======================

This page lists the integration algorithms currently implemented.

.. _vegas-label:

Vegas
=====

This implementation of the Vegas algorithm closely follow the description of the importance sampling in the original `Vegas <https://doi.org/10.1016/0021-9991(78)90004-9>`_ paper.

An integration with the Vegas algorithm can be performed using the ``VegasFlow`` class.
Initializing the integrator requires to provide a number of dimensions with which initialize the grid and a target number of calls per iterations.

.. code-block:: python
    from vegasflow.vflow import VegasFlow
    vegas_instance = VegasFlow(dimensions, ncalls)

Once that is generated it is possible to register an integrand by calling the compile method.

.. code-block:: python

    def example_integrand(x, **kwargs):
        return x

    vegas_instance.compile(example_integrand)

Once this process has been performed we can start computing the result by simply calling the run_integration method to which we need to provided a number of iterations.
After each iteration the grid will be refined, producing more points (and hence reducing the error) in the regions where the integrand is larger.

.. code-block:: python
    result = vegas_instance.run_integration(n_iter)

It is often useful to freeze the grid to compute the integration several times with a frozen grid, in order to do that we provide the freeze_grid method. Note that freezing the grid forces a recompilation of the integrand which means the first iteration after freezing can potentially be slow, after which it will become much faster as before as the part of the graph dedicated to the adjusting of the grid is dropped.

.. code-block:: python
    vegas_instance.freeze_grid()


.. autoclass:: vegasflow.vflow.VegasFlow
    :members: freeze_grid, unfreeze_grid, refine_grid, run_event

