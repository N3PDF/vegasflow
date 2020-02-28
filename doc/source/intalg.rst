.. _intalg-label:

======================
Integration algorithms
======================

This page lists the integration algorithms currently implemented.

.. _vegas-label:

VegasFlow
=========

This implementation of the Vegas algorithm closely follow the description of the importance sampling in the original `Vegas <https://doi.org/10.1016/0021-9991(78)90004-9>`_ paper.

An integration with the Vegas algorithm can be performed using the ``VegasFlow`` class.
Initializing the integrator requires to provide a number of dimensions with which initialize the grid and a target number of calls per iterations.

.. code-block:: python

    from vegasflow.vflow import VegasFlow
    vegas_instance = VegasFlow(dimensions, ncalls)

Once that is generated it is possible to register an integrand by calling the ``compile`` method.

.. code-block:: python

    def example_integrand(x, **kwargs):
        return x

    vegas_instance.compile(example_integrand)

Once this process has been performed we can start computing the result by simply calling the ``run_integration`` method to which we need to provided a number of iterations.
After each iteration the grid will be refined, producing more points (and hence reducing the error) in the regions where the integrand is larger.

.. code-block:: python

    result = vegas_instance.run_integration(n_iter)

The output variable, in this example named ``result``, is a tuple variable where the first element is the result of the integration while the second element is the error of the integration.

It is often useful to freeze the grid to compute the integration several times with a frozen grid, in order to do that we provide the ``freeze_grid`` method. Note that freezing the grid forces a recompilation of the integrand which means the first iteration after freezing can potentially be slow, after which it will become much faster as before as the part of the graph dedicated to the adjusting of the grid is dropped.

.. code-block:: python

    vegas_instance.freeze_grid()

On a related note, it is possible to save and load the grid from and to json files, in order to do that we can use the ``save_grid`` and ``load_grid`` methods at any point in the calculation.
Note, however, that loading a new grid will destroy the current grid.

.. code-block:: python

    json_file = "my_grid.json"
    vegas_instance.save_grid(json_file)
    vegas_instance.load_grid(json_file)

.. autoclass:: vegasflow.vflow.VegasFlow
    :show-inheritance:
    :members: freeze_grid, unfreeze_grid, save_grid, load_grid

 
PlainFlow
=========

We provide a very rudimentary Monte Carlo integrator which we name PlainFlow.
This provides a easy example on how to implement a new integration algorithm.

The usage pattern is similar to :ref:`vegas-label`.

.. code-block:: python

    from vegasflow.plain import PlainFlow
    plain_instance = PlainFlow(dimensions, ncalls)
    plain_instance.compile(example_integrand)
    plain_instance.run_integration(n_iter)


.. autoclass:: vegasflow.plain.PlainFlow
    :show-inheritance:
