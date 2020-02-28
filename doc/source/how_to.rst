.. _howto-label:

==========
How to use
==========

``vegasflow`` is a python library that provides a number of functions to perform Monte Carlo integration of some functions.


.. code-block:: python

    import tensorflow as tf

    @tf.function
    def example_integrand(xarr, n_dim=None):
        c = tf.constant(0.1, dtype=tf.float64)
        s = tf.reduce_sum(xarr)
        result = tf.pow(c/s)
        return result

    dimensions = 3
    ncalls = int(1e7)
    # Create an instance of the VegasFlow class
    vegas_instance = VegasFlow(dimensions, ncalls)
    # Compile the function to be integrated
    vegas_instance.compile(example_integrand)
    # Compute the result after a number of iterations
    n_iter = 5
    result = vegas_instance.run_integration(n_iter)

We also provide a convenience wrapper ``vegas_wrapper`` that allows to run the whole thing in one go.

.. code-block:: python

    result = vegas_wrapper(example_integrand, dimensions, n_iter, ncalls)

