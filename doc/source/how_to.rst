.. _howto-label:

==========
How to use
==========

``vegasflow`` is a python library that provides a number of functions to perform Monte Carlo integration of some functions.

.. contents::
   :local:
   :depth: 1


A first VegasFlow integration
=============================

Prototyping in ``VegasFlow`` is easy, the best results are obtained when the
integrands are written using TensorFlow primitives.
Below we show one example where we create a TF constant (using ``tf.constant``) and then we use the sum and power functions.

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

Running in distributed systems
==============================

``vegasflow`` implements an easy interface to distributed system via
the `dask <https://dask.org/>`_ library.
In order to enable it is enough by calling the ``set_distribute`` method
of the instantiated integrator class.
This method takes a `dask_jobqueue <https://jobqueue.dask.org/en/latest/>`_
to send the jobs to.

An example can be found in the `examples/cluster_dask.py <https://github.com/N3PDF/vegasflow/blob/master/examples/cluster_dask.py>`_ file where
the `SLURM <https://slurm.schedmd.com/documentation.html>`_ cluster is used as an example

.. note:: When the distributing capabilities of dask are being useful, ``VegasFlow`` "forfeits" control of the devices in which to run, trusting ``TensorFlow``'s defaults. To run, for instance, two GPUs in one single node while using dask the user should send two separate dask jobs, each targetting a different GPU.

Global configuration
====================

Verbosity
---------

Tensorflow is very verbose by default.
When ``vegasflow`` is imported the environment variable ``TF_CPP_MIN_LOG_LEVEL``
is set to 1, hiding most warnings.
If you want to recover the usual Tensorflow logging level you can
set your enviroment to ``export TF_CPP_MIN_LOG_LEVEL=0``.

Choosing integration device
---------------------------

The ``CUDA_VISIBLE_DEVICES`` environment variable will tell Tensorflow
(and thus VegasFlow) in which device it should run.
If the variable is not set, it will default to use all (and only) GPUs available.
In order to use the CPU you can hide the GPU by setting
``export CUDA_VISIBLE_DEVICES=""``.

If you have a set-up with more than one GPU you can select which one you will
want to use for the integration by setting the environment variable to the
right device, e.g., ``export CUDA_VISIBLE_DEVICES=0``.





.. _eager-label:

Eager Vs Graph-mode
-------------------

When performing computational expensive tasks Tensorflow's graph mode is preferred.
When compiling you will notice the first iteration of the integration takes a bit longer, this is normal
and it's due to the creation of the graph.
Subsequent iterations will be faster.

Graph-mode however is not debugger friendly as the code is read only once, when compiling the graph.
You can however enable Tensorflow's `eager execution <https://www.tensorflow.org/guide/eager>`_.
With eager mode the code is run sequentially as you would expect with normal python code,
this will allow you to throw in instances of ``pdb.set_trace()``.
In order to enable eager mode include these lines at the top of your program:

.. code-block:: python

    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    
or if you are using versions of Tensorflow older than 2.3:

.. code-block:: python

    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)


Eager mode also enables the usage of the library as a `standard` python library
allowing you to integrate non-tensorflow integrands.
These integrands, as they are not understood by tensorflow, are not run using
GPU kernels while the rest of ``VegasFlow`` will still be run on GPU if possible.


Histograms
==========

A commonly used feature in Monte Carlo calculations is the generation of histograms.
In order to generate them while at the same time keeping all the features of ``vegasflow``,
such as GPU computing, it is necessary to ensure the histogram generation is also wrapped with the ``@tf.function`` directive.

Below we show one such example (how the histogram is actually generated and saved is up to the user).
The first step is to create a ``Variable`` tensor which will be used to fill the histograms.
This is a crucial step (and the only fixed step) as this tensor will be accumulated internally by ``VegasFlow``.


.. code-block:: python

    from vegasflow.utils import consume_array_into_indices
    fzero = tf.constant(0.0, dtype=tf.float64)
    fone = tf.constant(1.0, dtype=tf.float64)
    HISTO_BINS = 2

    cumulator_tensor = tf.Variable(tf.zeros(HISTO_BINS, dtype=DTYPE))

    @tf.function
    def histogram_collector(results, variables):
        """ This function will receive a tensor (result)
        and the variables corresponding to those integrand results 
        In the example integrand below, these corresponds to 
            `final_result` and `histogram_values` respectively.
        `current_histograms` instead is the current value of the histogram
        which will be overwritten """
        # Fill a histogram with HISTO_BINS (2) bins, (0 to 0.5, 0.5 to 1)
        # First generate the indices with TF
        indices = tf.histogram_fixed_width_bins(
            variables, [fzero, fone], nbins=HISTO_BINS
        )
        t_indices = tf.transpose(indices)
        # Then consume the results with the utility we provide
        partial_hist = consume_array_into_indices(results, t_indices, HISTO_BINS)
        # Then update the results of current_histograms
        new_histograms = partial_hist + current_histograms
        cummulator_tensor.assign(new_histograms)

    @tf.function
    def integrand_example(xarr, n_dim=None, weight=fone):
        # some complicated calculation that generates 
        # a final_result and some histogram values:
        final_result = tf.constant(42, dtype=tf.float64)
        histogram_values = xarr
        histogram_collector(final_result * weight, histogram_values)
        return final_result

Finally we can normally call ``vegasflow``, remembering to pass down the accumulator tensor, which will be filled in with the histograms.
Note that here we are only filling one histograms and so the histogram tuple contains only one element, but any number of histograms can be filled.


.. code-block:: python

    histogram_tuple = (cumulator_tensor,)
    results = mc_instance.run_integration(n_iter, histograms=histogram_tuple)


We ship an example of an integrand which generates histograms in the github repository: `here <https://github.com/N3PDF/vegasflow/blob/master/examples/histogram_ex.py>`_.




