.. _howto-label:

=====
Usage
=====

``vegasflow`` is a python library that provides a number of functions to perform Monte Carlo integration of some functions.
In this guide we do our best to explain the steps to follow in order to perform a successful calculation with VegasFlow.
If, after reading this, you have any doubts, questions (or ideas for improvements!) please, don't hesistate to contact us [opening an issue](https://github.com/N3PDF/vegasflow/issues/new?assignees=&body=I%20have%20a%20question%20about%20VegasFlow...&labels=question) in Github.

.. contents::
   :local:
   :depth: 1


Integrating with VegasFlow
==========================

Basic usage
^^^^^^^^^^^

Integrating a function with ``VegasFlow`` is done in three basic steps:

1. **Instantiating an integrator**: At the time of instantiation it is necessary to provide
   a number of dimensions and a number of calls per iteration.
   The reason for giving this information beforehand is to allow for optimization.

.. code-block:: python

    from vegasflow import VegasFlow
    
    dims = 3
    n_calls = int(1e7)
    vegas_instance = VegasFlow(dims, n_calls)

2. **Compiling the integrand**: The integrand needs to be given to the integrator for compilation.
Compilation serves a dual purposes, it first registers the integrand and then it compiles it
using the ``tf.function`` decorator.

.. code-block:: python

    import tensorflow as tf
    
    def example_integrand(xarr, n_dim=none, weight=none):
      s = tf.reduce_sum(xarr, axis=1)
      result = tf.pow(0.1/s, 2)
      return result
      
    vegas_instance.compile(example_integrand)

3. **Running the integration**: Once everything is in place, we just need to inform the integrator of the number of
   iterations we want.

.. code-block:: python

    n_iter = 5
    result = vegas_instance.run_integration(n_iter)Q


Constructing the integrand
^^^^^^^^^^^^^^^^^^^^^^^^^^
Note that the ``example_integrand`` contained only ``TensorFlow`` operations.
All ``VegasFlow`` integrands such, in principle, depend only on python primitives
and ``TensorFlow`` operations, otherwise the code cannot be compiled and as a result it cannot
run on GPU u other ``TensorFlow``-supported hardware accelerators.

It is possible, however (and often useful when prototyping) integrate functions not
based on ``TensorFlow``, by passing the ``compilable`` flag at compile time.
This will spare the compilation of the integrand (while maintaining the compilation of
the integration algorithm).

.. code-block:: python

    import numpy as np
    
    def example_integrand(xarr, n_dim=None, weight=None):
      s = np.pow(xarr, axis=1)
      result = np.square(0.1/s)
      return result
      
    vegas_instance.compile(example_integrand, compilable=False)


.. note:: Integrands must accept the keyword arguments ``n_dim`` and ``weight``, as the integrators
   will try to pass those argument when convenient. It is however possible to construct integrands
   that accept only the array of random number ``xarr``, see :ref:`simple-label`


It is also possible to completely avoid compilation,
by leveraging ``TensorFlow``'s `eager execution <https://www.tensorflow.org/guide/eager>`_ as
explained at :ref:`eager-label`.

Choosing the correct types
^^^^^^^^^^^^^^^^^^^^^^^^^^

A common pitfall when writing ``TensorFlow``-compilable integrands is to mix different precision types.
If a function is compiled with a 32-bit float input not only it won't work when called with a 64-bit
float, but it will catastrophically fail.
The types in ``VegasFlow`` can be controlled via :ref:`environ-label` but we also provide the
``float_me`` and ``int_me`` function in order to ensure that all variables in the program have consistent
types.

These functions are wrappers around ``tf.cast`` `🔗 <https://www.tensorflow.org/api_docs/python/tf/cast>`__.

.. code-block:: python

    from vegasflow import float_me, int_me
    import tensorflow as tf
    
    constant = float_me(0.1)
    
    def example_integrand(xarr, n_dim=None, weight=None):
      s = tf.reduce_sum(xarr, axis=1)
      result = tf.pow(constant/s, 2)
      return result
      
    vegas_instance.compile(example_integrand)



Integration wrappers
^^^^^^^^^^^^^^^^^^^^

Although manually instantiating the integrator allows for a better fine-grained control
of the integration, it is also possible to use wrappers which automatically do most of the work
behind the scenes.

.. code-block:: python

   from vegasflow import vegas_wrapper
   
   result = vegas_wrapper(example_integrand, dims, n_iter, n_calls, compilable=False)


The full list of integration algorithms and wrappers can be consulted at: :ref:`intalg-label`.


Tips and Tricks
===============

.. _simple-label:

Improving results by simplifying the integrand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above example the integrand receives the keyword arguments `n_dim` and `xjac`.
Although these are useful for instance for filling histograms or dimensional-dependent integrands,
these extra argument can harm the performance of the integration when they are not being used.

It is possible to instantiate ``VegasFlow`` algorithms with ``simplify_signature``.
In this case the integrand will only receive the array of random numbers and, in exchange for this
loss of flexibility, the function will be retraced less often.
For more details in what function retracing entails we direct to the `TensorFlow documentation <https://www.tensorflow.org/api_docs/python/tf/function>`_.

.. code-block:: python

    from vegasflow import VegasFlow, float_me
    import tensorflow as tf

    def example_integrand(xarr):
        c = float_me(0.1)
        s = tf.reduce_sum(xarr)
        result = tf.pow(c/s)
        return result

    dimensions = 3
    n_calls = int(1e7)
    # Create an instance of the VegasFlow class
    vegas_instance = VegasFlow(dimensions, n_calls, simplify_signature = True)
    # Compile the function to be integrated
    vegas_instance.compile(example_integrand)
    # Compute the result after a number of iterations
    n_iter = 5
    result = vegas_instance.run_integration(n_iter)
    

Seeding the random number generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Seeding operations in ``TensorFlow`` is not always trivial.
We include in all integrator the method ``set_seed`` which is a wrapper to
``TensorFlow``'s own `seed method <https://www.tensorflow.org/api_docs/python/tf/random/set_seed>`_.

.. code-block:: python

    from vegasflow import VegasFlow

    vegas_instance = VegasFlow(dimensions, n_calls)
    vegas_instance.set_seed(7)


This is equivalent to:

.. code-block:: python

    from vegasflow import VegasFlow
    import tensorflow as tf
    
    vegas_instance = VegasFlow(dimensions, n_calls)
    tf.random.set_seed(7)
    

This seed is what ``TensorFlow`` calls a global seed and is then used to generate operation-level seeds.
In graph mode (:ref:`eager-label`) all top level ``tf.functions`` branch out
of the same initial state.
As a consequence, if we were to run two separate instances of ``VegasFlow``,
despite running sequentially, they would both run with the same seed.
Note that it only occurs if the seed is manually set.

.. code-block:: python

    from vegasflow import vegas_wrapper
    import tensorflow as tf
    
    tf.random.set_seed(7)
    result_1 = vegas_wrapper(example_integrand, dims, n_iter, n_calls)
    result_2 = vegas_wrapper(example_integrand, dims, n_iter, n_calls)
    assert result_1 == result_2
    

The way ``TensorFlow`` seeding works can be consulted here `here <https://www.tensorflow.org/api_docs/python/tf/random/set_seed>`_.

.. note:: Even when using seed, reproducibility is not guaranteed between two different versions of TensorFlow.


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
^^^^^^^^^

``VegasFlow`` uses the internal logging capabilities of python by
creating a new logger handled named ``vegasflow``.
You can modify the behavior of the logger as with any sane python library with the following lines:

.. code-block:: python

  import logging
  
  log_dict = {
        "0" : logging.ERROR,
        "1" : logging.WARNING,
        "2" : logging.INFO,
        "3" : logging.DEBUG
        }
  logger_vegasflow = logging.getLogger('vegasflow')
  logger_vegasflow.setLevel(log_dict["0"])
  
Where the log level can be any level defined in the ``log_dict`` dictionary.

Since ``VegasFlow`` is to be interfaced with non-python code it is also
possible to control the behaviour through the environment variable ``VEGASFLOW_LOG_LEVEL``, in that case any of the keys in ``log_dict`` can be used. For instance:

.. code-block:: bash
  
  export VEGASFLOW_LOG_LEVEL=1

will suppress all logger information other than ``WARNING`` and ``ERROR``.



.. _environ-label:

Environment
^^^^^^^^^^^

``VegasFlow`` is based on ``TensorFlow`` and as such all environment variable that
have an effect on ``TensorFlow``s behavior will also have an effect on ``VegasFlow``.

Here we describe only some of what we found to be the most useful variables.
For a complete description on the variables controlling the GPU-behavior of ``TensorFlow`` please refer to
the `nvidia official documentation <https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#variablestf>`_.

- ``TF_CPP_MIN_LOG_LEVEL``: controls the ``TensorFlow`` logging level. It is set to 1 by default so that only errors are printed.
- ``VEGASFLOW_LOG_LEVEL``: controls the ``VegasFlow`` logging level. Set to 3 by default so that everything is printed.
- ``VEGASFLOW_FLOAT``: controls the ``VegasFlow`` float precision. Default is 64 for 64-bits. Accepts: 64, 32.
- ``VEGASFLOW_INT``: controls the ``VegasFlow`` integer precision. Default is 32 for 32-bits. Accepts: 64, 32.


Choosing integration device
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

When performing computational expensive tasks Tensorflow's graph mode is preferred.
When compiling you will notice the first iteration of the integration takes a bit longer, this is normal
and it's due to the creation of the graph.
Subsequent iterations will be faster.

Graph-mode however is not debugger friendly as the code is read only once, when compiling the graph.
You can however enable ``Tensorflow``'s `eager execution <https://www.tensorflow.org/guide/eager>`_.
With eager mode the code is run sequentially as you would expect with normal python code,
this will allow you for instance throw in instances of ``pdb.set_trace()``.
In order to use eager execution we provide the ``run_eager`` wrapper.

.. code-block:: python

   from vegasflow import run_eager
   
   run_eager() # Enable eager-mode
   run_eager(False) # Disable


This is a wrapper around the following lines of code:

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
    from vegasflow.configflow import fzero, fone, int_me, DTYPE
    
    HISTO_BINS = int_me(2)
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

Generate conditions
===================

A very common case when integrating using Monte Carlo method is to add non trivial cuts to the
integration space.
It is not obvious how to implement cuts in a consistent manner in GPU or using ``TensorFlow``
routines when we have to combine several conditions.
We provide the ``generate_condition_function``  auxiliary function which generates
a ``TensorFlow``-compiled function for the necessary number of conditions.

For instance, let's take the case of a parton collision simulation, in which
we want to constraint the phase space of the two final state particles to the region
in which the two particles have a transverse momentum above 15 GeV or any of them
a rapidity below 4.

We first generate the condition we want to apply using ``generate_condition_function``.

.. code-block:: python

    from vegasflow.utils import generate_condition_function
    
    f_cond = generate_condition_function(3, condition = ['and', 'or'])


Now we can use the ``f_cond`` function in our integrand.
This ``f_cond`` function accepts three arguments and returns a mask of all of them
and the ``True`` indices.

.. code-block:: python

    import tensorflow as tf
    from vegasflow import vegas_wrapper
    
    def two_particle(xarr, **kwargs):
        # Complicated calculation of phase space
        pt_jet_1 = xarr[:,0]*100 + 5
        pt_jet_2 = xarr[:,1]*100 + 5
        rapidity = xarr[:,2]*50
        # Generate the conditions
        c_1 = pt_jet_1 > 15
        c_2 = pt_jet_2 > 15
        c_3 = rapidity < 4
        mask, idx = f_cond(c_1, c_2, c_3)
        # Now we can mask away the unwanted results
        good_vals = tf.boolean_mask(xarr[:,3], mask, axis=0)
        # Perform very complicated calculation
        result = tf.square(good_vals)
        # Return a sparse tensor so that only the actual results have a value
        ret = tf.scatter_nd(idx, result, shape=c_1.shape)
        return ret
      
    result = vegas_wrapper(two_particle, 4, 3, 100, compilable=False)
    
Note that we use the mask to remove the values that are not part of the phase space.
If the phase space to be integrated is much smaller than the integration region,
removing unwanted values can have a huge impact in the calculation from the
point of view of speed and memory, so we recommend removing them instead of just
zeroing them.

The resulting array, however, must have one value per event, so before returning
back the array to ``VegasFlow`` we use ``tf.scatter_nd`` to create a sparse tensor
where all values are set to 0 except the indices defined in ``idx`` that
have the values defined by ``result``.
