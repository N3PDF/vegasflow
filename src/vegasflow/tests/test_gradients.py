"""
    Tests the gradients of the different algorithms
"""

import numpy as np
from pytest import mark
import tensorflow as tf

from vegasflow import PlainFlow, VegasFlow, VegasFlowPlus, float_me, run_eager


def generate_integrand(variable):
    """Generate an integrand that depends on an input variable"""

    def example_integrand(x):
        y = tf.reduce_sum(x, axis=1)
        return y * variable

    return example_integrand


def generate_differentiable_function(iclass, integrand, dims=3, n_calls=int(1e5), i_kwargs=None):
    """Generates a function that depends on the result of a Monte Carlo integral
    of ``integrand`` using the class iclass (in differentiable form) as integrator
    """
    if i_kwargs is None:
        i_kwargs = {}
    integrator_instance = iclass(dims, n_calls, verbose=False, **i_kwargs)
    integrator_instance.compile(integrand)
    # Train
    _ = integrator_instance.run_integration(2)
    # Now make it differentiable/compilable
    runner = integrator_instance.make_differentiable()

    def some_complicated_function(x):
        integration_result, *_ = runner()
        return x * integration_result

    compiled_fun = tf.function(some_complicated_function)
    # Compile the function
    _ = compiled_fun(float_me(4.0))
    return compiled_fun


def wrapper_test(iclass, x_point=5.0, alpha=10, integrator_kwargs=None):
    """Wrapper for all integrators"""
    # Create a variable
    z = tf.Variable(float_me(1.0))
    # Create an integrand that depends on this variable
    integrand = generate_integrand(z)
    # Now create a function that depends on its integration
    fun = generate_differentiable_function(iclass, integrand, i_kwargs=integrator_kwargs)

    x0 = float_me(x_point)
    with tf.GradientTape() as tape:
        tape.watch(x0)
        y1 = fun(x0)

    grad_1 = tape.gradient(y1, x0)

    # Change the value of the variable
    z.assign(z.numpy() * alpha)

    with tf.GradientTape() as tape:
        tape.watch(x0)
        y2 = fun(x0)

    grad_2 = tape.gradient(y2, x0)

    # Test that the gradient works as expected
    np.testing.assert_allclose(grad_1 * alpha, grad_2, rtol=1e-2)


@mark.parametrize("algorithm", [VegasFlowPlus, VegasFlow, PlainFlow])
def test_gradient(algorithm):
    """ "Test one can compile and generate gradients with the different algorithms"""
    wrapper_test(algorithm)


def test_gradient_VegasflowPlus_adaptive():
    """ "Test one can compile and generate gradients with VegasFlowPlus"""
    wrapper_test(VegasFlowPlus, integrator_kwargs={"adaptive": True})
