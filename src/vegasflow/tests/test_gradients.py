"""
    Tests the gradients of the different algorithms
"""

from vegasflow import float_me
from vegasflow import VegasFlow, VegasFlowPlus, PlainFlow
import tensorflow as tf
import numpy as np


def generate_integrand(variable):
    """Generate an integrand that depends on an input variable"""

    def example_integrand(x):
        y = tf.reduce_sum(x, axis=1)
        return y * variable

    return example_integrand


def generate_differentiable_function(iclass, integrand, dims=3, n_calls=int(1e5)):
    """Generates a function that depends on the result of a Monte Carlo integral
    of ``integrand`` using the class iclass (in differentiable form) as integrator
    """
    integrator_instance = iclass(dims, n_calls, verbose=False)
    runner = integrator_instance.make_differentiable()
    integrator_instance.compile(integrand)

    def some_complicated_function(x):
        integration_result, *_ = runner()
        return x * integration_result

    compiled_fun = tf.function(some_complicated_function)
    # Compile
    _ = compiled_fun(float_me(4.0))
    # Train
    _ = integrator_instance.run_integration(2)
    return compiled_fun


def wrapper_test(iclass, x_point=5.0, alpha=10):
    """Wrapper for all integrators"""
    # Create a variable
    z = tf.Variable(float_me(1.0))
    # Create an integrand that depends on this variable
    integrand = generate_integrand(z)
    # Now create a function that depends on its integration
    fun = generate_differentiable_function(iclass, integrand)

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


def test_gradient_Vegasflow():
    """"Test one can compile and generate gradients with VegasFlow"""
    wrapper_test(VegasFlow)

# 
# def test_gradient_VegasflowPlus():
#     """"Test one can compile and generate gradients with VegasFlowPlus"""
#     wrapper_test(VegasFlowPlus)


def test_gradient_PlainFlow():
    """"Test one can compile and generate gradients with PlainFlow"""
    wrapper_test(PlainFlow)
