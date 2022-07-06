"""
    Checks that the different integation algorithms
    are able to run and don't produce a crazy result
"""

""" Test a run with a simple function to make sure
everything works """
import json
import tempfile
import pytest
import numpy as np
from vegasflow.configflow import DTYPE, run_eager
from vegasflow.vflow import VegasFlow
from vegasflow.plain import PlainFlow
from vegasflow.vflowplus import VegasFlowPlus
from vegasflow import plain_sampler, vegas_sampler
import tensorflow as tf

# Test setup
dim = 2
ncalls = np.int32(1e4)
n_iter = 4


def example_integrand(xarr, weight=None):
    """Example function that integrates to 1"""
    n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


def instance_and_compile(Integrator, mode=0, integrand_function=example_integrand):
    """Wrapper for convenience"""
    if mode == 0:
        integrand = integrand_function
    elif mode == 1:

        def integrand(xarr, n_dim=None):
            return integrand_function(xarr)

    elif mode == 2:

        def integrand(xarr):
            return integrand_function(xarr)

    elif mode == 3:

        def integrand(xarr, n_dim=None, weight=None):
            return integrand_function(xarr, weight=None)

    int_instance = Integrator(dim, ncalls)
    int_instance.compile(integrand)
    return int_instance


def check_is_one(result, sigmas=3):
    """Wrapper for convenience"""
    res = result[0]
    err = np.mean(result[1] * sigmas)
    # Check that it passes by {sigmas} number of sigmas
    np.testing.assert_allclose(res, 1.0, atol=err)


def test_VegasFlow():
    """Test VegasFlow class, importance sampling algorithm"""
    for mode in range(4):
        vegas_instance = instance_and_compile(VegasFlow, mode)
        _ = vegas_instance.run_integration(n_iter)
        vegas_instance.freeze_grid()
        result = vegas_instance.run_integration(n_iter)
        check_is_one(result)

    # Change the number of events
    vegas_instance.n_events = 2 * ncalls
    new_result = vegas_instance.run_integration(n_iter)
    check_is_one(new_result)

    # Unfreeze the grid
    vegas_instance.unfreeze_grid()
    new_result = vegas_instance.run_integration(n_iter)
    check_is_one(new_result)

    # And change the number of calls again
    vegas_instance.n_events = 3 * ncalls
    new_result = vegas_instance.run_integration(n_iter)
    check_is_one(new_result)


def test_VegasFlow_save_grid():
    """Test the grid saving feature of vegasflow"""
    tmp_filename = tempfile.mktemp()
    vegas_instance = instance_and_compile(VegasFlow)
    # Run an iteration so the grid is not trivial
    _ = vegas_instance.run_integration(1)
    current_grid = vegas_instance.divisions.numpy()
    # Save and load the grid from the file
    vegas_instance.save_grid(tmp_filename)
    with open(tmp_filename, "r") as f:
        json_grid = np.array(json.load(f)["grid"])
    np.testing.assert_equal(current_grid, json_grid)


def test_VegasFlow_load_grid():
    tmp_filename = tempfile.mktemp()
    # Get the information from the vegas_instance
    vegas_instance = instance_and_compile(VegasFlow)
    grid_shape = vegas_instance.divisions.shape
    tmp_grid = np.random.rand(*grid_shape)
    # Save into some rudimentary json file
    jdict = {"grid": tmp_grid.tolist()}
    with open(tmp_filename, "w") as f:
        json.dump(jdict, f)
    # Try to load it
    vegas_instance.load_grid(file_name=tmp_filename)
    # Check that the loading did work
    loaded_grid = vegas_instance.divisions.numpy()
    np.testing.assert_equal(loaded_grid, tmp_grid)
    # Now try to load a numpy array directly instead
    tmp_grid = np.random.rand(*grid_shape)
    vegas_instance.load_grid(numpy_grid=tmp_grid)
    loaded_grid = vegas_instance.divisions.numpy()
    np.testing.assert_equal(loaded_grid, tmp_grid)
    # Now check that the errors also work
    jdict["BINS"] = 0
    with open(tmp_filename, "w") as f:
        json.dump(jdict, f)
    # Check that it fails if the number of bins is different
    with pytest.raises(ValueError):
        vegas_instance.load_grid(file_name=tmp_filename)
    # Check that it fails if the number of dimensons is different
    jdict["dimensions"] = -4
    with open(tmp_filename, "w") as f:
        json.dump(jdict, f)
    with pytest.raises(ValueError):
        vegas_instance.load_grid(file_name=tmp_filename)


def test_PlainFlow():
    # We could use hypothesis here instead of this loop
    for mode in range(4):
        plain_instance = instance_and_compile(PlainFlow, mode)
        result = plain_instance.run_integration(n_iter)
        check_is_one(result)

    # Use the last instance to check that changing the number of events
    # don't change the result
    plain_instance.n_events = 2 * ncalls
    new_result = plain_instance.run_integration(n_iter)
    check_is_one(new_result)


def helper_rng_tester(sampling_function, n_events):
    """Ensure the random number generated have the correct shape
    Return the random numbers and the jacobian"""
    rnds, _, px = sampling_function(n_events)
    np.testing.assert_equal(rnds.shape, (n_events, dim))
    return rnds, px


def test_rng_generation(n_events=100):
    """Test that the random generation genrates the correct type of arrays"""
    plain_sampler_instance = instance_and_compile(PlainFlow)
    _, px = helper_rng_tester(plain_sampler_instance.generate_random_array, n_events)
    np.testing.assert_equal(px.numpy(), 1.0 / n_events)
    vegas_sampler_instance = instance_and_compile(VegasFlow)
    vegas_sampler_instance.run_integration(2)
    _, px = helper_rng_tester(vegas_sampler_instance.generate_random_array, n_events)
    np.testing.assert_equal(px.shape, (n_events,))
    # Test the wrappers
    p = plain_sampler(example_integrand, dim, n_events, training_steps=2, return_class=True)
    _ = helper_rng_tester(p.generate_random_array, n_events)
    v = vegas_sampler(example_integrand, dim, n_events, training_steps=2)
    _ = helper_rng_tester(v, n_events)


def test_VegasFlowPlus_ADAPTIVE_SAMPLING():
    """Test Vegasflow with Adaptive Sampling on (the default)"""
    for mode in range(4):
        vflowplus_instance = instance_and_compile(VegasFlowPlus, mode)
        result = vflowplus_instance.run_integration(n_iter)
        check_is_one(result)


def test_VegasFlowPlus_NOT_ADAPTIVE_SAMPLING():
    """Test Vegasflow with Adaptive Sampling off (non-default)"""
    vflowplus_instance = VegasFlowPlus(dim, ncalls, adaptive=False)
    vflowplus_instance.compile(example_integrand)
    result = vflowplus_instance.run_integration(n_iter)
    check_is_one(result)
