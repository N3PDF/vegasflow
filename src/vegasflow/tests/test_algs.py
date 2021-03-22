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
import tensorflow as tf
from vegasflow.configflow import DTYPE
from vegasflow.vflow import VegasFlow
from vegasflow.plain import PlainFlow

# Test setup
dim = 2
ncalls = np.int32(1e5)
n_iter = 4


@tf.function
def example_integrand(xarr, n_dim=None, weight=None):
    """ Example function that integrates to 1 """
    if n_dim is None:
        n_dim = xarr.shape[0]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


def instance_and_compile(Integrator):
    """ Wrapper for convenience """
    int_instance = Integrator(dim, ncalls)
    int_instance.compile(example_integrand)
    return int_instance


def check_is_one(result, sigmas=3):
    """ Wrapper for convenience """
    res = result[0]
    err = result[1] * sigmas
    # Check that it passes by {sigmas} number of sigmas
    np.testing.assert_allclose(res, 1.0, atol=err)


def test_VegasFlow():
    vegas_instance = instance_and_compile(VegasFlow)
    _ = vegas_instance.run_integration(n_iter)
    vegas_instance.freeze_grid()
    result = vegas_instance.run_integration(n_iter)
    check_is_one(result)


def test_VegasFlow_save_grid():
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
    plain_instance = instance_and_compile(PlainFlow)
    result = plain_instance.run_integration(n_iter)
    check_is_one(result)


def test_rng_generation(n_events=100):
    plain_sampler = instance_and_compile(PlainFlow)
    rnds, _, px = plain_sampler.generate_random_array(n_events)
    np.testing.assert_equal(rnds.shape, (100,2))
    np.testing.assert_equal(px.numpy(), 1.0/n_events)
    vegas_sampler = instance_and_compile(VegasFlow)
    vegas_sampler.run_integration(2)
    rnds, _, px = vegas_sampler.generate_random_array(n_events)
    np.testing.assert_equal(rnds.shape, (100,2))
    np.testing.assert_equal(px.shape, (100,))
