"""
    Miscellaneous tests that don't really fit anywhere else
"""
import pytest
import numpy as np

from vegasflow import VegasFlow, VegasFlowPlus, PlainFlow
import tensorflow as tf

from .test_algs import instance_and_compile, check_is_one


def _vector_integrand(xarr, weight=None):
    res = tf.square((xarr - 1.0) ** 2)
    return tf.exp(-res) / 0.845


def _wrong_integrand(xarr):
    """Integrand with the wrong output shape"""
    return tf.reduce_sum(xarr)


def _simple_integrand(xarr):
    """Integrand f(x) = x"""
    return tf.reduce_prod(xarr, axis=1)


def _simple_integral(xmin, xmax):
    """Integated version of simple_ingrand"""
    xm = np.array(xmin) ** 2 / 2.0
    xp = np.array(xmax) ** 2 / 2.0
    return np.prod(xp - xm)


def _wrong_vector_integrand(xarr):
    """Vector integrand with the wrong output shape"""
    return tf.transpose(xarr)


@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("alg", [VegasFlow, PlainFlow])
def test_working_vectorial(alg, mode):
    """Check that the algorithms that accept integrating vectorial functions can really do so"""
    inst = instance_and_compile(alg, mode=mode, integrand_function=_vector_integrand)
    result = inst.run_integration(2)
    check_is_one(result, sigmas=4)


@pytest.mark.parametrize("alg", [VegasFlowPlus])
def test_notworking_vectorial(alg):
    """Check that the algorithms that do not accept vectorial functions fail appropriately"""
    with pytest.raises(NotImplementedError):
        _ = instance_and_compile(alg, integrand_function=_vector_integrand)


def test_check_wrong_main_dimension():
    """Check that an error is raised  by VegasFlow
    if the main dimension is > than the dimensionality of the integrand"""
    inst = VegasFlow(3, 100, main_dimension=5)
    with pytest.raises(ValueError):
        inst.compile(_vector_integrand)


@pytest.mark.parametrize("wrong_fun", [_wrong_vector_integrand, _wrong_integrand])
def test_wrong_shape(wrong_fun):
    """Check that an error is raised by the compilation if the integrand has the wrong shape"""
    with pytest.raises(ValueError):
        _ = instance_and_compile(PlainFlow, integrand_function=wrong_fun)


@pytest.mark.parametrize("alg", [PlainFlow, VegasFlow, VegasFlowPlus])
def test_integration_limits(alg, ncalls=int(1e4)):
    """Test an integration where the integration limits are modified"""
    dims = np.random.randint(1, 5)
    xmin = -1.0 + np.random.rand(dims) * 2.0
    xmax = 3.0 + np.random.rand(dims)
    inst = alg(dims, ncalls, xmin=xmin, xmax=xmax)
    inst.compile(_simple_integrand)
    result = inst.run_integration(5)
    expected_result = _simple_integral(xmin, xmax)
    check_is_one(result, target_result=expected_result)


def test_integration_limits_checks():
    """Test that the errors for wrong limits actually work"""
    # use hypothesis to check other corner cases
    with pytest.raises(ValueError):
        PlainFlow(1, 10, xmin=[10], xmax=[1])
    with pytest.raises(ValueError):
        PlainFlow(1, 10, xmin=[10])
    with pytest.raises(ValueError):
        PlainFlow(1, 10, xmax=[10])
    with pytest.raises(ValueError):
        PlainFlow(2, 10, xmin=[0], xmax=[1])
    with pytest.raises(ValueError):
        PlainFlow(2, 10, xmin=[0, 1], xmax=[1])
