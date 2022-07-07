"""
    Miscellaneous tests that don't really fit anywhere else
"""
import pytest
from vegasflow import VegasFlow, VegasFlowPlus, PlainFlow
import tensorflow as tf

from .test_algs import instance_and_compile, check_is_one


def _vector_integrand(xarr, weight=None):
    res = tf.square((xarr - 1.0) ** 2)
    return tf.exp(-res) / 0.845


def _wrong_integrand(xarr):
    """Integrand with the wrong output shape"""
    return tf.reduce_sum(xarr)


def _wrong_vector_integrand(xarr):
    """Vector integrand with the wrong output shape"""
    return tf.transpose(xarr)


def test_working_vectorial():
    """Check that the algorithms that accept integrating vectorial functions can really do so"""
    for alg in [VegasFlow, PlainFlow]:
        for mode in range(4):
            inst = instance_and_compile(alg, mode=mode, integrand_function=_vector_integrand)
            result = inst.run_integration(2)
            check_is_one(result)


def test_notworking_vectorial():
    """Check that the algorithms that do not accept vectorial functions fail appropriately"""
    for alg in [VegasFlowPlus]:
        with pytest.raises(NotImplementedError):
            _ = instance_and_compile(alg, integrand_function=_vector_integrand)


def test_check_wrong_main_dimension():
    """Check that an error is raised  by VegasFlow
    if the main dimension is > than the dimensionality of the integrand"""
    inst = VegasFlow(3, 100, main_dimension=5)
    with pytest.raises(ValueError):
        inst.compile(_vector_integrand)


def test_wrong_shape():
    """Check that an error is raised by the compilation if the integrand has the wrong shape"""
    for wrong_fun in [_wrong_vector_integrand, _wrong_integrand]:
        with pytest.raises(ValueError):
            _ = instance_and_compile(PlainFlow, integrand_function=wrong_fun)
