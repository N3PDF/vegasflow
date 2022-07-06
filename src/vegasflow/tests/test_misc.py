"""
    Miscellaneous tests that don't really fit anywhere else
"""
import pytest
from vegasflow import VegasFlow, VegasFlowPlus, PlainFlow
import tensorflow as tf

from .test_algs import instance_and_compile, check_is_one


def _multidim_integrand(xarr, weight=None):
    res = tf.square((xarr - 1.0) ** 2)
    return tf.exp(-res) / 0.845


def test_working_vectorial():
    """Check that the algorithms that accept integrating vectorial functions can really do so"""
    for alg in [VegasFlow, PlainFlow]:
        for mode in range(4):
            inst = instance_and_compile(alg, mode=mode, integrand_function=_multidim_integrand)
            result = inst.run_integration(2)
            check_is_one(result)


def test_notworking_vectorial():
    """Check that the algorithms that do not accept vectorial functions fail appropriately"""
    for alg in [VegasFlowPlus]:
        with pytest.raises(NotImplementedError):
            inst = instance_and_compile(alg, integrand_function=_multidim_integrand)
