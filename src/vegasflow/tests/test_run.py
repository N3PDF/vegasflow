""" Test a run with a simple function to make sure
everything works """
import numpy as np
import tensorflow as tf
from vegasflow.configflow import DTYPE
from vegasflow.vflow import VegasFlow

# Test setup
dim = 2
ncalls = np.int32(1e4)
n_iter = 3


def test_run():
    @tf.function
    def lepage(xarr, n_dim=None):
        """Lepage test function"""
        if n_dim is None:
            n_dim = xarr.shape[-1]
        a = tf.constant(0.1, dtype=DTYPE)
        n100 = tf.cast(100 * n_dim, dtype=DTYPE)
        pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
        coef = tf.reduce_sum(tf.range(n100 + 1))
        coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
        coef -= (n100 + 1) * n100 / 2.0
        return pref * tf.exp(-coef)

    vegas_instance = VegasFlow(dim, ncalls)
    vegas_instance.compile(lepage)
    result = vegas_instance.run_integration(n_iter)
    res = result[0]
    np.testing.assert_almost_equal(res, 1.000, decimal=2)
