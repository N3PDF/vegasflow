"""
    Example: basic integration

    Basic example using the vegas_wrapper helper
"""

from vegasflow import VegasFlow, float_me
import time
import numpy as np
import tensorflow as tf
from vegasflow.vflow import vegas_wrapper
from vegasflow.plain import plain_wrapper 


# MC integration setup
dim = 4
ncalls = np.int32(1e5)
n_iter = 5


def symgauss(xarr):
    """symgauss test function"""
    n_dim = xarr.shape[-1]
    a = float_me(0.1)
    n100 = float_me(100 * n_dim)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


if __name__ == "__main__":
    """Testing several different integrations"""
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    vegas_instance = VegasFlow(dim, ncalls, simplify_signature=True)
    vegas_instance.compile(symgauss)
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")
