"""
    Example: basic integration

    Basic example using the vegas_wrapper helper
"""

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from vegasflow.configflow import DTYPE, run_eager
run_eager(True)

import time
import numpy as np
from vegasflow.vflow import vegas_wrapper
from vegasflow.plain import plain_wrapper 
from vegasflow.rtbm import rtbm_wrapper


# MC integration setup
dim = 4
ncalls = np.int32(1e5)
n_iter = 3


@tf.function
def symgauss(xarr, n_dim=None, **kwargs):
    """symgauss test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


if __name__ == "__main__":
    """Testing several different integrations"""
    print(f"RTBM MC, ncalls={ncalls}:")
    start = time.time()
    ncalls = ncalls
    r = rtbm_wrapper(symgauss, dim, n_iter, ncalls)
    end = time.time()
    print(f"RTBM took: time (s): {end-start}")

    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    ncalls = ncalls
    r = vegas_wrapper(symgauss, dim, n_iter, ncalls)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")

#     print(f"Plain MC, ncalls={ncalls}:")
#     start = time.time()
#     r = plain_wrapper(symgauss, dim, n_iter, ncalls)
#     end = time.time()
#     print(f"Plain took: time (s): {end-start}")
