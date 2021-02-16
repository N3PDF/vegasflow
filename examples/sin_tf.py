"""
    Example: basic integration

    Basic example using the vegas_wrapper helper
"""

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
from vegasflow import run_eager, float_me
run_eager(True)

import time
import numpy as np
from vegasflow.vflow import vegas_wrapper
from vegasflow.rtbm import rtbm_wrapper


# MC integration setup
dim = 3
ncalls = np.int32(1e4)
n_iter = 12
tf_pi = float_me(np.pi)


@tf.function
def sin_fun(xarr, **kwargs):
    """symgauss test function"""
    res = tf.pow(tf.sin(2.0*xarr*tf_pi),2)
    return tf.reduce_prod(res, axis=1)


if __name__ == "__main__":
    """Testing several different integrations"""
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    ncalls = ncalls
    r = vegas_wrapper(sin_fun, dim, n_iter, ncalls)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")

    print(f"RTBM MC, ncalls={ncalls}:")
    start = time.time()
    ncalls = ncalls
    rt = rtbm_wrapper(sin_fun, dim, n_iter, ncalls)
    end = time.time()
    print(f"RTBM took: time (s): {end-start}")

    print(f"Result computed by Vegas: {r}")
    print(f"Result computed by RTBM: {rt}")


