"""
    Example: basic integration

    Basic example using the vegas_wrapper helper
"""

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
from vegasflow import run_eager, float_me
run_eager(True)

import time
import numpy as np
from vegasflow.vflow import vegas_wrapper
from vegasflow.rtbm import RTBMFlow


# MC integration setup
dim = 3
ncalls = np.int32(1e3)
n_iter = 5
tf_pi = float_me(np.pi)


@tf.function
def sin_fun(xarr, **kwargs):
    """symgauss test function"""
    res = tf.pow(tf.sin(xarr*tf_pi*4.0),2)
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
    rtbm = RTBMFlow(n_dim=dim, n_events=ncalls, gaussian=True, train=True, n_hidden=2)
    rtbm.compile(sin_fun)
    rtbm.run_integration(4)
    end = time.time()
    print(f"RTBM took: time (s): {end-start}")

#     print(f"Result computed by Vegas: {r}")
#     print(f"Result computed by RTBM: {rt}")


