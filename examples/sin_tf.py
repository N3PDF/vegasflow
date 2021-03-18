"""
    Example: basic integration

    Basic example using the vegas_wrapper helper
"""

from vegasflow import run_eager, float_me
import tensorflow as tf
run_eager(True)

import time
import numpy as np
from vegasflow.vflow import vegas_wrapper
from vegasflow.rtbm import RTBMFlow


# MC integration setup
dim = 1
ncalls = np.int32(1e4)
n_iter = 5
tf_pi = float_me(np.pi)


@tf.function
def sin_fun(xarr, **kwargs):
    """symgauss test function"""
    res = tf.pow(tf.sin(xarr*tf_pi),2)
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
    rtbm = RTBMFlow(n_dim=dim, n_events=ncalls, train=True, n_hidden=1)
    rtbm.compile(sin_fun)
    rt = rtbm.run_integration(4)
    end = time.time()
    print(f"RTBM took: time (s): {end-start}")

    print(f"Result computed by Vegas: {r}")
    print(f"Result computed by RTBM: {rt}")


