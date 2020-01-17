# Place your function here
import time
import numpy as np
import tensorflow as tf
from vegasflow.configflow import DTYPE, DTYPEINT
from vegasflow.vflow import vegas_wrapper


# MC integration setup
dim = 4
ncalls = np.int32(1e6)
n_iter = 5


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


if __name__ == "__main__":
    """Testing a basic integration"""
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    r = vegas_wrapper(lepage, dim, n_iter, ncalls)
    end = time.time()
    print(f"time (s): {end-start}")
