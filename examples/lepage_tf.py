# Place your function here
import time
import numpy as np
import tensorflow as tf
from vegasflow.configflow import DTYPE, DTYPEINT
from vegasflow.vflow import vegas_wrapper
from vegasflow.plain import plain_wrapper 


# MC integration setup
dim = 4
ncalls = np.int32(1e5)
n_iter = 5


@tf.function
def lepage(xarr, n_dim=None, **kwargs):
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
    """Testing several different integrations"""
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    r = vegas_wrapper(lepage, dim, n_iter, ncalls)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")

    print(f"Plain MC, ncalls={ncalls}:")
    start = time.time()
    r = plain_wrapper(lepage, dim, n_iter, ncalls)
    end = time.time()
    print(f"Plain took: time (s): {end-start}")
