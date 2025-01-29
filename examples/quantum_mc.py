"""
Run integration with qibo as the backend
"""

from vegasflow.quantum import quantum_wrapper
from vegasflow import float_me
import time
import numpy as np
import tensorflow as tf


# MC integration setup
dim = 2
ncalls = int(1e2)
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
    result = quantum_wrapper(symgauss, dim, n_iter, ncalls)
    end = time.time()
