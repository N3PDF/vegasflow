"""
    Example: eager mode integrand

    Running a non-tensorflow integrand using VegasFlow
"""

from vegasflow.configflow import DTYPE, DTYPEINT
import time
import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)
from vegasflow.vflow import vegas_wrapper


# MC integration setup
dim = 4
ncalls = np.int32(1e5)
n_iter = 5


def symgauss(xarr, n_dim=None, **kwargs):
    """symgauss test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = 0.1
    pref = pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = np.sum(np.arange(1, 101))
    # Tensorflow variable will be casted down by numpy
    # you can directly access their numpy representation with .numpy()
    xarr_sq = np.square((xarr - 1.0 / 2.0) / a)
    coef += np.sum(xarr_sq, axis=1)
    coef -= 100.0 * 101.0 / 2.0
    return pref * np.exp(-coef)


if __name__ == "__main__":
    """Testing several different integrations"""
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    ncalls = 10 * ncalls
    r = vegas_wrapper(symgauss, dim, n_iter, ncalls, compilable=True)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")
