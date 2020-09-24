# Asian option like integral
# From https://doi.org/10.1016/S0885-064X(03)00003-7
# Equation (14)

from vegasflow.configflow import DTYPE, DTYPEINT
from vegasflow import vegas_wrapper

from tensorflow.python.ops.distributions.special_math import ndtri
import time
import numpy as np
import tensorflow as tf

# MC integration setup
d =  16
ncalls = np.int32(2e6)
n_iter = 5

# Asian Option setup
T = tf.constant(1.0, dtype=DTYPE)
r = tf.constant(1.0, dtype=DTYPE)
sigma = tf.constant(1.0, dtype=DTYPE)
sigma2 = tf.square(sigma)
S0 = tf.constant(1.0, dtype=DTYPE)
t = tf.constant(tf.ones(shape=(ncalls,), dtype=DTYPE))
sqrtdt = tf.constant(1.0, dtype=DTYPE)
K = tf.constant(0.0, dtype=DTYPE)
e = tf.exp(tf.constant(-1*r*T, dtype=DTYPE))
zero = tf.constant(0.0, dtype=DTYPE)


@tf.function
def example_integrand(xarr, n_dim=None, **kwargs):
    """Asian options test function"""
    sum1 = tf.reduce_sum(ndtri(xarr), axis=1)
    a = S0 * tf.exp((r-sigma2/2) + sigma*sqrtdt*sum1)
    arg = 1 / d * tf.reduce_sum(a)
    return e*tf.maximum(zero, arg-K)


if __name__ == "__main__":
    """Testing a basic integration"""
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    r = vegas_wrapper(example_integrand, d, n_iter, ncalls)
    end = time.time()
    print(f"time (s): {end-start}")
