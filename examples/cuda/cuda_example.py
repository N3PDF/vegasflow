from vegasflow.configflow import DTYPE, DTYPEINT
import time
import numpy as np
import tensorflow as tf
from vegasflow.plain import plain_wrapper 

# MC integration setup
dim = 4
ncalls = np.int32(1e4)
n_iter = 5

integrand_module = tf.load_op_library('./integrand.so')

@tf.function
def wrapper_integrand(xarr, **kwargs):
    return integrand_module.integrand_op(xarr)

@tf.function
def fully_python_integrand(xarr, **kwargs):
    return tf.reduce_sum(xarr, axis=1)

if __name__ == "__main__":
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    ncalls = 10*ncalls
    r = plain_wrapper(wrapper_integrand, dim, n_iter, ncalls)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")
