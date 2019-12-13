# Place your function here
import numpy as np
import tensorflow as tf

DIM = 4
DTYPE = tf.float64
DTYPEINT = tf.int32

# MC integration setup
setup = {
    'xlow': np.array([0]*DIM, dtype=np.float32),
    'xupp': np.array([1]*DIM, dtype=np.float32),
    'ncalls': np.int32(1e7),
    'dim': DIM
}


# test function
@tf.function
def MC_INTEGRAND(xarr):
    """Le page test function"""
    a = 0.1
    n = DIM
    n100 = 100*n
    pref = np.power(1.0/a/np.sqrt(np.pi), n)
    coef = tf.cast(tf.reduce_sum(tf.range(n100+1)), dtype=DTYPE)
    coef +=  tf.reduce_sum(tf.square( (xarr-tf.constant(1.0/2.0))/a), axis=0)
    coef -= (n100+1)*n100/2.0
    return pref*tf.exp(-coef)
