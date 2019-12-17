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
    'ncalls': np.int32(2e6),
    'dim': DIM
}

@tf.function
def MC_INTEGRAND(xarr, n_dim=None):
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
