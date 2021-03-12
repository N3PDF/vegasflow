"""
    Retracing example in VegasFlowPlus
"""

from vegasflow import VegasFlowPlus, VegasFlow, StratifiedFlow, PlainFlow
from vegasflow.configflow import DTYPE, DTYPEINT,run_eager, float_me
import time
import numpy as np
import tensorflow as tf
#import pineappl

# MC integration setup
dim = 2
ncalls = np.int32(1e3)
n_iter = 5

#@tf.function(input_signature=[
#                    tf.TensorSpec(shape=[None,dim], dtype=DTYPE),
#                    tf.TensorSpec(shape=[], dtype=DTYPE),
#                   tf.TensorSpec(shape=[None], dtype=DTYPE)
#                ]
#            )
def symgauss(xarr, n_dim=None,weight=None, **kwargs):
    """symgauss test function"""
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

    #run_eager()
    vegas_instance = VegasFlowPlus(dim, ncalls,adaptive=True)
    vegas_instance.compile(symgauss)
    vegas_instance.run_integration(n_iter)
