"""
Define some constants, header style
"""
# Most of this can be moved to a yaml file without loss of generality
import tensorflow as tf

# Define the tensorflow numberic types
DTYPE = tf.float64
DTYPEINT = tf.int32

# Define some default parameters for Vegas
BINS_MAX = 50
ALPHA = 1.5

# Limit the logistics of the itnegration
EVENTS_LIMIT = int(1e6)

# Create wrappers in order to have numbers of the correct type
def int_me(i):
    return tf.constant(i, dtype=DTYPEINT)


def float_me(i):
    return tf.constant(i, dtype=DTYPE)


ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)
