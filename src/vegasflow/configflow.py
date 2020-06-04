"""
Define some constants, header style
"""
# Most of this can be moved to a yaml file without loss of generality
import tensorflow as tf

# Define the tf.numberic types
DTYPE = tf.float64
DTYPEINT = tf.int32

# Define some default parameters for Vegas
BINS_MAX = 50
ALPHA = 1.5

# Set up the logistics of the integration
# Events Limit limits how many events are done in one single run of the event_loop
# set it lower if hitting memory problems
MAX_EVENTS_LIMIT = int(1e7)
# Select the list of devices to look for
DEFAULT_ACTIVE_DEVICES = ["GPU"]  # , 'CPU']

# Create wrappers in order to have numbers of the correct type
def int_me(i):
    """ Casts any interger to DTYPEINT """
    return tf.cast(i, dtype=DTYPEINT)


def float_me(i):
    """ Cast any float to DTYPE """
    return tf.cast(i, dtype=DTYPE)


ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)
