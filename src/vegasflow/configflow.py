"""
Define some constants, header style
"""
# Most of this can be moved to a yaml file without loss of generality
import os
import logging
import numpy as np

# Some global parameters
BINS_MAX = 50
ALPHA = 1.5
BETA = 0.75 # Vegas +
TECH_CUT = 1e-8

# Set up the logistics of the integration
# set the limits lower if hitting memory problems

# Events Limit limits how many events are done in one single run of the event_loop
MAX_EVENTS_LIMIT = int(1e6)
# Maximum number of evaluation per hypercube for VegasFlowPlus
MAX_NEVAL_HCUBE = int(1e4)

# Select the list of devices to look for
DEFAULT_ACTIVE_DEVICES = ["GPU"]  # , 'CPU']

# Log levels
LOG_DICT = {"0": logging.ERROR, "1": logging.WARNING, "2": logging.INFO, "3": logging.DEBUG}

# Read the VEGASFLOW environment variables
_log_level_idx = os.environ.get("VEGASFLOW_LOG_LEVEL")
_data_path = os.environ.get("VEGASFLOW_DATA_PATH")
_float_env = os.environ.get("VEGASFLOW_FLOAT", "64")
_int_env = os.environ.get("VEGASFLOW_INT", "32")


# Logging
_bad_log_warning = None
if _log_level_idx not in LOG_DICT:
    _bad_log_warning = _log_level_idx
    _log_level_idx = None

if _log_level_idx is None:
    # If no log level is provided, set some defaults
    _log_level = LOG_DICT["2"]
    _tf_log_level = LOG_DICT["0"]
else:
    _log_level = _tf_log_level = LOG_DICT[_log_level_idx]

# Configure logging
logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(_log_level)

# Create and format the log handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(_log_level)
_console_format = logging.Formatter("[%(levelname)s] (%(name)s) %(message)s")
_console_handler.setFormatter(_console_format)
logger.addHandler(_console_handler)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf

tf.get_logger().setLevel(_tf_log_level)

# set the precision type
if _float_env == "64":
    DTYPE = tf.float64
    FMAX = tf.constant(np.finfo(np.float64).max, dtype=DTYPE)
elif _float_env == "32":
    DTYPE = tf.float32
    FMAX = tf.constant(np.finfo(np.float32).max, dtype=DTYPE)
else:
    DTYPE = tf.float64
    FMAX = tf.constant(np.finfo(np.float64).max, dtype=DTYPE)
    logger.warning(f"VEGASFLOW_FLOAT={_float_env} not understood, defaulting to 64 bits")

if _int_env == "64":
    DTYPEINT = tf.int64
elif _int_env == "32":
    DTYPEINT = tf.int32
else:
    DTYPEINT = tf.int64
    logger.warning(f"VEGASFLOW_INT={_int_env} not understood, defaulting to 64 bits")


def run_eager(flag=True):
    """Wrapper around `run_functions_eagerly`
    When used no function is compiled
    """
    if tf.__version__ < "2.3.0":
        tf.config.experimental_run_functions_eagerly(flag)
    else:
        tf.config.run_functions_eagerly(flag)


FMAX = tf.constant(np.finfo(np.float64).max, dtype=DTYPE)
# The wrappers below transform tensors and array to the correct type
def int_me(i):
    """Cast the input to the `DTYPEINT` type"""
    return tf.cast(i, dtype=DTYPEINT)


def float_me(i):
    """Cast the input to the `DTYPE` type"""
    return tf.cast(i, dtype=DTYPE)


ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)
