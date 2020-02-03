"""
    This module contains tensorflow_compiled utilities
"""

import tensorflow as tf
from vegasflow.configflow import DTYPEINT, fzero

#@tf.function
def consume_array_into_indices(input_arr, indices, result_size):
    """ """
    all_bins = tf.range(result_size, dtype = DTYPEINT)
    eq = tf.transpose(tf.equal(indices, all_bins))
    res_tmp = tf.where(eq, input_arr, fzero)
    final_result = tf.reduce_sum(res_tmp, axis = 1)
    return final_result
