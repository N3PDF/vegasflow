""" Test the utilities """

import numpy as np
import tensorflow as tf

from vegasflow.configflow import DTYPEINT
from vegasflow.utils import consume_array_into_indices


def test_consume_array_into_indices():
    # Select the size
    size_in = np.random.randint(5, 100)
    size_out = np.random.randint(1, size_in-3)
    # Generate the input array and the indices
    input_array = np.random.rand(size_in)
    indices = np.random.randint(0, size_out, size=size_in)
    # Make them into TF
    tf_input = tf.constant(input_array)
    tf_indx = tf.constant(indices.reshape(-1,1), dtype=DTYPEINT)
    result = consume_array_into_indices(tf_input, tf_indx, size_out)
    # Check that no results were lost
    np.testing.assert_almost_equal(np.sum(input_array), np.sum(result))
    # Check that the arrays in numpy produce the same in numpy
    check_result = np.zeros(size_out)
    for val, i in zip(input_array, indices):
        check_result[i] += val
    np.testing.assert_allclose(check_result, result)
