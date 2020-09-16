""" Test the utilities """

import numpy as np
import tensorflow as tf
import pytest

from vegasflow.configflow import DTYPEINT
from vegasflow.utils import consume_array_into_indices

from vegasflow.utils import generate_condition_function


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

def util_check(np_mask, tf_mask, tf_ind):
    np.testing.assert_array_equal(np_mask, tf_mask)
    np.testing.assert_array_equal(np_mask.nonzero()[0], tf_ind)

def test_generate_condition_function():
    """ Tests generate_condition_function and its errors """
    masks = 4 # Always > 2
    vals = 15
    np_masks = np.random.randint(2, size=(masks, vals), dtype=np.bool)
    tf_masks = [tf.constant(i, dtype=tf.bool) for i in np_masks]
    # Generate the functions for and and or
    f_and = generate_condition_function(masks, 'and')
    f_or = generate_condition_function(masks, 'or')
    # Get the numpy and tf results
    np_ands = np.all(np_masks, axis=0)
    np_ors = np.any(np_masks, axis=0)
    tf_ands, idx_ands = f_and(*tf_masks)
    tf_ors, idx_ors = f_or(*tf_masks)
    # Check the values are the same
    util_check(np_ands, tf_ands, idx_ands)
    util_check(np_ors, tf_ors, idx_ors)
    # Check a combination
    f_comb = generate_condition_function(3, ['and', 'or'])
    np_comb = np_masks[0] & np_masks[1] | np_masks[2]
    tf_comb, idx_comb = f_comb(*tf_masks[:3])
    util_check(np_comb, tf_comb, idx_comb)
    # Check failures
    with pytest.raises(ValueError):
        generate_condition_function(1, 'and')
    with pytest.raises(ValueError):
        generate_condition_function(5, 'bad_condition')
    with pytest.raises(ValueError):
        generate_condition_function(5, ['or', 'and'])
    with pytest.raises(ValueError):
        generate_condition_function(3, ['or', 'bad_condition'])
