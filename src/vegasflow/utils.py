"""
    This module contains tensorflow_compiled utilities
"""

import tensorflow as tf

from vegasflow.configflow import DTYPE, DTYPEINT, float_me, fzero, int_me


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
    ]
)
def consume_array_into_indices(input_arr, indices, result_size):
    """
    Accumulate the input tensor `input_arr` into an output tensor of
    size `result_size`. The accumulation occurs according to the array
    of `indices`.

    For instance, `input_array` = [a,b,c,d] and vector column `indices` = [[0,1,0,0]].T
    (with `result_size` = 2) will result in a final_result: (a+c+d, b)

    Parameters
    ----------
    `input_arr`
        Array of results to be consumed
    `indices`
        Indices of the bins in which to accumulate the input array
    `result_size`
        size of the output array

    Returns
    -------
    `final_result`
        Array of size `result_size`
    """
    all_bins = tf.range(result_size, dtype=DTYPEINT)
    eq = tf.transpose(tf.equal(indices, all_bins))
    res_tmp = tf.where(eq, input_arr, fzero)
    final_result = tf.reduce_sum(res_tmp, axis=1)
    return final_result


def py_consume_array_into_indices(input_arr, indices, result_size):
    """
    Python interface wrapper for ``consume_array_into_indices``.
    It casts the possible python-object input into the correct tensorflow types.
    """
    return consume_array_into_indices(float_me(input_arr), int_me(indices), int_me(result_size))


def generate_condition_function(n_mask, condition="and"):
    """Generates a function that takes a number of masks
    and returns a combination of all n_masks for the given condition.

    It is possible to pass a list of allowed conditions, in that case
    the length of the list should be n_masks - 1 and will be apply
    sequentially.

    Note that for 2 masks you can directly use & and |

    >>> from vegasflow.utils import generate_condition_function
    >>> import tensorflow as tf
    >>> f_cond = generate_condition_function(2, condition='or')
    >>> t_1 = tf.constant([True, False, True])
    >>> t_2 = tf.constant([False, False, True])
    >>> full_mask, indices = f_cond(t_1, t_2)
    >>> print(f"{full_mask=}\n{indices=}")
    full_mask=<tf.Tensor: shape=(3,), dtype=bool, numpy=array([ True, False,  True])>
    indices=<tf.Tensor: shape=(2, 1), dtype=int32, numpy=
    array([[0],
        [2]], dtype=int32)>
    >>> f_cond = generate_condition_function(3, condition=['or', 'and'])
    >>> t_1 = tf.constant([True, False, True])
    >>> t_2 = tf.constant([False, False, True])
    >>> t_3 = tf.constant([True, False, False])
    >>> full_mask, indices = f_cond(t_1, t_2, t_3)
    >>> print(f"{full_mask=}\n{indices=}")
    full_mask=<tf.Tensor: shape=(3,), dtype=bool, numpy=array([ True, False, False])>
    indices=<tf.Tensor: shape=(1, 1), dtype=int32, numpy=array([[0]], dtype=int32)>



    Parameters
    ----------
    `n_mask`: int
        Number of masks the function should accept
    `condition`: str (default='and')
        Condition to apply to all masks. Accepted values are: and, or

    Returns
    -------
    `condition_to_idx`: function
        function(*masks) -> full_mask, true indices
    """
    allowed_conditions = {"and": tf.math.logical_and, "or": tf.math.logical_or}
    allo = list(allowed_conditions.keys())

    # Check that the user is not asking for anything weird
    if n_mask < 2:
        raise ValueError("At least two masks needed to generate a wrapper")

    if isinstance(condition, str):
        if condition not in allowed_conditions:
            raise ValueError(f"Wrong condition {condition}, allowed values are {allo}")
        is_list = False
    else:
        if len(condition) != n_mask - 1:
            raise ValueError(f"Wrong number of conditions for {n_mask} masks: {len(condition)}")
        for cond in condition:
            if cond not in allowed_conditions:
                raise ValueError(f"Wrong condition {cond}, allowed values are {allo}")
        is_list = True

    def py_condition(*masks):
        """Receives a list of conditions and returns a result mask
        and the list of indices in which the result mask is True

        Returns
        -------
            `res`: tf.bool
                Mask that combines all masks
            `indices`: tf.int
                Indices in which `res` is True
        """
        if is_list:
            res = masks[0]
            for i, cond in enumerate(condition):
                res = allowed_conditions[cond](res, masks[i + 1])
        elif condition == "and":
            res = tf.math.reduce_all(masks, axis=0)
        elif condition == "or":
            res = tf.math.reduce_any(masks, axis=0)
        indices = int_me(tf.where(res))
        return res, indices

    signature = n_mask * [tf.TensorSpec(shape=[None], dtype=tf.bool)]

    condition_to_idx = tf.function(py_condition, input_signature=signature)
    return condition_to_idx
