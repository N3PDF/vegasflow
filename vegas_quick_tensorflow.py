#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup, DTYPE, DTYPEINT
import numpy as np
import tensorflow as tf

BINS_MAX = 50
ALPHA = 1.5


# Define some constants
n_dim = setup["dim"]


def int_me(i):
    return tf.constant(i, dtype=DTYPEINT)


def float_me(i):
    return tf.constant(i, dtype=DTYPE)


ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)

shape_rn = tf.TensorSpec(shape=(None, n_dim), dtype=DTYPE)
shape_sub = tf.TensorSpec(shape=(n_dim, BINS_MAX), dtype=DTYPE)


@tf.function(input_signature=[shape_rn, shape_sub])
def generate_random_array(rnds, divisions):
    """
        Generates the Vegas random array for any number of events

        Parameters
        ----------
            rnds: array shaped (None, n_dim)
            divisions: array shaped (n_dim, BINS_MAX)

        Returns
        -------
            x: array (None, n_dim)
            div_index: array (None, n_dim)
            w: array (None,)
    """
    reg_i = fzero
    reg_f = fone
    # Get the corresponding random number
    xn = BINS_MAX * (fone - rnds)
    int_xn = tf.maximum(
        tf.cast(0, DTYPEINT), tf.minimum(tf.cast(xn, DTYPEINT), BINS_MAX)
    )
    # In practice int_xn = int(xn)-1 unless xn < 1
    aux_rand = xn - tf.cast(int_xn, dtype=DTYPE)
    # Now get the indices that will be used with this subdivision
    # If the index is 0, we cannot get the index-1 so...
    ind_f = tf.transpose(int_xn)
    ind_i = tf.maximum(ind_f - 1, 0)
    gather_f = tf.gather(divisions, ind_f, batch_dims=1)
    gather_i_tmp = tf.gather(divisions, ind_i, batch_dims=1)
    # Now the ones that had a "fake 0" need to be set to 0
    ind_is_0 = tf.equal(ind_f, 0)
    gather_i = tf.where(ind_is_0, fzero, gather_i_tmp)
    # Now compute the random number for this dimension
    x_ini = tf.transpose(gather_i)
    xdelta = tf.transpose(gather_f) - x_ini
    rand_x = x_ini + xdelta * aux_rand
    x = reg_i + rand_x * (reg_f - reg_i)
    weights = tf.reduce_prod(xdelta * BINS_MAX, axis=1)
    return x, int_xn, weights


@tf.function
def quick_integrand(xarr, n_dim=None):
    """Le page test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


@tf.function
def refine_grid_per_dimension(t_res_sq, subdivisions):
    """
        Modifies the boundaries for the vegas grid for a given dimension

        Parameters
        ----------
            `t_res_sq`: tensor
                array of results squared per bin
            `subdivision`: tensor
                current boundaries for the grid

        Returns
        -------
            `new_divisions`: tensor
                array with the new boundaries of the grid
    """
    # Define some constants
    paddings = tf.constant([[1, 1],])
    tmp_meaner = tf.fill([BINS_MAX - 2,], float_me(3.0))
    meaner = tf.pad(tmp_meaner, paddings, constant_values=2.0)
    # Pad the vector of results
    res_padded = tf.pad(t_res_sq, paddings)
    # First we need to smear out the array of results squared
    smeared_tensor_tmp = tf.math.accumulate_n(
        (res_padded[1:-1], res_padded[2:], res_padded[:-2])
    )
    smeared_tensor = tf.maximum(smeared_tensor_tmp / meaner, float_me(1e-30))
    # Now we refine the grid according to
    # journal of comp phys, 27, 192-203 (1978) G.P. Lepage
    sum_t = tf.reduce_sum(smeared_tensor)
    log_t = tf.math.log(smeared_tensor)
    aux_t = (1.0 - smeared_tensor / sum_t) / (tf.math.log(sum_t) - log_t)
    wei_t = tf.pow(aux_t, ALPHA)
    ave_t = tf.reduce_sum(wei_t) / BINS_MAX

    ###### Auxiliary functions for the while loop
    @tf.function
    def while_check(bin_weight, *args):
        """ Checks whether the bin has enough weight
        to beat the average """
        return bin_weight < ave_t

    @tf.function
    def while_body(bin_weight, n_bin, cur, prev):
        """ Fills the bin weight until it surpassed the avg
        once it's done, returns the limits of the last bin """
        n_bin += 1
        bin_weight += wei_t[n_bin]
        prev = cur
        cur = subdivisions[n_bin]
        return bin_weight, n_bin, cur, prev
    ###########################

    # And now resize all bins
    new_bins = []
    # Auxiliary variables
    bin_weight = fzero
    n_bin = -1
    cur = fzero
    prev = fzero
    for _ in range(BINS_MAX - 1):
        bin_weight, n_bin, cur, prev = tf.while_loop(
            while_check,
            while_body,
            (bin_weight, n_bin, cur, prev),
            parallel_iterations=1,
        )
        bin_weight -= ave_t
        delta = (cur - prev) * bin_weight / wei_t[n_bin]
        new_bins.append(cur - delta)
    new_bins.append(fone)

    new_divisions = tf.stack(new_bins)
    return new_divisions


@tf.function
def consume_results(res2, indices):
    """ Consumes the results squared to
    generate the array-per-bin of results """
    all_bins = tf.range(BINS_MAX, dtype=DTYPEINT)
    eq = tf.transpose(tf.equal(indices, all_bins))
    res_tmp = tf.where(eq, res2, fzero)
    arr_res2 = tf.reduce_sum(res_tmp, axis=1)
    return arr_res2


def vegas(n_dim, n_iter, n_events):
    """
    # Arguments in:
        n_dim: number of dimensions
        n_iter: number of iterations
        n_events: number of events per iteration


    # Returns:
        - integral value
        - error
    """
    # Initialize constant variables, we can use python numbers here
    xjac = 1.0 / n_events

    # Initialize variable variables
    subdivision_np = np.linspace(1 / BINS_MAX, 1, BINS_MAX)
    divisions_np = subdivision_np.repeat(n_dim).reshape(-1, n_dim).T
    divisions = tf.Variable(divisions_np, dtype=DTYPE)

    # "allocate" arrays
    all_results = []

    # Loop of iterations
    for iteration in range(n_iter):
        start = time.time()

        # Generate all random number for this iteration
        rnds = tf.random.uniform((n_events, n_dim), minval=0, maxval=1, dtype=DTYPE)

        # Pass them through the Vegas digestion
        x, ind, w = generate_random_array(rnds, divisions)

        # Now compute the integrand
        tmp = xjac * w * quick_integrand(x, n_dim=n_dim)

        # Compute the final result for this iteration
        res = tf.reduce_sum(tmp)
        # Compute the error
        tmp2 = tf.square(tmp)
        res2 = tf.reduce_sum(tmp2)
        err_tmp2 = (n_events * res2 - tf.square(res)) / (n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))

        # Rebin Vegas
        for j in range(n_dim):
            arr_res2 = consume_results(tmp2, ind[:, j : j + 1])
            new_divisions = refine_grid_per_dimension(arr_res2, divisions[j, :])
            divisions[j, :].assign(new_divisions)

        # Print the results
        end = time.time()
        print(f"Results for {iteration} {res:.5f} +/- {sigma:.5f} (took {end-start} s)")
        all_results.append((res, sigma))

    # Compute the final results
    aux_res = 0.0
    weight_sum = 0.0
    for result in all_results:
        res = result[0]
        sigma = result[1]
        wgt_tmp = 1.0 / pow(sigma, 2)
        aux_res += res * wgt_tmp
        weight_sum += wgt_tmp

    final_result = aux_res / weight_sum
    sigma = np.sqrt(1.0 / weight_sum)
    print(f" > Final results: {final_result.numpy()} +/- {sigma}")
    return final_result, sigma


if __name__ == "__main__":
    """Testing a basic integration"""
    ncalls = setup["ncalls"]
    xlow = setup["xlow"]
    xupp = setup["xupp"]

    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    r = vegas(n_dim, 5, ncalls)
    end = time.time()
    print(f"time (s): {end-start}")
