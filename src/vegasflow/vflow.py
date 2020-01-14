#!/usr/env python
"""
    This module contains the VegasFlow class and all its auxuliary functions

    The main interfaces of this class are the class `VegasFlow` and the
    `vegas_wrapper`
"""
import numpy as np
import tensorflow as tf

from vegasflow.configflow import DTYPE, DTYPEINT, fone, fzero, float_me
from vegasflow.configflow import BINS_MAX, ALPHA
from vegasflow.monte_carlo import MonteCarloFlow


# Auxiliary functions for Vegas
@tf.function
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
    smeared_tensor_tmp = res_padded[1:-1] + res_padded[2:] + res_padded[:-2]
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


####### VegasFlow
class VegasFlow(MonteCarloFlow):
    """
    Implementation of the adaptative sampling algorithm Vegas
    """

    def __init__(self, n_dim, n_events, train = True):
        super().__init__(n_dim, n_events)

        # If training is True, the grid will be changed after every iteration
        # otherwise it will be frozen
        self.train = train

        # Initialize grid
        subdivision_np = np.linspace(1 / BINS_MAX, 1, BINS_MAX)
        divisions_np = subdivision_np.repeat(n_dim).reshape(-1, n_dim).T
        self.divisions = tf.Variable(divisions_np, dtype=DTYPE)

    def freeze_grid(self):
        self.train = False

    def unfreeze_grid(self):
        self.train = True

    def generate_accumulators(self):
        res = fzero
        res2 = fzero
        self.arr_res2 = tf.Variable(tf.zeros((self.n_dim, BINS_MAX), dtype = DTYPE))
        return res, res2

    @tf.function
    def post_integration(self):
        for j in range(self.n_dim):
            new_divisions = refine_grid_per_dimension(self.arr_res2[j,:], self.divisions[j, :])
            self.divisions[j, :].assign(new_divisions)

    def _run_event(self, integrand, ncalls = None, acc = None):
        """ Runs one event of Vegas"""
        if acc is None:
            raise ValueError("The _run_event method of Vegas needs the result from generate_accumulators")

        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        n_dim = self.n_dim
        divisions = self.divisions
        xjac = self.xjac

        # Generate all random number for this iteration
        rnds = tf.random.uniform((n_events, n_dim), minval=0, maxval=1, dtype=DTYPE)

        # Pass them through the Vegas digestion
        x, ind, w = generate_random_array(rnds, divisions)

        # Now compute the integrand
        tmp = xjac * w * integrand(x, n_dim=n_dim)
        tmp2 = tf.square(tmp)

        # Compute the final result for this sub-iteration and accumulate it
        res, res2 = acc
        res += tf.reduce_sum(tmp)
        res2 += tf.reduce_sum(tmp2)

        if self.train:
            # Rebin Vegas
            for j in range(self.n_dim):
                new_arr2 = consume_results(tmp2, ind[:, j:j+1]) # + self.arr_res2[j, :] # TODO ????????
                self.arr_res2[j, :].assign( new_arr2 )

        return res, res2

    def _run_iteration(self):
        """ Runs one iteration of the Vegas integrator """
        # Generate the accumulators
        acc_res, acc_res2 = self.generate_accumulators()
        # Compute the result
        res, res2 = self.run_event(acc = (acc_res, acc_res2))
        # Compute the error
        err_tmp2 = (self.n_events * res2 - tf.square(res)) / (self.n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))
        # Act post integration
        self.post_integration()
        return res, sigma


def vegas_wrapper(integrand, n_dim, n_iter, total_n_events):
    """ Convenience wrapper

    Parameters
    ----------
        `integrand`: tf.function
        `n_dim`: number of dimensions
        `n_iter`: number of iterations
        `n_events`: number of events per iteration

    Returns
    -------
        `final_result`: integral value
        `sigma`: monte carlo error
    """
    vegas_instance = VegasFlow(n_dim, total_n_events)
    vegas_instance.compile(integrand)
    return vegas_instance.run_integration(n_iter)
