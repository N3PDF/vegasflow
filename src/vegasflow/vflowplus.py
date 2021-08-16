#!/usr/env python
"""
    Implementation of vegas+ algorithm:

    adaptive importance sampling + adaptive stratified sampling
"""
from itertools import product
import logging
import tensorflow as tf
import numpy as np
from vegasflow.configflow import DTYPE, DTYPEINT, fone, fzero, float_me, ione, int_me, BINS_MAX, BETA, MAX_NEVAL_HCUBE
from vegasflow.monte_carlo import wrapper
from vegasflow.vflow import VegasFlow
from vegasflow.utils import consume_array_into_indices


logger = logging.getLogger(__name__)

FBINS = float_me(BINS_MAX)


@tf.function(input_signature=3 * [tf.TensorSpec(shape=[None, None], dtype=DTYPE)])
def _compute_x(x_ini, xn, xdelta):
    """ Helper function for ``generate_samples_in_hypercubes`` """
    aux_rand = xn - tf.math.floor(xn)
    return x_ini + xdelta * aux_rand


# same as vegasflow generate_random_array
@tf.function(input_signature=2*[tf.TensorSpec(shape=[None, None], dtype=DTYPE)])
def _digest(xn, divisions):
    ind_i = tf.cast(xn, DTYPEINT)
    # Get the value of the left and right sides of the bins
    ind_f = ind_i + ione
    x_ini = tf.gather(divisions, ind_i, batch_dims=1)
    x_fin = tf.gather(divisions, ind_f, batch_dims=1)
    # Compute the width of the bins
    xdelta = x_fin - x_ini
    return ind_i, x_ini, xdelta


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPEINT),
        tf.TensorSpec(shape=[None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[None, None], dtype=DTYPEINT),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
    ]
)
def generate_samples_in_hypercubes(rnds, n_strat, n_ev, hypercubes, divisions):
    """Receives an array of random numbers 0 and 1 and
    distribute them in each hypercube according to the
    number of samples in each hypercube specified by n_ev

    Parameters
    ----------
        `rnds`: tensor of random number between 0 and 1
        `n_strat`: tensor with number of stratifications in each dimension
        `n_ev`: tensor containing number of sample per hypercube
        `hypercubes`: tensor containing all different hypercube
        `divisions`: vegas grid

    Returns
    -------
        `x` : random numbers collocated in hypercubes
        `w` : weight of each event
        `ind`: division index in which each (n_dim) set of random numbers fall
        `segm` : segmentantion for later computations
    """

    indices = tf.repeat(tf.range(tf.shape(hypercubes, out_type=DTYPEINT)[0]), n_ev)
    points = float_me(tf.gather(hypercubes, indices))
    n_evs = float_me(tf.gather(n_ev, indices))
    xn = tf.transpose((points + tf.transpose(rnds)) * FBINS / float_me(n_strat))
    segm = indices

    ind_i, x_ini, xdelta = _digest(xn, divisions)
    # Compute the random number between 0 and 1
    # This is the heavy part of the calc

    x = _compute_x(x_ini, xn, xdelta)
    # Compute the random number between the limits
    #     x = reg_i + rand_x * (reg_f - reg_i)
    # and the weight
    weights = tf.reduce_prod(xdelta * FBINS, axis=0)
    final_weights = weights / n_evs
    x_t = tf.transpose(x)
    int_xn = tf.transpose(ind_i)

    return x_t, int_xn, final_weights, segm


class VegasFlowPlus(VegasFlow):
    """
    Implementation of the VEGAS+ algorithm
    """

    def __init__(self, n_dim, n_events, train=True, adaptive=None, **kwargs):
        _ = kwargs.setdefault("events_limit", n_events)
        super().__init__(n_dim, n_events, train, **kwargs)

        self.init_calls = n_events

        # naive check not to use adaptive if n_dim > 13
        if n_dim > 13 and adaptive == None:
            self.adaptive = False
        else:
            self.adaptive = adaptive

        # Initialize stratifications
        if self.adaptive:
            neval_eff = int(self.n_events / 2)
            self.n_strat = tf.math.floor(tf.math.pow(neval_eff / 2, 1 / n_dim))
        else:
            neval_eff = self.n_events
            self.n_strat = tf.math.floor(tf.math.pow(neval_eff / 2, 1 / n_dim))

        if tf.math.pow(self.n_strat, n_dim) > MAX_NEVAL_HCUBE:
            self.n_strat = tf.math.floor(tf.math.pow(1e4, 1/n_dim))

        self.n_strat = int_me(self.n_strat)

        # Initialize hypercubes
        hypercubes_one_dim = np.arange(0, int(self.n_strat))
        hypercubes = [list(p) for p in product(hypercubes_one_dim, repeat=int(n_dim))]
        self.hypercubes = tf.convert_to_tensor(hypercubes, dtype=DTYPEINT)

        if len(hypercubes) != int(tf.math.pow(self.n_strat, n_dim)):
            raise ValueError("Hypercubes are not equal to n_strat^n_dim")

        self.min_neval_hcube = int(neval_eff // len(hypercubes))
        if self.min_neval_hcube < 2:
            self.min_neval_hcube = 2

        self.n_ev = tf.fill([1, len(hypercubes)], self.min_neval_hcube)
        self.n_ev = tf.cast(tf.reshape(self.n_ev, [-1]), dtype=DTYPEINT)
        self._n_events = int(tf.reduce_sum(self.n_ev))
        self.my_xjac = float_me(1 / len(hypercubes))

        if self.adaptive:
            logger.warning("Variable number of events requires function signatures all across")

    def redistribute_samples(self, arr_var):
        """Receives an array with the variance of the integrand in each
        hypercube and recalculate the samples per hypercube according
        to VEGAS+ algorithm"""
        damped_arr_sdev = tf.pow(arr_var, BETA / 2)
        new_n_ev = tf.maximum(
            self.min_neval_hcube,
            damped_arr_sdev * self.init_calls / 2 / tf.reduce_sum(damped_arr_sdev),
        )
        self.n_ev = int_me(new_n_ev)
        self.n_events = int(tf.reduce_sum(self.n_ev))

    def _run_event(self, integrand, ncalls=None, n_ev=None):
        # NOTE: needs to receive both ncalls and n_ev
        
        n_events = ncalls

        tech_cut = 1e-8
        # Generate all random number for this iteration
        rnds = tf.random.uniform(
            (self.n_dim, n_events), minval=tech_cut, maxval=1.0 - tech_cut, dtype=DTYPE
        )

        # Pass random numbers in hypercubes
        x, ind, w, segm = generate_samples_in_hypercubes(
            rnds, self.n_strat, n_ev, self.hypercubes, self.divisions,
        )

        # compute integrand
        xjac = self.my_xjac * w
        tmp = xjac * integrand(x, weight=xjac)
        tmp2 = tf.square(tmp)

        # tensor containing resummed component for each hypercubes
        ress = tf.math.segment_sum(tmp, segm)
        ress2 = tf.math.segment_sum(tmp2, segm)

        Fn_ev = tf.cast(n_ev, DTYPE)
        arr_var = ress2 * Fn_ev - tf.square(ress)

        arr_res2 = []
        if self.train:
            # If the training is active, save the result of the integral sq
            for j in range(self.n_dim):
                arr_res2.append(
                    consume_array_into_indices(tmp2, ind[: , j : j + 1], int_me(self.grid_bins - 1))
                )
            arr_res2 = tf.reshape(arr_res2, (self.n_dim, -1))

        return ress, arr_var, arr_res2

    def _iteration_content(self):
        ress, arr_var, arr_res2 = self.run_event(n_ev=self.n_ev)

        Fn_ev = tf.cast(self.n_ev, DTYPE)
        sigmas2 = tf.maximum(arr_var, fzero)
        res = tf.reduce_sum(ress)
        sigma2 = tf.reduce_sum(sigmas2 / (Fn_ev - fone))
        sigma = tf.sqrt(sigma2)

        # If adaptive is active redistributes samples
        if self.adaptive:
            self.redistribute_samples(arr_var)

        if self.train:
            self.refine_grid(arr_res2)
        return res, sigma

    def run_event(self, tensorize_events=None, **kwargs):
        """ Tensorizes the number of events so they are not python or numpy primitives if self.adaptive=True"""
        return super().run_event(tensorize_events=self.adaptive, **kwargs)


def vegasflowplus_wrapper(integrand, n_dim, n_iter, total_n_events, **kwargs):
    """Convenience wrapper

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
    return wrapper(VegasFlowPlus, integrand, n_dim, n_iter, total_n_events, **kwargs)
