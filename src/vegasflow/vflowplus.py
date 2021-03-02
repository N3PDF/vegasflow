#!/usr/env python
"""
    Implementation of vegas+ algorithm:
        adaptive importance sampling + adaptive stratified sampling
          
"""
from itertools import chain, repeat, product
import tensorflow as tf
import numpy as np
from vegasflow.configflow import DTYPE, DTYPEINT, fone, fzero, float_me, ione, int_me, BINS_MAX
from vegasflow.monte_carlo import wrapper
from vegasflow.vflow import VegasFlow
from vegasflow.utils import consume_array_into_indices


N_STRAT_MIN = 4
BETA = 0.75
FBINS = float_me(BINS_MAX)


@tf.function
def generate_samples_in_hypercubes(randoms, n_strat, n_ev, hypercubes, divisions):
    """Receives an array of random numbers 0 and 1 and
    distribute them in each hypercube according to the
    number of sample in each hypercube specified by n_ev

    Parameters
    ----------
        `rnds`: tensor of random number betweenn 0 and 1
        `n_strat`: tensor with number of stratifications in each dimension
        `n_ev`: tensor containing number of sample per each hypercube
        `hypercubes`: tensor containing all different hypercube
        `divisions`: vegas grid

    Returns
    -------
        `x` : random numbers collocated in hypercubes
        `w` : weight of each event
        div_index: division index in which each (n_dim) set of random numbers fall
        `segm` : segmentantion for later computations
    """
    points = tf.repeat(hypercubes, n_ev, axis=0)
    xn = tf.transpose((points+tf.transpose(randoms))*FBINS/float_me(n_strat))
    segm = tf.cast(tf.repeat(tf.range(fzero,
                                      tf.shape(hypercubes)[0]),
                             n_ev),
                   DTYPEINT)

    # same as vegasflow generate_random_array
    @tf.function
    def digest(xn):
        ind_i = tf.cast(xn, DTYPEINT)
        # Get the value of the left and right sides of the bins
        ind_f = ind_i + ione
        x_ini = tf.gather(divisions, ind_i, batch_dims=1)
        x_fin = tf.gather(divisions, ind_f, batch_dims=1)
        # Compute the width of the bins
        xdelta = x_fin - x_ini
        return ind_i, x_ini, xdelta

    ind_i, x_ini, xdelta = digest(xn)
    # Compute the random number between 0 and 1
    # This is the heavy part of the calc

    @tf.function
    def compute_x(x_ini, xn, xdelta):
        aux_rand = xn - tf.math.floor(xn)
        return x_ini + xdelta * aux_rand

    x = compute_x(x_ini, xn, xdelta)
    # Compute the random number between the limits
    #     x = reg_i + rand_x * (reg_f - reg_i)
    # and the weight
    weights = tf.reduce_prod(xdelta * FBINS, axis=0)
    x_t = tf.transpose(x)
    int_xn = tf.transpose(ind_i)
    return x_t, int_xn, weights, segm


class VegasFlowPlus(VegasFlow):
    """
    Implementation of the VEGAS+ algorithm
    """

    def __init__(self, n_dim, n_events, train=True, adaptive=True, **kwargs):
        super().__init__(n_dim, n_events, train, **kwargs)

        self.adaptive = adaptive

        # Initialize stratifications
        self.n_strat = tf.math.floor(tf.math.pow(self.n_events/N_STRAT_MIN, 1/n_dim))
        self.n_strat = int_me(self.n_strat)

        # Initialize hypercubes
        hypercubes_one_dim = np.arange(0, int(self.n_strat))
        hypercubes = [list(p) for p in product(hypercubes_one_dim, 
                                               repeat=int(n_dim))]
        self.hypercubes = tf.convert_to_tensor(hypercubes, dtype=DTYPE)

        if len(hypercubes) != int(tf.math.pow(self.n_strat, n_dim)):
            raise ValueError("Hypercubes problem!")

        # Initialize n_ev
        n_ev = int_me(tf.math.floordiv(self.n_events, tf.math.pow(self.n_strat, n_dim)))
        n_ev = tf.math.maximum(n_ev, 2)
        self.n_ev = tf.fill([1, len(hypercubes)], n_ev)
        self.n_ev = tf.reshape(self.n_ev, [-1])

        self.n_events = int(tf.reduce_sum(self.n_ev))
        self.xjac = 1 / self.n_events

    def redistribute_samples(self, arr_var):
        """Receives an array with the variance of the integrand in each
        hypercube and recalculate the samples per hypercube according
        to VEGAS+ algorithm"""
        
        # neval_eff = int(self.n_events/2)
        damped_arr_sdev = tf.pow(arr_var, BETA/2)
        new_n_ev = tf.maximum(2,
                              damped_arr_sdev * self.n_events/tf.reduce_sum(damped_arr_sdev)/2)
        self.n_ev = int_me(tf.math.floor(new_n_ev))
        self.n_events = int(tf.reduce_sum(self.n_ev))
        self.xjac = 1 / self.n_events
    
    def _run_event(self, integrand, ncalls=None):

        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        tech_cut = 1e-8
        # Generate all random number for this iteration
        rnds = tf.random.uniform(
            (self.n_dim, n_events), minval=tech_cut, maxval=1.0 - tech_cut, dtype=DTYPE
        )

        # Pass random numbers in hypercubes
        
        x, ind, w, segm = generate_samples_in_hypercubes(rnds,
                                                         self.n_strat,
                                                         self.n_ev,
                                                         self.hypercubes,
                                                         self.divisions)
        
        # compute integrand
        xjac = self.xjac * w
        if self.simplify_signature:
            tmp = xjac * integrand(x)
        else:
            tmp = xjac * integrand(x, n_dim=self.n_dim, weight=xjac)
        tmp2 = tf.square(tmp)    

        # tensor containing resummed component for each hypercubes
        ress = tf.math.segment_sum(tmp, segm)
        ress2 = tf.math.segment_sum(tmp2, segm)

        Fn_ev = tf.cast(self.n_ev, DTYPE)
        arr_var = (ress2 - tf.square(ress)/Fn_ev)/(Fn_ev - fone)

        arr_res2 = []
        if self.train:
            # If the training is active, save the result of the integral sq
            for j in range(self.n_dim):
                arr_res2.append(
                    consume_array_into_indices(tmp2, ind[ :, j : j + 1], int_me(self.grid_bins - 1))
                )
            arr_res2 = tf.reshape(arr_res2, (self.n_dim, -1))
        
        return ress, arr_var, arr_res2
    
    def _iteration_content(self):
        
        # print("_iteration_content_vfp")
        ress, arr_var, arr_res2 = self.run_event()
        Fn_ev = tf.cast(self.n_ev, DTYPE)
        # compute variance for each hypercube
        sigmas2 = tf.maximum(arr_var, fzero)
       
        res = tf.reduce_sum(ress)

        sigma2 = tf.reduce_sum(sigmas2*Fn_ev)
        sigma = tf.sqrt(sigma2)

        # If adaptive is active redistributes samples
        if self.adaptive:
            self.redistribute_samples(arr_var)

        if self.train:
            self.refine_grid(arr_res2)
        return res, sigma


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
