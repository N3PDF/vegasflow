"""
    Implementation of the adaptive stratified sampling
"""
from itertools import product
import tensorflow as tf
import numpy as np
from vegasflow.configflow import DTYPE, fone, fzero, DTYPEINT, float_me, int_me
from vegasflow.monte_carlo import MonteCarloFlow, wrapper

BETA = 0.75


@tf.function
def generate_random_array(rnds, n_strat, n_ev, hypercubes):
    """Receives an array of random numbers 0 and 1 and
    distribute them in each hypercube according to the
    number of sample in each hypercube specified by n_ev

    Parameters
    ----------
        `rnds`: tensor of random number betweenn 0 and 1
        `n_strat`: tensor with number of stratifications in each dimension
        `n_ev`: tensor containing number of sample per each hypercube
        `hypercubes`: tensor containing all different hypercube

    Returns
    -------
        `x` : random numbers collocated in hypercubes
        `w` : weight of each event
        `segm` : segmentantion for later computations
    """
    # stratification width
    delta_y = tf.cast(1/n_strat, DTYPE)
    random = rnds*delta_y
    points = tf.cast(tf.repeat(hypercubes, n_ev, axis=0), DTYPE)
    x = points*delta_y + random
    w = tf.cast(tf.repeat(1/n_ev, n_ev), DTYPE)
    segm = tf.cast(tf.repeat(tf.range(fzero,
                                      tf.shape(hypercubes)[0]),
                             n_ev),
                   DTYPEINT)

    return x, w, segm


class StratifiedFlow(MonteCarloFlow):
    """
        Monte Carlo integrator with Adaptive Stratified Sampling.
    """

    def __init__(self, n_dim, n_events, adaptive=True, **kwargs):
        super().__init__(n_dim, n_events, **kwargs)

        self.init_calls = n_events
        self.adaptive = adaptive

        # Initialize stratifications
        if self.adaptive:
            neval_eff = int(self.n_events/2)
            self.n_strat = tf.math.floor(tf.math.pow(neval_eff/2, 1/n_dim))
        else:
            neval_eff = self.n_events
            self.n_strat = tf.math.floor(tf.math.pow(neval_eff/2, 1/n_dim))

        self.n_strat = int_me(self.n_strat)

        # Initialize hypercubes
        hypercubes_one_dim = np.arange(0, int(self.n_strat))
        hypercubes = [list(p) for p in product(hypercubes_one_dim,
                                               repeat=int(n_dim))]
        self.hypercubes = tf.convert_to_tensor(hypercubes)

        if len(hypercubes) != int(tf.math.pow(self.n_strat, n_dim)):
            raise ValueError("Hypercubes problem!")

        # Set min evaluations per hypercube
        self.min_neval_hcube = int(neval_eff // len(hypercubes))
        if self.min_neval_hcube < 2:
            self.min_neval_hcube = 2

        # Initialize n_ev
        self.n_ev = tf.fill([1, len(hypercubes)], self.min_neval_hcube)
        self.n_ev = tf.reshape(self.n_ev, [-1])
        self.n_events = int(tf.reduce_sum(self.n_ev))
        self.xjac = float_me(1/len(hypercubes))

    def redistribute_samples(self, arr_var):
        """Receives an array with the variance of the integrand in each
        hypercube and recalculate the samples per hypercube according
        to the VEGAS+ algorithm"""
         
        damped_arr_sdev = tf.pow(arr_var, BETA / 2)
        new_n_ev = tf.maximum(self.min_neval_hcube,
                              damped_arr_sdev
                              * self.init_calls / 2
                              / tf.reduce_sum(damped_arr_sdev))
        self.n_ev = int_me(tf.math.floor(new_n_ev))
        self.n_events = int(tf.reduce_sum(self.n_ev))
    
    def _run_event(self, integrand, ncalls=None):

        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        # Generate all random numbers
        rnds = tf.random.uniform((n_events, self.n_dim), minval=0, maxval=1,
                                 dtype=DTYPE)
         
        # Pass random numbers in hypercubes
        x, w, segm = generate_random_array(rnds,
                                           self.n_strat,
                                           self.n_ev,
                                           self.hypercubes)
        
        # compute integrand
        xjac = self.xjac * w
        tmp = integrand(x, n_dim=self.n_dim, weight=xjac) * xjac
        tmp2 = tf.square(tmp)

        # tensor containing resummed component for each hypercubes
        ress = tf.math.segment_sum(tmp, segm)
        ress2 = tf.math.segment_sum(tmp2, segm)

        Fn_ev = tf.cast(self.n_ev, DTYPE)
        arr_var = ress2 * Fn_ev - tf.square(ress)

        return ress, arr_var

    def _run_iteration(self):

        ress, arr_var = self.run_event()

        Fn_ev = tf.cast(self.n_ev, DTYPE)
        sigmas2 = tf.maximum(arr_var, fzero)
        res = tf.reduce_sum(ress)
        sigma2 = tf.reduce_sum(sigmas2/(Fn_ev-fone))
        sigma = tf.sqrt(sigma2)

        # If adaptive is True redistributes samples
        if self.adaptive:
            self.redistribute_samples(arr_var)

        return res, sigma


def stratified_wrapper(*args):
    """ Wrapper around PlainFlow """
    return wrapper(StratifiedFlow, *args)
