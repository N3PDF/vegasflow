"""
    Plain implementation of the plainest possible MonteCarlo
"""

from vegasflow.configflow import DTYPE, fone, fzero, DTYPEINT,DTYPE
from vegasflow.monte_carlo import MonteCarloFlow, wrapper
import tensorflow as tf

import numpy as np
from itertools import chain,repeat,product
N_STRAT_MIN = 4
BETA = 0.75


@tf.function
def generate_random_array(rnds,n_strat,n_ev,hypercubes):
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
    delta_y = tf.cast(1/n_strat,DTYPE)

    random = rnds*delta_y
    points = tf.cast(tf.repeat(hypercubes,n_ev,axis=0),DTYPE)
    n_dim = int(tf.shape(hypercubes)[1])
    
    x = points*delta_y + random - tf.cast([delta_y] * n_dim,DTYPE)
    w =  tf.cast(tf.repeat(1/n_ev,n_ev),DTYPE)
    segm  = tf.cast(tf.repeat(tf.range(fzero,tf.shape(hypercubes)[0]),n_ev),DTYPEINT)
    return x,w,segm 


class StratifiedFlow(MonteCarloFlow):
    """
        Simple Monte Carlo integrator with Stratified Sampling.
    """

    def __init__(self, n_dim, n_events,adaptive=True, **kwargs):
        super().__init__(n_dim, n_events, **kwargs)

        self.adaptive = adaptive

        # Initialize stratifications
        self.n_strat = tf.math.floor(tf.math.pow(n_events/N_STRAT_MIN, 1/n_dim))
        substratification_np = np.linspace(0,1,int(self.n_strat)+1)
        stratifications_np = substratification_np.repeat(n_dim).reshape(-1, n_dim).T
        self.stratifications = tf.Variable(stratifications_np, dtype=DTYPE)
        # Initialize hypercubes
        hypercubes_one_dim = np.arange(1,int(self.n_strat)+1)
        hypercubes = [list(p) for p in product(hypercubes_one_dim, repeat=int(n_dim))]
        hypercubes_tf=tf.convert_to_tensor(hypercubes)
        self.hypercubes = hypercubes_tf

        #Initialize n_ev
        n_ev = tf.math.floor(n_events/tf.math.pow(self.n_strat,n_dim))
        self.n_ev=tf.fill([1 ,len(hypercubes)], int(n_ev))
        self.n_ev = tf.reshape(self.n_ev,[-1])

        #correction of self.n_events due to samples per hypercube
        self.n_events = int(n_ev*tf.math.pow(self.n_strat,n_dim))

    def redistribute_samples(self,arr_var):
        """Receives an array with the variance of the integrand in each hypercube
        and recalculate the samples per hypercube according to VEGAS+ algorithm"""
        
        damped_arr_var = tf.pow(arr_var,BETA)
        new_n_ev = tf.maximum(2,damped_arr_var * self.n_events/tf.reduce_sum(damped_arr_var))
        self.n_ev = tf.math.floor(new_n_ev)
    
        
    def _run_event(self, integrand, ncalls=None):

        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        # Generate all random numbers 
        rnds = tf.random.uniform((n_events, self.n_dim), minval=0, maxval=1, dtype=DTYPE)
         
        # Pass random numbers in hypercubes
        x, w, segm = generate_random_array(rnds,self.n_strat,self.n_ev,self.hypercubes)
        
        # compute integrand
        xjac = w
        tmp = integrand(x, n_dim=self.n_dim, weight=xjac) * xjac
        tmp2 = tf.square(tmp)

        # tensor containing resummed component for each hypercubes
        ress = tf.math.segment_sum(tmp,segm)
        ress2 = tf.math.segment_sum(tmp2,segm)

        #if adaptive save variance of each hypercube
        arr_var = None
        if self.adaptive:
            hypercube_volume = tf.cast(tf.math.pow(1/self.n_strat, self.n_dim),DTYPE)
            arr_var = ress2 * tf.cast(self.n_ev,DTYPE)* tf.square(hypercube_volume) - tf.square(ress*hypercube_volume)
        
        return ress, ress2, arr_var


    def _run_iteration(self):

        ress, raw_ress2, arr_var = self.run_event()

        # compute variance for each hypercube
        ress2 = raw_ress2 * tf.cast(self.n_ev,DTYPE)
        err_tmp2s = (ress2 - tf.square(ress))/(tf.cast(self.n_ev,DTYPE)-fone)
        sigmas2 = tf.maximum(err_tmp2s,fzero)

        
        res = tf.reduce_sum(ress) / tf.cast(tf.math.pow(self.n_strat,self.n_dim),DTYPE)
        sigma2 = tf.reduce_sum(sigmas2)/tf.cast(tf.math.pow(self.n_strat,self.n_dim),DTYPE)/self.n_events    
        sigma = tf.sqrt(sigma2)

        # If adaptive is active redistributes samples
        if self.adaptive:
            self.redistribute_samples(arr_var)

        return res, sigma


def plain_wrapper(*args):
    """ Wrapper around PlainFlow """
    return wrapper(StratifiedFlow, *args)




@tf.function
def symgauss(xarr, n_dim=None, **kwargs):
    """symgauss test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


if __name__ == "__main__":

    import time
    # MC integration setup
    dim = 4
    ncalls = int(1e3)
    n_iter = 5

    start = time.time()
    vegas_instance = StratifiedFlow(dim, ncalls,simplify_signature=True)
    vegas_instance.compile(symgauss)
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")