#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup
import numpy as np
import tensorflow as tf

DTYPE = tf.float32
BINS_MAX = 50
ALPHA = 1.5

def internal_rand():
    """ Generates a random number """
    return np.random.uniform(0,1)

def generate_random_array(n_dim, divisions, x, div_index):
    """
    Generates a random array
    # Arguments in:
        - n_dim: number of dimensions
        - divisions: an array defining the grid divisions

    # Arguments out:
        - x: array of dimension n_dim with the random number
        - div_index: array of dimension n_dim with the subdivision
                     of each point

    # Returns:
        - wgt: weight of the point
    """
    reg_i = 0.0
    reg_f = 1.0
    wgt = 1.0
    for i in range(n_dim):
        rn = internal_rand()
        # Get a random number randomly assigned to a subdivision
        xn = BINS_MAX*(1.0 - rn)
        int_xn = max(0, min(int(xn), BINS_MAX))
        # In practice int_xn = int(xn)-1 unless xn < 1
        aux_rand = xn - int_xn
        if int_xn == 0:
            x_ini = 0.0
        else:
            x_ini = divisions[i, int_xn - 1]
        xdelta = divisions[i, int_xn] - x_ini
        rand_x = x_ini + xdelta*aux_rand
        x[i] = reg_i + rand_x*(reg_f - reg_i)
        wgt *= xdelta*BINS_MAX
        div_index[i] = int_xn
    return wgt

def rebin(rw, rc, subdivisions):
    """ broken from function above to use it for initialiation """
    k = -1
    dr = 0.0
    aux = []
    for i in range(BINS_MAX-1):
        old_xi = 0.0
        while rc > dr:
            k += 1
            dr += rw[k]
        if k > 0:
            old_xi = subdivisions[k-1]
        old_xf = subdivisions[k]
        dr -= rc
        delta_x = old_xf-old_xi
        aux.append(old_xf - delta_x*(dr / rw[k]))
    aux.append(1.0)
    for i, tmp in enumerate(aux):
        subdivisions[i] = tmp


def refine_grid(res_sq, subdivisions):
    """
    Resize the grid
    # Arguments in:
        - res_sq: array with the accumulative sum for each division for one dim
    # Arguments inout:
        - subdivisions: the array the defining the vegas grid divisions for one dim
    """
    # First we smear out the array div_sq, where we have store
    # the value of f^2 for each sub_division for each dimension
    aux = [
            (res_sq[0] + res_sq[1])/2.0
            ]
    for i in range(1, BINS_MAX-1):
        tmp = (res_sq[i-1] + res_sq[i] + res_sq[i+1])/3.0
        if tmp < 1e-30:
            tmp = 1e-30
        aux.append(tmp)
    tmp = (res_sq[BINS_MAX-2] + res_sq[BINS_MAX-1])/2.0
    aux.append(tmp)
    aux_sum = np.sum(np.array(aux))
    # Now we refine the grid according to
    # journal of comp phys, 27, 192-203 (1978) G.P. Lepage
    rw = []
    for res in aux:
        tmp = pow( (1.0 - res/aux_sum)/(np.log(aux_sum) - np.log(res)), ALPHA )
        rw.append(tmp)
    rw = np.array(rw)
    rc = np.sum(rw)/BINS_MAX
    rebin(rw, rc, subdivisions)

def vegas(n_dim, n_iter, n_events, results, sigmas):
    """
    # Arguments in:
        n_dim: number of dimensions
        n_iter: number of iterations
        n_events: number of events per iteration

    # Arguments out:
        results: array with all results by iteration
        sigmas: array with all errors by iteration

    # Returns:
        - integral value
        - error
    """
    # Initialize variables
    xjac = 1.0/n_events
    divisions = np.zeros( (n_dim, BINS_MAX) )
    divisions[:, 0] = 1.0
    # Do a fake initialization at the begining
    rw_tmp = np.ones(BINS_MAX)
    for i in range(n_dim):
        rebin(rw_tmp, 1.0/BINS_MAX, divisions[i])

    # "allocate" arrays
    x = np.zeros(n_dim, dtype=np.float32)
    div_index = np.zeros(n_dim, dtype = np.int64)
    all_results = []

    # Loop of iterations
    for k in range(n_iter):
        res = 0.0
        res2 = 0.0
        arr_res2 = np.zeros( (n_dim, BINS_MAX) )

        for i in range(n_events):
            xwgt = generate_random_array(n_dim, divisions, x, div_index)
            wgt = xjac*xwgt

            # Call the integrand
            tmp = wgt*MC_INTEGRAND(x)
            tmp2 = pow(tmp, 2)

            res += tmp
            res2 += tmp2

            for j, ind in enumerate(div_index):
                arr_res2[j, ind] += tmp2

        err_tmp2 = max((n_events*res2 - res**2)/(n_events-1.0), 1e-30)
        sigma = np.sqrt(err_tmp2)

        print("Results for interation {0}: {1} +/- {2}".format(k+1, res, sigma))
        results[k] = res
        sigmas[k] = sigma
        all_results.append( (res, sigma) )
        for j in range(n_dim):
            refine_grid(arr_res2[j], divisions[j])

    # Compute the final results
    aux_res = 0.0
    weight_sum = 0.0
    for result in all_results:
        res = result[0]
        sigma = result[1]
        wgt_tmp = 1.0/pow(sigma, 2)
        aux_res += res*wgt_tmp
        weight_sum += wgt_tmp

    final_result = aux_res/weight_sum
    sigma = np.sqrt(1.0/weight_sum)
    return final_result, sigma

class make_vegas:
    """A Vegas MC integrator using importance sampling"""
    def __init__(self, dim, xl = None, xu = None):
        self.dim = dim
        # At the moment we save xl, xu but it is not used
        self.xl = xl
        self.xu = xu


    def integrate(self, iters = 5, calls = 1e4):
        results = np.zeros(iters)
        sigmas = np.zeros(iters)
        r = vegas(self.dim, iters, calls, results, sigmas)
        return r


if __name__ == '__main__':
    """Testing a basic integration"""
    ncalls = setup['ncalls']
    xlow = setup['xlow']
    xupp = setup['xupp']
    dim = setup['dim']

    print(f'VEGAS MC numba, ncalls={ncalls}:')
    start = time.time()
    v = make_vegas(dim=dim, xl = xlow, xu = xupp)
    r = v.integrate(calls=ncalls)
    end = time.time()
    print(r)
    print(f'time (s): {end-start}')
