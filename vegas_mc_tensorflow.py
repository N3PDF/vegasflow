#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup, DTYPE, DTYPEINT
import numpy as np
import tensorflow as tf

BINS_MAX = 50
ALPHA = 1.5

@tf.function
def generate_random_array(shape, divisions, x_ini, xdelta):
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
    reg_i = tf.zeros(shape, dtype=DTYPE)
    reg_f = tf.ones(shape, dtype=DTYPE)
    rn = tf.random.uniform(shape, minval=0, maxval=1, dtype=DTYPE)
    xn = BINS_MAX*(1.0 - rn)
    int_xn = tf.maximum(tf.cast(0, DTYPEINT),
                        tf.minimum(tf.cast(xn, DTYPEINT), BINS_MAX))
    aux_rand = xn - tf.cast(int_xn, dtype=DTYPE)
    for i in tf.range(x_ini.shape[0], dtype=DTYPEINT):
        for j in tf.range(x_ini.shape[1], dtype=DTYPEINT):
            if int_xn[i,j] > 0:
                x_ini[i,j].assign(divisions[i, int_xn[i,j] - 1])
            xdelta[i,j].assign(divisions[i, int_xn[i,j]])
    xdelta.assign_sub(x_ini)
    rand_x = x_ini + xdelta*aux_rand
    x = reg_i + rand_x*(reg_f - reg_i)
    wgt = tf.reduce_prod(xdelta*BINS_MAX, axis=0)
    div_index = int_xn
    return x, wgt, div_index


def rebin(rw, rc, subdivisions, dim):
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
            old_xi = subdivisions[dim, k-1]
        old_xf = subdivisions[dim, k]
        dr -= rc
        delta_x = old_xf-old_xi
        aux.append(old_xf - delta_x*(dr / rw[k]))
    aux.append(1.0)
    subdivisions[dim,:].assign(aux)

def refine_grid(res_sq, subdivisions, dim):
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
    rebin(rw, rc, subdivisions, dim)

@tf.function
def loop(n_dim, n_events, arr_res2, div_index, tmp2):
    for j in tf.range(n_dim, dtype=DTYPEINT):
        for z in tf.range(n_events, dtype=DTYPEINT):
            arr_res2[j, div_index[j,z]].assign(arr_res2[j, div_index[j,z]]+tmp2[z])

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
    xjac = tf.constant(1.0/n_events, dtype=DTYPE)
    divisions = tf.Variable(tf.zeros((n_dim, BINS_MAX), dtype=DTYPE))
    divisions[:, 0].assign(tf.ones(n_dim, dtype=DTYPE))

    # Do a fake initialization at the begining
    rw_tmp = tf.Variable(tf.ones(BINS_MAX, dtype=DTYPE))

    rc = tf.constant(1.0/BINS_MAX, dtype=DTYPE)
    for i in tf.range(n_dim):
        rebin(rw_tmp, rc, divisions, i)
    # "allocate" arrays
    all_results = []

    # Loop of iterations
    for k in range(n_iter):
        res = 0.0
        res2 = 0.0
        arr_res2 = tf.Variable(tf.zeros((n_dim, BINS_MAX), dtype=DTYPE))

        shape = (n_dim, n_events)
        x_ini = tf.Variable(tf.zeros(shape, dtype=DTYPE))
        xdelta = tf.Variable(tf.zeros(shape, dtype=DTYPE))
        x, xwgt, div_index = generate_random_array(shape, divisions, x_ini, xdelta)

        wgt = xjac*xwgt
        tmp = wgt*MC_INTEGRAND(x)
        tmp2 = tf.square(tmp)

        res = tf.reduce_sum(tmp)
        res2 = tf.reduce_sum(tmp2)

        loop(n_dim, n_events, arr_res2, div_index, tmp2)

        err_tmp2 = tf.maximum((n_events*res2 - res**2)/(n_events-1.0), 1e-30)
        sigma = tf.sqrt(err_tmp2)
        print("Results for interation {0}: {1} +/- {2}".format(k+1, res, sigma))
        results[k] = res
        sigmas[k] = sigma
        all_results.append( (res, sigma) )
        for j in range(n_dim):
            refine_grid(arr_res2[j], divisions, j)

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
    return final_result.numpy(), sigma

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
