#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup, DTYPE, DTYPEINT
import numpy as np
import tensorflow as tf

BINS_MAX = 50
ALPHA = 1.5


# Define some constants
n_dim = setup['dim']
def int_me(i):
    return tf.constant(i, dtype = DTYPEINT)
def float_me(i):
    return tf.constant(i, dtype = DTYPE)
ione = int_me(1)
izero = int_me(0)
fone = float_me(1)
fzero = float_me(0)

shape_rn = tf.TensorSpec(shape=(None, n_dim), dtype=DTYPE)
shape_sub = tf.TensorSpec(shape=(n_dim, BINS_MAX), dtype = DTYPE)

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
    xn = BINS_MAX*(fone - rnds)
    int_xn = tf.maximum(tf.cast(0, DTYPEINT),
                       tf.minimum(tf.cast(xn, DTYPEINT), BINS_MAX))
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
    rand_x = x_ini + xdelta*aux_rand
    x = reg_i + rand_x*(reg_f - reg_i)
    weights = tf.reduce_prod(xdelta*BINS_MAX, axis=1)
    return x, int_xn, weights

@tf.function
def quick_integrand(xarr, n_dim = None):
    """Le page test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100*n_dim, dtype=DTYPE)
    pref = tf.pow(1.0/a/np.sqrt(np.pi), n_dim)
    coef = fzero
    for i in range(n100+1):
        coef += i
    for i in range(n_dim):
        coef += tf.pow( (xarr[:,i]-1.0/2.0)/a, 2)
    coef -= (n100+1)*n100/2.0
    return pref*tf.exp(-coef)

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
def consume_results(res2, indices):
    """ Consumes the results squared to
    generate the array-per-bin of results """
    all_bins = tf.range(BINS_MAX, dtype = DTYPEINT)
    eq = tf.transpose(tf.equal(indices, all_bins))
    res_tmp = tf.where(eq, res2, fzero)
    arr_res2 = tf.reduce_sum(res_tmp, axis = 1)
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
    xjac = 1.0/n_events

    # Initialize variable variables
    subdivision_np = np.linspace(1/BINS_MAX, 1, BINS_MAX)
    divisions_np = subdivision_np.repeat(n_dim).reshape(-1, n_dim).T
    divisions = tf.Variable(divisions_np, dtype = DTYPE)

    # "allocate" arrays
    all_results = []

    # Loop of iterations
    for iteration in range(n_iter):
        start = time.time()

        # Generate all random number for this iteration
        rnds = tf.random.uniform((n_events,n_dim), minval=0, maxval=1, dtype=DTYPE)

        # Pass them through the Vegas digestion
        x, ind, w = generate_random_array(rnds, divisions)

        # Now compute the integrand
        tmp = xjac*w*quick_integrand(x, n_dim = n_dim)

        # Compute the final result for this iteration
        res = tf.reduce_sum(tmp)
        # Compute the error
        tmp2 = tf.square(tmp)
        res2 = tf.reduce_sum(tmp2)
        err_tmp2 = (n_events*res2 - tf.square(res))/(n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))

        # Rebin Vegas
        all_bins = tf.range(BINS_MAX, dtype = DTYPEINT)
        for j in range(n_dim):
            arr_res2 = consume_results(tmp2, ind[:, j:j+1])
            refine_grid(arr_res2.numpy(), divisions, j)

# This seems to be faster but only for big number of events
#             arr_res2 = []
#             for i in range(BINS_MAX):
#                 mask = tf.equal(ind, i)[:,j]
#                 res_tmp = tf.reduce_sum(tf.boolean_mask(tmp2, mask))
#                 arr_res2.append(res_tmp)
#             arr_res2 = np.array(arr_res2)
#             refine_grid(arr_res2, divisions, j)


        # Print the results
        end = time.time()
        print(f"Results for {iteration} {res} +/- {sigma} (took {end-start} s)")
        all_results.append( (res, sigma) )

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


if __name__ == '__main__':
    """Testing a basic integration"""
    ncalls = setup['ncalls']
    xlow = setup['xlow']
    xupp = setup['xupp']

    print(f'VEGAS MC, ncalls={ncalls}:')
    start = time.time()
    r = vegas(n_dim, 5, ncalls)
    end = time.time()
    print(r)
    print(f'time (s): {end-start}')
