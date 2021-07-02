"""
    Example: basic integration

    Basic example using the vegas_wrapper helper
"""

from vegasflow import run_eager, float_me
import tensorflow as tf

import time
import numpy as np
from vegasflow.vflow import vegas_wrapper
from vegasflow.rtbm import RTBMFlow


# MC integration setup
dim = 2
ncalls = np.int32(1e2)
n_iter = 5
n_hidden = 1
tf_pi = float_me(np.pi)
npeaks = 2.0


@tf.function
def sin_fun(xarr, **kwargs):
    """symgauss test function"""
    res = tf.pow(tf.sin(npeaks * xarr * tf_pi), 2)
    return tf.reduce_prod(res, axis=1)


integrand = sin_fun
# from simgauss_tf import symgauss as integrand

if __name__ == "__main__":
    # Testing several different integrations
    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    ncalls = ncalls
    _, vegas_instance = vegas_wrapper(integrand, dim, n_iter, ncalls, return_instance=True)
    vegas_instance.freeze_grid()
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")

    print(f"RTBM MC, ncalls={ncalls}:")
    start = time.time()
    rtbm = RTBMFlow(n_dim=dim, n_events=ncalls, train=True, n_hidden=n_hidden, generations=500)
    rtbm.compile(integrand)
    _ = rtbm.run_integration(n_iter)
    rtbm.freeze()
    end = time.time()
    print(f"RTBM took: time (s): {end-start}")

    parameters = rtbm._rtbm.get_parameters()
    param_file = f"PARAMS_for_ndim={dim}_{npeaks}xpeaks_hidden={n_hidden}.npy"
    np.save(param_file, parameters)
    print(f" > Saved parameters to {param_file}")

    print("Results with frozen grids")
    r = vegas_instance.run_integration(5)
    rt = rtbm.run_integration(5)
    print(f"Result computed by Vegas: {r[0]:.5f} +- {r[1]:.6f}")
    print(f"Result computed by RTBM:  {rt[0]:.5f} +- {rt[1]:.6f}")

# Notes
# For 1 and 2 dimensions is enough with n_hidden=1 to get a better per event error
# For 3 dimensions we need to go to n_hidden=2
# We might be actually overfitting big time because after a few iterations the integration stops being so good
#   a good way of testing this would be plotting the distribution of points one gets after each training!
#   or fitting wrongly anyway
