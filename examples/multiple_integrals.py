#!/usr/bin/env python3
"""
    Example of using VegasFlow to compute an arbitrary number of integrals
    Note that it should only be used for similarly-behaved integrands.

    The example integrands are variations of the Genz functions definged in
    Novak et al, 1999 (J. of Comp and Applied Maths, 112 (1999) 215-228 and implemented from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.8452&rep=rep1&type=pdf
"""
import vegasflow
from vegasflow.configflow import DTYPE
import tensorflow as tf
import numpy as np

NDIM = 4
NPARAMS = 10
a1 = 0.1
a2 = 0.5

RESULT_ARRAY = []
ERRORS_ARRAY = []

w = np.random.rand(NDIM)


def generate_genz(a):
    """Generates the genz_oscillatory function for a given parameter a"""

    def genz_oscillatory(xrandr, **kwargs):
        """
        The input xrandr has shape (nevents, ndim)
        the output of the function is (nevents,)
        """
        res = tf.einsum("ij, j->i", xrandr, w) + 2.0 * np.pi * a
        return (tf.cos(res) + 1.0) / 2.0

    return genz_oscillatory


def generate_multiple_genz(full_a):
    """Compute the genz oscillatory function for multiple values of a at once"""
    # Make sure that full_a is a tensor
    # _and_ add a dummy event axis
    a = tf.constant(np.reshape(full_a, (1, -1)), dtype=DTYPE)

    def genz_oscillatory_multiple(xrandr, **kwargs):
        """
        The input xrandr has shape (nevents, ndim)
        the output of the function is (nevents, nparams)
        """
        raw_res = tf.einsum("ij, j->i", xrandr, w)
        # Add a dummy parameter-axis
        res = tf.reshape(raw_res, (-1, 1)) + 2.0 * np.pi * a
        return (tf.cos(res) + 1.0) / 2.0

    return genz_oscillatory_multiple


test1 = generate_genz(a1)
test2 = generate_multiple_genz([a1, a2])

# Generate 5 events
rval = np.random.rand(5, NDIM)
# Check the shape of the two functions
r1 = test1(rval)
r2 = test2(rval)
print(f"Shape of single-parameter call: {r1.numpy().shape}")
print(f"Shape of multiparameter-parameter call: {r2.numpy().shape}")

# Now we can generate an integrand for this
generate_single_integrand = generate_genz

# And run it with VegasFlow for a given value of a=0.1
result_single_1 = vegasflow.vegas_wrapper(
    generate_single_integrand(a1), NDIM, n_iter=5, total_n_events=int(1e4)
)
result_single_2 = vegasflow.vegas_wrapper(
    generate_single_integrand(a2), NDIM, n_iter=5, total_n_events=int(1e4)
)

# Now, we cannot do the same for the multiple parameter as VegasFlow is expecting an escalar result!
# So we need to save the results for each integrand somehow, let's use a "digest function"


def digest_function(all_results, all_wgts):
    # Receive the results from the computing device
    # and associated event weights
    results = all_results.numpy()
    wgts = all_wgts.numpy().reshape(-1)
    # Now get the weighted average of the result (the MC estimate) per parameter
    final_result = np.einsum("ij, i->j", results, wgts)
    # And compute the errors
    sq_results = np.einsum("ij, i->j", results**2, wgts**2) * len(wgts)
    errors = np.abs(final_result**2 - sq_results) / (len(wgts) - 1)
    print(f"{final_result} +- {np.sqrt(errors)}")
    RESULT_ARRAY.append(final_result)
    ERRORS_ARRAY.append(errors)
    return 0.0


def generate_multiple_integrand(full_a):

    genz_function = generate_multiple_genz(full_a)

    # Clean the previous results (if any)
    while RESULT_ARRAY:
        RESULT_ARRAY.pop()
        ERRORS_ARRAY.pop()

    def integrand(xrandr, weight=1.0):
        res = genz_function(xrandr)

        # Store the individual result
        tf.py_function(digest_function, [res, weight], Tout=DTYPE)

        # Return the result for the first parameter
        return res[:, 0]

    return integrand


vegasflow.vegas_wrapper(
    generate_multiple_integrand([a1, a2]), NDIM, n_iter=5, total_n_events=int(1e4)
)

res_mul_1, res_mul_2 = np.average(RESULT_ARRAY, axis=0)

err_mul_1, err_mul_2 = np.sqrt(
    1 / np.sum(1 / np.array(ERRORS_ARRAY) ** 2, axis=0) / len(ERRORS_ARRAY)
)

print(
    f"""
The single calculation obtained:
{result_single_1[0]:.4} +- {result_single_1[1]:.4} for a={a1} and
{result_single_2[0]:.4} +- {result_single_2[1]:.4} for a={a2}

while the multiple integration found:
{res_mul_1:.4} +- {err_mul_1:.4} for a={a1} and
{res_mul_2:.4} +- {err_mul_2:.4} for a={a2}
"""
)
