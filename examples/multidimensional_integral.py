"""
    Example: basic multidimensional integral

    The output of the function is not an scalar but a vector v.
    In this case it is necessary to tell Vegas which is the main dimension
    (i.e., the output dimension the grid should adapt to)

    Note that the integrand should have an output of the same shape as the tensor of random numbers
    the shape of the tensor of random numbers and of the output is (nevents, ndim) 
"""
from vegasflow import VegasFlow, run_eager

run_eager()
import tensorflow as tf

# MC integration setup
dim = 3
ncalls = int(1e4)
n_iter = 5


@tf.function
def test_function(xarr):
    res = tf.square((xarr - 1.0) ** 2)
    return tf.exp(-res[:, 1])


if __name__ == "__main__":
    print("Testing a multidimensional integration")
    vegas = VegasFlow(dim, ncalls)
    vegas.compile(test_function)
    all_results, all_err = vegas.run_integration(2)
    try:
        for result, error in zip(all_results, all_err):
            print(f"{result = :.5} +- {error:.5}")
    except TypeError:
        # So that the example works also if the integrand is made scalar
        result = all_results
        error = all_err
        print(f"{result = :.5} +- {error:.5}")
