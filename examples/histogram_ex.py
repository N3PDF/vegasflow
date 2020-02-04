"""
    Example of integration of a simple function
    With the filling of a histogram
"""

import time
import numpy as np
import tensorflow as tf
from vegasflow.configflow import DTYPE, DTYPEINT, fzero, fone
from vegasflow.plain import PlainFlow
from vegasflow.vflow import VegasFlow
from vegasflow.utils import consume_array_into_indices

# MC integration setup
dim = 3
ncalls = np.int32(1e5)
n_iter = 5
hst_dim = 2
HISTO_BINS = 2


def generate_integrand(cummulator_tensor):
    """ 
    This function will generate an integrand function
    which will already hold a reference to the tensor to accumulate
    """

    @tf.function
    def histogram_collector(results, variables):
        """ This function will receive a tensor (result)
        and the variables corresponding to those integrand results 
        In the example integrand below, these corresponds to 
            `final_result` and `histogram_values` respectively.
        `current_histograms` instead is the current value of the histogram
        which will be overwritten """
        # Fill a histogram with HISTO_BINS (2) bins, (0 to 0.5, 0.5 to 1)
        # First generate the indices with TF
        indices = tf.histogram_fixed_width_bins(
            variables, [fzero, fone], nbins=HISTO_BINS
        )
        t_indices = tf.transpose(indices)
        # Then consume the results with the utility we provide
        partial_hist = consume_array_into_indices(results, t_indices, HISTO_BINS)
        # Then update the results of current_histograms
        new_histograms = partial_hist + current_histograms
        cummulator_tensor.assign(new_histograms)

    def integrand_example(xarr, n_dim=None, weight=fone):
        """ Example of function which saves histograms """
        if n_dim is None:
            n_dim = xarr.shape[-1]
        if n_dim < hst_dim:
            raise ValueError(
                f"The number of dimensions has to be greater than {hst_dim} for this example"
            )
        a = tf.constant(0.1, dtype=DTYPE)
        n100 = tf.cast(100 * n_dim, dtype=DTYPE)
        pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
        coef = tf.reduce_sum(tf.range(n100 + 1))
        coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
        coef -= (n100 + 1) * n100 / 2.0
        final_result = pref * tf.exp(-coef)
        # Collect the value for the histogram collector in a tuple
        histogram_values = (xarr[:, hst_dim],)
        histogram_collector(final_result * weight, histogram_values)
        return final_result

    return integrand_example


if __name__ == "__main__":
    """Testing histogram generation"""
    print(f"Plain MC, ncalls={ncalls}:")
    start = time.time()
    # First we create the tensor in which to accumulate the histogra
    # This part is completely free
    current_histograms = tf.Variable(tf.zeros(HISTO_BINS, dtype=DTYPE))
    integrand_example = generate_integrand(current_histograms)
    mc_instance = PlainFlow(dim, ncalls)
    mc_instance.compile(integrand_example, compilable=True)
    # Pass the histogram variables to the integration
    # so it can be filled only once per iteration
    # This needs to be a tuple/list of tensor variables
    # as they will be emptied at the end of each iteration
    histogram_tuple = (current_histograms,)
    results = mc_instance.run_integration(n_iter, histograms=histogram_tuple)
    r = results[0]
    s = results[1]
    # At the end of the integration the variable `current_histograms` is filled
    # with the weighted accumulation of the histograms per iteration
    # while the result of the histogram each iteration can be accessed through
    # the history of the integrator
    results_per_iteration = mc_instance.history
    end = time.time()
    print(f"Plain took: time (s): {end-start}")
    print(f"Final result: {r} +/- {s}")
    print(f"Final histogram: {current_histograms}")
