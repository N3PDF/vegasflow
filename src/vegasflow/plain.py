"""
    Plain implementation of the plainest possible MonteCarlo
"""

from vegasflow.configflow import DTYPE, DTYPEINT, fone, fzero, float_me
from vegasflow.monte_carlo import MonteCarloFlow, wrapper
import tensorflow as tf


class PlainFlow(MonteCarloFlow):
    """
        Simple Monte Carlo integrator.
    """

    def _run_event(self, integrand, ncalls=None):
        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        # Jacobian
        xjac = 1.0 / self.n_events
        # Generate all random number for this iteration
        rnds = tf.random.uniform(
            (n_events, self.n_dim), minval=0, maxval=1, dtype=DTYPE
        )
        # Compute the integrand
        tmp = integrand(rnds, n_dim=self.n_dim, weight=xjac) * xjac
        tmp2 = tf.square(tmp)
        # Accumulate the current result
        res = tf.reduce_sum(tmp)
        res2 = tf.reduce_sum(tmp2)
        return res, res2

    def _run_iteration(self):
        res, raw_res2 = self.run_event()
        res2 = raw_res2 * self.n_events
        # Compute the error
        err_tmp2 = (res2 - tf.square(res)) / (self.n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))
        return res, sigma


def plain_wrapper(*args):
    return wrapper(PlainFlow, *args)


if __name__ == "__main__":
    import numpy as np
    import time

    @tf.function
    def lepage(xarr, n_dim=None):
        """Lepage test function"""
        if n_dim is None:
            n_dim = xarr.shape[-1]
        a = tf.constant(0.1, dtype=DTYPE)
        n100 = tf.cast(100 * n_dim, dtype=DTYPE)
        pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
        coef = tf.reduce_sum(tf.range(n100 + 1))
        coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
        coef -= (n100 + 1) * n100 / 2.0
        return pref * tf.exp(-coef)

    """Testing a basic integration"""
    ncalls = int(1e5)
    dim = 4
    n_iter = 4
    print(f"Plain MC, ncalls={ncalls}:")
    start = time.time()
    r = plain_wrapper(lepage, dim, n_iter, ncalls)
    end = time.time()
    print(f"time (s): {end-start}")
