"""
    Plain implementation of the plainest possible MonteCarlo
"""

import tensorflow as tf

from vegasflow.configflow import fone, fzero
from vegasflow.monte_carlo import MonteCarloFlow, sampler, wrapper


class PlainFlow(MonteCarloFlow):
    """
    Simple Monte Carlo integrator.
    """

    _CAN_RUN_VECTORIAL = True

    def _run_event(self, integrand, ncalls=None):
        if ncalls is None:
            n_events = self.n_events
        else:
            n_events = ncalls

        # Generate all random number for this iteration
        rnds, xjac = self._generate_random_array(n_events)

        # Compute the integrand
        tmp = integrand(rnds, weight=xjac) * xjac
        tmp2 = tf.square(tmp)

        # Accommodate multidimensional output by ensuring that only the event axis is accumulated
        res = tf.reduce_sum(tmp, axis=0)
        res2 = tf.reduce_sum(tmp2, axis=0)

        return res, res2

    def _run_iteration(self):
        res, raw_res2 = self.run_event()
        res2 = raw_res2 * self.n_events
        # Compute the error
        err_tmp2 = (res2 - tf.square(res)) / (self.n_events - fone)
        sigma = tf.sqrt(tf.maximum(err_tmp2, fzero))
        return res, sigma


def plain_wrapper(*args, **kwargs):
    """Wrapper around PlainFlow"""
    return wrapper(PlainFlow, *args, **kwargs)


def plain_sampler(*args, **kwargs):
    """Wrapper sampler around PlainFlow"""
    return sampler(PlainFlow, *args, **kwargs)
